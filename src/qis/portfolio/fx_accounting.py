"""Explicit self-financed equity, local-debt and FX-forward cash accounting.

Unlike a normalized payoff index, this ledger keeps spendable cash, funded units,
local financing and unsettled contracts separate. It extends the native funded
backtester for contracts that instruments_carry cannot represent; existing funded
backtest semantics are unchanged. All economic events occur on supplied dates.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from qis.fx.fx_rates_data import FxRatesData
from qis.fx.fx_returns import FxCashAccrual, FxCashPolicy
from qis.utils.df_ops import multiply_df_by_dt

__all__ = ["FxAccountSpec", "FxAccountData", "backtest_fx_account"]


@dataclass(frozen=True)
class FxAccountSpec:
    """Explicit financing, collateral and sizing choices for an FX account.

    Attributes:
        reference_ccy: Currency of capital, cash, contract payments and NAV.
        initial_capital: Positive initial collateral/capital, not exposure notional.
        financing: Local matched borrowing/deposits, or reference-cash asset funding.
        notional_rule: Size targets on fixed initial capital or current marked NAV.
        cash_rate_policy: Reference benchmark on signed cash, or explicitly no interest.
        missing_quotes: Reject missing marks, or carry valuation only without trading.
        cash_policy: PERIOD_OPENING with ACT/365F on the valuation grid; no
            provider-basis conversion. MONTHLY_INDEX is not supported by this ledger.
    """

    reference_ccy: str
    initial_capital: float
    financing: Literal["local", "reference"]
    notional_rule: Literal["fixed", "current_nav"]
    cash_rate_policy: Literal["reference", "zero"]
    missing_quotes: Literal["raise", "carry_valuation"] = "raise"
    cash_policy: FxCashPolicy = FxCashPolicy()

    def __post_init__(self) -> None:
        """Reject incomplete policies rather than choosing an implicit accounting contract."""
        if not isinstance(self.reference_ccy, str) or not self.reference_ccy.strip():
            raise ValueError("reference_ccy must be a currency ID")
        if not np.isfinite(self.initial_capital) or self.initial_capital <= 0:
            raise ValueError("Positive initial capital is required, even at zero net exposure")
        choices = {
            "financing": ("local", "reference"),
            "notional_rule": ("fixed", "current_nav"),
            "cash_rate_policy": ("reference", "zero"),
            "missing_quotes": ("raise", "carry_valuation"),
        }
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"Unknown {name}")
        if (not isinstance(self.cash_policy, FxCashPolicy) or self.cash_policy.quote_lag != 1
                or self.cash_policy.accrual is not FxCashAccrual.PERIOD_OPENING):
            raise ValueError("Account cash requires PERIOD_OPENING, opening quotes and ACT/365F")


@dataclass(frozen=True, init=False)
class FxAccountData:
    """Owned account histories and event receipts, in explicit monetary units.

    Each public field read makes a deep copy to preserve mutation isolation. For
    reporting loops, read once (for example, ``nav = account.nav``) and reuse that
    snapshot; the cost of each read is proportional to the field's size.

    Attributes:
        spec: Accounting policy, including initial reference capital.
        nav: Marked reference-currency capital, including unsettled receivables.
        accounts: Cash, asset, financing and contract values whose sum is NAV.
        pnl: Additive reference-currency P&L by economic leg, not percentage returns.
        asset_units: Equity units after each date's trades, held until the next trade.
        local_balances: Signed per-asset financing balances in that asset's currency.
        cash_flows: Settlements, asset trades and opening-cash interest in reference units.
        contracts: Executed strikes, signed local quantities, fixing and payment receipts.
        contract_values: Reference marks by contract; zero before trade and after payment.
        diagnostics: Stale-mark flags and accounting reconciliation errors.
        provenance: Reset, settlement, rate, financing and notional assumptions.
    """

    spec: FxAccountSpec
    _fields: dict

    def __init__(self, spec: FxAccountSpec, **fields) -> None:
        """Own the completed results; public reads return independent pandas objects."""
        object.__setattr__(self, "spec", spec)
        object.__setattr__(self, "_fields", deepcopy(fields))

    def __getattr__(self, name: str):
        """Return a copy of a documented result field without exposing internal state."""
        fields = object.__getattribute__(self, "_fields")
        if name in fields:
            return deepcopy(fields[name])
        raise AttributeError(name)

    def to_portfolio_data(self):
        """Export positive funded NAV for QIS reporting without adding cash a second time.

        Returns:
            NAV-only PortfolioData. Detailed monetary attribution stays in this ledger.

        Raises:
            ValueError: If capital is nonpositive; signed monetary P&L remains inspectable.
        """
        from qis.portfolio.portfolio_data import PortfolioData

        nav = self.nav
        if (nav <= 0).any():
            raise ValueError("Nonpositive capital cannot be reported as a funded return index")
        return PortfolioData(nav=nav)


def _dates(index: pd.Index, name: str) -> None:
    """Validate explicit chronological daily event boundaries without silent rounding."""
    if (
        not isinstance(index, pd.DatetimeIndex)
        or index.hasnans
        or not index.is_unique
        or not index.is_monotonic_increasing
        or index.tz is not None
        or not index.equals(index.normalize())
    ):
        raise ValueError(f"{name} needs unique ordered timezone-naive daily dates")


def _panel(value: pd.DataFrame, name: str, allow_empty_columns: bool = False) -> None:
    """Validate a nonempty numeric panel and its immutable label order."""
    if (
        not isinstance(value, pd.DataFrame)
        or value.index.empty
        or (value.columns.empty and not allow_empty_columns)
    ):
        raise ValueError(f"{name} must be a nonempty DataFrame")
    _dates(value.index, name)
    if not value.columns.is_unique or value.columns.hasnans:
        raise ValueError(f"{name} requires unique nonmissing columns")
    if np.isinf(value.to_numpy(dtype=float)).any():
        raise ValueError(f"{name} contains infinite values")


def _discount(rate: float, date: pd.Timestamp, end: pd.Timestamp) -> float:
    """Flat simple discount from the supplied current annual quote, ACT/365F."""
    gross = 1.0 + rate * (end - date).days / 365.0
    if not np.isfinite(gross) or gross <= 0:
        raise ValueError("Cash/forward discount gross must be positive")
    return 1.0 / gross


def _schedule(
    forwards: pd.DataFrame | None, dates: pd.DatetimeIndex, currencies: list[str], reference: str
) -> pd.DataFrame:
    """Validate explicit signed contract schedules, including dates beyond the sample end."""
    columns = [
        "trade_date",
        "fixing_date",
        "settlement_date",
        "currency",
        "notional_weight",
        "notional_basis",
    ]
    if forwards is None:
        return pd.DataFrame(columns=columns)
    if not isinstance(forwards, pd.DataFrame) or list(forwards.columns) != columns:
        raise ValueError(f"Forward columns must be {columns}, in this order")
    result = forwards.copy(deep=True)
    if not result.index.is_unique or result.index.hasnans:
        raise ValueError("Forward contract IDs must be unique and nonmissing")
    for name in columns[:3]:
        result[name] = pd.to_datetime(result[name])
        values = pd.DatetimeIndex(result[name])
        if values.hasnans or values.tz is not None or not values.equals(values.normalize()):
            raise ValueError("Forward dates must be explicit timezone-naive daily dates")
        if any(d < dates[0] or (d <= dates[-1] and d not in dates) for d in values):
            raise ValueError("Every in-sample forward event must occur on the valuation grid")
    if (result.fixing_date <= result.trade_date).any() or (
        result.settlement_date < result.fixing_date
    ).any():
        raise ValueError("Require trade < fixing <= settlement")
    if (
        not set(result.currency).issubset(set(currencies) - {reference})
        or not set(result.notional_basis).issubset({"capital", "asset_currency_value"})
        or not np.isfinite(result.notional_weight.astype(float)).all()
    ):
        raise ValueError("Invalid forward currency, notional basis or signed weight")
    result["notional_weight"] = result.notional_weight.astype(float)
    return result


# backtest_model_portfolio cannot carry local debt or zero-cost contracts with fixing/payment legs.
def backtest_fx_account(
    local_prices: pd.DataFrame,
    local_ccys: pd.Series,
    target_weights: pd.DataFrame,
    fx_data: FxRatesData,
    *,
    spec: FxAccountSpec,
    local_settlement_dates: pd.DatetimeIndex,
    spot_is_observed: pd.DataFrame,
    forwards: pd.DataFrame | None = None,
) -> FxAccountData:
    """Run a self-financed cash ledger for signed cash equities and cash-settled FX forwards.

    Args:
        local_prices: Local total-return levels on the valuation/cash-reset grid. Units
            are held between target dates; these are funded levels, never excess indices.
            An empty asset-column panel supports a pure forward/cash account.
        local_ccys: Exact static asset-to-quote-currency map. Resolve subunits, ADRs and
            currency changes upstream; a currency-spell DataFrame is deliberately rejected.
        target_weights: Signed asset notionals divided by the chosen capital base at each
            trade. Rows must be actual valuation dates; include the opening date explicitly.
            A zero row closes assets. Weights need not sum to zero or one.
        fx_data: Supplied USD-per-currency spots and annual rate benchmarks.
            FxRatesData spots are USD per currency; cross rates divide by reference spot.
        spec: Explicit capital, local/reference financing, sizing and cash policy.
        local_settlement_dates: Dates on which local asset-plus-financing P&L is swept to
            reference cash and financing reset to minus the held local asset value. Required
            explicitly, even when empty. Reference financing requires an empty schedule.
        spot_is_observed: Explicit boolean quote-availability panel on the valuation grid,
            covering every used currency. FxRatesData has already forward-filled spots;
            it cannot recover original missing-quote provenance. False permits valuation
            carry only under the selected policy, never execution or fixing.
        forwards: Optional contract table with columns trade_date, fixing_date,
            settlement_date, currency, notional_weight, notional_basis, and unique contract
            IDs. Positive weight buys local currency forward; negative hedges it. Capital
            sizing uses spec.notional_rule; asset_currency_value sizes on net marked equity
            exposure in that currency after same-date equity trades. Fixing/payment dates
            can differ. A new row is an explicit roll; no implicit early termination.

    Returns:
        Owned FxAccountData with cash, financing, unsettled marks, units, monetary P&L,
        cash transfers, contract receipts and exact reconciliation diagnostics.

    Raises:
        ValueError: On invalid labels/dates, unavailable execution/fixing quotes, missing
            rates, nonpositive discount factors or nonpositive capital at NAV-sized resets.

    The model permits symmetric borrowing/lending at supplied benchmarks and unrestricted
    reference collateral cash, without costs, haircuts, basis, external flows or margin calls.
    Before fixing, forwards use current flat simple CIP curves to fixing and reference
    discounting to payment. At fixing q*(spot-strike) becomes a nominal reference receivable;
    it earns no cash interest before payment. Event order is accrual, marking/fixing/payment,
    local settlement, equity trade, new forward. Reporting resampling never resets a contract.
    """
    if not isinstance(spec, FxAccountSpec):
        raise TypeError("spec must be FxAccountSpec")
    _panel(local_prices, "local_prices", allow_empty_columns=True)
    _panel(target_weights, "target_weights", allow_empty_columns=True)
    dates, assets = local_prices.index, local_prices.columns
    if len(dates) < 2:
        raise ValueError("Supply at least two valuation dates")
    if (
        not isinstance(local_ccys, pd.Series)
        or not local_ccys.index.equals(assets)
        or local_ccys.isna().any()
        or any(not isinstance(c, str) or not c.strip() for c in local_ccys)
    ):
        raise ValueError("local_ccys must be an exact static labelled currency map")
    if (
        not target_weights.columns.equals(assets)
        or target_weights.index[0] != dates[0]
        or not target_weights.index.isin(dates).all()
        or not np.isfinite(target_weights.to_numpy()).all()
    ):
        raise ValueError("Targets require exact assets, opening date and finite on-grid weights")
    _dates(local_settlement_dates, "local_settlement_dates")
    if not local_settlement_dates.isin(dates).all():
        raise ValueError("Local settlements must occur on the valuation grid")
    if spec.financing == "reference" and len(local_settlement_dates):
        raise ValueError("Reference financing has no local principal settlements")
    raw_spots, raw_rates = fx_data.fx_spots, fx_data.domestic_rates
    _panel(raw_spots, "fx_spots")
    _panel(raw_rates, "domestic_rates")
    currencies = list(dict.fromkeys([spec.reference_ccy, *local_ccys.tolist()]))
    if forwards is not None and "currency" in forwards:
        currencies = list(dict.fromkeys([*currencies, *forwards.currency.tolist()]))
    if not set(currencies).issubset(raw_spots) or not set(currencies).issubset(raw_rates):
        raise ValueError("Missing spot or rate currency")
    contracts = _schedule(forwards, dates, currencies, spec.reference_ccy)
    if (
        not isinstance(spot_is_observed, pd.DataFrame)
        or not spot_is_observed.index.equals(dates)
        or not spot_is_observed.columns.is_unique
        or not set(currencies).issubset(spot_is_observed.columns)
        or spot_is_observed[currencies].isna().any().any()
        or not all(pd.api.types.is_bool_dtype(t) for t in spot_is_observed[currencies].dtypes)
    ):
        raise ValueError("Supply exact-grid boolean spot_is_observed for all used currencies")
    observed = spot_is_observed[currencies].copy()
    spot_quotes = raw_spots.reindex(dates)[currencies].where(observed)
    if (spot_quotes <= 0).any().any() or (local_prices <= 0).any().any():
        raise ValueError("Price and spot marks must be positive when observed")
    price_missing, spot_missing = local_prices.isna(), spot_quotes.isna()
    if spec.missing_quotes == "raise" and (price_missing.any().any() or spot_missing.any().any()):
        raise ValueError("Missing valuation quote")
    prices = local_prices.ffill()
    spots = spot_quotes.ffill()
    if prices.isna().any().any() or spots.isna().any().any():
        raise ValueError("Missing opening valuation quote; no future backfill")
    spots = spots.div(spots[spec.reference_ccy], axis=0)
    rates = raw_rates[currencies].reindex(dates, method="ffill")
    if rates.isna().any().any():
        raise ValueError("Missing as-of rate quote; no future backfill")
    accruals = multiply_df_by_dt(
        raw_rates[currencies],
        dates=dates,
        lag=1,
        is_actual_calendar_dt=True,
        annualization_factor=365.0,
    )
    if not np.isfinite(accruals.to_numpy()).all() or (accruals <= -1.0).any().any():
        raise ValueError("Invalid opening cash accrual")
    mapped = np.array([currencies.index(c) for c in local_ccys], dtype=int)
    reference = spec.reference_ccy
    n = len(assets)
    units, loans = np.zeros(n), np.zeros(n)
    cash = float(spec.initial_capital)
    prior_nav, prior_fwd = cash, 0.0
    account_rows, pnl_rows, flow_rows, diagnostic_rows = [], [], [], []
    units_rows, loans_rows, contract_rows = [], [], []
    active, receipts = {}, []
    trades_by_date = {date: list(group.iterrows())
                      for date, group in contracts.groupby("trade_date", sort=False)}
    spot_values = spots.to_numpy(dtype=float)
    currency_positions = {ccy: position for position, ccy in enumerate(spots.columns)}
    for step, date in enumerate(dates):
        p = prices.iloc[step].to_numpy(dtype=float)
        s = spot_values[step]
        asset_spot = s[mapped]
        r = rates.iloc[step]
        period = accruals.iloc[step]
        flows = dict(
            cash_interest=0.0, local_settlement=0.0, equity_trades=0.0, forward_settlement=0.0
        )
        pnl = dict(
            asset_price_fx=0.0,
            financing_interest=0.0,
            financing_fx=0.0,
            forward=0.0,
            cash_interest=0.0,
        )
        if step:
            old_p = prices.iloc[step - 1].to_numpy(dtype=float)
            old_s = spot_values[step - 1, mapped]
            pnl["asset_price_fx"] = float(np.sum(units * (p * asset_spot - old_p * old_s)))
            pnl["financing_fx"] = float(np.sum(loans * (asset_spot - old_s)))
            interest = loans * period.iloc[mapped].to_numpy()
            pnl["financing_interest"] = float(np.sum(interest * asset_spot))
            loans += interest
            if spec.cash_rate_policy == "reference":
                flows["cash_interest"] = cash * period[reference]
                cash += flows["cash_interest"]
                pnl["cash_interest"] = flows["cash_interest"]
        marks = dict.fromkeys(contracts.index, 0.0)
        for identity, contract in active.items():
            row = contract["schedule"]
            ccy = row.currency
            if date == row.fixing_date:
                if spot_missing.loc[date, [ccy, reference]].any():
                    raise ValueError(f"Missing FX fixing quote for {identity}")
                contract["fixed_payoff"] = contract["quantity"] * (
                    s[currency_positions[ccy]] - contract["strike"]
                )
            if date >= row.settlement_date:
                if not contract["paid"]:
                    amount = contract["fixed_payoff"]
                    cash += amount
                    flows["forward_settlement"] += amount
                    contract["paid"] = True
            elif date >= row.fixing_date:
                marks[identity] = contract["fixed_payoff"] * _discount(
                    r[reference], date, row.settlement_date
                )
            else:
                fair_forward = (
                    s[currency_positions[ccy]]
                    * _discount(r[ccy], date, row.fixing_date)
                    / _discount(r[reference], date, row.fixing_date)
                )
                marks[identity] = (
                    contract["quantity"]
                    * (fair_forward - contract["strike"])
                    * (_discount(r[reference], date, row.settlement_date))
                )
        fwd_value = sum(marks.values())
        pnl["forward"] = fwd_value + flows["forward_settlement"] - prior_fwd
        nav_before = cash + float(np.sum(units * p * asset_spot + loans * asset_spot)) + fwd_value
        if date in local_settlement_dates:
            affected = (units != 0.0) | (loans != 0.0)
            missing_fx = spot_missing.loc[date, local_ccys.to_numpy()].to_numpy()
            if (
                price_missing.iloc[step].to_numpy()[affected].any()
                or missing_fx[affected].any()
                or spot_missing.loc[date, reference]
            ):
                raise ValueError("Cannot settle local financing using stale price/FX marks")
            flows["local_settlement"] = float(np.sum((units * p + loans) * asset_spot))
            cash += flows["local_settlement"]
            loans = -units * p
        if date in target_weights.index:
            weights = target_weights.loc[date].to_numpy(dtype=float)
            base = spec.initial_capital if spec.notional_rule == "fixed" else nav_before
            if spec.notional_rule == "current_nav" and base <= 0:
                raise ValueError("Cannot size an equity reset from nonpositive capital")
            desired = base * weights / (p * asset_spot)
            changes = desired - units
            affected = changes != 0.0
            missing_fx = spot_missing.loc[date, local_ccys.to_numpy()].to_numpy()
            if (
                price_missing.iloc[step].to_numpy()[affected].any()
                or missing_fx[affected].any()
                or spot_missing.loc[date, reference]
            ):
                raise ValueError("Cannot trade equities using stale price/FX marks")
            if spec.financing == "local":
                loans -= changes * p
            else:
                flows["equity_trades"] = -float(np.sum(changes * p * asset_spot))
                cash += flows["equity_trades"]
            units = desired
        for identity, row in trades_by_date.get(date, ()):
            ccy = row.currency
            if spot_missing.loc[date, [ccy, reference]].any():
                raise ValueError("Cannot open a forward using stale FX marks")
            if row.notional_basis == "capital":
                base = spec.initial_capital if spec.notional_rule == "fixed" else nav_before
                if spec.notional_rule == "current_nav" and base <= 0:
                    raise ValueError("Cannot size a forward reset from nonpositive capital")
            else:
                selected = local_ccys.eq(ccy).to_numpy()
                if price_missing.iloc[step].to_numpy()[selected & (units != 0.0)].any():
                    raise ValueError("Cannot size a principal hedge on stale equity marks")
                base = float(np.sum(units[selected] * p[selected] * asset_spot[selected]))
            quantity = row.notional_weight * base / s[currency_positions[ccy]]
            strike = (
                s[currency_positions[ccy]]
                * _discount(r[ccy], date, row.fixing_date)
                / (_discount(r[reference], date, row.fixing_date))
            )
            active[identity] = dict(
                schedule=row, quantity=quantity, strike=strike, fixed_payoff=np.nan, paid=False
            )
            receipts.append(
                dict(
                    contract_id=identity,
                    trade_date=date,
                    fixing_date=row.fixing_date,
                    settlement_date=row.settlement_date,
                    currency=ccy,
                    reference_notional=quantity * s[currency_positions[ccy]],
                    local_quantity=quantity,
                    strike=strike,
                )
            )
        asset_value = float(np.sum(units * p * asset_spot))
        financing_value = float(np.sum(loans * asset_spot))
        nav = cash + asset_value + financing_value + fwd_value
        error = nav - prior_nav - sum(pnl.values())
        scale = max(1.0, abs(nav), abs(cash), abs(asset_value), abs(financing_value))
        if abs(error) > 1e-10 * scale or abs(nav - nav_before) > 1e-10 * scale:
            raise AssertionError("Self-financing identity failed")
        account_rows.append(
            dict(
                cash=cash,
                asset_value=asset_value,
                local_financing_value=financing_value,
                forward_value=fwd_value,
                nav=nav,
            )
        )
        pnl_rows.append(pnl)
        flow_rows.append(flows)
        diagnostic_rows.append(
            dict(
                reconciliation_error=error,
                stale_price_marks=int(price_missing.iloc[step].sum()),
                stale_fx_marks=int(spot_missing.iloc[step].sum()),
            )
        )
        units_rows.append(units.copy())
        loans_rows.append(loans.copy())
        contract_rows.append(marks)
        prior_nav, prior_fwd = nav, fwd_value
    executed = pd.DataFrame(receipts)
    if not executed.empty:
        executed = executed.set_index("contract_id")
        executed["fixed_payoff"] = [active[i]["fixed_payoff"] for i in executed.index]
        executed["paid"] = [active[i]["paid"] for i in executed.index]
    accounts = pd.DataFrame(account_rows, index=dates)
    return FxAccountData(
        spec,
        nav=accounts.nav.rename("FX account"),
        accounts=accounts,
        pnl=pd.DataFrame(pnl_rows, index=dates),
        cash_flows=pd.DataFrame(flow_rows, index=dates),
        asset_units=pd.DataFrame(units_rows, index=dates, columns=assets),
        local_balances=pd.DataFrame(loans_rows, index=dates, columns=assets),
        contracts=executed,
        contract_values=pd.DataFrame(contract_rows, index=dates, columns=contracts.index),
        diagnostics=pd.DataFrame(diagnostic_rows, index=dates),
        provenance=dict(
            quote="reference units per local currency; FxRatesData USD crosses",
            asset_input="funded local total-return levels; embedded hedges remain in input",
            local_ccys=local_ccys.copy(),
            cash_policy=spec.cash_policy,
            cash_reset_dates=dates.copy(),
            local_settlement_dates=local_settlement_dates.copy(),
            target_dates=target_weights.index.copy(),
            forward_schedule=contracts.copy(),
            borrowing="symmetric supplied benchmark; unrestricted signed cash; no external flows",
            collateral="reference cash; no haircuts, segregated margin or margin calls",
            forward_valuation="flat simple CIP to fixing; reference discount to payment",
            settlement="fixed nominal reference payoff; no cash interest before payment",
            costs="no transaction costs, borrow fees or cross-currency basis",
            rate_provenance=spec.cash_policy.rate_provenance,
            spot_is_observed=observed,
        ),
    )

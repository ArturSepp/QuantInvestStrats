"""Typed FX return requests and owned results, independent of synthetic wealth indices.

Funded returns and signed payoffs per opening reference notional have distinct
coordinate contracts. Synthetic capital is optional and must be explicitly normalized.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd

from qis.perfstats.returns import to_returns, returns_to_nav
from qis.fx.fx_hedging import (
    compute_local_and_fx_return,
    _hedged_simple_return,
    _compute_forward_hedge_cost,
)

from qis.fx.fx_rates_data import FxRatesData
from qis.fx.cash_index import compute_monthly_cash_index


class FxReturnType(str, Enum):
    """Economic object, independently of its numerical coordinate.

    Attributes:
        LOCAL_TOTAL: Funded asset total return in its quote currency.
        LOCAL_EXCESS: Local total return relative to local cash.
        REFERENCE_TOTAL: Principal-hedged funded asset return in reference currency.
        REFERENCE_EXCESS: Reference total return relative to reference cash.
        FX_SPOT: Change in reference-per-local spot; no carry or funding.
        LOCAL_EXCESS_TRANSLATED: Locally financed asset P&L per opening reference notional.
        FX_FORWARD: Long-local forward P&L per opening reference value of local notional.
    """

    LOCAL_TOTAL = "local_total"
    LOCAL_EXCESS = "local_excess"
    REFERENCE_TOTAL = "reference_total"
    REFERENCE_EXCESS = "reference_excess"
    FX_SPOT = "fx_spot"
    LOCAL_EXCESS_TRANSLATED = "local_excess_translated"
    FX_FORWARD = "fx_forward"


class FxReturnCoordinate(str, Enum):
    """Coordinate with a kind-dependent, explicitly validated meaning.

    Attributes:
        SIMPLE: Simple total/spot return, arithmetic excess, or signed P&L per notional.
        CASH_RELATIVE_SIMPLE: Excess only: (1+R)/(1+cash)-1.
        LOG: Log total/spot return, or log1p(R)-log1p(cash) for excess.
    """

    SIMPLE = "simple"
    CASH_RELATIVE_SIMPLE = "cash_relative_simple"
    LOG = "log"


class FxCashAccrual(str, Enum):
    """Cash measurement account, independent of return and hedge frequencies.

    Attributes:
        PERIOD_OPENING: Opening quote held over each requested return interval.
        MONTHLY_INDEX: One calendar month-end-reset ACT/365F cash index per currency.
    """

    PERIOD_OPENING = "period_opening"
    MONTHLY_INDEX = "monthly_index"


@dataclass(frozen=True)
class FxCashPolicy:
    """Cash accrual policy; quote provenance does not imply provider normalization.

    Attributes:
        quote_lag: Boundary lag; 1 is opening, 0 is closing descriptive analytics.
        day_count: ACT/365 Fixed only; source annual decimals remain unchanged.
        rate_provenance: Source/normalization description of the annual input quotes.
        accrual: Excess-return measurement only; never changes forward pricing or financing.
    """

    quote_lag: int = 1
    day_count: str = "ACT/365F"
    rate_provenance: str = "FxRatesData.domestic_rates; annual decimals; provider basis unchanged"

    accrual: FxCashAccrual = FxCashAccrual.PERIOD_OPENING

    def __post_init__(self):
        """Reject unsupported day counts, missing lineage and invalid boundary lags."""
        if not isinstance(self.accrual, FxCashAccrual):
            raise TypeError("accrual must be FxCashAccrual")
        if self.accrual is FxCashAccrual.MONTHLY_INDEX and self.quote_lag != 1:
            raise ValueError("MONTHLY_INDEX requires opening quotes (quote_lag=1)")
        if self.day_count != "ACT/365F":
            raise ValueError("Only ACT/365F cash is supported")
        if (
            isinstance(self.quote_lag, bool)
            or not isinstance(self.quote_lag, int)
            or self.quote_lag < 0
        ):
            raise ValueError("quote_lag must be a nonnegative integer")
        if not self.rate_provenance.strip():
            raise ValueError("rate_provenance must describe the supplied annual quotes")


@dataclass(frozen=True)
class FxIndexNormalization:
    """Explicit synthetic index normalization, not a self-financing account engine.

    Attributes:
        initial_value: Positive index/capital at the boundary before the first payoff.
        notional_to_capital: Opening payoff notional divided by capital, strictly positive.
        notional_rule: Capital-proportional compounds; fixed accumulates initial-capital P&L.
        settlement_frequency: Must equal the result cadence; no implicit daily settlement.
        reinvestment: Reinvest each interval for capital-proportional; none for fixed.
        cash_treatment: As represented in the return object; no cash is added a second time.
        missing_returns: Explicitly skip gaps or stop at the first missing post-inception payoff.
    """

    initial_value: float
    notional_to_capital: float
    notional_rule: Literal["capital_proportional", "fixed"]
    settlement_frequency: str
    reinvestment: Literal["each_interval", "none"]
    cash_treatment: Literal["as_returned"]
    missing_returns: Literal["skip", "stop"]

    def __post_init__(self):
        """Reject unspecified or contradictory sizing and settlement policies."""
        if not np.isfinite(self.initial_value) or self.initial_value <= 0:
            raise ValueError("initial_value must be finite and positive")
        if not np.isfinite(self.notional_to_capital) or self.notional_to_capital <= 0:
            raise ValueError("notional_to_capital must be finite and positive")
        expected = {"capital_proportional": "each_interval", "fixed": "none"}
        if self.notional_rule not in expected or self.reinvestment != expected[self.notional_rule]:
            raise ValueError("notional_rule and reinvestment are inconsistent")
        if self.cash_treatment != "as_returned" or self.missing_returns not in ("skip", "stop"):
            raise ValueError("Declare cash_treatment=as_returned and missing_returns=skip or stop")
        pd.tseries.frequencies.to_offset(self.settlement_frequency)


@dataclass(frozen=True)
class FxAssetReturnsRequest:
    """Asset-panel request with one economic kind and one coordinate.

    Attributes:
        prices: Native price levels, with unique dates and asset columns.
        local_ccys: Currency ID, asset Series, or dated currency-spell DataFrame.
        return_type: Local/reference total/excess or LOCAL_EXCESS_TRANSLATED.
        coordinate: SIMPLE, LOG, or excess-only CASH_RELATIVE_SIMPLE; P&L requires SIMPLE.
        reference_ccy: Required for reference returns and translated P&L; absent for local.
        hedge_ratios: Additional principal hedge; scalar, asset Series or dated DataFrame.
            Must be zero or missing for local returns and LOCAL_EXCESS_TRANSLATED; missing
            means no additional hedge on these kinds. Reference-currency assets have no FX
            exposure: their effective opening hedge is zero, also in opening_hedge_ratios.
            Missing foreign-asset hedges still mask reference-return observations.
        frequency: Reset/return cadence, scalar or per-asset Series.
        cash_policy: Cash quote timing, day count and provenance.
        zero_return_to_nan: Explicit observation filter, applied after all calculations.
        embedded_hedge_policy: Prices include existing hedges; supplied hedges are additional.
        currency_changes: Reject currency changes or explicitly mask crossing intervals.
        diagnostics: Include compact currency-level cash, spot and hedge-cost legs.
        normalization: Optional synthetic index specification; absent by default.
    """

    prices: pd.DataFrame
    local_ccys: str | pd.Series | pd.DataFrame
    return_type: FxReturnType
    coordinate: FxReturnCoordinate
    reference_ccy: str | None = None
    hedge_ratios: float | pd.Series | pd.DataFrame = 0.0
    frequency: str | pd.Series = "ME"
    cash_policy: FxCashPolicy = field(default_factory=FxCashPolicy)
    zero_return_to_nan: bool = False
    embedded_hedge_policy: Literal[
        "prices_include_existing_hedges; ratios_are_additional"
    ] = "prices_include_existing_hedges; ratios_are_additional"
    currency_changes: Literal["reject", "mask_boundary"] = "reject"
    diagnostics: bool = False
    normalization: FxIndexNormalization | None = None

    def __post_init__(self):
        """Own the supplied pandas inputs so later caller edits cannot alter the request."""
        for name in ("prices", "local_ccys", "hedge_ratios", "frequency"):
            value = getattr(self, name)
            if isinstance(value, (pd.DataFrame, pd.Series)):
                object.__setattr__(self, name, value.copy(deep=True))


@dataclass(frozen=True)
class FxCurrencyReturnsRequest:
    """Currency-panel request, with no invented asset price panel.

    Attributes:
        currencies: Distinct currency IDs to quote against reference_ccy.
        reference_ccy: Reference denomination of each spot.
        return_type: FX_SPOT or exact FX_FORWARD under matched-period synthetic CIP.
        coordinate: SIMPLE or LOG for spot; forward P&L requires SIMPLE.
        frequency: Spot observation/return cadence.
        diagnostics: Include compact spot diagnostics.
        normalization: Optional explicit normalized index; no added cash or collateral.
        cash_policy: Annual rate provenance and ACT/365F; forwards require opening quotes.
    """

    currencies: tuple[str, ...]
    reference_ccy: str
    return_type: FxReturnType = FxReturnType.FX_SPOT
    coordinate: FxReturnCoordinate = FxReturnCoordinate.SIMPLE
    frequency: str = "ME"
    diagnostics: bool = False
    normalization: FxIndexNormalization | None = None
    cash_policy: FxCashPolicy = field(default_factory=FxCashPolicy)

    def __post_init__(self):
        """Own currency IDs as an immutable tuple."""
        object.__setattr__(self, "currencies", tuple(self.currencies))


@dataclass(frozen=True)
class FxReturnsData:
    """Owned return values and their economic/timing contract.

    Attributes:
        return_type: Economic kind, independent of representation.
        coordinate: Numerical coordinate of values.
        frequency: Actual reset/return cadence; periods records its boundaries.
        reference_ccy: Reference denomination, or None for native local returns.
        cash_policy: Annual quote lineage, boundary lag and ACT/365F convention.
        embedded_hedge_policy: Meaning of the effective additional hedge ratios.
        normalization: Explicit index policy, or None when no index was requested.
        values: Copy of the date-by-asset/currency return panel.
        periods: Copy of opening/closing interval boundaries, end-labelled.
        local_ccys: Copy of as-of currency-spell metadata on the result grid.
        currency_spells: Copy of the original dated currency history, including off-grid changes.
        opening_hedge_ratios: Copy of effective opening principal hedges.
        diagnostics: Optional owned currency-level leg copies.
        synthetic_index: Optional compounded index or accumulated capital-plus-P&L.
        index_kind: None, compounded_index, or accumulated_pnl_with_initial_capital.
        payoff_basis: Denominator and funding meaning of the returned numbers.
    """

    return_type: FxReturnType
    coordinate: FxReturnCoordinate
    frequency: str
    reference_ccy: str | None
    cash_policy: FxCashPolicy
    embedded_hedge_policy: str
    normalization: FxIndexNormalization | None
    _values: pd.DataFrame = field(repr=False)
    _local_ccys: pd.DataFrame = field(repr=False)
    _opening_hedge_ratios: pd.DataFrame = field(repr=False)
    _diagnostics: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)
    _currency_spells: pd.DataFrame | None = field(default=None, repr=False)
    _synthetic_index: pd.DataFrame | None = field(default=None, repr=False, init=False)

    def __post_init__(self):
        """Own every mutable result and construct an index only when explicitly requested."""
        for name in ("_values", "_local_ccys", "_opening_hedge_ratios"):
            object.__setattr__(self, name, getattr(self, name).copy(deep=True))
        object.__setattr__(
            self,
            "_diagnostics",
            {key: value.copy(deep=True) for key, value in self._diagnostics.items()},
        )
        spells = self._local_ccys if self._currency_spells is None else self._currency_spells
        object.__setattr__(self, "_currency_spells", spells.copy(deep=True))
        if self.normalization is not None:
            object.__setattr__(self, "_synthetic_index", self.to_index(self.normalization))

    @property
    def values(self) -> pd.DataFrame:
        """Return an owned copy, never the internal mutable panel."""
        return self._values.copy(deep=True)

    @property
    def local_ccys(self) -> pd.DataFrame:
        """Return effective quote-currency spells on the valuation grid."""
        return self._local_ccys.copy(deep=True)

    @property
    def currency_spells(self) -> pd.DataFrame:
        """Return owned original dated spells, retaining intra-interval changes for audit."""
        return self._currency_spells.copy(deep=True)

    @property
    def opening_hedge_ratios(self) -> pd.DataFrame:
        """Return effective opening hedges; a closing change affects the next interval."""
        return self._opening_hedge_ratios.copy(deep=True)

    @property
    def periods(self) -> pd.DataFrame:
        """Return the actual reset boundaries, with no opening for the initial valuation."""
        dates = self._values.index
        return pd.DataFrame(
            {"opening": pd.Series(dates, index=dates).shift(1), "closing": dates}, index=dates
        )

    @property
    def diagnostics(self) -> dict[str, pd.DataFrame]:
        """Return copies of requested compact currency legs."""
        return {key: value.copy(deep=True) for key, value in self._diagnostics.items()}

    @property
    def synthetic_index(self) -> pd.DataFrame | None:
        """Return the optional explicitly normalized index, otherwise None."""
        return None if self._synthetic_index is None else self._synthetic_index.copy(deep=True)

    @property
    def index_kind(self) -> str | None:
        """Distinguish compounded wealth from fixed-notional accumulated capital and P&L."""
        if self.normalization is None:
            return None
        return (
            "compounded_index"
            if self.normalization.notional_rule == "capital_proportional"
            else "accumulated_pnl_with_initial_capital"
        )

    @property
    def payoff_basis(self) -> str:
        """Describe the denominator without asserting a return on zero initial capital."""
        currency = self.reference_ccy or "each asset local currency"
        if self.return_type is FxReturnType.LOCAL_EXCESS_TRANSLATED:
            return (
                f"locally financed asset P&L per unit opening reference notional in {currency}; "
                "local debt repaid at settlement; no added reference funding"
            )
        if self.return_type is FxReturnType.FX_FORWARD:
            return (
                f"long-local forward P&L per unit opening reference value of local notional "
                f"in {currency}; no initial capital, collateral or added funding implied"
            )
        if self.return_type in _EXCESS:
            if self.coordinate is FxReturnCoordinate.SIMPLE:
                return f"arithmetic asset-minus-cash payoff per unit opening capital in {currency}"
            return f"asset wealth relative to cash growth in {currency}; no added funding"
        if self.return_type is FxReturnType.FX_SPOT:
            return f"spot change per unit opening converted currency value in {currency}; no carry"
        return f"funded asset total return per unit opening capital in {currency}"

    def to_index(self, normalization: FxIndexNormalization) -> pd.DataFrame:
        """Construct a synthetic index under explicit sizing, cash and missing-data policies.

        Args:
            normalization: Full capital/notional/settlement specification. Cash is already
                included or subtracted according to return_type; this never adds funding.

        Returns:
            Compounded index or initial capital plus accumulated fixed-notional P&L.

        Raises:
            ValueError: Cadences differ or compounded gross increments are nonpositive.
        """
        if normalization.settlement_frequency != self.frequency:
            raise ValueError("settlement_frequency must equal the result frequency")
        values = (
            np.expm1(self._values)
            if self.coordinate is FxReturnCoordinate.LOG
            else self._values.copy()
        )
        values = values * normalization.notional_to_capital
        fixed = normalization.notional_rule == "fixed"
        if not fixed and (values <= -1).any().any():
            raise ValueError(
                "Compounded index requires strictly positive normalized gross increments"
            )
        # The existing QIS NAV primitive owns compounding and ragged inception behavior.
        index = returns_to_nav(
            values,
            is_log_returns=False,
            init_period=0,
            constant_trade_level=fixed,
            ffill_between_nans=normalization.missing_returns == "skip",
        )
        index *= normalization.initial_value
        if normalization.missing_returns == "stop":
            started = values.notna().cummax()
            unknown = (started & values.isna()).cummax()
            index = index.mask(unknown)
        return index


_EXCESS = (FxReturnType.LOCAL_EXCESS, FxReturnType.REFERENCE_EXCESS)
_REFERENCE = (FxReturnType.REFERENCE_TOTAL, FxReturnType.REFERENCE_EXCESS)
_NOTIONAL_PAYOFFS = (FxReturnType.LOCAL_EXCESS_TRANSLATED, FxReturnType.FX_FORWARD)
_CURRENCY_KINDS = (FxReturnType.FX_SPOT, FxReturnType.FX_FORWARD)


def _dated(obj: pd.DataFrame | pd.Series, name: str) -> None:
    """Validate unique ordered timestamp axes without changing observation policies."""
    if (
        not isinstance(obj.index, pd.DatetimeIndex)
        or obj.index.hasnans
        or not obj.index.is_unique
        or not obj.index.is_monotonic_increasing
    ):
        raise ValueError(f"{name} requires ordered unique nonmissing timestamps")


def _asof(obj: pd.Series | pd.DataFrame, dates: pd.DatetimeIndex):
    """Retain off-grid updates before sampling, without reading future observations."""
    _dated(obj, "dated metadata")
    return obj.ffill().reindex(dates, method="ffill")


def _metadata(value, columns, dates, name: str) -> pd.DataFrame:
    """Broadcast static metadata or as-of align dated values with exact asset coverage."""
    if isinstance(value, pd.DataFrame):
        if not value.columns.is_unique or set(columns) - set(value.columns):
            raise ValueError(f"{name} must cover every asset exactly once")
        return _asof(value.loc[:, columns], dates)
    if isinstance(value, pd.Series):
        if not value.index.is_unique or set(columns) - set(value.index):
            raise ValueError(f"{name} must cover every asset exactly once")
        return pd.DataFrame({asset: value.loc[asset] for asset in columns}, index=dates)
    return pd.DataFrame(value, index=dates, columns=columns)


def _excess_cash(data, dates, currency, frequency, policy):
    """Sample the measurement account without changing any contracted hedge cash."""
    if policy.accrual is FxCashAccrual.PERIOD_OPENING:
        return data._get_period_cash_returns(dates, currency, frequency, False, policy.quote_lag)
    index = compute_monthly_cash_index(data, currency, dates)
    cash = index / index.shift(1) - 1.0
    # Direct one-month accrual avoids cancellation and retains exact monthly parity.
    if len(dates) > 1:
        opening = dates[:-1]
        closing = dates[1:]
        single_month = opening.is_month_end & (closing == opening + pd.offsets.MonthEnd(1))
        native = data._get_period_cash_returns(dates, currency, frequency, False, 1)
        selected = closing[single_month]
        cash.loc[selected] = native.loc[selected].where(index.loc[selected].notna())
    if len(cash):
        cash.iloc[0] = 0.0
    return cash


def _coordinate(
    total: pd.Series,
    cash: pd.Series,
    kind: FxReturnType,
    coordinate: FxReturnCoordinate,
    log_total: pd.Series | None = None,
) -> pd.Series:
    """Transform a completed total payoff; never take log1p of arithmetic excess."""
    if coordinate is FxReturnCoordinate.LOG:
        if (total <= -1).any():
            raise ValueError("Log returns require strictly positive terminal wealth")
        log_total = np.log1p(total) if log_total is None else log_total
        return log_total - np.log1p(cash) if kind in _EXCESS else log_total
    if kind in _EXCESS:
        return (
            (total - cash)
            if coordinate is FxReturnCoordinate.SIMPLE
            else (1.0 + total) / (1.0 + cash) - 1.0
        )
    return total


def _asset_result(
    data: FxRatesData, request: FxAssetReturnsRequest, frequency: str, assets: list
) -> FxReturnsData:
    """Compute one cadence using the existing price and principal-hedge kernels."""
    prices = request.prices.loc[:, assets]
    # Metadata dates need the union for dated switches and off-grid hedge decisions.
    currency_source = _metadata(request.local_ccys, assets, prices.index, "local_ccys")
    if currency_source.isna().any().any():
        raise ValueError("local_ccys must be known at every supplied price date")
    outputs, currencies, hedges = {}, {}, {}
    for asset in assets:
        spells = (
            request.local_ccys[asset].ffill()
            if isinstance(request.local_ccys, pd.DataFrame)
            else currency_source[asset]
        )
        kinds = spells.dropna().unique()
        if len(kinds) > 1 and request.currency_changes == "reject":
            raise ValueError("Currency spell changes require currency_changes=mask_boundary")
        pieces, spell_hedges, spell_currencies = [], [], []
        for local in kinds:
            if not isinstance(local, str) or not local.strip():
                raise ValueError("local_ccys must contain nonblank currency IDs")
            foreign = (
                request.return_type in _REFERENCE
                or request.return_type is FxReturnType.LOCAL_EXCESS_TRANSLATED
            ) and local != request.reference_ccy
            if foreign:
                cross = data.get_local_to_reference_fx_rate(local, request.reference_ccy)
                total, spot = compute_local_and_fx_return(prices[asset], cross, frequency, False)
            else:
                total = to_returns(prices[asset], freq=frequency, is_log_returns=False)
                spot = pd.Series(0.0, index=total.index)
            dates = total.index
            ccy = _metadata(request.local_ccys, [asset], dates, "local_ccys")[asset]
            h = _metadata(request.hedge_ratios, [asset], dates, "hedge_ratios")[asset].shift(1)
            if h.dropna().map(lambda value: not np.isfinite(value)).any():
                raise ValueError("hedge_ratios must be finite or missing")
            if request.return_type is FxReturnType.LOCAL_EXCESS_TRANSLATED:
                # compute_futures_fx_adjusted_returns consumes price gains. Reconstructing
                # cash-relative excess NAVs for that kernel would replace arithmetic R-c by q.
                cash = data._get_period_cash_returns(dates, local, frequency, False, 1)
                result = (total - cash) * (1.0 + spot)
                h = pd.Series(0.0, index=dates)
            else:
                if foreign:
                    # CIP always uses opening quotes, independently of descriptive excess lag.
                    lc = data._get_period_cash_returns(dates, local, frequency, False, 1)
                    rc = data._get_period_cash_returns(
                        dates, request.reference_ccy, frequency, False, 1
                    )
                    premium = (1.0 + lc) / (1.0 + rc) - 1.0
                    # Preserve the validated log-premium evaluation order. Algebraically identical
                    # rounding changes can propagate through constrained fitted/portfolio solvers.
                    log_premium = request.coordinate is FxReturnCoordinate.LOG
                    priced_premium = np.log1p(premium) if log_premium else premium
                    total = _hedged_simple_return(
                        total, spot, h, priced_premium, is_log_premium=log_premium
                    )
                    log_total = None
                else:
                    # Preserve native log-level differencing and rounding for same-currency paths.
                    log_total = to_returns(prices[asset], freq=frequency, is_log_returns=True)
                    h = pd.Series(0.0, index=dates)
                cash_ccy = request.reference_ccy if request.return_type in _REFERENCE else local
                cash = (
                    _excess_cash(data, dates, cash_ccy, frequency, request.cash_policy)
                    if request.return_type in _EXCESS
                    else pd.Series(0.0, index=dates)
                )
                result = _coordinate(
                    total, cash, request.return_type, request.coordinate, log_total
                )
            if len(kinds) > 1:
                # Never infer a currency-rebasing factor from two differently denominated prices.
                changes = spells.ne(spells.shift()).astype(int).cumsum()
                count = _asof(changes, dates)
                valid_spell = ccy.eq(local) & count.eq(count.shift(1))
                result = result.where(valid_spell)
                h = h.where(valid_spell)
            pieces.append(result.rename(asset))
            spell_currencies.append(ccy)
            spell_hedges.append(h.where(ccy.eq(local)))
        outputs[asset] = pd.concat(pieces, axis=1, sort=True).sum(axis=1, min_count=1)
        currencies[asset] = spell_currencies[0]
        hedges[asset] = pd.concat(spell_hedges, axis=1, sort=True).sum(axis=1, min_count=1)
    values = pd.DataFrame(outputs)
    if request.zero_return_to_nan:
        values = values.replace(0.0, np.nan)
    diagnostics = {}
    if request.diagnostics:
        ids = list(pd.unique(pd.DataFrame(currencies).to_numpy().ravel()))
        ids = [ccy for ccy in ids if pd.notna(ccy)]
        if request.reference_ccy is not None:
            ids = list(dict.fromkeys([*ids, request.reference_ccy]))
        diagnostics["cash"] = pd.DataFrame(
            {
                ccy: _excess_cash(data, values.index, ccy, frequency, request.cash_policy)
                for ccy in ids
            }
        )
        if request.reference_ccy is not None:
            diagnostics["spot"] = pd.DataFrame(
                {
                    ccy: to_returns(
                        data.get_local_to_reference_fx_rate(ccy, request.reference_ccy),
                        freq=frequency,
                        is_log_returns=False,
                    ).reindex(values.index)
                    for ccy in ids
                }
            )
            ref = data._get_period_cash_returns(
                values.index, request.reference_ccy, frequency, False, 1
            )
            diagnostics["hedge_cost"] = pd.DataFrame(
                {
                    ccy: 1
                    - (1 + ref)
                    / (1 + data._get_period_cash_returns(values.index, ccy, frequency, False, 1))
                    for ccy in ids
                }
            )
    return FxReturnsData(
        return_type=request.return_type,
        coordinate=request.coordinate,
        frequency=frequency,
        reference_ccy=request.reference_ccy,
        cash_policy=request.cash_policy,
        embedded_hedge_policy=request.embedded_hedge_policy,
        normalization=request.normalization,
        _values=values,
        _local_ccys=pd.DataFrame(currencies),
        _opening_hedge_ratios=pd.DataFrame(hedges),
        _diagnostics=diagnostics,
        _currency_spells=(request.local_ccys.loc[:, assets]
                          if isinstance(request.local_ccys, pd.DataFrame) else None),
    )


def _currency_result(data: FxRatesData, request: FxCurrencyReturnsRequest) -> FxReturnsData:
    """Settle currency spot or exact long-local forward P&L on the requested reset grid."""
    is_forward = request.return_type is FxReturnType.FX_FORWARD
    spot = pd.DataFrame(
        {
            ccy: to_returns(
                data.get_local_to_reference_fx_rate(ccy, request.reference_ccy),
                freq=request.frequency,
                is_log_returns=request.coordinate is FxReturnCoordinate.LOG,
            )
            for ccy in request.currencies
        }
    )
    values = spot
    diagnostics = {"spot": spot} if request.diagnostics else {}
    if is_forward:
        ids = list(dict.fromkeys((*request.currencies, request.reference_ccy)))
        cash = pd.DataFrame(
            {
                ccy: data._get_period_cash_returns(spot.index, ccy, request.frequency, False, 1)
                for ccy in ids
            }
        )
        premium = pd.DataFrame(
            {
                ccy: (1.0 + cash[ccy]) / (1.0 + cash[request.reference_ccy]) - 1.0
                for ccy in request.currencies
            }
        )
        cost = premium.apply(_compute_forward_hedge_cost, axis=0)
        values = spot + cost
        if request.diagnostics:
            diagnostics.update(
                cash=cash,
                forward_premium=premium,
                hedge_cost=cost,
                forward_to_spot=1.0 / (1.0 + premium),
            )
    return FxReturnsData(
        return_type=request.return_type,
        coordinate=request.coordinate,
        frequency=request.frequency,
        reference_ccy=request.reference_ccy,
        cash_policy=request.cash_policy,
        embedded_hedge_policy=("not applicable: long-local currency forward"
                               if is_forward else "not applicable: spot only"),
        normalization=request.normalization,
        _values=values,
        _local_ccys=pd.DataFrame({ccy: ccy for ccy in values}, index=values.index),
        _opening_hedge_ratios=pd.DataFrame(0.0, index=values.index, columns=values.columns),
        _diagnostics=diagnostics,
    )


def compute_fx_returns(
    data: FxRatesData, request: FxAssetReturnsRequest | FxCurrencyReturnsRequest
) -> dict[str, FxReturnsData]:
    """Compute typed FX analytics, grouped by cadence, without an automatic NAV.

    Args:
        data: Existing USD-per-local spot and annual-decimal domestic-rate container.
        request: Explicit asset-panel or currency-panel request, kind and coordinate.

    Returns:
        Cadence-to-result mapping; every result owns its values and metadata.

    Raises:
        ValueError: Invalid kind/coordinate, dates, labels, currency or index policy.
        TypeError: Request or enum arguments are not the declared types.
    """
    if not isinstance(request, (FxAssetReturnsRequest, FxCurrencyReturnsRequest)):
        raise TypeError("Use FxAssetReturnsRequest or FxCurrencyReturnsRequest")
    if not isinstance(request.return_type, FxReturnType) or not isinstance(
        request.coordinate, FxReturnCoordinate
    ):
        raise TypeError("return_type and coordinate must be FX enum members")
    if not isinstance(request.cash_policy, FxCashPolicy):
        raise TypeError("cash_policy must be FxCashPolicy")
    if (request.cash_policy.accrual is FxCashAccrual.MONTHLY_INDEX
            and request.return_type not in _EXCESS):
        raise ValueError("MONTHLY_INDEX is only an excess-return measurement policy")
    if request.return_type in _NOTIONAL_PAYOFFS:
        if request.coordinate is not FxReturnCoordinate.SIMPLE:
            raise ValueError(
                "Notional payoffs require SIMPLE; normalize an index before taking logs"
            )
        if request.cash_policy.quote_lag != 1:
            raise ValueError(
                "Financed and forward payoffs require opening cash quotes (quote_lag=1)"
            )
    if (
        request.coordinate is FxReturnCoordinate.CASH_RELATIVE_SIMPLE
        and request.return_type not in _EXCESS
    ):
        raise ValueError("CASH_RELATIVE_SIMPLE requires an excess return kind")
    if isinstance(request, FxCurrencyReturnsRequest):
        if request.return_type not in _CURRENCY_KINDS:
            raise ValueError("Currency requests support FX_SPOT or FX_FORWARD")
        if (
            not request.currencies
            or len(set(request.currencies)) != len(request.currencies)
            or any(not isinstance(ccy, str) or not ccy.strip() for ccy in request.currencies)
        ):
            raise ValueError("currencies must be nonempty unique currency IDs")
        if not request.reference_ccy:
            raise ValueError("reference_ccy is required for currency requests")
        return {request.frequency: _currency_result(data, request)}
    if request.return_type in _CURRENCY_KINDS:
        raise ValueError("Use a currency request for FX_SPOT or FX_FORWARD")
    needs_reference = (
        request.return_type in _REFERENCE
        or request.return_type is FxReturnType.LOCAL_EXCESS_TRANSLATED
    )
    if needs_reference != bool(request.reference_ccy):
        raise ValueError("reference_ccy is required only for reference returns or translated P&L")
    if request.currency_changes not in ("reject", "mask_boundary"):
        raise ValueError("currency_changes must be reject or mask_boundary")
    if request.embedded_hedge_policy != "prices_include_existing_hedges; ratios_are_additional":
        raise ValueError(
            "Only explicit additional hedges on supplied price histories are supported"
        )
    if not isinstance(request.prices, pd.DataFrame):
        raise TypeError("prices must be a DataFrame")
    _dated(request.prices, "prices")
    if request.prices.empty or not request.prices.columns.is_unique:
        raise ValueError("prices requires nonempty unique asset columns")
    if request.return_type not in _REFERENCE:
        h = np.asarray(request.hedge_ratios)
        if np.any(h[pd.notna(h)] != 0.0):
            raise ValueError(
                "Local returns and translated local excess require zero or missing hedges"
            )
    if isinstance(request.frequency, str):
        groups = {request.frequency: request.prices.columns.tolist()}
    elif isinstance(request.frequency, pd.Series):
        freq = request.frequency.reindex(request.prices.columns)
        if not request.frequency.index.is_unique or freq.isna().any():
            raise ValueError("frequency must cover every asset exactly once")
        groups = {str(key): values.index.tolist() for key, values in freq.groupby(freq)}
    else:
        raise TypeError("frequency must be a string or asset Series")
    return {freq: _asset_result(data, request, freq, assets) for freq, assets in groups.items()}

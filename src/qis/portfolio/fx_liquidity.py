"""Cash and eligible collateral at FX payment cutoffs, alongside a marked QIS account."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from qis.portfolio.fx_accounting import FxAccountData


_KEYS = ["currency", "agreement"]
_FLOW_KINDS = [
    "cash_flow",
    "funding",
    "initial_margin",
    "collateralised_vm",
    "received_vm",
    "settled_vm",
]
_LEDGER_FLOWS = ["cash_interest", "local_settlement", "equity_trades"]


@dataclass
class FxLiquidityResult:
    """Cutoff balances in payment currency and closing valuations in reference currency.

    Attributes:
        reconciliation: Per-cutoff roll-forward for each currency and agreement.
        summary: Lowest free cash and largest funding gap per agreement and currency.
        currency_summary: Largest simultaneous funding gap across segregated agreements in
            each payment currency, with its first binding time.
        valuation: Original closing cash, forward marks and NAV from the supplied account.
    """

    reconciliation: pd.DataFrame
    summary: pd.DataFrame
    currency_summary: pd.DataFrame
    valuation: pd.DataFrame


def _close(date):
    """End of the ledger valuation day on the common timezone-naive operational clock."""
    return date + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)


def _timestamp(value):
    """Require an explicit, finite local-clock timestamp."""
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tz is not None:
        raise ValueError("Cutoffs must be valid timezone-naive timestamps on a common clock")
    return stamp


def _key(value):
    """Require a currency or agreement label that remains unambiguous in exports."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Currencies and agreements must be nonempty strings")
    return value


def _schedule(frame, columns, optional, first, last):
    """Validate operational records without changing the supplied frame or ordering."""
    if frame is None:
        return pd.DataFrame(columns=columns)
    if (
        not isinstance(frame, pd.DataFrame)
        or not frame.columns.is_unique
        or not set(columns) <= set(frame)
        or set(frame) - set(columns) - set(optional)
    ):
        raise ValueError(f"Expected columns {columns} and optional {optional}")
    frame = frame.copy()
    frame["at"] = pd.DatetimeIndex([_timestamp(v) for v in frame["at"]])
    if ((frame["at"] < first) | (frame["at"] > last)).any():
        raise ValueError("Operational events must lie within the ledger valuation span")
    for label in _KEYS:
        frame[label] = frame[label].map(_key)
    return frame


def _opening_balances(opening_cash, reference, treasury, capital):
    """Allocate the reference capital once across the supplied cash pools."""
    if opening_cash is None:
        opening_cash = pd.Series(
            [capital], index=pd.MultiIndex.from_tuples([(reference, treasury)])
        )
    if (
        not isinstance(opening_cash.index, pd.MultiIndex)
        or opening_cash.index.nlevels != 2
        or not opening_cash.index.is_unique
    ):
        raise ValueError("Opening cash needs distinct (currency, agreement) keys")
    balances = {(_key(c), _key(a)): float(v) for (c, a), v in opening_cash.items()}
    if not np.isfinite(list(balances.values())).all() or not np.isclose(
        sum(v for (c, _), v in balances.items() if c == reference), capital, rtol=1e-12, atol=1e-9
    ):
        raise ValueError("Reference opening cash must reconcile to the ledger initial capital")
    return balances


def _forward_payments(contracts, contract_terms, treasury, reference, vm):
    """Match early settled VM to its own executed contract before assigning the residual payment."""
    if contract_terms is None:
        terms = pd.DataFrame({"agreement": treasury}, index=contracts.index)
    else:
        if (
            not contract_terms.index.is_unique
            or not contract_terms.columns.is_unique
            or "agreement" not in contract_terms
            or set(contract_terms) - {"agreement", "settlement_at"}
            or set(contract_terms.index) != set(contracts.index)
        ):
            raise ValueError("Provide agreement terms for each executed contract exactly once")
        terms = contract_terms.reindex(contracts.index).copy()
    terms["agreement"] = terms.agreement.map(_key)
    cutoffs = []
    for contract, row in contracts.iterrows():
        value = terms.at[contract, "settlement_at"] if "settlement_at" in terms else pd.NaT
        stamp = _close(row.settlement_date) if pd.isna(value) else _timestamp(value)
        if stamp.normalize() != row.settlement_date:
            raise ValueError("Settlement cutoff must be on the ledger settlement date")
        cutoffs.append(stamp)
    terms["settlement_at"] = pd.DatetimeIndex(cutoffs)
    for row in vm.itertuples(index=False):
        identity = row.contract_id
        if (
            identity not in terms.index
            or row.currency != reference
            or row.agreement != terms.at[identity, "agreement"]
            or row.at < contracts.at[identity, "trade_date"]
            or row.at > terms.at[identity, "settlement_at"]
        ):
            raise ValueError(
                "Settled VM must match a live contract, settlement currency and agreement"
            )

    # The ledger pays gross; only the remaining contractual balance is due at maturity.
    vm_by_contract = vm.groupby("contract_id", sort=False).amount.sum()
    payments = []
    for identity, row in contracts.iterrows():
        if row.paid:
            stamp, group = terms.at[identity, "settlement_at"], terms.at[identity, "agreement"]
            payments.extend(
                [
                    (stamp, reference, group, "gross_settlement", row.fixed_payoff),
                    (stamp, reference, group, "vm_offset", -vm_by_contract.get(identity, 0.0)),
                ]
            )
    return terms, payments


def _ledger_payments(raw, ledger_flow_times, treasury, reference, dates, end):
    """Record cash at its usable cutoff and reconcile intervening settlement receivables."""
    if ledger_flow_times is not None:
        if (
            not ledger_flow_times.index.is_unique
            or not ledger_flow_times.columns.is_unique
            or not ledger_flow_times.index.isin(dates).all()
            or set(ledger_flow_times) - set(_LEDGER_FLOWS)
        ):
            raise ValueError(
                "Cash timing overrides must use ledger flow labels and valuation dates"
            )
    payments = []
    for name in _LEDGER_FLOWS:
        for date, amount in raw[name].items():
            stamp = _close(date)
            if (
                ledger_flow_times is not None
                and name in ledger_flow_times
                and date in ledger_flow_times.index
            ):
                override = ledger_flow_times.at[date, name]
                if pd.notna(override):
                    stamp = _timestamp(override)
                    if stamp.normalize() < date:
                        raise ValueError("Ledger cash cannot be available before its posting date")
            if not amount:
                continue
            if stamp.normalize() > date:
                payments.append((_close(date), reference, treasury, "deferred_change", amount))
                if stamp <= end:
                    payments.append((stamp, reference, treasury, "deferred_change", -amount))
            if stamp <= end:
                payments.append((stamp, reference, treasury, name, amount))
    return payments


def compute_fx_liquidity(
    account: FxAccountData,
    flows: pd.DataFrame | None = None,
    *,
    collateral: pd.DataFrame | None = None,
    opening_cash: pd.Series | None = None,
    ledger_agreement: str = "Treasury",
    contract_terms: pd.DataFrame | None = None,
    ledger_flow_times: pd.DataFrame | None = None,
    payment_cutoffs: pd.DatetimeIndex | None = None,
) -> FxLiquidityResult:
    """Reconcile FX margin and payment needs in each ring-fenced currency and agreement.

    Args:
        account: Completed QIS FX account. Cash, contract marks and actual maturity
            payments come from this ledger; its valuation and rates are preserved.
        flows: Rows (at, currency, agreement, kind, amount), with contract_id required
            only for settled_vm. Positive cash_flow/funding/settled_vm denotes cash in.
            Positive initial_margin/collateralised_vm posts collateral; negative releases.
            Positive received_vm segregates a counterparty cash deposit; negative returns it.
            All amounts are in the bucket currency. External flows exclude ledger cash flows.
        collateral: Rows (at, currency, agreement, value, haircut, eligible). At each
            bucket timestamp these replace the entire dedicated noncash inventory, in
            payment-currency units. Post eligible securities first at value times one minus
            haircut. Reallocate cash collateral at the cutoff when capacity changes.
        opening_cash: Optional Series indexed by (currency, agreement) allocating initial
            cash before ledger trades. Reference buckets must sum to ledger initial capital.
            Other currency balances are supplementary liquidity outside the marked account.
        ledger_agreement: Default opening cash pool and recipient of non-forward ledger flows.
        contract_terms: Optional frame indexed by every executed contract ID, with agreement
            and optional settlement_at on its ledger maturity date. Default is the ledger
            agreement and end-of-day settlement. Settled VM uses that same bucket and contract.
        ledger_flow_times: Optional frame indexed by ledger date with timestamp overrides for
            cash_interest, equity_trades or local_settlement. Each is an aggregate daily flow,
            usable at day-end unless timed. Later value dates defer cash, including beyond
            the reporting span. All timestamps use the same timezone-naive clock.
        payment_cutoffs: Additional timestamps at which to report balances, even if no flow occurs.

    Returns:
        Cutoff reconciliation, agreement and currency summaries, and original marked
        valuations. Negative closing free cash is an unfunded requirement. Same-time events
        net within a bucket. Unused noncash collateral cannot fund cash-settled obligations.
    """
    reference, capital = account.spec.reference_ccy, account.spec.initial_capital
    treasury = _key(ledger_agreement)
    valuation = account.accounts[["cash", "forward_value", "nav"]].copy()
    dates = valuation.index
    closes = _close(dates)
    balances = _opening_balances(opening_cash, reference, treasury, capital)
    events = _schedule(
        flows, ["at", *_KEYS, "kind", "amount"], ["contract_id"], dates[0], closes[-1]
    )
    if not events.kind.isin(_FLOW_KINDS).all():
        raise ValueError("Unknown liquidity flow kind")
    events["amount"] = events.amount.astype(float)
    if not np.isfinite(events.amount).all():
        raise ValueError("Flow amounts must be finite")
    if "contract_id" not in events:
        events["contract_id"] = None
    settled = events.kind.eq("settled_vm")
    if not events.contract_id.notna().eq(settled).all():
        raise ValueError("Only settled_vm requires a contract_id")

    inventory = _schedule(
        collateral, ["at", *_KEYS, "value", "haircut", "eligible"], [], dates[0], closes[-1]
    )
    amounts = inventory[["value", "haircut"]].astype(float)
    if (
        not np.isfinite(amounts).all().all()
        or amounts.value.lt(0).any()
        or not amounts.haircut.between(0, 1).all()
        or not inventory.eligible.map(lambda v: isinstance(v, (bool, np.bool_))).all()
    ):
        raise ValueError(
            "Collateral requires finite nonnegative values, [0, 1] haircuts and booleans"
        )
    inventory["capacity"] = (amounts.value * (1 - amounts.haircut)).where(
        inventory.eligible.astype(bool), 0.0
    )
    terms, payments = _forward_payments(
        account.contracts, contract_terms, treasury, reference, events.loc[settled]
    )
    payments.extend(
        _ledger_payments(
            account.cash_flows, ledger_flow_times, treasury, reference, dates, closes[-1]
        )
    )
    fields = ["at", *_KEYS, "kind", "amount"]
    records = events[fields]
    if payments:
        payment_records = pd.DataFrame(payments, columns=fields)
        # Empty frames can change dtype inference across pandas versions.
        if records.empty:
            records = payment_records
        else:
            records = pd.concat([records, payment_records], ignore_index=True)
    buckets = dict.fromkeys(
        [
            *balances,
            (reference, treasury),
            *((reference, group) for group in terms.agreement),
            *records[_KEYS].itertuples(index=False, name=None),
            *inventory[_KEYS].itertuples(index=False, name=None),
        ]
    )
    cutoffs = pd.DatetimeIndex(
        [] if payment_cutoffs is None else [_timestamp(value) for value in payment_cutoffs]
    )
    if ((cutoffs < dates[0]) | (cutoffs > closes[-1])).any():
        raise ValueError("Payment cutoffs must lie within the ledger valuation span")
    times = (
        closes.union(dates[:1])
        .union(cutoffs)
        .union(pd.DatetimeIndex(records["at"]))
        .union(pd.DatetimeIndex(inventory["at"]))
        .unique()
        .sort_values()
    )
    index = pd.MultiIndex.from_tuples(
        [(t, c, a) for t in times for c, a in buckets], names=["at", *_KEYS]
    )
    cols = [*_LEDGER_FLOWS, *_FLOW_KINDS, "gross_settlement", "vm_offset", "deferred_change"]
    table = (
        records.groupby(["at", *_KEYS, "kind"])
        .amount.sum()
        .unstack("kind")
        .reindex(index=index, columns=cols)
        .fillna(0.0)
        .astype(float)
    )
    table.columns.name = None
    capacity = inventory.groupby(["at", *_KEYS]).capacity.sum().reindex(index)
    table["eligible_capacity"] = capacity.groupby(level=_KEYS).ffill().fillna(0.0)
    groups = table.groupby(level=_KEYS)
    positions = groups[["initial_margin", "collateralised_vm", "received_vm"]].cumsum()
    if (positions < -1e-9).any().any():
        raise ValueError("Collateral release exceeds the posted or received balance in its bucket")
    table["pending_ledger_cash"] = groups.deferred_change.cumsum()
    table["segregated_receipts"] = positions.received_vm
    posted = positions.initial_margin + positions.collateralised_vm
    table["pledged_credit"] = np.minimum(posted, table.eligible_capacity)
    table["restricted_cash"] = posted - table.pledged_credit
    table["unused_credit"] = table.eligible_capacity - table.pledged_credit
    table["net_settlement"] = table.gross_settlement + table.vm_offset
    previous = table.restricted_cash.groupby(level=_KEYS).shift(fill_value=0.0)
    table["cash_margin_transfer"] = table.restricted_cash - previous
    money = [*_LEDGER_FLOWS, "funding", "cash_flow", "settled_vm", "net_settlement"]
    cash_delta = table[money].sum(axis=1) - table.cash_margin_transfer
    opening = pd.Series([balances.get(key, 0.0) for key in index.droplevel("at")], index=index)
    table["closing_free_cash"] = opening + cash_delta.groupby(level=_KEYS).cumsum()
    table["opening_free_cash"] = (
        table.closing_free_cash.groupby(level=_KEYS).shift().fillna(opening)
    )
    table["outstanding_settled_vm"] = (
        (table.settled_vm + table.vm_offset).groupby(level=_KEYS).cumsum()
    )
    table["shortfall"] = (-table.closing_free_cash).clip(lower=0)
    marks = account.contract_values
    table["ledger_forward_mark"] = np.nan
    for ccy, agreement in buckets:
        if ccy == reference:
            ids = terms.index[terms.agreement.eq(agreement)]
            table.loc[(closes, ccy, agreement), "ledger_forward_mark"] = (
                marks[ids].sum(axis=1).to_numpy()
            )
    # Recover the underlying ledger cash after stripping operational flows and early VM.
    extra = (table.cash_flow + table.funding).groupby(level=_KEYS).cumsum()
    implied = (
        table.closing_free_cash
        + table.restricted_cash
        + table.pending_ledger_cash
        - table.outstanding_settled_vm
        - extra
    )
    cash = implied.xs(reference, level="currency").groupby(level="at").sum().reindex(closes)
    if not np.allclose(cash.to_numpy(), valuation.cash, rtol=1e-12, atol=1e-12 * max(1.0, capital)):
        raise ValueError("Reference-currency cash does not reconcile to the supplied ledger")
    summary = _summarize(table)
    by_currency = table.groupby(level=["at", "currency"]).agg(
        closing_free_cash=("closing_free_cash", "sum"), shortfall=("shortfall", "sum")
    )
    return FxLiquidityResult(table, summary, _summarize(by_currency), valuation)


def _summarize(table):
    """Summarize concurrent deficits, preserving the first cutoff at the largest shortfall."""
    groups = table.groupby(level=table.index.names[1:], sort=False)
    summary = groups.agg(
        minimum_free_cash=("closing_free_cash", "min"), peak_shortfall=("shortfall", "max")
    )
    summary["binding_cutoff"] = pd.to_datetime(
        [
            key[0] if peak > 0 else pd.NaT
            for key, peak in zip(groups.shortfall.idxmax(), summary.peak_shortfall, strict=True)
        ]
    )
    return summary

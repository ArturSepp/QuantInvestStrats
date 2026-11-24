"""Independent funding and settlement references for a cutoff-driven FX liquidity overlay."""

import numpy as np
import pandas as pd
import pytest

from qis.fx.fx_rates_data import FxRatesData
from qis.portfolio.fx_accounting import FxAccountSpec, backtest_fx_account
from qis.portfolio.fx_liquidity import compute_fx_liquidity


DATES = pd.date_range("2025-02-03", periods=5)
KEYS = ["at", "currency", "agreement", "kind", "amount", "contract_id"]


def _book(forwards=None, rates=0.0, reference="USD", capital=100.0):
    """Price known, opposite forward quantities with an independent, zero-cost funded book."""
    spots = pd.DataFrame({"USD": 1.0, "EUR": [1.0, 1.1, 1.2, 1.2, 1.2], "CHF": 0.8}, index=DATES)
    fx = FxRatesData(spots, pd.DataFrame(rates, index=DATES, columns=spots.columns))
    empty = pd.DataFrame(index=DATES)
    orders = None
    if forwards:
        orders = pd.DataFrame(
            [(DATES[0], DATES[2], DATES[-1], "EUR", w, "capital") for w in forwards.values()],
            columns=[
                "trade_date",
                "fixing_date",
                "settlement_date",
                "currency",
                "notional_weight",
                "notional_basis",
            ],
            index=forwards,
        )
    return backtest_fx_account(
        empty,
        pd.Series(dtype=object),
        empty.iloc[:1],
        fx,
        forwards=orders,
        spec=FxAccountSpec(reference, capital, "local", "fixed", "reference"),
        local_settlement_dates=pd.DatetimeIndex([]),
        spot_is_observed=spots.notna(),
    )


def _time(day, hour):
    """Return a precise common-clock cutoff."""
    return DATES[day] + pd.Timedelta(hours=hour)


def _flows(*rows):
    """Build signed operational events."""
    return pd.DataFrame(rows, columns=KEYS)


def test_zero_calls_reconcile_to_marked_account():
    """Idle cash and open marks are distinct, including for a non-USD reference account."""
    for ref, contracts in [("USD", None), ("CHF", {"short": -0.6})]:
        ledger = _book(contracts, rates=0.05, reference=ref)
        result = compute_fx_liquidity(ledger, payment_cutoffs=[_time(2, 9)])
        bucket = result.reconciliation.xs((ref, "Treasury"), level=["currency", "agreement"])
        daily = bucket.groupby(bucket.index.normalize()).last()
        np.testing.assert_allclose(daily.closing_free_cash, ledger.accounts.cash, atol=1e-11)
        np.testing.assert_allclose(daily.ledger_forward_mark, ledger.accounts.forward_value)
        np.testing.assert_allclose(result.valuation.nav, ledger.nav)
        assert pd.isna(bucket.loc[_time(2, 9), "ledger_forward_mark"])
        assert result.summary.peak_shortfall.eq(0).all()
        assert result.summary.binding_cutoff.isna().all()


def test_initial_margin_transfer_and_release():
    """A 5% posting on USD 50,000 transfers exactly USD 2,500 and creates no P&L."""
    book = _book(capital=50_000)
    at, release = _time(1, 10), _time(4, 13)
    result = compute_fx_liquidity(
        book,
        _flows(
            (at, "USD", "Treasury", "initial_margin", 2_500.0, None),
            (release, "USD", "Treasury", "initial_margin", -2_500.0, None),
        ),
    )
    table = result.reconciliation.droplevel(["currency", "agreement"])
    assert table.loc[
        at, ["opening_free_cash", "restricted_cash", "closing_free_cash"]
    ].tolist() == [50_000.0, 2_500.0, 47_500.0]
    assert table.loc[release, "cash_margin_transfer"] == -2_500.0
    np.testing.assert_array_equal(table.closing_free_cash + table.restricted_cash, 50_000.0)
    np.testing.assert_array_equal(book.pnl.sum(axis=1), 0.0)


def test_settled_vm_and_collateralised_vm_for_opposite_contracts():
    """Maturity pays -20 and +20 gross, but the early -12 belongs only to the short."""
    book = _book({"short": -1.0, "long": 1.0})
    at = _time(1, 10)
    late, early = _time(4, 16), _time(4, 9)
    terms = pd.DataFrame(
        {"agreement": ["A", "B"], "settlement_at": [early, late]}, index=["short", "long"]
    )
    opening = pd.Series({("USD", "A"): 30.0, ("USD", "B"): 70.0})
    result = compute_fx_liquidity(
        book,
        _flows(
            (at, "USD", "A", "settled_vm", -12.0, "short"),
            (at, "USD", "B", "collateralised_vm", 10.0, None),
            (late, "USD", "B", "collateralised_vm", -10.0, None),
        ),
        opening_cash=opening,
        contract_terms=terms,
    )
    a = result.reconciliation.xs(("USD", "A"), level=["currency", "agreement"])
    b = result.reconciliation.xs(("USD", "B"), level=["currency", "agreement"])
    np.testing.assert_allclose(
        a.loc[early, ["gross_settlement", "vm_offset", "net_settlement", "closing_free_cash"]],
        [-20, 12, -8, 10],
    )
    np.testing.assert_allclose(
        b.loc[late, ["net_settlement", "vm_offset", "restricted_cash", "closing_free_cash"]],
        [20, 0, 0, 90],
    )
    assert a.loc[at, "outstanding_settled_vm"] == -12 and a.iloc[-1].outstanding_settled_vm == 0
    assert b.loc[at, "closing_free_cash"] == 60
    # Aggregate ledger settlement is zero on every day, making contract-level linkage essential.
    np.testing.assert_allclose(book.cash_flows.forward_settlement, 0.0)


def test_cutoffs_haircuts_ineligible_collateral_and_segregation():
    """Cash and eligible securities can pay a margin call, but securities cannot settle VM."""
    book = _book({"short": -1.0})
    at, change, maturity = _time(1, 9), _time(1, 14), _time(4, 15)
    inventory = pd.DataFrame(
        [
            (at, "USD", "A", 50.0, 0.2, True),
            (at, "USD", "A", 500.0, 0.0, False),
            (change, "USD", "A", 50.0, 0.5, True),
            (at, "EUR", "B", 1_000.0, 0.0, True),
        ],
        columns=["at", "currency", "agreement", "value", "haircut", "eligible"],
    )
    terms = pd.DataFrame({"agreement": ["A"], "settlement_at": [maturity]}, index=["short"])
    result = compute_fx_liquidity(
        book,
        _flows(
            (at, "USD", "A", "initial_margin", 50.0, None),
            (at, "USD", "A", "received_vm", 200.0, None),
            (at, "USD", "A", "settled_vm", -15.0, "short"),
            (at, "EUR", "B", "cash_flow", -60.0, None),
            (change, "USD", "A", "funding", 10.0, None),
            (maturity, "USD", "A", "initial_margin", -50.0, None),
            (maturity, "USD", "A", "received_vm", -200.0, None),
        ),
        collateral=inventory,
        contract_terms=terms,
        opening_cash=pd.Series({("USD", "A"): 20.0, ("USD", "B"): 80.0}),
    )
    a = result.reconciliation.xs(("USD", "A"), level=["currency", "agreement"])
    assert a.loc[
        at, ["eligible_capacity", "restricted_cash", "segregated_receipts", "closing_free_cash"]
    ].tolist() == [40.0, 10.0, 200.0, -5.0]
    assert a.loc[change, ["restricted_cash", "closing_free_cash"]].tolist() == [25.0, -10.0]
    assert a.loc[maturity, "net_settlement"] == pytest.approx(-5.0)
    assert result.currency_summary.loc["USD", "peak_shortfall"] == 10.0
    assert result.currency_summary.loc["EUR", "peak_shortfall"] == 60.0
    assert result.summary.loc[("USD", "A"), "binding_cutoff"] == change
    assert result.currency_summary.loc["EUR", "binding_cutoff"] == at


def test_equity_sale_value_date_and_hedge_payment():
    """Rebuild both the USD asset sale and EUR hedge from units and observed spot."""
    spot = pd.Series([1.0, 1.15, 1.15, 1.15, 1.15], index=DATES)
    spots = pd.DataFrame({"USD": 1.0, "EUR": spot}, index=DATES)
    shares, capital = 80_000, 8_750_000.0
    prices = pd.DataFrame({"asset": 100.0}, index=DATES)
    initial_value = shares * 100.0
    target = pd.DataFrame({"asset": initial_value / capital}, index=DATES[:2])
    order = pd.DataFrame(
        [
            dict(
                trade_date=DATES[0],
                fixing_date=DATES[2],
                settlement_date=DATES[-1],
                currency="EUR",
                notional_weight=-1.0,
                notional_basis="asset_currency_value",
            )
        ],
        index=["hedge"],
    )
    book = backtest_fx_account(
        prices,
        pd.Series({"asset": "EUR"}),
        target,
        FxRatesData(spots, pd.DataFrame(0.0, index=DATES, columns=spots.columns)),
        spec=FxAccountSpec("USD", capital, "reference", "fixed", "zero"),
        local_settlement_dates=pd.DatetimeIndex([]),
        spot_is_observed=spots.notna(),
        forwards=order,
    )
    vm = initial_value * (1 - spot.iloc[1])
    call, sale = _time(1, 10), _time(3, 15)
    flows = _flows((call, "USD", "Treasury", "settled_vm", vm, "hedge"))
    timings = pd.DataFrame({"equity_trades": [sale]}, index=DATES[1:2])
    out = compute_fx_liquidity(
        book, flows, ledger_flow_times=timings, payment_cutoffs=[_time(3, 14), _time(4, 10)]
    )
    table = out.reconciliation.droplevel(["currency", "agreement"])
    daily = table.groupby(table.index.normalize()).last()
    cash = capital - initial_value
    proceeds = shares * 100.0 * spot.iloc[1] - initial_value
    np.testing.assert_allclose(
        daily.closing_free_cash,
        [cash, cash + vm, cash + vm, cash + vm + proceeds, cash + vm + proceeds],
    )
    np.testing.assert_allclose(daily.pending_ledger_cash, [0, proceeds, proceeds, 0, 0])
    np.testing.assert_allclose(daily.ledger_forward_mark, [0, vm, vm, vm, 0], atol=1e-8)
    np.testing.assert_allclose(out.valuation.nav, capital)
    assert out.summary.iloc[0].peak_shortfall == pytest.approx(450_000.0)
    assert out.summary.iloc[0].binding_cutoff == call
    assert table.loc[_time(3, 14), "closing_free_cash"] == pytest.approx(-450_000.0)
    assert table.loc[sale, "closing_free_cash"] == pytest.approx(750_000.0)
    assert table.iloc[-1].net_settlement == pytest.approx(0, abs=1e-8)
    # A value date beyond the history receives no premature credit.
    timings.iloc[0, 0] = DATES[-1] + pd.Timedelta(days=2)
    later = compute_fx_liquidity(book, flows, ledger_flow_times=timings).reconciliation.iloc[-1]
    assert later.pending_ledger_cash == pytest.approx(proceeds)
    assert later.closing_free_cash == pytest.approx(-450_000.0)


def test_currency_summary_uses_simultaneous_agreement_deficits():
    """Opposite cash balances cannot net, and peaks at different times cannot be added."""
    a, b, c = _time(1, 9), _time(1, 10), _time(1, 11)
    result = compute_fx_liquidity(
        _book(),
        _flows(
            (a, "USD", "A", "initial_margin", 40.0, None),
            (b, "USD", "A", "initial_margin", -40.0, None),
            (b, "USD", "B", "initial_margin", 50.0, None),
            (c, "USD", "A", "initial_margin", 10.0, None),
        ),
    )
    assert result.summary.loc[("USD", "A"), "peak_shortfall"] == 40
    assert result.summary.loc[("USD", "B"), "peak_shortfall"] == 50
    assert result.currency_summary.loc["USD"].tolist() == [40, 60, c]


@pytest.mark.parametrize(
    "currency,agreement,contract,stamp",
    [
        ("EUR", "Treasury", "short", _time(1, 10)),
        ("USD", "other", "short", _time(1, 10)),
        ("USD", "Treasury", "unknown", _time(1, 10)),
        ("USD", "Treasury", "short", _time(4, 16)),
    ],
)
def test_invalid_settled_vm_is_rejected(currency, agreement, contract, stamp):
    """Never borrow a payment from another currency, agreement, contract or maturity."""
    terms = pd.DataFrame(
        {"agreement": "Treasury", "settlement_at": [_time(4, 10)]}, index=["short"]
    )
    with pytest.raises(ValueError, match="live contract"):
        compute_fx_liquidity(
            _book({"short": -1.0}),
            _flows((stamp, currency, agreement, "settled_vm", -10.0, contract)),
            contract_terms=terms,
        )


def test_text_and_missing_maturity_cutoffs_keep_exact_payment_times():
    """A supplied string cutoff and an omitted nanosecond close coexist in one schedule."""
    book = _book({"short": -1.0, "long": 1.0})
    terms = pd.DataFrame(
        {
            "agreement": ["A", "B"],
            "settlement_at": pd.array(["2025-02-07 10:00", None], dtype="string"),
        },
        index=["short", "long"],
    )
    table = compute_fx_liquidity(book, contract_terms=terms).reconciliation
    end = DATES[-1] + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    assert table.loc[(_time(4, 10), "USD", "A"), "net_settlement"] == pytest.approx(-20)
    assert table.loc[(end, "USD", "B"), "net_settlement"] == pytest.approx(20)


def test_input_and_release_validation():
    """Invalid time, haircut, eligibility, opening cash or over-release fails explicitly."""
    ledger = _book()
    at = _time(1, 10)
    frame = _flows((at, "USD", "Treasury", "initial_margin", -1.0, None))
    with pytest.raises(ValueError, match="release exceeds"):
        compute_fx_liquidity(ledger, frame)
    with pytest.raises(ValueError, match="Operational events"):
        compute_fx_liquidity(ledger, frame.assign(at=DATES[0] - pd.Timedelta(days=1)))
    with pytest.raises(ValueError, match="timezone"):
        compute_fx_liquidity(ledger, frame.assign(at=at.tz_localize("UTC")))
    inventory = pd.DataFrame(
        [dict(at=at, currency="USD", agreement="Treasury", value=1.0, haircut=1.1, eligible=True)]
    )
    for bad in [inventory, inventory.assign(haircut=0.1, eligible="yes")]:
        with pytest.raises(ValueError, match="Collateral requires"):
            compute_fx_liquidity(ledger, collateral=bad)
    with pytest.raises(ValueError, match="opening cash"):
        compute_fx_liquidity(ledger, opening_cash=pd.Series({("USD", "Treasury"): 99.0}))
    with pytest.raises(ValueError, match="posting date"):
        compute_fx_liquidity(
            ledger, ledger_flow_times=pd.DataFrame({"equity_trades": [DATES[0]]}, index=DATES[1:2])
        )

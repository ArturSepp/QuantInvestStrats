"""Independent cash-flow, native-engine and event-order references for FX accounts."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import qis


COLUMNS = [
    "trade_date",
    "fixing_date",
    "settlement_date",
    "currency",
    "notional_weight",
    "notional_basis",
]


def account(prices, ccys, weights, fx, **kwargs):
    """Pass explicitly observed synthetic quotes to the public account contract."""
    kwargs.setdefault("spot_is_observed", fx.fx_spots.notna().reindex(prices.index))
    return qis.backtest_fx_account(prices, ccys, weights, fx, **kwargs)


def sample():
    """Irregular leap-year dates, varying rates and two funded foreign total-return assets."""
    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-04", "2024-03-31"])
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, 108.0, 120.0], "B": [80.0, 76.0, 78.0, 81.0]}, index=dates
    )
    spots = pd.DataFrame(
        {"USD": 1.0, "EUR": [1.1, 1.2, 1.18, 1.25], "CHF": [1.05, 1.02, 1.03, 1.08]}, index=dates
    )
    rates = pd.DataFrame(
        {
            "USD": [0.06, 0.08, 0.07, 0.03],
            "EUR": [0.03, 0.05, 0.04, 0.02],
            "CHF": [0.01, 0.02, 0.015, 0.01],
        },
        index=dates,
    )
    return prices, pd.Series({"A": "EUR", "B": "CHF"}), qis.FxRatesData(spots, rates)


def spec(**kwargs):
    """An explicit default for tests, not an implicit library financing policy."""
    return qis.FxAccountSpec(
        "USD",
        100.0,
        **dict(financing="local", notional_rule="fixed", cash_rate_policy="reference", **kwargs),
    )


def run(weights=(0.8, -0.8), policy=None, settlements=None, forwards=None):
    """Run the supplied signed book on the fixture grid."""
    prices, ccys, fx = sample()
    targets = pd.DataFrame([weights], index=prices.index[:1], columns=prices.columns)
    return account(
        prices,
        ccys,
        targets,
        fx,
        spec=policy or spec(),
        local_settlement_dates=pd.DatetimeIndex([]) if settlements is None else settlements,
        forwards=forwards,
    )


@pytest.mark.parametrize("weights", [(0.8, -0.8), (0.8, -0.3), (-0.6, -0.3)])
def test_signed_local_book_matches_closed_cash_and_debt_balances(weights):
    """Zero, positive and negative net exposure all carry real local balances and unit drift."""
    p, ccys, fx = sample()
    result = run(weights)
    s = fx.fx_spots[ccys.to_list()].to_numpy()
    rate = fx.domestic_rates
    u = 100.0 * np.array(weights) / (p.iloc[0].to_numpy() * s[0])
    dt = np.diff(p.index.to_numpy()).astype("timedelta64[D]").astype(float) / 365.0
    growth = np.vstack(
        [
            np.ones(2),
            np.cumprod(1.0 + rate[ccys.to_list()].iloc[:-1].to_numpy() * dt[:, None], axis=0),
        ]
    )
    loans = -u * p.iloc[0].to_numpy() * growth
    cash = 100.0 * np.r_[1.0, np.cumprod(1.0 + rate.USD.iloc[:-1].to_numpy() * dt)]
    expected = cash + np.sum((u * p.to_numpy() + loans) * s, axis=1)
    np.testing.assert_allclose(result.nav, expected, rtol=1e-13, atol=1e-12)
    np.testing.assert_allclose(result.local_balances, loans, atol=1e-12)
    np.testing.assert_allclose(result.asset_units, np.tile(u, (len(p), 1)))
    np.testing.assert_allclose(result.pnl.sum(axis=1).iloc[1:], np.diff(expected), atol=1e-12)
    assert not np.isclose(np.sum(u * p.iloc[-1].to_numpy() * s[-1]), 100.0 * sum(weights))


def test_local_settlement_resets_debt_and_cash_sweep_without_changing_units():
    """Two independently financed holding intervals reconcile with the intermediate sweep."""
    p, ccys, fx = sample()
    dates = p.index
    result = run(settlements=dates[[1, 3]])
    u = (
        100.0
        * np.array([0.8, -0.8])
        / (p.iloc[0].to_numpy() * fx.fx_spots[ccys].iloc[0].to_numpy())
    )
    rates = fx.domestic_rates
    s = fx.fx_spots[ccys.to_list()]
    dt = (dates[1] - dates[0]).days / 365
    first = np.sum(
        u
        * (p.iloc[1] - p.iloc[0] * (1.0 + rates.iloc[0][ccys].to_numpy() * dt))
        * s.iloc[1].to_numpy()
    )
    cash1 = 100.0 * (1.0 + rates.USD.iloc[0] * dt) + first
    remaining = np.diff(dates[1:].to_numpy()).astype("timedelta64[D]").astype(float) / 365
    local_growth = np.prod(
        1.0 + rates[ccys.to_list()].iloc[1:-1].to_numpy() * remaining[:, None], axis=0
    )
    second = np.sum(u * (p.iloc[-1] - p.iloc[1] * local_growth) * s.iloc[-1].to_numpy())
    expected = cash1 * np.prod(1.0 + rates.USD.iloc[1:-1].to_numpy() * remaining) + second
    assert result.nav.iloc[-1] == pytest.approx(expected, abs=1e-12)
    assert result.cash_flows.local_settlement.iloc[1] == pytest.approx(first)
    np.testing.assert_allclose(result.local_balances.loc[dates[1]], -u * p.iloc[1])
    assert result.accounts.cash.iloc[-1] == pytest.approx(expected)


@pytest.mark.parametrize("weights", [(0.8, -0.8), (0.8, -0.3), (-0.6, -0.3)])
def test_single_interval_agrees_with_typed_translated_payoff(weights):
    """The account adds collateral interest once to the signed locally financed P&L."""
    prices, ccys, fx = sample()
    prices = prices.iloc[:2]
    targets = pd.DataFrame([weights], index=prices.index[:1], columns=prices.columns)
    result = account(
        prices, ccys, targets, fx, spec=spec(), local_settlement_dates=prices.index[-1:]
    )
    payoff = qis.compute_fx_returns(
        fx,
        qis.FxAssetReturnsRequest(
            prices,
            ccys,
            qis.FxReturnType.LOCAL_EXCESS_TRANSLATED,
            qis.FxReturnCoordinate.SIMPLE,
            reference_ccy="USD",
            frequency="ME",
        ),
    )["ME"].values.iloc[1]
    c = 0.06 * 29 / 365
    assert result.nav.iloc[-1] == pytest.approx(100.0 * (1 + c) + 100.0 * np.dot(weights, payoff))


@pytest.mark.parametrize("rule", ["fixed", "current_nav"])
def test_reference_funded_special_case_matches_existing_native_backtester(rule):
    """The additive extension preserves native unit/cash behavior when no FX contract is needed."""
    p, ccys, fx = sample()
    targets = pd.DataFrame(
        [[0.8, -0.4], [0.4, 0.9], [0.0, 0.0]], index=p.index[[0, 1, 3]], columns=p.columns
    )
    policy = replace(spec(), financing="reference", notional_rule=rule)
    actual = account(p, ccys, targets, fx, spec=policy, local_settlement_dates=pd.DatetimeIndex([]))
    converted = p * pd.DataFrame({a: fx.fx_spots[c] for a, c in ccys.items()})
    native = qis.backtest_model_portfolio(
        converted,
        targets,
        initial_nav=100.0,
        funding_rate=fx.domestic_rates.USD,
        constant_trade_level=100.0 if rule == "fixed" else None,
    )
    np.testing.assert_allclose(actual.nav, native.nav, atol=1e-12)
    np.testing.assert_allclose(actual.asset_units, native.units, atol=1e-12)


def pure_forward(fx, contracts, policy=None):
    """A genuine FX-only account does not fabricate dummy funded asset prices."""
    empty = pd.DataFrame(index=fx.fx_spots.index)
    return account(
        empty,
        pd.Series(dtype=object),
        empty.iloc[:1],
        fx,
        spec=policy or spec(),
        local_settlement_dates=pd.DatetimeIndex([]),
        forwards=contracts,
    )


def test_delayed_settlement_is_a_receivable_not_spendable_cash_or_a_second_payoff():
    """Fixing, present value and payment are compared with separate closed-form cash flows."""
    _, _, fx = sample()
    d = fx.fx_spots.index
    forwards = pd.DataFrame(
        [[d[0], d[1], d[2], "EUR", 1.0, "capital"]], columns=COLUMNS, index=["first"]
    )
    result = pure_forward(fx, forwards)
    q = 100.0 / 1.1
    strike = 1.1 * (1 + 0.06 * 29 / 365) / (1 + 0.03 * 29 / 365)
    payoff = q * (1.2 - strike)
    cash_fix = 100.0 * (1 + 0.06 * 29 / 365)
    receivable = payoff / (1 + 0.08 * 4 / 365)
    assert result.accounts.cash.iloc[1] == pytest.approx(cash_fix)
    assert result.contract_values["first"].iloc[1] == pytest.approx(receivable)
    assert result.nav.iloc[1] == pytest.approx(cash_fix + receivable)
    cash_paid = cash_fix * (1 + 0.08 * 4 / 365) + payoff
    assert result.accounts.cash.iloc[2] == pytest.approx(cash_paid)
    assert result.nav.iloc[2] == pytest.approx(cash_paid)
    assert result.cash_flows.forward_settlement.iloc[1] == 0
    assert result.cash_flows.forward_settlement.iloc[2] == pytest.approx(payoff)
    assert result.cash_flows.forward_settlement.sum() == pytest.approx(payoff)
    assert result.contracts.loc["first", "paid"]
    assert result.pnl.cash_interest.iloc[2] == pytest.approx(cash_fix * 0.08 * 4 / 365)


def test_principal_hedge_roll_uses_current_holdings_after_equity_trade():
    """A roll has a fresh strike/notional; the old contract and delayed payment remain separate."""
    p, ccys, fx = sample()
    d = p.index
    weights = pd.DataFrame([[1.0, 0.0], [0.5, 0.0]], index=d[[0, 1]], columns=p.columns)
    contracts = pd.DataFrame(
        [
            [d[0], d[1], d[2], "EUR", -1.0, "asset_currency_value"],
            [d[1], d[3], d[3], "EUR", -0.5, "asset_currency_value"],
        ],
        columns=COLUMNS,
        index=["old", "new"],
    )
    policy = replace(spec(), financing="reference")
    result = account(
        p,
        ccys,
        weights,
        fx,
        spec=policy,
        local_settlement_dates=pd.DatetimeIndex([]),
        forwards=contracts,
    )
    assert result.contracts.loc["old", "local_quantity"] == pytest.approx(-100 / 1.1)
    assert result.contracts.loc["new", "local_quantity"] == pytest.approx(-0.5 * 50 / 1.2)
    old_strike = 1.1 * (1 + 0.06 * 29 / 365) / (1 + 0.03 * 29 / 365)
    assert result.contracts.loc["old", "fixed_payoff"] == pytest.approx(
        -100 / 1.1 * (1.2 - old_strike)
    )
    assert result.cash_flows.forward_settlement.iloc[2] != 0
    assert result.contract_values.old.iloc[1] != 0
    assert result.contract_values.new.iloc[1] == 0
    assert result.diagnostics.reconciliation_error.abs().max() < 1e-12


def test_full_funded_hedge_matches_exact_reference_total_return():
    """One-period principal hedging agrees with QIS's typed funded return, without extra cash."""
    p, ccys, fx = sample()
    p, ccys = p.iloc[:2, :1], ccys.iloc[:1]
    d = p.index
    contract = pd.DataFrame(
        [[d[0], d[1], d[1], "EUR", -1.0, "asset_currency_value"]], columns=COLUMNS, index=["hedge"]
    )
    result = account(
        p,
        ccys,
        pd.DataFrame(1.0, index=d[:1], columns=p.columns),
        fx,
        spec=replace(spec(), financing="reference"),
        local_settlement_dates=pd.DatetimeIndex([]),
        forwards=contract,
    )
    exact = qis.compute_fx_returns(
        fx,
        qis.FxAssetReturnsRequest(
            p,
            ccys,
            qis.FxReturnType.REFERENCE_TOTAL,
            qis.FxReturnCoordinate.SIMPLE,
            reference_ccy="USD",
            hedge_ratios=1.0,
            frequency="ME",
        ),
    )["ME"].values.iloc[1, 0]
    assert result.nav.iloc[-1] == pytest.approx(100 * (1 + exact))


def test_overlay_counterexample_removes_closing_notional_carry_error():
    """An already-unfunded payoff earns collateral interest once, unlike the carry-flag recipe."""
    d = pd.to_datetime(["2024-01-31", "2024-02-29"])
    fx = qis.FxRatesData(
        pd.DataFrame({"USD": 1.0, "EUR": [1.0, 1.1]}, index=d),
        pd.DataFrame({"USD": 0.06, "EUR": 0.06}, index=d),
    )
    contracts = pd.DataFrame([[d[0], d[1], d[1], "EUR", 1.0, "capital"]], columns=COLUMNS)
    correct = pure_forward(fx, contracts)
    assert correct.nav.iloc[-1] == pytest.approx(110.0 + 100 * 0.06 * 29 / 365)
    old = qis.backtest_model_portfolio(
        pd.DataFrame({"Overlay": [100.0, 110.0]}, index=d),
        pd.DataFrame({"Overlay": [1.0]}, index=d[:1]),
        funding_rate=fx.domestic_rates.USD,
        instruments_carry=pd.DataFrame({"Overlay": 0.06}, index=d),
    )
    assert (old.nav.iloc[-1] - correct.nav.iloc[-1]) * 100 == pytest.approx(4.76712328767)


def test_prefix_causality_and_owned_outputs():
    """Later prices/rates do not enter earlier accounting and returned fields are copied."""
    p, ccys, fx = sample()
    d = p.index
    contracts = pd.DataFrame([[d[0], d[2], d[3], "EUR", -1.0, "capital"]], columns=COLUMNS)
    full = run(forwards=contracts)
    short = account(
        p.iloc[:2],
        ccys,
        pd.DataFrame([[0.8, -0.8]], index=d[:1], columns=p.columns),
        fx,
        spec=spec(),
        local_settlement_dates=pd.DatetimeIndex([]),
        forwards=contracts,
    )
    pd.testing.assert_series_equal(full.nav.iloc[:2], short.nav)
    edited = full.accounts
    edited.iloc[:] = 0
    assert full.accounts.nav.iloc[0] == 100
    assert full.to_portfolio_data().nav.iloc[0] == 100


@pytest.mark.parametrize("event", ["trade", "local_settlement", "fixing"])
def test_stale_marks_cannot_execute_an_event(event):
    """An explicit valuation-only fill never authorizes trading, financing settlement or fixing."""
    p, ccys, fx = sample()
    d = p.index
    p.loc[d[1], "A"] = np.nan
    spots = fx.fx_spots.copy()
    spots.loc[d[1], "EUR"] = np.nan
    observed = spots.notna()
    fx = qis.FxRatesData(spots, fx.domestic_rates)
    w = pd.DataFrame([[0.8, -0.8]], index=d[:1], columns=p.columns)
    settlements = pd.DatetimeIndex([])
    forwards = None
    if event == "trade":
        w.loc[d[1]] = [0.0, 0.0]
    elif event == "local_settlement":
        settlements = d[1:2]
    else:
        forwards = pd.DataFrame([[d[0], d[1], d[2], "EUR", 1.0, "capital"]], columns=COLUMNS)
    with pytest.raises(ValueError, match="stale|fixing quote"):
        account(
            p,
            ccys,
            w,
            fx,
            spec=replace(spec(), missing_quotes="carry_valuation"),
            local_settlement_dates=settlements,
            forwards=forwards,
            spot_is_observed=observed,
        )


def test_valuation_fill_preserves_units_and_flags_missing_marks():
    """Stale valuation retains units without forced liquidation or invented returns."""
    p, ccys, fx = sample()
    p.iloc[1, 0] = np.nan
    w = pd.DataFrame([[0.8, -0.8]], index=p.index[:1], columns=p.columns)
    with pytest.raises(ValueError, match="Missing valuation quote"):
        account(p, ccys, w, fx, spec=spec(), local_settlement_dates=pd.DatetimeIndex([]))
    result = account(
        p,
        ccys,
        w,
        fx,
        spec=replace(spec(), missing_quotes="carry_valuation"),
        local_settlement_dates=pd.DatetimeIndex([]),
    )
    np.testing.assert_array_equal(result.asset_units.iloc[0], result.asset_units.iloc[1])
    assert result.diagnostics.stale_price_marks.iloc[1] == 1
    assert result.diagnostics.reconciliation_error.abs().max() < 1e-12


def test_signed_losses_remain_valid_but_negative_capital_cannot_compound():
    """A large monetary loss is not discarded by log/index conventions."""
    d = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"])
    fx = qis.FxRatesData(
        pd.DataFrame({"USD": 1.0, "EUR": [1.0, 0.2, 0.3]}, index=d),
        pd.DataFrame({"USD": 0.0, "EUR": 0.0}, index=d),
    )
    contracts = pd.DataFrame([[d[0], d[1], d[1], "EUR", 3.0, "capital"]], columns=COLUMNS)
    result = pure_forward(fx, contracts)
    assert result.nav.iloc[1] == pytest.approx(-140.0)
    with pytest.raises(ValueError, match="Nonpositive capital"):
        result.to_portfolio_data()
    contracts.loc[1] = [d[1], d[2], d[2], "EUR", 1.0, "capital"]
    with pytest.raises(ValueError, match="nonpositive capital"):
        pure_forward(fx, contracts, replace(spec(), notional_rule="current_nav"))


@pytest.mark.parametrize("change", ["capital", "cash_lag", "labels", "rate", "schedule"])
def test_invalid_account_inputs_fail_explicitly(change):
    """No hidden assumptions or filling future data can make an invalid account executable."""
    p, ccys, fx = sample()
    policy, forwards = spec(), None
    with pytest.raises(ValueError):
        if change == "capital":
            policy = replace(policy, initial_capital=0.0)
        elif change == "cash_lag":
            policy = replace(policy, cash_policy=qis.FxCashPolicy(quote_lag=0))
        elif change == "labels":
            ccys = ccys.iloc[::-1]
        elif change == "rate":
            fx = qis.FxRatesData(fx.fx_spots, fx.domestic_rates.iloc[1:])
        else:
            forwards = pd.DataFrame(
                [[p.index[0], p.index[0], p.index[1], "EUR", 1.0, "capital"]], columns=COLUMNS
            )
        account(
            p,
            ccys,
            pd.DataFrame([[1.0, -1.0]], index=p.index[:1], columns=p.columns),
            fx,
            spec=policy,
            local_settlement_dates=pd.DatetimeIndex([]),
            forwards=forwards,
        )


@pytest.mark.parametrize("rule,expected", [("fixed", 120.0), ("current_nav", 121.0)])
def test_capital_sizing_at_forward_roll_is_explicit(rule, expected):
    """Each roll uses the selected reference capital base, not the old local quantity."""
    d = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"])
    fx = qis.FxRatesData(
        pd.DataFrame({"USD": 1.0, "EUR": [1.0, 1.1, 1.21]}, index=d),
        pd.DataFrame({"USD": 0.0, "EUR": 0.0}, index=d),
    )
    contracts = pd.DataFrame(
        [[d[0], d[1], d[1], "EUR", 1.0, "capital"], [d[1], d[2], d[2], "EUR", 1.0, "capital"]],
        columns=COLUMNS,
    )
    result = pure_forward(fx, contracts, replace(spec(), notional_rule=rule))
    assert result.nav.iloc[-1] == pytest.approx(expected)


def test_local_sweep_cadence_changes_the_strategy_and_conversion_of_cash_flows():
    """Daily local P&L conversion and endpoint financing are intentionally distinct contracts."""
    d = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    p = pd.DataFrame({"A": [100.0, 110.0, 100.0]}, index=d)
    fx = qis.FxRatesData(
        pd.DataFrame({"USD": 1.0, "EUR": [1.0, 2.0, 1.0]}, index=d),
        pd.DataFrame({"USD": 0.0, "EUR": 0.0}, index=d),
    )
    w = pd.DataFrame({"A": [1.0]}, index=d[:1])
    args = (p, pd.Series({"A": "EUR"}), w, fx)
    daily = account(*args, spec=spec(), local_settlement_dates=d[1:])
    endpoint = account(*args, spec=spec(), local_settlement_dates=d[-1:])
    assert daily.nav.iloc[-1] == pytest.approx(100.0 + 10.0 * 2.0 - 10.0 * 1.0)
    assert endpoint.nav.iloc[-1] == pytest.approx(100.0)


def test_non_usd_reference_uses_true_cross_rates_and_its_own_collateral_rate():
    """A CHF capital account uses CHF-per-EUR, not the raw USD-per-EUR quote."""
    p, ccys, fx = sample()
    p = p.iloc[:2]
    w = pd.DataFrame([[0.8, -0.3]], index=p.index[:1], columns=p.columns)
    result = account(
        p,
        ccys,
        w,
        fx,
        spec=replace(spec(), reference_ccy="CHF"),
        local_settlement_dates=p.index[-1:],
    )
    local = p.iloc[1].to_numpy() / p.iloc[0].to_numpy() - 1.0
    cash = np.array([0.03, 0.01]) * 29 / 365
    eur_chf = (1.2 / 1.02) / (1.1 / 1.05)
    expected = 100.0 * (1.0 + 0.01 * 29 / 365) + 100.0 * np.dot(
        [0.8, -0.3], (local - cash) * [eur_chf, 1.0]
    )
    assert result.nav.iloc[-1] == pytest.approx(expected)


def test_zero_collateral_interest_policy_is_not_silently_overridden():
    """A contractual zero cash yield leaves forward pricing and payoff unchanged."""
    _, _, fx = sample()
    d = fx.fx_spots.index
    schedule = pd.DataFrame([[d[0], d[1], d[1], "EUR", 1.0, "capital"]], columns=COLUMNS)
    result = pure_forward(fx, schedule, replace(spec(), cash_rate_policy="zero"))
    assert result.pnl.cash_interest.eq(0).all()
    assert result.nav.iloc[-1] == pytest.approx(100.0 + result.contracts.fixed_payoff.sum())


def test_observation_mask_is_required_even_when_fx_data_has_filled_the_gap():
    """An ambiguous quote history cannot assert that a filled value was executable."""
    p, ccys, fx = sample()
    with pytest.raises(ValueError, match="spot_is_observed"):
        qis.backtest_fx_account(
            p,
            ccys,
            pd.DataFrame([[1.0, -1.0]], index=p.index[:1], columns=p.columns),
            fx,
            spec=spec(),
            local_settlement_dates=pd.DatetimeIndex([]),
            spot_is_observed=None,
        )


def test_prefixing_mark_matches_discounted_currency_legs():
    """Price independently through local and reference zero-coupon legs before fixing."""
    _, _, fx = sample()
    d = fx.fx_spots.index
    contracts = pd.DataFrame(
        [[d[0], d[2], d[3], "EUR", -1.0, "capital"]], columns=COLUMNS, index=["hedge"]
    )
    result = pure_forward(fx, contracts)
    q = -100.0 / 1.1
    strike = 1.1 * (1 + 0.06 * 33 / 365) / (1 + 0.03 * 33 / 365)
    local_leg = 1.2 / (1 + 0.05 * 4 / 365)
    reference_leg = strike / (1 + 0.08 * 4 / 365)
    payment_adjustment = (1 + 0.08 * 4 / 365) / (1 + 0.08 * 31 / 365)
    expected = q * (local_leg - reference_leg) * payment_adjustment
    assert result.contract_values.hedge.iloc[1] == pytest.approx(expected, abs=1e-12)
    assert result.cash_flows.forward_settlement.iloc[:2].eq(0).all()
    assert result.accounts.cash.iloc[1] == pytest.approx(100 * (1 + 0.06 * 29 / 365))


def test_future_quotes_cannot_change_prior_forward_marks_or_asset_accounts():
    """Perturb future prices, FX and rates while comparing all prior balances and marks."""
    p, ccys, fx = sample()
    d = p.index
    targets = pd.DataFrame([[0.8, -0.3]], index=d[:1], columns=p.columns)
    contracts = pd.DataFrame(
        [[d[0], d[2], d[3], "EUR", -1.0, "asset_currency_value"]],
        columns=COLUMNS, index=["hedge"],
    )
    kwargs = dict(spec=spec(), local_settlement_dates=pd.DatetimeIndex([]), forwards=contracts)
    original = account(p, ccys, targets, fx, **kwargs)
    p2, s2, r2 = p.copy(), fx.fx_spots.copy(), fx.domestic_rates.copy()
    p2.iloc[2:] *= 1.7
    s2.loc[d[2]:, ["EUR", "CHF"]] *= 0.6
    r2.iloc[2:] += 0.15
    changed = account(p2, ccys, targets, qis.FxRatesData(s2, r2), **kwargs)
    for field in ("accounts", "asset_units", "local_balances", "contract_values", "cash_flows"):
        pd.testing.assert_frame_equal(getattr(original, field).iloc[:2],
                                      getattr(changed, field).iloc[:2])
    assert original.contracts.strike.equals(changed.contracts.strike)
    assert original.nav.iloc[-1] != changed.nav.iloc[-1]

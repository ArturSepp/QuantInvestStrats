"""Exact notional FX payoffs checked against independent settlement cash flows."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import qis


def market():
    """Monthly market with deliberately material financing and FX changes."""
    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"])
    spots = pd.DataFrame({"EUR": [1.0, 1.2, 0.9], "USD": 1.0}, index=dates)
    rates = pd.DataFrame(
        {"EUR": [0.05 * 365 / 29, 0.03, 0.08], "USD": [0.02 * 365 / 29, 0.01, 0.07]}, index=dates
    )
    return qis.FxRatesData(spots, rates), pd.DataFrame(
        {"equity": [100.0, 110.0, 104.0]}, index=dates
    )


def translated(fx, prices, **kwargs):
    """Request the arithmetic locally financed payoff without synthetic wealth."""
    return qis.compute_fx_returns(
        fx,
        qis.FxAssetReturnsRequest(
            prices,
            "EUR",
            qis.FxReturnType.LOCAL_EXCESS_TRANSLATED,
            qis.FxReturnCoordinate.SIMPLE,
            "USD",
            **kwargs,
        ),
    )["ME"]


def forward(fx, **kwargs):
    """Request the arithmetic long-local forward payoff."""
    return qis.compute_fx_returns(
        fx,
        qis.FxCurrencyReturnsRequest(
            ("EUR",), "USD", return_type=qis.FxReturnType.FX_FORWARD, **kwargs
        ),
    )["ME"]


def normalization(frequency="ME", scale=1.0, fixed=False):
    """Fully specified synthetic-capital policy for testing optional indices."""
    return qis.FxIndexNormalization(
        initial_value=100.0,
        notional_to_capital=scale,
        notional_rule="fixed" if fixed else "capital_proportional",
        settlement_frequency=frequency,
        reinvestment="none" if fixed else "each_interval",
        cash_treatment="as_returned",
        missing_returns="stop",
    )


def test_translated_excess_is_direct_financed_cashflow():
    """110 of stock less 105 of local debt, translated at 1.2, pays 6 per 100."""
    fx, prices = market()
    result = translated(fx, prices)
    assert result.values.equity.iloc[1] == pytest.approx((110.0 - 105.0) * 1.2 / 100.0)
    assert result.values.equity.iloc[1] != pytest.approx((1.1 / 1.05 - 1.0) * 1.2)
    assert result.synthetic_index is None
    assert "notional" in result.payoff_basis and "financed" in result.payoff_basis


def test_forward_is_terminal_spot_minus_opening_forward_over_opening_spot():
    """The reference-per-local forward is S0 times reference/local cash growth."""
    fx, _ = market()
    result = forward(fx, diagnostics=True)
    fwd = 1.0 * 1.02 / 1.05
    assert result.values.EUR.iloc[1] == pytest.approx((1.2 - fwd) / 1.0)
    assert result.synthetic_index is None
    assert result.diagnostics["hedge_cost"].EUR.iloc[1] == pytest.approx(1.0 - fwd)
    assert result.diagnostics["forward_to_spot"].EUR.iloc[1] == pytest.approx(fwd)
    assert "notional" in result.payoff_basis and "forward" in result.payoff_basis


@pytest.mark.parametrize("hedge", [-0.5, 0.0, 0.5, 1.0, 1.05, 1.7])
def test_exact_reference_excess_decompositions(hedge):
    """Both exact simple-space identities reconcile with independently settled hedges."""
    fx, prices = market()
    p = translated(fx, prices).values.equity.iloc[1]
    phi = forward(fx).values.EUR.iloc[1]
    got = qis.compute_fx_returns(
        fx,
        qis.FxAssetReturnsRequest(
            prices,
            "EUR",
            qis.FxReturnType.REFERENCE_EXCESS,
            qis.FxReturnCoordinate.SIMPLE,
            "USD",
            hedge_ratios=hedge,
        ),
    )["ME"].values.equity.iloc[1]
    terminal = 110.0 * 1.2 + 100.0 * hedge * (1.02 / 1.05 - 1.2)
    expected = terminal / 100.0 - 1.0 - 0.02
    kappa = 1.0 - 1.02 / 1.05
    assert got == pytest.approx(expected, abs=5e-16)
    assert got == pytest.approx(p + (1.05 - hedge) * phi, abs=5e-16)
    assert got == pytest.approx(
        0.1 - 0.05 + (1 - hedge) * phi + 0.1 * 0.2 + kappa * 0.05, abs=5e-16
    )


@pytest.mark.parametrize("frequency", ["B", "ME", "QE"])
@pytest.mark.parametrize("local,reference", [("EUR", "USD"), ("USD", "EUR"), ("EUR", "EUR")])
def test_forward_actual_days_opening_rates_and_pair_orientation(frequency, local, reference):
    """Independent settlement prices cover negative rates, leap days and inverse pairs."""
    dates = pd.bdate_range("2023-12-29", "2024-07-01")
    spots = pd.DataFrame(
        {"EUR": 1.1 * np.exp(0.0003 * np.arange(len(dates))), "USD": 1.0}, index=dates
    )
    rates = pd.DataFrame({"EUR": -0.03, "USD": 0.07}, index=dates)
    fx = qis.FxRatesData(spots, rates)
    result = qis.compute_fx_returns(
        fx,
        qis.FxCurrencyReturnsRequest(
            (local,), reference, qis.FxReturnType.FX_FORWARD, frequency=frequency
        ),
    )[frequency]
    cross = qis.df_asfreq(spots[local] / spots[reference], freq=frequency)
    dt = cross.index.to_series().diff().dt.days / 365.0
    cl = rates[local].iloc[0] * dt
    cr = rates[reference].iloc[0] * dt
    forward_price = cross.shift(1) * (1.0 + cr) / (1.0 + cl)
    expected = (cross - forward_price) / cross.shift(1)
    np.testing.assert_allclose(result.values[local], expected, atol=5e-15, equal_nan=True)
    assert result.periods.index.equals(cross.index)


def test_only_opening_information_prices_a_forward():
    """A rate announced at settlement reprices only the following reset interval."""
    fx, _ = market()
    before = forward(fx).values
    fx.domestic_rates.iloc[1, :] = [0.6, 0.2]
    after = forward(fx).values
    pd.testing.assert_frame_equal(before.iloc[:2], after.iloc[:2], check_exact=True)
    assert before.EUR.iloc[2] != after.EUR.iloc[2]
    truncated = qis.FxRatesData(fx.fx_spots.iloc[:2], fx.domestic_rates.iloc[:2])
    pd.testing.assert_frame_equal(forward(truncated).values, after.iloc[:2], check_freq=False)


def test_unknown_opening_financing_is_not_zero_or_backfilled():
    """A later known quote must not finance an earlier asset or forward payoff."""
    fx, prices = market()
    fx.domestic_rates.iloc[0, 0] = np.nan
    assert pd.isna(translated(fx, prices).values.equity.iloc[1])
    assert pd.isna(forward(fx).values.EUR.iloc[1])
    assert pd.notna(forward(fx).values.EUR.iloc[2])


@pytest.mark.parametrize("kind", ["LOCAL_EXCESS_TRANSLATED", "FX_FORWARD"])
def test_signed_losses_remain_available_without_capital_or_log(kind):
    """A loss below -100% of notional remains valid P&L and need not have a NAV."""
    fx, prices = market()
    fx.fx_spots.EUR = [1.0, 2.0, 1.0]
    prices.equity = [100.0, 10.0, 90.0]
    fx.domestic_rates.EUR = 0.0
    fx.domestic_rates.USD = 4.0 * 365 / 29
    result = translated(fx, prices) if kind == "LOCAL_EXCESS_TRANSLATED" else forward(fx)
    assert result.values.iloc[1, 0] < -1.0
    assert result.synthetic_index is None
    with pytest.raises(ValueError, match="positive"):
        result.to_index(normalization())
    fixed = result.to_index(normalization(fixed=True))
    assert fixed.iloc[1, 0] == pytest.approx(100.0 * (1.0 + result.values.iloc[1, 0]))


@pytest.mark.parametrize("kind", ["LOCAL_EXCESS_TRANSLATED", "FX_FORWARD"])
@pytest.mark.parametrize("coordinate", ["LOG", "CASH_RELATIVE_SIMPLE"])
def test_notional_payoffs_reject_unfunded_log_coordinates(kind, coordinate):
    """An index policy does not silently turn signed P&L into a funded/log return."""
    fx, prices = market()
    if kind == "LOCAL_EXCESS_TRANSLATED":
        request = qis.FxAssetReturnsRequest(
            prices,
            "EUR",
            getattr(qis.FxReturnType, kind),
            getattr(qis.FxReturnCoordinate, coordinate),
            "USD",
            normalization=normalization(),
        )
    else:
        request = qis.FxCurrencyReturnsRequest(
            ("EUR",),
            "USD",
            getattr(qis.FxReturnType, kind),
            getattr(qis.FxReturnCoordinate, coordinate),
            normalization=normalization(),
        )
    with pytest.raises(ValueError, match="SIMPLE"):
        qis.compute_fx_returns(fx, request)


@pytest.mark.parametrize("kind", ["LOCAL_EXCESS_TRANSLATED", "FX_FORWARD"])
def test_normalized_log_index_transforms_completed_scaled_payoff(kind):
    """Log normalized wealth is log1p(scaled completed P&L), never log spot plus log carry."""
    fx, prices = market()
    result = translated(fx, prices) if kind == "LOCAL_EXCESS_TRANSLATED" else forward(fx)
    nav = result.to_index(normalization(scale=0.5))
    got = qis.to_returns(nav, is_log_returns=True)
    expected = np.log1p(0.5 * result.values)
    np.testing.assert_allclose(got, expected, atol=6e-16, equal_nan=True)
    assert result.synthetic_index is None


@pytest.mark.parametrize("kind", ["LOCAL_EXCESS_TRANSLATED", "FX_FORWARD"])
def test_financed_payoffs_require_opening_cash_policy(kind):
    """Closing quote descriptive analytics is not an opening financing contract."""
    fx, prices = market()
    with pytest.raises(ValueError, match="opening"):
        if kind == "LOCAL_EXCESS_TRANSLATED":
            translated(fx, prices, cash_policy=qis.FxCashPolicy(quote_lag=0))
        else:
            forward(fx, cash_policy=qis.FxCashPolicy(quote_lag=0))


def test_translated_payoff_rejects_added_asset_hedge():
    """The defined locally financed payoff has no extra principal hedge."""
    fx, prices = market()
    with pytest.raises(ValueError, match="hedge"):
        translated(fx, prices, hedge_ratios=0.5)


def test_mixed_cadence_and_dated_currency_spells_for_translated_payoff():
    """The new kind retains the existing cadence/spell contract and masks unit changes."""
    dates = pd.bdate_range("2023-12-29", "2024-07-01")
    fx = qis.FxRatesData(
        pd.DataFrame({"EUR": 1.2, "USD": 1.0}, index=dates),
        pd.DataFrame({"EUR": 0.03, "USD": 0.05}, index=dates),
    )
    prices = pd.DataFrame(
        {"a": np.arange(len(dates)) + 100.0, "b": np.arange(len(dates)) + 200.0}, index=dates
    )
    spells = pd.DataFrame(
        {"a": ["EUR", "USD"], "b": ["EUR", "EUR"]},
        index=pd.to_datetime(["2023-12-29", "2024-02-12"]),
    )
    request = qis.FxAssetReturnsRequest(
        prices,
        spells,
        qis.FxReturnType.LOCAL_EXCESS_TRANSLATED,
        qis.FxReturnCoordinate.SIMPLE,
        "USD",
        frequency=pd.Series({"a": "ME", "b": "QE"}),
        currency_changes="mask_boundary",
    )
    results = qis.compute_fx_returns(fx, request)
    assert set(results) == {"ME", "QE"}
    assert pd.isna(results["ME"].values.loc["2024-02-29", "a"])
    assert pd.notna(results["ME"].values.loc["2024-03-31", "a"])
    pd.testing.assert_frame_equal(results["ME"].currency_spells, spells[["a"]])


def test_premium_to_exact_cost_delta_is_separate_from_day_count():
    """Holding quotes, days and spot fixed isolates -f^2/(1+f), with no ACT/365 change."""
    fx, _ = market()
    result = forward(fx, diagnostics=True)
    f = result.diagnostics["forward_premium"].EUR
    x = result.diagnostics["spot"].EUR
    np.testing.assert_allclose(
        result.values.EUR - (x + f), (-f * f / (1 + f)).where(x.notna()), atol=2e-16, equal_nan=True
    )


def test_optional_forward_index_carries_its_explicit_normalization():
    """Attached indices carry capital/settlement meaning, while diagnostics remain owned."""
    fx, _ = market()
    result = forward(fx, normalization=normalization(), diagnostics=True)
    assert result.index_kind == "compounded_index"
    assert result.normalization.initial_value == 100.0
    assert result.synthetic_index.iloc[0, 0] == 100.0
    original = result.diagnostics["forward_premium"]
    changed = result.diagnostics
    changed["forward_premium"].iloc[:, :] = 99.0
    pd.testing.assert_frame_equal(result.diagnostics["forward_premium"], original)
    with pytest.raises(ValueError, match="frequency"):
        result.to_index(replace(normalization(), settlement_frequency="B"))


@pytest.mark.parametrize("method", ["get_daily_carry_local_return", "get_fx_total_return_nav",
                                    "get_carry_fx_return_nav"])
def test_legacy_currency_analytics_warn_and_preserve_premium_history(method):
    """Retirement names the exact replacement without silently relabelling old economics."""
    fx, _ = market()
    kwargs = {"is_normalise_by_spot_vol": False} if method == "get_carry_fx_return_nav" else {}
    with pytest.warns(DeprecationWarning, match="deprecated in QIS 5.46.0.*FX_FORWARD.*6.0.0"):
        got = getattr(fx, method)("EUR", "USD", **kwargs)
    f = 1.05/1.02-1.
    if method == "get_daily_carry_local_return":
        assert got[1].iloc[1] == pytest.approx(f)
    elif method == "get_fx_total_return_nav":
        assert got.iloc[1] == pytest.approx(1.+.2+f)
    else:
        assert got.iloc[1] == pytest.approx(1.+f)

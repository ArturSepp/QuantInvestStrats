"""ACT/365 Fixed contracts shared by FX conversion and portfolio cash funding."""

import numpy as np
import pandas as pd
import pytest
import qis
from qis.fx.fx_rates_data import FxRatesData


def _data(dates):
    """Build supplied annual rates and spot levels on an explicit calendar."""
    return FxRatesData(
        fx_spots=pd.DataFrame({"USD": 1.0, "EUR": np.linspace(1.0, 1.2, len(dates))}, index=dates),
        domestic_rates=pd.DataFrame({"USD": 0.06, "EUR": 0.09}, index=dates),
    )


@pytest.mark.parametrize(
    "start,end,freq,days",
    [
        ("2024-01-31", "2024-02-29", "ME", 29),
        ("2023-01-31", "2023-02-28", "ME", 28),
        ("2024-03-01", "2024-03-04", "B", 3),
        ("2024-03-31", "2024-06-30", "QE", 91),
        ("2024-01-01", "2025-01-01", "YS", 366),
    ],
)
def test_fx_cash_agrees_with_actual_funding(start, end, freq, days):
    """FX subtraction equals independent calendar accrual and native funding."""
    dates = pd.DatetimeIndex([start, end])
    data = _data(dates)
    cash = data._get_period_cash_returns(dates, "USD", freq, False)
    expected = 0.06 * days / 365.0
    assert cash.iloc[0] == 0.0
    np.testing.assert_allclose(cash.iloc[-1], expected, rtol=0, atol=1e-15)
    generic = -qis.compute_excess_returns(pd.Series(0.0, index=dates), data.domestic_rates["USD"])
    np.testing.assert_allclose(cash, generic, rtol=0, atol=1e-15)
    funded = qis.backtest_model_portfolio(
        prices=pd.DataFrame({"asset": 100.0}, index=dates),
        weights=pd.DataFrame({"asset": [0.0]}, index=dates[:1]),
        funding_rate=data.domestic_rates["USD"],
    )
    np.testing.assert_allclose(
        funded.nav.iloc[-1] / funded.nav.iloc[0] - 1, expected, rtol=0, atol=1e-14
    )


@pytest.mark.parametrize("is_log", [False, True])
@pytest.mark.parametrize("hedge", [0.0, 0.4, 1.0])
def test_hedged_excess_matches_independent_dated_delivery_payoff(is_log, hedge):
    """Use opening quotes and actual maturity even when closing rates jump."""
    dates = pd.date_range("2024-01-31", periods=4, freq="ME")
    data = _data(dates)
    data.domestic_rates["USD"] = [0.06, 0.5, 0.02, 0.9]
    data.domestic_rates["EUR"] = [0.09, 0.2, 0.03, 0.8]
    prices = pd.Series([100.0, 110.0, 105.0, 112.0], index=dates, name="asset")
    actual = qis.compute_fx_returns(
        data,
        qis.FxAssetReturnsRequest(
            prices=(prices).to_frame(),
            local_ccys="EUR",
            return_type=qis.FxReturnType.REFERENCE_EXCESS
            if True
            else qis.FxReturnType.REFERENCE_TOTAL,
            coordinate=qis.FxReturnCoordinate.LOG if is_log else qis.FxReturnCoordinate.SIMPLE,
            frequency="ME",
            cash_policy=qis.FxCashPolicy(quote_lag=1),
            reference_ccy="USD",
            hedge_ratios=hedge,
            zero_return_to_nan=False,
        ),
    )["ME"].values.iloc[:, 0]
    dt = dates.to_series().diff().dt.days / 365.0
    c_l = data.domestic_rates["EUR"].shift(1) * dt
    c_b = data.domestic_rates["USD"].shift(1) * dt
    x = data.fx_spots.EUR / data.fx_spots.EUR.shift(1)
    gross = (prices / prices.shift(1)) * x - hedge * (x - (1 + c_b) / (1 + c_l))
    expected = np.log(gross) - np.log1p(c_b) if is_log else gross - 1 - c_b
    pd.testing.assert_series_equal(actual, expected.rename(prices.name), rtol=0, atol=1e-14)


def test_explicit_forward_intervals_retain_weekend_maturity():
    """Friday quotes priced for month-end intervals use each complete month."""
    quotes = pd.to_datetime(["2024-02-29", "2024-03-28", "2024-04-30"])
    data = _data(quotes)
    dates = pd.date_range("2024-02-29", periods=3, freq="ME")
    actual = data.get_forward_rate_for_local_ccy("EUR", "USD", period_dates=dates)
    dt = np.array([31.0, 30.0]) / 365.0
    np.testing.assert_allclose(
        actual.iloc[:-1], (1 + 0.09 * dt) / (1 + 0.06 * dt) - 1, rtol=0, atol=1e-15
    )
    assert pd.isna(actual.iloc[-1])


def test_cash_nav_and_forward_quotes_keep_existing_reset_grid():
    """Business-day resets accrue three days over a weekend without changing f to kappa."""
    dates = pd.to_datetime(["2024-03-01", "2024-03-04", "2024-03-05"])
    data = _data(dates)
    cash = data.get_local_rate(freq="B", local_ccy="EUR")
    expected = pd.Series([0.0, 0.09 * 3 / 365, 0.09 / 365], index=dates, name="EUR")
    pd.testing.assert_series_equal(cash, expected, check_freq=False)
    nav = data.build_local_cash_nav("EUR", freq="B")
    np.testing.assert_allclose(nav, (1 + expected).cumprod(), atol=1e-15, rtol=0)
    carry = data.get_forward_rate_for_local_ccy("EUR", "USD", period_dates=dates).shift(1)
    dt = pd.Series([0.0, 3 / 365, 1 / 365], index=dates)
    np.testing.assert_allclose(
        carry.iloc[1:], ((1 + 0.09 * dt) / (1 + 0.06 * dt) - 1).iloc[1:], atol=1e-15, rtol=0
    )


def test_missing_opening_cash_does_not_become_free_funding():
    """Initial cash is zero; an unknown earned interval leaves cash NAV unknown."""
    dates = pd.date_range("2024-01-31", periods=4, freq="ME")
    data = _data(dates)
    data.domestic_rates["USD"] = [np.nan, -0.02, -0.03, -0.04]
    cash = data._get_period_cash_returns(dates, "USD", "ME", False)
    assert cash.iloc[0] == 0 and pd.isna(cash.iloc[1])
    np.testing.assert_allclose(cash.iloc[2], -0.02 * 31 / 365)
    nav = data.build_local_cash_nav("USD", freq="ME")
    assert nav.iloc[0] == 1.0 and nav.iloc[1:].isna().all()
    data.domestic_rates.iloc[1, 0] = -20.0
    with pytest.raises(ValueError, match="positive"):
        data._get_period_cash_returns(dates, "USD", "ME", True)


def test_closing_quote_cannot_change_completed_cash_interval():
    """A closing-date rate update never alters the preceding cash accrual."""
    dates = pd.date_range("2024-01-31", periods=2, freq="ME")
    data = _data(dates)
    before = data._get_period_cash_returns(dates, "USD", "ME", False)
    data.domestic_rates.iloc[-1, 0] = 9.0
    after = data._get_period_cash_returns(dates, "USD", "ME", False)
    pd.testing.assert_series_equal(before, after)


def test_empty_cash_and_explicit_forward_grids_remain_empty():
    """No completed reset interval yields an empty result, not a fabricated accrual."""
    data = _data(pd.to_datetime(["2024-01-10", "2024-01-11"]))
    assert data.get_local_rate(freq="QE", local_ccy="USD").empty
    result = data.get_forward_rate_for_local_ccy("EUR", "USD", period_dates=pd.DatetimeIndex([]))
    assert result.empty and result.name == "EUR-USD"

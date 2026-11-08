"""ACT/365 Fixed contracts shared by FX conversion and portfolio cash funding."""
import numpy as np
import pandas as pd
import pytest
import qis
from qis.market_data.fx_rates_data import FxRatesData


def _data(dates):
    """Build supplied annual rates and spot levels on an explicit calendar."""
    return FxRatesData(
        fx_spots=pd.DataFrame({'USD': 1., 'EUR': np.linspace(1., 1.2, len(dates))},
                             index=dates),
        domestic_rates=pd.DataFrame({'USD': .06, 'EUR': .09}, index=dates))


@pytest.mark.parametrize('start,end,freq,days', [
    ('2024-01-31', '2024-02-29', 'ME', 29),
    ('2023-01-31', '2023-02-28', 'ME', 28),
    ('2024-03-01', '2024-03-04', 'B', 3),
    ('2024-03-31', '2024-06-30', 'QE', 91),
    ('2024-01-01', '2025-01-01', 'YS', 366),
])
def test_fx_cash_agrees_with_actual_funding(start, end, freq, days):
    """FX subtraction equals independent calendar accrual and native funding."""
    dates = pd.DatetimeIndex([start, end])
    data = _data(dates)
    cash = data._get_period_cash_returns(dates, 'USD', freq, False)
    expected = .06 * days / 365.
    assert cash.iloc[0] == 0.
    np.testing.assert_allclose(cash.iloc[-1], expected, rtol=0, atol=1e-15)
    generic = -qis.compute_excess_returns(
        pd.Series(0., index=dates), data.domestic_rates['USD'])
    np.testing.assert_allclose(cash, generic, rtol=0, atol=1e-15)
    funded = qis.backtest_model_portfolio(
        prices=pd.DataFrame({'asset': 100.}, index=dates),
        weights=pd.DataFrame({'asset': [0.]}, index=dates[:1]),
        funding_rate=data.domestic_rates['USD'])
    np.testing.assert_allclose(funded.nav.iloc[-1] / funded.nav.iloc[0] - 1,
                               expected, rtol=0, atol=1e-14)


@pytest.mark.parametrize('is_log', [False, True])
@pytest.mark.parametrize('hedge', [0., .4, 1.])
def test_hedged_excess_matches_independent_dated_delivery_payoff(is_log, hedge):
    """Use opening quotes and actual maturity even when closing rates jump."""
    dates = pd.date_range('2024-01-31', periods=4, freq='ME')
    data = _data(dates)
    data.domestic_rates['USD'] = [.06, .5, .02, .9]
    data.domestic_rates['EUR'] = [.09, .2, .03, .8]
    prices = pd.Series([100., 110., 105., 112.], index=dates, name='asset')
    _, actual = data.compute_performance_of_local_ccy_asset_in_reference_ccy(
        prices, hedge, 'EUR', 'USD', freq='ME', is_log_returns=is_log,
        is_excess_returns=True)
    dt = dates.to_series().diff().dt.days / 365.
    c_l = data.domestic_rates['EUR'].shift(1) * dt
    c_b = data.domestic_rates['USD'].shift(1) * dt
    x = data.fx_spots.EUR / data.fx_spots.EUR.shift(1)
    gross = (prices / prices.shift(1)) * x - hedge * (x - (1+c_b)/(1+c_l))
    expected = np.log(gross)-np.log1p(c_b) if is_log else gross-1-c_b
    pd.testing.assert_series_equal(actual, expected.rename(None), rtol=0, atol=1e-14)


def test_explicit_forward_intervals_retain_weekend_maturity():
    """Friday quotes priced for month-end intervals use each complete month."""
    quotes = pd.to_datetime(['2024-02-29', '2024-03-28', '2024-04-30'])
    data = _data(quotes)
    dates = pd.date_range('2024-02-29', periods=3, freq='ME')
    actual = data.get_forward_rate_for_local_ccy('EUR', 'USD', period_dates=dates)
    dt = np.array([31., 30.]) / 365.
    np.testing.assert_allclose(actual.iloc[:-1], (1+.09*dt)/(1+.06*dt)-1,
                               rtol=0, atol=1e-15)
    assert pd.isna(actual.iloc[-1])


def test_cash_nav_and_daily_carry_keep_existing_reset_grid():
    """Business-day resets accrue three days over a weekend without changing f to kappa."""
    dates = pd.to_datetime(['2024-03-01', '2024-03-04', '2024-03-05'])
    data = _data(dates)
    cash = data.get_local_rate(freq='B', local_ccy='EUR')
    expected = pd.Series([0., .09*3/365, .09/365], index=dates, name='EUR')
    pd.testing.assert_series_equal(cash, expected, check_freq=False)
    nav = data.build_local_cash_nav('EUR', freq='B')
    np.testing.assert_allclose(nav, (1+expected).cumprod(), atol=1e-15, rtol=0)
    _, carry = data.get_daily_carry_local_return('EUR', 'USD')
    dt = pd.Series([0., 3/365, 1/365], index=dates)
    np.testing.assert_allclose(carry, (1+.09*dt)/(1+.06*dt)-1, atol=1e-15, rtol=0)


def test_missing_opening_cash_does_not_become_free_funding():
    """Initial cash is zero; an unknown earned interval leaves cash NAV unknown."""
    dates = pd.date_range('2024-01-31', periods=4, freq='ME')
    data = _data(dates)
    data.domestic_rates['USD'] = [np.nan, -.02, -.03, -.04]
    cash = data._get_period_cash_returns(dates, 'USD', 'ME', False)
    assert cash.iloc[0] == 0 and pd.isna(cash.iloc[1])
    np.testing.assert_allclose(cash.iloc[2], -.02*31/365)
    nav = data.build_local_cash_nav('USD', freq='ME')
    assert nav.iloc[0] == 1. and nav.iloc[1:].isna().all()
    data.domestic_rates.iloc[1, 0] = -20.
    with pytest.raises(ValueError, match='positive'):
        data._get_period_cash_returns(dates, 'USD', 'ME', True)


def test_closing_quote_cannot_change_completed_cash_interval():
    """A closing-date rate update never alters the preceding cash accrual."""
    dates = pd.date_range('2024-01-31', periods=2, freq='ME')
    data = _data(dates)
    before = data._get_period_cash_returns(dates, 'USD', 'ME', False)
    data.domestic_rates.iloc[-1, 0] = 9.
    after = data._get_period_cash_returns(dates, 'USD', 'ME', False)
    pd.testing.assert_series_equal(before, after)


def test_empty_cash_and_explicit_forward_grids_remain_empty():
    """No completed reset interval yields an empty result, not a fabricated accrual."""
    data = _data(pd.to_datetime(['2024-01-10', '2024-01-11']))
    assert data.get_local_rate(freq='QE', local_ccy='USD').empty
    result = data.get_forward_rate_for_local_ccy(
        'EUR', 'USD', period_dates=pd.DatetimeIndex([]))
    assert result.empty and result.name == 'EUR-USD'


# Legacy behavior is intentionally regression-tested until removal in QIS 6.0.0.
pytestmark = pytest.mark.filterwarnings(
    'ignore:compute_.*deprecated in QIS 5.45.0:DeprecationWarning')

pytestmark = [pytestmark, pytest.mark.filterwarnings(
    "ignore:get_.*deprecated in QIS 5.46.0:DeprecationWarning")]

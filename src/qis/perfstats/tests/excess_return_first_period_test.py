"""The first return period of every excess-return path accrues cash and is compounded.

The cash return of the period ``(t-1, t]`` is the annual rate known at the return date ``t-1``
times the ACT/365 fraction of the period, and the first return date accrues nothing. A rate
series that starts on the first price date therefore covers every period, and the per-annum
excess return compounds the whole history over the same elapsed years ``Y`` as the per-annum
return. Expected values are computed independently with NumPy.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import qis
from qis.datasets import generate_synthetic_universe
from qis.perfstats.returns import (compute_excess_returns, compute_pa_excess_compounded_returns,
                                   get_excess_returns_nav)


def _pa_excess_reference(simple_returns: np.ndarray, dates: pd.DatetimeIndex,
                         rate: float) -> float:
    """Per-annum compounded excess return over the whole sample, first period included.

    Args:
        simple_returns: returns of the periods ending on dates[1:]
        dates: return dates, the first of which anchors the NAV at one
        rate: constant annual cash rate accrued ACT/365

    Returns:
        the per-annum excess return with Y = days / 365.25
    """
    days = np.diff(dates.to_numpy()).astype('timedelta64[D]').astype(float)
    excess = simple_returns - rate * days / 365.0
    years = (dates[-1] - dates[0]).days / 365.25
    return float(np.prod(1.0 + excess) ** (1.0 / years) - 1.0)


def test_pa_excess_return_compounds_the_first_period() -> None:
    """A rate series starting on the first return date leaves no period uncharged."""
    dates = pd.date_range('2019-12-31', periods=25, freq='ME')
    returns = pd.Series(np.r_[0.0, np.full(24, 0.01)], index=dates, name='asset')
    rates = pd.Series(0.12, index=dates, name='cash')

    actual = compute_pa_excess_compounded_returns(returns=returns, rates_data=rates,
                                                  first_date=dates[0])

    expected = _pa_excess_reference(returns.to_numpy()[1:], dates, 0.12)
    assert abs(actual - expected) < 1e-14


def test_daily_rates_on_monthly_returns_use_the_previous_month_end_quote() -> None:
    """Each month is charged the daily quote known at the previous month-end.

    The rate steps from 36.5% to 73% on 15 February and to 146% on 15 March. February is
    charged 36.5% for its 29 days and March 73% for its 31 days: the quotes of 31 January and
    29 February, not the second-to-last daily quotes of February and March.
    """
    daily = pd.bdate_range('2024-01-01', '2024-03-29')
    rates = pd.Series(0.365, index=daily)
    rates.loc[rates.index >= pd.Timestamp('2024-02-15')] = 0.73
    rates.loc[rates.index >= pd.Timestamp('2024-03-15')] = 1.46
    month_ends = pd.DatetimeIndex(['2024-01-31', '2024-02-29', '2024-03-31'])
    returns = pd.Series([0.0, 0.10, 0.20], index=month_ends)

    excess = compute_excess_returns(returns=returns, rates_data=rates)

    expected = pd.Series([0.0, 0.10 - 0.365 * 29 / 365, 0.20 - 0.73 * 31 / 365],
                         index=month_ends)
    pd.testing.assert_series_equal(excess, expected, check_exact=False, rtol=0.0, atol=1e-15)


def test_late_rates_shorten_the_window_consistently() -> None:
    """Periods before the first quote have no excess return and are not annualised.

    The rates start on 31 March 2020, so the first period with a known rate ends on 1 April.
    The per-annum excess return compounds from 31 March and divides by the years from 31 March,
    with a warning that names the later start.
    """
    dates = pd.date_range('2020-01-01', '2021-12-31', freq='D')
    rng = np.random.default_rng(7)
    returns = pd.Series(np.r_[0.0, 0.0004 + 0.01 * rng.standard_normal(len(dates) - 1)],
                        index=dates, name='asset')
    rates = pd.Series(0.03, index=pd.date_range('2020-03-31', '2021-12-31', freq='D'))

    with pytest.warns(UserWarning, match='rates_data'):
        actual = compute_pa_excess_compounded_returns(returns=returns, rates_data=rates,
                                                      first_date=dates[0])

    window = dates[dates >= pd.Timestamp('2020-03-31')]
    expected = _pa_excess_reference(returns.loc[window].to_numpy()[1:], window, 0.03)
    assert abs(actual - expected) < 1e-14


def test_get_excess_returns_nav_keeps_the_first_period() -> None:
    """The excess NAV is defined on the first date and includes the first period's return."""
    dates = pd.bdate_range('2024-01-01', periods=5)
    prices = pd.Series([100.0, 110.0, 99.0, 99.0, 108.9], index=dates, name='asset')
    funding = pd.Series(0.365, index=pd.bdate_range('2023-12-29', periods=6))

    nav = get_excess_returns_nav(prices=prices, funding_rate=funding, freq='B')

    days = np.diff(dates.to_numpy()).astype('timedelta64[D]').astype(float)
    excess = prices.to_numpy()[1:] / prices.to_numpy()[:-1] - 1.0 - 0.001 * days
    path = np.r_[1.0, np.cumprod(1.0 + excess)]
    expected = pd.Series(path * prices.iloc[-1] / path[-1], index=dates, name='asset')
    pd.testing.assert_series_equal(nav, expected, check_exact=False, rtol=1e-14, check_freq=False)


@pytest.fixture(scope='module')
def universe_prices() -> pd.DataFrame:
    """Clean synthetic prices from 2 January 2014 to 31 December 2025."""
    universe = generate_synthetic_universe(start='2014-01-02', end='2025-12-31', seed=20260725,
                                           apply_quirks=False)
    return universe.prices[['SEQ_US', 'SCM_GLD']]


def _table(prices: pd.DataFrame, cash: pd.Series) -> pd.DataFrame:
    """Risk-adjusted table on month-end volatility with a cash series."""
    params = qis.PerfParams(freq='ME', return_type=qis.ReturnTypes.LOG, rates_data=cash)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return qis.compute_ra_perf_table(prices=prices, perf_params=params)


def test_ra_perf_table_excess_columns_do_not_depend_on_an_earlier_rate_start(
        universe_prices: pd.DataFrame) -> None:
    """A flat 2% rate from the first price date gives the same table as one from before it.

    The native `PA_EXCESS_RETURN` is recomputed from the daily prices: SEQ_US -0.704% and
    SCM_GLD 6.323% a year. The rate starting on the first price date previously dropped the
    first day and gave -0.651% and 6.265%.
    """
    from_first = _table(universe_prices, pd.Series(0.02, index=pd.bdate_range('2014-01-02',
                                                                                '2025-12-31')))
    from_before = _table(universe_prices, pd.Series(0.02, index=pd.bdate_range('2013-12-02',
                                                                                 '2025-12-31')))
    stats = [qis.PerfStat.PA_EXCESS_RETURN, qis.PerfStat.AN_LOG_EXCESS_RETURN,
             qis.PerfStat.SHARPE_EXCESS, qis.PerfStat.SHARPE_LOG_EXCESS,
             qis.PerfStat.CALMAR_RATIO, qis.PerfStat.SORTINO_RATIO]
    for stat in stats:
        np.testing.assert_allclose(from_first[stat.to_str()].astype(float),
                                   from_before[stat.to_str()].astype(float), rtol=0.0, atol=1e-14)

    dates = universe_prices.index
    expected = [_pa_excess_reference(universe_prices[asset].pct_change().to_numpy()[1:], dates,
                                     0.02) for asset in universe_prices.columns]
    actual = from_first[qis.PerfStat.PA_EXCESS_RETURN.to_str()].astype(float).to_numpy()
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(actual, [-0.00704, 0.06323], atol=5e-6)
    np.testing.assert_allclose(from_first[qis.PerfStat.AN_LOG_EXCESS_RETURN.to_str()].astype(float),
                               np.log1p(expected), rtol=0.0, atol=1e-14)


def test_ra_perf_table_monthly_rates_from_the_first_month_end(
        universe_prices: pd.DataFrame) -> None:
    """Monthly 2% cash from 31 January 2014 charges the first sampled month.

    `SHARPE_EXCESS` of SCM_GLD is the per-annum excess return on the month-end boundaries from
    31 January 2014 over `VOL`, 0.3782; dropping February 2014 previously gave 0.3925.
    """
    cash = pd.Series(0.02, index=pd.date_range('2014-01-31', '2025-12-31', freq='ME'))
    table = _table(universe_prices, cash)

    month_end = universe_prices.resample('ME').last()
    simple = month_end.pct_change().to_numpy()[1:]
    log_r = np.log(month_end).diff().to_numpy()[1:]
    vol = np.sqrt(12.0) * log_r.std(axis=0, ddof=1)
    pa_excess = np.array([_pa_excess_reference(simple[:, j], month_end.index, 0.02)
                          for j in range(simple.shape[1])])
    sharpe = table[qis.PerfStat.SHARPE_EXCESS.to_str()].astype(float).to_numpy()
    np.testing.assert_allclose(sharpe, pa_excess / vol, rtol=0.0, atol=1e-12)
    assert abs(sharpe[1] - 0.3782) < 5e-5

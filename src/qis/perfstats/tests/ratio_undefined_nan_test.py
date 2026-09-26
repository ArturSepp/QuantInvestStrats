"""Undefined ratio columns of the risk-adjusted table are missing, not infinite or zero.

Three ratios have a denominator that can vanish or be undefined on a legitimate history:

- ``CALMAR_RATIO`` divides by ``MAX_DD``, which is zero for a NAV that never falls. The code
  previously returned minus infinity for a positive return, the wrong sign as well as undefined.
- ``SORTINO_RATIO`` divides by ``DOWNSIDE_VOL``, the sample standard deviation of the negative
  returns, which needs at least two of them. The column previously reported 0.0 downside
  volatility and an infinite Sortino ratio for fewer than two negative returns.
- ``MAX_DD_VOL`` divides by ``VOL``, which is missing for a single sampled return and zero for a
  constant sampled price. The column previously reported 0.0 for both.

Expected values are computed directly with NumPy from the month-end prices.
"""

import numpy as np
import pandas as pd

# qis
from qis.perfstats.config import PerfParams, PerfStat
from qis.perfstats.perf_stats import compute_ra_perf_table


_DATES = pd.date_range('2020-01-31', periods=25, freq='ME')
_PARAMS = PerfParams(freq='ME', freq_drawdown='ME')


def _rising_prices() -> pd.Series:
    """Return a month-end NAV that rises every month, so its maximum drawdown is zero."""
    growth = 1.01 + 0.002 * np.sin(np.arange(24))
    return pd.Series(100.0 * np.concatenate(([1.0], np.cumprod(growth))), index=_DATES,
                     name='rising')


def _value(table: pd.DataFrame, stat: PerfStat) -> float:
    """One statistic of the single-asset table as a float."""
    return float(table.iloc[0][stat.to_str()])


def test_calmar_ratio_is_missing_without_a_drawdown() -> None:
    """A NAV that never falls has no drawdown, so the Calmar ratio is undefined."""
    table = compute_ra_perf_table(prices=_rising_prices(), perf_params=_PARAMS)
    assert _value(table, PerfStat.MAX_DD) == 0.0
    assert _value(table, PerfStat.PA_EXCESS_RETURN) > 0.0
    assert np.isnan(_value(table, PerfStat.CALMAR_RATIO))


def test_calmar_ratio_is_unchanged_with_a_drawdown() -> None:
    """With a drawdown the ratio is the excess p.a. return over the absolute maximum drawdown."""
    prices = _rising_prices()
    prices.iloc[10:] *= 0.9
    table = compute_ra_perf_table(prices=prices, perf_params=_PARAMS)
    values = prices.to_numpy()
    max_dd = float(np.min(values / np.maximum.accumulate(values) - 1.0))
    np.testing.assert_allclose(_value(table, PerfStat.MAX_DD), max_dd, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(_value(table, PerfStat.CALMAR_RATIO),
                               _value(table, PerfStat.PA_EXCESS_RETURN) / abs(max_dd),
                               rtol=1e-14)


def test_sortino_ratio_is_missing_with_fewer_than_two_negative_returns() -> None:
    """Zero or one negative return leaves the downside volatility and the Sortino undefined."""
    no_loss = _rising_prices()
    one_loss = _rising_prices()
    one_loss.iloc[10:] *= 0.98
    for prices, n_negative in ((no_loss, 0), (one_loss, 1)):
        returns = np.diff(np.log(prices.to_numpy()))
        assert int(np.sum(returns < 0.0)) == n_negative
        table = compute_ra_perf_table(prices=prices, perf_params=_PARAMS)
        assert np.isnan(_value(table, PerfStat.DOWNSIDE_VOL))
        assert np.isnan(_value(table, PerfStat.SORTINO_RATIO))


def test_sortino_ratio_with_two_negative_returns_is_finite() -> None:
    """Two distinct negative returns give a defined downside volatility and Sortino ratio."""
    prices = _rising_prices()
    prices.iloc[10:] *= 0.98
    prices.iloc[15:] *= 0.97
    table = compute_ra_perf_table(prices=prices, perf_params=_PARAMS)
    returns = np.diff(np.log(prices.to_numpy()))
    downside = np.sqrt(12.0) * np.std(returns[returns < 0.0], ddof=1)
    np.testing.assert_allclose(_value(table, PerfStat.DOWNSIDE_VOL), downside, rtol=1e-12)
    assert np.isfinite(_value(table, PerfStat.SORTINO_RATIO))


def test_max_dd_vol_is_missing_when_volatility_is_undefined_or_zero() -> None:
    """One sampled return has no volatility; a constant sampled price has zero volatility."""
    single = pd.Series([100.0, 90.0], index=_DATES[:2], name='single')
    table = compute_ra_perf_table(prices=single, perf_params=_PARAMS)
    assert np.isnan(_value(table, PerfStat.VOL))
    np.testing.assert_allclose(_value(table, PerfStat.MAX_DD), -0.1, rtol=0.0, atol=1e-14)
    assert np.isnan(_value(table, PerfStat.MAX_DD_VOL))

    constant = pd.Series(100.0, index=_DATES[:12], name='constant')
    table = compute_ra_perf_table(prices=constant, perf_params=_PARAMS)
    assert _value(table, PerfStat.VOL) == 0.0
    assert np.isnan(_value(table, PerfStat.MAX_DD_VOL))

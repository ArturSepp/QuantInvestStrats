"""A coarse drawdown grid keeps each asset's final observation.

``PerfParams.freq_drawdown`` samples levels on complete calendar boundaries. Before this fix the
trailing incomplete period was dropped, so with ``freq_drawdown='ME'`` a fall inside the current
month was invisible to ``MAX_DD`` and ``CURRENT_DD``: the "current" drawdown referred to the last
month-end. The drawdown grid now appends each asset's own final observation when it lies between
boundaries, so ``CURRENT_DD`` is the drawdown at the last observation. The volatility grid still
uses complete boundaries only.

Expected values are computed directly from the month-end levels and the final level with NumPy.
"""

import numpy as np
import pandas as pd

# qis
from qis.perfstats.config import PerfParams, PerfStat
from qis.perfstats.perf_stats import compute_ra_perf_table


_DATES = pd.bdate_range('2020-01-01', '2021-06-15')


def _crash_in_last_month() -> pd.Series:
    """Daily levels rising steadily, then 30% lower from the first business day of June 2021."""
    prices = pd.Series(100.0 * np.exp(0.0003 * np.arange(len(_DATES))), index=_DATES,
                       name='crash')
    prices.loc['2021-06-01':] *= 0.7
    return prices


def _expected(levels: np.ndarray) -> tuple:
    """Maximum drawdown, current drawdown and worst simple return of a level path."""
    drawdowns = levels / np.maximum.accumulate(levels) - 1.0
    returns = levels[1:] / levels[:-1] - 1.0
    return float(drawdowns.min()), float(drawdowns[-1]), float(returns.min())


def test_month_end_drawdown_grid_sees_a_fall_in_the_current_month() -> None:
    """MAX_DD and CURRENT_DD include the final, off-grid observation."""
    prices = _crash_in_last_month()
    table = compute_ra_perf_table(prices=prices, perf_params=PerfParams(freq_drawdown='ME'))
    month_end = prices.resample('ME').last()
    # resample labels the incomplete June at 2021-06-30; the observation is on 2021-06-15
    levels = month_end.to_numpy()
    max_dd, current_dd, worst = _expected(levels)
    row = table.loc['crash']
    np.testing.assert_allclose(row[PerfStat.MAX_DD.to_str()], max_dd, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(row[PerfStat.CURRENT_DD.to_str()], current_dd, rtol=0.0,
                               atol=1e-12)
    np.testing.assert_allclose(row[PerfStat.WORST.to_str()], worst, rtol=0.0, atol=1e-12)
    assert row[PerfStat.CURRENT_DD.to_str()] < -0.29


def test_volatility_grid_still_uses_complete_boundaries() -> None:
    """The partial month enters the drawdown grid only, not the volatility sample."""
    prices = _crash_in_last_month()
    table = compute_ra_perf_table(prices=prices, perf_params=PerfParams(freq_drawdown='ME'))
    month_end = prices.resample('ME').last().iloc[:-1]  # complete months only
    vol = np.sqrt(12.0) * np.diff(np.log(month_end.to_numpy())).std(ddof=1)
    np.testing.assert_allclose(table.loc['crash', PerfStat.VOL.to_str()], vol, rtol=1e-12)
    assert table.loc['crash', PerfStat.NUM_OBS.to_str()] == len(month_end) - 1


def test_terminated_asset_keeps_its_own_final_observation_only() -> None:
    """A terminated column adds its own final level; its neighbour is not resampled by it."""
    prices = _crash_in_last_month().to_frame()
    prices['neighbour'] = 100.0 * np.exp(0.0002 * np.arange(len(_DATES)))
    prices.loc[prices.index > pd.Timestamp('2021-03-17'), 'crash'] = np.nan
    prices.loc['2021-03-10':'2021-03-17', 'crash'] *= 0.8
    table = compute_ra_perf_table(prices=prices, perf_params=PerfParams(freq_drawdown='ME'))

    crash = prices['crash'].dropna()
    crash_levels = np.append(crash.resample('ME').last().iloc[:-1].to_numpy(), crash.iloc[-1])
    max_dd, current_dd, worst = _expected(crash_levels)
    np.testing.assert_allclose(table.loc['crash', PerfStat.MAX_DD.to_str()], max_dd, atol=1e-12)
    np.testing.assert_allclose(table.loc['crash', PerfStat.CURRENT_DD.to_str()], current_dd,
                               atol=1e-12)
    np.testing.assert_allclose(table.loc['crash', PerfStat.WORST.to_str()], worst, atol=1e-12)

    isolated = compute_ra_perf_table(prices=prices[['neighbour']],
                                     perf_params=PerfParams(freq_drawdown='ME'))
    for stat in (PerfStat.MAX_DD, PerfStat.CURRENT_DD, PerfStat.WORST, PerfStat.BEST):
        np.testing.assert_allclose(table.loc['neighbour', stat.to_str()],
                                   isolated.loc['neighbour', stat.to_str()], rtol=0.0, atol=0.0)


def test_calendar_day_default_is_unchanged() -> None:
    """On the default 'D' grid the final observation is already a grid date."""
    prices = _crash_in_last_month()
    table = compute_ra_perf_table(prices=prices, perf_params=PerfParams())
    max_dd, current_dd, worst = _expected(prices.to_numpy())
    np.testing.assert_allclose(table.loc['crash', PerfStat.MAX_DD.to_str()], max_dd, atol=1e-12)
    np.testing.assert_allclose(table.loc['crash', PerfStat.CURRENT_DD.to_str()], current_dd,
                               atol=1e-12)

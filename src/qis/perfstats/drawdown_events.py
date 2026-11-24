"""Compare two observed NAVs over drawdowns dated by a reference benchmark."""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from qis.perfstats.perf_stats import compute_drawdowns_stats_table
from qis.utils.dates import TimePeriod


def compute_benchmark_drawdown_events(
    strategy_nav: pd.Series,
    benchmark_nav: pd.Series,
    min_depth: float = 0.10,
    named_windows: Mapping[str, TimePeriod] | None = None,
) -> pd.DataFrame:
    """Return simple peak-to-trough and peak-to-recovery returns on benchmark dates.

    Args:
        strategy_nav: Observed positive strategy levels. An absent endpoint leaves that
            strategy return missing, without changing the benchmark event or the other return.
        benchmark_nav: Observed positive reference levels. Missing quotes are omitted before
            episode detection. Restrict both inputs to the reporting span before calling.
        min_depth: Positive loss magnitude in [0, 1). Only strictly deeper events qualify
            (ignoring floating-point roundoff at equality).
        named_windows: Optional labels mapped to observed benchmark peak and trough dates.
            These replace automatic episodes. The trough must be a minimum and the peak a
            maximum of its supplied window. Recovery is the first subsequent observed level
            at or above that local peak, within the supplied sample.

    Returns:
        Events sorted by depth then peak date. Dates are benchmark observations. For an
        open event, end is the latest benchmark date and recovery and recovery returns
        are missing. All four return columns are unannualised decimal simple returns.
    """
    if not np.isfinite(min_depth) or not 0 <= min_depth < 1:
        raise ValueError('min_depth must be a finite loss magnitude in [0, 1)')
    for nav in (strategy_nav, benchmark_nav):
        if not isinstance(nav, pd.Series) or not isinstance(nav.index, pd.DatetimeIndex):
            raise TypeError('NAVs must be Series with a DatetimeIndex')
        if not nav.index.is_unique or nav.index.hasnans:
            raise ValueError('NAV dates must be unique and observed')
        observed = nav.dropna()
        if not np.isfinite(observed).all() or (observed <= 0).any():
            raise ValueError('Observed NAVs must be finite and positive')
    benchmark = benchmark_nav.dropna().sort_index()
    if named_windows is None:
        episodes = compute_drawdowns_stats_table(benchmark, freq=None)
        events = episodes[['start', 'trough', 'end']].rename(columns={'start': 'peak'}).copy()
        events['recovery'] = events['end'].where(episodes['is_recovered'], pd.NaT)
        events['name'] = ''
    else:
        records = []
        for name, period in named_windows.items():
            peak, trough = period.start, period.end
            if peak not in benchmark.index or trough not in benchmark.index or peak >= trough:
                raise ValueError(f'{name}: supply observed benchmark peak and later trough dates')
            window = benchmark.loc[peak:trough]
            if window.iloc[0] != window.max() or window.iloc[-1] != window.min():
                raise ValueError(f'{name}: endpoints must be the peak and trough of the window')
            after = benchmark.loc[trough:].iloc[1:]
            recovery = after[after.ge(window.iloc[0])].first_valid_index()
            end = benchmark.index[-1] if recovery is None else recovery
            records.append((peak, trough, end, recovery, name))
        events = pd.DataFrame(records, columns=['peak', 'trough', 'end', 'recovery', 'name'])

    for point in ('peak', 'trough', 'recovery', 'end'):
        events[point] = pd.array(events[point], dtype=benchmark.index.dtype)
    for label, nav in [('benchmark', benchmark), ('strategy', strategy_nav)]:
        opening = events['peak'].map(nav)
        for point in ('trough', 'recovery'):
            closing = events[point].map(nav)
            events[f'{label}_{point}_return'] = (closing - opening).div(opening).astype(float)
    loss = -events['benchmark_trough_return']
    at_cutoff = np.isclose(loss, min_depth, atol=0, rtol=8 * np.finfo(float).eps)
    qualifies = loss.gt(min_depth) & ~at_cutoff
    return events.loc[qualifies].sort_values(
        ['benchmark_trough_return', 'peak'], kind='stable').reset_index(drop=True)

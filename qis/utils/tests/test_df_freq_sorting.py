"""
``df_asfreq`` on a panel whose index is not in chronological order.

pandas 3.0 changed ``pd.concat``: with ``axis=1`` and ``sort=False`` (the default) the union of
two non-identical DatetimeIndexes is left in appearance order, so the dates carried only by the
second frame are appended after the last date of the first. Under pandas 2.2 the same call
returned a sorted union. A panel joined from a benchmark series and a strategy nav on different
calendars therefore arrives at ``df_asfreq`` unsorted, and two things went wrong: the
pre-reindex ffill carried a later price onto an earlier date, and ``reindex(method='ffill')``
raised ``ValueError: index must be monotonic increasing or decreasing`` from inside pandas.

To confirm these checks can fail, remove the ``sort_index`` guard at the top of ``df_asfreq``:
both checks below fail with that ValueError, raised from ``pandas/core/indexes/base.py``. That
was run before this file was committed. The look-ahead check is the second half of the guard's
job and pins where it sits: sorting after the pre-reindex ffill rather than before it would
leave the December price of the following year on 2020-12-31.
"""
# packages
import numpy as np
import pandas as pd
# qis / project
from qis.utils.df_freq import df_asfreq

# dates the benchmark leg does not carry; 2020-12-31 is a sampled date at 'YE' and 'ME', so a
# value filled onto it in row order is visible in the resampled output rather than hidden
BENCHMARK_GAPS: pd.DatetimeIndex = pd.DatetimeIndex(['2020-07-15', '2020-12-31'])


def make_unsorted_panel() -> pd.DataFrame:
    """two price columns on different calendars, joined the way pandas 3.0 joins them.

    Prices are 1, 2, 3, ... along the business-day index, so any value filled from the wrong
    row is identifiable by size alone.

    Returns:
        panel of 'benchmark' and 'strategy' whose index is in appearance order rather than
        chronological order
    """
    index = pd.bdate_range(start='2020-01-01', end='2021-12-31')
    strategy = pd.Series(np.arange(1.0, len(index) + 1.0), index=index, name='strategy')
    benchmark = strategy.drop(index=BENCHMARK_GAPS).rename('benchmark')
    # the union pd.concat(axis=1, sort=False) produces in pandas 3.0: the gap dates last
    appearance_order = benchmark.index.append(BENCHMARK_GAPS)
    panel = pd.concat([benchmark, strategy], axis=1, sort=False)
    return panel.reindex(index=appearance_order)  # pinned, so pandas 2.2 sees the same order


def test_unsorted_panel_is_unsorted() -> None:
    """the fixture states the condition under test, so it cannot pass by being well-formed"""
    panel = make_unsorted_panel()
    assert not panel.index.is_monotonic_increasing
    assert panel.index.is_unique


def test_df_asfreq_matches_the_sorted_panel() -> None:
    """resampling is independent of the row order the panel arrives in"""
    panel = make_unsorted_panel()
    resampled = df_asfreq(df=panel, freq='ME', include_start_date=True, include_end_date=True)
    expected = df_asfreq(df=panel.sort_index(), freq='ME',
                         include_start_date=True, include_end_date=True)
    pd.testing.assert_frame_equal(resampled, expected)
    assert resampled.index.is_monotonic_increasing


def test_df_asfreq_does_not_fill_a_later_price_onto_an_earlier_date() -> None:
    """the sampled value is the last observation at or before the date, not the terminal one.

    Reference computed the other way round: ``Series.asof`` on the chronologically sorted
    column, which does not go through ``reindex`` at all.
    """
    panel = make_unsorted_panel()
    resampled = df_asfreq(df=panel, freq='ME', include_start_date=True, include_end_date=True)
    sorted_benchmark = panel['benchmark'].sort_index()
    for date in (pd.Timestamp('2020-07-31'), pd.Timestamp('2020-12-31')):
        assert np.isclose(resampled.loc[date, 'benchmark'], sorted_benchmark.asof(date)), \
            (f"df_asfreq reports {resampled.loc[date, 'benchmark']} on {date.date()}, "
             f"asof on the sorted column gives {sorted_benchmark.asof(date)}")
    assert resampled.loc[pd.Timestamp('2020-12-31'), 'benchmark'] < sorted_benchmark.iloc[-1]

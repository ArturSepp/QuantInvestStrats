"""
``prices_at_freq`` on a panel whose index is not in chronological order.

With ``freq`` set, the work is done by ``df_asfreq``, which sorts - see
``qis/utils/tests/test_df_freq_sorting.py`` for the pandas 3.0 concat change that lets an
unsorted panel get this far. With ``freq=None`` the ffill happens here instead, in row order,
and the dates a column does not carry sit after the last row of the panel: the fill then takes
the terminal price backwards onto them, and every return differenced from those rows is wrong
without anything raising.

To confirm these checks can fail, remove the ``sort_index`` guard at the top of
``prices_at_freq``: the fill returns 522.0 on 2020-12-31 where the benchmark last traded at
260.0, and both checks below fail. That was run before this file was committed.
"""
# packages
import numpy as np
import pandas as pd
# qis / project
from qis.perfstats.returns import prices_at_freq, to_returns

BENCHMARK_GAPS: pd.DatetimeIndex = pd.DatetimeIndex(['2020-07-15', '2020-12-31'])


def make_unsorted_panel() -> pd.DataFrame:
    """two price columns on different calendars, in the row order pandas 3.0 joins them in.

    Returns:
        panel of 'benchmark' and 'strategy' whose index is in appearance order rather than
        chronological order
    """
    index = pd.bdate_range(start='2020-01-01', end='2021-12-31')
    strategy = pd.Series(np.arange(1.0, len(index) + 1.0), index=index, name='strategy')
    benchmark = strategy.drop(index=BENCHMARK_GAPS).rename('benchmark')
    appearance_order = benchmark.index.append(BENCHMARK_GAPS)
    panel = pd.concat([benchmark, strategy], axis=1, sort=False)
    return panel.reindex(index=appearance_order)  # pinned, so pandas 2.2 sees the same order


def test_prices_at_freq_fills_in_date_order() -> None:
    """with freq=None the ffill runs here, so the panel is sorted before it"""
    panel = make_unsorted_panel()
    filled = prices_at_freq(prices=panel, freq=None, ffill_nans=True)
    assert filled.index.is_monotonic_increasing
    pd.testing.assert_frame_equal(filled, prices_at_freq(prices=panel.sort_index(), freq=None))
    sorted_benchmark = panel['benchmark'].sort_index()
    for date in BENCHMARK_GAPS:
        assert np.isclose(filled.loc[date, 'benchmark'], sorted_benchmark.asof(date)), \
            (f"fill reports {filled.loc[date, 'benchmark']} on {date.date()}, "
             f"asof on the sorted column gives {sorted_benchmark.asof(date)}")


def test_to_returns_is_independent_of_row_order() -> None:
    """daily returns off the unsorted panel match the ones off the sorted panel"""
    panel = make_unsorted_panel()
    pd.testing.assert_frame_equal(to_returns(prices=panel, freq=None),
                                  to_returns(prices=panel.sort_index(), freq=None))

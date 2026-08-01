"""
``compute_periodic_returns`` on a panel whose index is not in chronological order.

The heatmap ffills and bfills the panel before resampling it, and both fill in row order. On the
appearance-order index pandas 3.0 returns from ``pd.concat(axis=1)`` - see
``qis/utils/tests/test_df_freq_sorting.py`` for what changed - the dates a column does not carry
sit after the last row, so the fill takes the terminal price backwards onto them. The annual
table then reports a return computed from a price the column did not have on that date, with
nothing raising. ``to_total_returns`` reads the first and last row and is wrong in the same way.

To confirm this check can fail, remove the ``sort_index`` guard at the top of
``compute_periodic_returns``: on the panel below the 2020 benchmark return moves from 260.0 to
522.0 - the terminal price over the first, since the fill puts the terminal price on 2020-12-31
- and the 2021 return from 1.0038 to exactly 0.0. Prices here count 1, 2, 3, ... so the returns
are large by construction; the point is which price lands on the sampling date. That was run
before this file was committed.
"""
# packages
import numpy as np
import pandas as pd
# qis / project
from qis.plots.derived.returns_heatmap import compute_periodic_returns

# 2020-12-31 is the 'YE' sampling date, so a price filled onto it in row order lands in the table
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


def test_periodic_returns_match_the_sorted_panel() -> None:
    """the annual table is independent of the row order the panel arrives in"""
    panel = make_unsorted_panel()
    data = compute_periodic_returns(prices=panel, freq='YE')
    expected = compute_periodic_returns(prices=panel.sort_index(), freq='YE')
    pd.testing.assert_frame_equal(data, expected)


def test_periodic_returns_carry_no_look_ahead_price() -> None:
    """the 2020 return uses the last benchmark price of 2020, not the price at the end of 2021.

    Reference computed the other way round: ``Series.asof`` on the chronologically sorted
    column, which goes through neither the fill nor the resample.
    """
    panel = make_unsorted_panel()
    data = compute_periodic_returns(prices=panel, freq='YE')
    sorted_benchmark = panel['benchmark'].sort_index()
    reference = sorted_benchmark.asof(pd.Timestamp('2020-12-31')) / sorted_benchmark.iloc[0] - 1.0
    reported = data.loc[pd.Timestamp('2020-12-31'), 'benchmark']
    assert np.isclose(reported, reference), \
        f"2020 benchmark return {reported:.4f} against {reference:.4f} computed with asof"

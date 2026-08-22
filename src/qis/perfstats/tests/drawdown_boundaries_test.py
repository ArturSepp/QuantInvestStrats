"""Regression coverage for complete drawdown episode boundaries.

A drawdown episode begins when its running peak level was first reached immediately before prices
fall and ends either at the observation that recovers that peak or at the final observation of an
ongoing drawdown. The fixtures below use short deterministic NAV paths so every date, depth,
duration, and recovery value can be calculated directly without reusing a QIS drawdown helper.
"""

import numpy as np
import pandas as pd

# qis
from qis.perfstats.perf_stats import compute_drawdowns_stats_table


# =============================================================================
# Shared deterministic fixtures
# =============================================================================

_DAILY_DATES = pd.date_range('2024-01-01', periods=9, freq='D')

_COMPLETE_AND_ONGOING_NAV = (100.0, 99.0, 50.0, 90.0, 100.0, 110.0, 99.0, 88.0, 90.0)
_TABLE_COLUMNS = [
    'start',
    'trough',
    'end',
    'max_dd',
    'days_dd',
    'days_to_trough',
    'days_recovery',
    'peak',
    'bottom',
    'recovery',
    'is_recovered',
]

_TOLERANCE = 1e-12


def _complete_and_ongoing_prices(name: str | None = 'nav') -> pd.Series:
    """Create one recovered drawdown followed by one ongoing drawdown.

    Args:
        name: Optional name assigned to the returned Series.

    Returns:
        New daily NAV Series containing the two independently specified episodes.
    """
    return pd.Series(_COMPLETE_AND_ONGOING_NAV, index=_DAILY_DATES, name=name)


def _assert_episode_table(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    """Compare a drawdown table while ignoring its non-contractual row index.

    Args:
        actual: Table returned by ``compute_drawdowns_stats_table``.
        expected: Independently constructed table in required episode order.
    """
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_exact=False,
        rtol=0.0,
        atol=_TOLERANCE,
    )


# =============================================================================
# Complete and ongoing episode boundaries
# =============================================================================

def test_compute_drawdowns_stats_table_includes_peak_and_recovery_boundaries() -> None:
    """Report complete boundaries for recovered and ongoing drawdowns.

    The first episode starts at the January 1 peak of 100, reaches 50 on January 3, and recovers
    to 100 on January 5. Its direct drawdown is ``50 / 100 - 1 = -50%`` over four sampled days.
    The second starts at the January 6 peak of 110, reaches 88 on January 8, and remains below its
    peak at 90 on January 9. It is therefore ongoing after three sampled days.
    """
    prices = _complete_and_ongoing_prices()
    original_prices = prices.copy(deep=True)
    expected = pd.DataFrame(
        {
            'start': [_DAILY_DATES[0], _DAILY_DATES[5]],
            'trough': [_DAILY_DATES[2], _DAILY_DATES[7]],
            'end': [_DAILY_DATES[4], _DAILY_DATES[8]],
            'max_dd': [50.0 / 100.0 - 1.0, 88.0 / 110.0 - 1.0],
            'days_dd': [4.0, 3.0],
            'days_to_trough': [2.0, 2.0],
            'days_recovery': [2.0, 1.0],
            'peak': [100.0, 110.0],
            'bottom': [50.0, 88.0],
            'recovery': [100.0, 90.0],
            'is_recovered': [True, False],
        },
        columns=pd.Index(_TABLE_COLUMNS),
    )

    actual = compute_drawdowns_stats_table(price=prices, freq=None)

    _assert_episode_table(actual, expected)
    pd.testing.assert_series_equal(prices, original_prices)


def test_compute_drawdowns_stats_table_supports_an_unnamed_series() -> None:
    """Make a Series name irrelevant to episode values and labels.

    A Series name is metadata, not part of the drawdown calculation. Removing ``"nav"`` must
    therefore produce the same complete table instead of using ``None`` as a failed column lookup.
    """
    named = compute_drawdowns_stats_table(price=_complete_and_ongoing_prices(), freq=None)

    unnamed = compute_drawdowns_stats_table(
        price=_complete_and_ongoing_prices(name=None),
        freq=None,
    )

    _assert_episode_table(unnamed, named)


# =============================================================================
# Ordering and empty boundaries
# =============================================================================

def test_compute_drawdowns_stats_table_orders_equal_depths_by_start_date() -> None:
    """Resolve equal maximum drawdowns chronologically before applying ``max_num``.

    The path has two exactly 20% recovered drawdowns: 100 to 80 and 120 to 96. Equal depths do not
    determine an order, so the earlier January 1 episode is the deterministic first row and the
    episode retained when ``max_num=1``.
    """
    dates = _DAILY_DATES[:6]
    prices = pd.Series((100.0, 80.0, 100.0, 120.0, 96.0, 120.0), index=dates, name='nav')

    actual = compute_drawdowns_stats_table(price=prices, freq=None)
    limited = compute_drawdowns_stats_table(price=prices, max_num=1, freq=None)

    assert actual['start'].to_list() == [dates[0], dates[3]]
    assert actual['end'].to_list() == [dates[2], dates[5]]
    np.testing.assert_allclose(
        actual['max_dd'].to_numpy(),
        np.asarray((80.0 / 100.0 - 1.0, 96.0 / 120.0 - 1.0)),
        rtol=0.0,
        atol=_TOLERANCE,
    )
    assert actual['is_recovered'].to_list() == [True, True]
    assert limited['start'].to_list() == [dates[0]]


def test_compute_drawdowns_stats_table_returns_empty_schema_without_an_episode() -> None:
    """Return the documented empty schema for monotonic and one-observation histories.

    A strictly increasing NAV never falls below its running peak, and a single observation cannot
    form a peak-to-trough interval. Neither input should manufacture an episode or lose the table's
    documented columns.
    """
    monotonic = pd.Series((100.0, 101.0, 102.0), index=_DAILY_DATES[:3], name='nav')
    one_observation = monotonic.iloc[:1]

    monotonic_table = compute_drawdowns_stats_table(price=monotonic, freq=None)
    one_observation_table = compute_drawdowns_stats_table(price=one_observation, freq=None)

    assert monotonic_table.empty
    assert one_observation_table.empty
    assert monotonic_table.columns.to_list() == _TABLE_COLUMNS
    assert one_observation_table.columns.to_list() == _TABLE_COLUMNS


# =============================================================================
# Sampling boundaries
# =============================================================================

def test_compute_drawdowns_stats_table_honors_native_and_requested_sampling() -> None:
    """Count durations on the requested observation grid without inventing native rows.

    The sparse path has observations on January 1, 3, and 5. With ``freq=None`` those are three
    native observations, so peak-to-recovery spans two sampled periods. With daily rebasing, the
    same dates span four daily periods. Both paths retain the same peak, trough, recovery, and 20%
    depth; only the explicitly requested sampling grid changes the duration.
    """
    dates = pd.to_datetime(('2024-01-01', '2024-01-03', '2024-01-05'))
    prices = pd.Series((100.0, 80.0, 100.0), index=dates, name='nav')

    native = compute_drawdowns_stats_table(price=prices, freq=None)
    daily = compute_drawdowns_stats_table(price=prices, freq='D')

    for actual in (native, daily):
        assert actual.loc[actual.index[0], 'start'] == dates[0]
        assert actual.loc[actual.index[0], 'trough'] == dates[1]
        assert actual.loc[actual.index[0], 'end'] == dates[2]
        np.testing.assert_allclose(
            actual['max_dd'].to_numpy(),
            np.asarray((80.0 / 100.0 - 1.0,)),
            rtol=0.0,
            atol=_TOLERANCE,
        )
        assert bool(actual.loc[actual.index[0], 'is_recovered'])

    assert native.loc[native.index[0], 'days_dd'] == 2.0
    assert native.loc[native.index[0], 'days_to_trough'] == 1.0
    assert native.loc[native.index[0], 'days_recovery'] == 1.0
    assert daily.loc[daily.index[0], 'days_dd'] == 4.0
    assert daily.loc[daily.index[0], 'days_to_trough'] == 2.0
    assert daily.loc[daily.index[0], 'days_recovery'] == 2.0


def test_compute_drawdowns_stats_table_preserves_calendar_days_for_weekly_sampling() -> None:
    """Keep the documented day units after sampling a price path weekly.

    A peak, trough, and recovery on consecutive Sundays span fourteen calendar days. Weekly
    sampling changes which observations define the episode, but the ``days_*`` columns continue
    to report elapsed calendar days rather than the number of weekly periods.
    """
    dates = pd.to_datetime(('2024-01-07', '2024-01-14', '2024-01-21'))
    prices = pd.Series((100.0, 80.0, 100.0), index=dates, name='nav')

    actual = compute_drawdowns_stats_table(price=prices, freq='W')

    assert actual.loc[actual.index[0], 'days_dd'] == 14.0
    assert actual.loc[actual.index[0], 'days_to_trough'] == 7.0
    assert actual.loc[actual.index[0], 'days_recovery'] == 7.0

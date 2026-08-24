"""Regression tests for ``append_time_series`` input-date ordering.

Plain time-series append treats each provider as a mapping from dates to values: storage order
must not change boundary selection, overlap diagnostics, or the returned chronology. The tests
use literal six-day expectations and independently calculated overlap differences. They cover a
mixed DataFrame containing shared, ragged, and one-sided histories, the equivalent Series path,
non-overlapping providers, stable duplicate handling, labels, column order, warnings, and caller
ownership.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from qis.perfstats.timeseries_bfill import append_time_series


# =============================================================================
# Shared deterministic timeline and panel schema
# =============================================================================

_DAILY_DATES = pd.date_range('2024-01-01', periods=6, freq='D')

_COLUMNS = ('shared', 'ragged', 'older_only', 'newer_only')
_SERIES_NAME = 'Asset A'


# =============================================================================
# Deterministic mixed-panel fixtures and independent references
# =============================================================================

def _make_mixed_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct overlapping providers containing every relevant column state.

    Returns:
        Newer and older DataFrames in chronological order. Tests deliberately permute fresh
        instances so each case remains independent.
    """
    older = pd.DataFrame(
        {
            'shared': (10.0, 20.0, 30.0, 40.0, 50.0),
            'ragged': (np.nan, 2.0, 3.0, 4.0, 5.0),
            'older_only': (1.0, 2.0, 3.0, 4.0, 5.0),
            'newer_only': (np.nan, np.nan, np.nan, np.nan, np.nan),
        },
        index=_DAILY_DATES[:5],
    )
    newer = pd.DataFrame(
        {
            'shared': (400.0, 500.0, 600.0),
            'ragged': (np.nan, 50.0, 60.0),
            'older_only': (np.nan, np.nan, np.nan),
            'newer_only': (40.0, 50.0, 60.0),
        },
        index=_DAILY_DATES[3:],
    )
    return newer, older


def _make_expected_mixed_result() -> pd.DataFrame:
    """Return the literal newer-precedence splice for the mixed providers.

    Returns:
        The older January 1-3 prefix followed by newer observations on January 4-6.
    """
    return pd.DataFrame(
        {
            'shared': (10.0, 20.0, 30.0, 400.0, 500.0, 600.0),
            'ragged': (np.nan, 2.0, 3.0, np.nan, 50.0, 60.0),
            'older_only': (1.0, 2.0, 3.0, np.nan, np.nan, np.nan),
            'newer_only': (np.nan, np.nan, np.nan, 40.0, 50.0, 60.0),
        },
        index=_DAILY_DATES,
    )


def _make_expected_overlap_diff() -> pd.Series:
    """Return independently calculated mean absolute overlap differences.

    Returns:
        Differences over January 4-5. The shared value is ``(360 + 450) / 2 = 405``;
        ragged has only January 5 jointly observed and therefore equals ``abs(5 - 50) = 45``.
    """
    return pd.Series(
        (405.0, 45.0, np.nan, np.nan),
        index=pd.Index(_COLUMNS),
    )


# =============================================================================
# Chronological overlap and mixed-panel contract
# =============================================================================

@pytest.mark.parametrize(
    ('older_positions', 'newer_positions'),
    (
        pytest.param((0, 1, 2, 3, 4), (0, 1, 2), id='sorted'),
        pytest.param((4, 0, 1, 2, 3), (0, 1, 2), id='older-boundary-first'),
        pytest.param((0, 1, 2, 3, 4), (2, 0, 1), id='newer-boundary-first'),
        pytest.param((4, 3, 2, 1, 0), (2, 1, 0), id='reversed'),
    ),
)
def test_append_time_series_normalizes_mixed_panel_date_order(
        older_positions: tuple[int, ...],
        newer_positions: tuple[int, ...],
) -> None:
    """Return one chronological splice and exact overlap diagnostic for every row order.

    The newer provider owns January 4-6, while January 1-3 comes from the older provider. The
    four-column fixture simultaneously proves finite shared history, ragged overlap, and both
    one-sided histories without allowing a correct column to hide another column's loss.

    Args:
        older_positions: Positional permutation applied to the older provider.
        newer_positions: Positional permutation applied to the newer provider.
    """
    newer, older = _make_mixed_inputs()
    newer = newer.iloc[list(newer_positions)]
    older = older.iloc[list(older_positions)]
    original_newer = newer.copy(deep=True)
    original_older = older.copy(deep=True)

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        actual, actual_diff = append_time_series(
            df_newer=newer,
            df_older=older,
            numerical_check_columns=list(_COLUMNS),
        )

    assert isinstance(actual, pd.DataFrame)
    assert isinstance(actual_diff, pd.Series)
    pd.testing.assert_frame_equal(
        actual,
        _make_expected_mixed_result(),
        check_exact=True,
        check_freq=False,
    )
    pd.testing.assert_series_equal(
        actual_diff,
        _make_expected_overlap_diff(),
        check_exact=True,
    )
    pd.testing.assert_frame_equal(newer, original_newer, check_exact=True)
    pd.testing.assert_frame_equal(older, original_older, check_exact=True)


def test_append_time_series_normalizes_nullable_mixed_panel_date_order() -> None:
    """Preserve the same mixed-panel contract for nullable ``Float64`` and ``pd.NA``.

    This uses the boundary-first older order that formerly discarded January 1-3. Converting the
    complete fixture and independent references to nullable dtype proves that missing placement,
    overlap arithmetic, and chronological normalization do not depend on NumPy-backed ``NaN``.
    """
    newer, older = _make_mixed_inputs()
    newer = newer.astype('Float64')
    older = older.astype('Float64').iloc[[4, 0, 1, 2, 3]]
    expected = _make_expected_mixed_result().astype('Float64')
    expected_diff = _make_expected_overlap_diff().astype('Float64')

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        actual, actual_diff = append_time_series(
            df_newer=newer,
            df_older=older,
            numerical_check_columns=list(_COLUMNS),
        )

    assert isinstance(actual, pd.DataFrame)
    assert isinstance(actual_diff, pd.Series)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True, check_freq=False)
    pd.testing.assert_series_equal(actual_diff, expected_diff, check_exact=True)


def test_append_time_series_normalizes_series_date_order() -> None:
    """Apply the same chronological overlap rule to a named Series.

    Older values ``[1, 2, 3]`` precede newer values ``[40, 50, 60]``. The two overlap differences
    are 36 and 45, so the independently calculated mean is exactly 40.5.
    """
    older = pd.Series(
        (5.0, 1.0, 4.0, 2.0, 3.0),
        index=_DAILY_DATES[[4, 0, 3, 1, 2]],
        name=_SERIES_NAME,
    )
    newer = pd.Series(
        (60.0, 40.0, 50.0),
        index=_DAILY_DATES[[5, 3, 4]],
        name=_SERIES_NAME,
    )
    expected = pd.Series(
        (1.0, 2.0, 3.0, 40.0, 50.0, 60.0),
        index=_DAILY_DATES,
        name=_SERIES_NAME,
    )
    expected_diff = pd.Series((40.5,), index=pd.Index([_SERIES_NAME]))

    actual, actual_diff = append_time_series(
        df_newer=newer,
        df_older=older,
        numerical_check_columns=[_SERIES_NAME],
    )

    assert isinstance(actual, pd.Series)
    assert isinstance(actual_diff, pd.Series)
    pd.testing.assert_series_equal(actual, expected, check_exact=True, check_freq=False)
    pd.testing.assert_series_equal(actual_diff, expected_diff, check_exact=True)


# =============================================================================
# Non-overlap and duplicate-date boundaries
# =============================================================================

def test_append_time_series_normalizes_nonoverlapping_providers() -> None:
    """Concatenate disjoint providers chronologically without inventing an overlap diagnostic."""
    older = pd.Series(
        (3.0, 1.0, 2.0),
        index=_DAILY_DATES[[2, 0, 1]],
        name=_SERIES_NAME,
    )
    newer = pd.Series(
        (6.0, 4.0, 5.0),
        index=_DAILY_DATES[[5, 3, 4]],
        name=_SERIES_NAME,
    )
    expected = pd.Series(
        (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        index=_DAILY_DATES,
        name=_SERIES_NAME,
    )

    actual, actual_diff = append_time_series(df_newer=newer, df_older=older)

    assert isinstance(actual, pd.Series)
    pd.testing.assert_series_equal(actual, expected, check_exact=True, check_freq=False)
    assert actual_diff is None


def test_append_time_series_preserves_last_supplied_duplicate_in_date_order() -> None:
    """Retain the last supplied value for an older duplicate while sorting distinct dates.

    January 2 is supplied first as 20 and later as 2. Stable chronological normalization retains
    that relative tie order, after which the established keep-last rule selects 2.
    """
    older = pd.Series(
        (3.0, 20.0, 1.0, 2.0),
        index=_DAILY_DATES[[2, 1, 0, 1]],
        name=_SERIES_NAME,
    )
    newer = pd.Series(
        (4.0, 5.0, 6.0),
        index=_DAILY_DATES[3:],
        name=_SERIES_NAME,
    )
    expected = pd.Series(
        (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        index=_DAILY_DATES,
        name=_SERIES_NAME,
    )

    actual, actual_diff = append_time_series(df_newer=newer, df_older=older)

    assert isinstance(actual, pd.Series)
    pd.testing.assert_series_equal(actual, expected, check_exact=True, check_freq=False)
    assert actual_diff is None

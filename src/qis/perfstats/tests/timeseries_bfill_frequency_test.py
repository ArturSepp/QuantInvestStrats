"""Regression tests for ``bfill_timeseries`` frequency-grid semantics.

Return gaps have different meanings under the three supported fill policies. ``None`` preserves
missing returns, ``to_zero`` treats a missing return as no change after that asset begins, and
``ffill`` explicitly repeats the last observed return. Price levels retain their established
forward-fill convention because an unreported level remains at its last observation.

The fixtures use literal expected grids rather than a QIS resampling helper. They also cover the
one- and two-observation boundaries where pandas cannot infer a frequency, ragged DataFrame
starts, Series/DataFrame consistency, labels, column order, and caller ownership.
"""

import numpy as np
import pandas as pd
import pytest

from qis.perfstats.timeseries_bfill import bfill_timeseries


# =============================================================================
# Shared deterministic timeline and comparison helper
# =============================================================================

_DAILY_DATES = pd.date_range("2024-01-01", periods=7, freq="D")
_ASSET_NAME = "Asset A"
_TOLERANCE = 1.0e-12


def _assert_series_and_frame_results(
        older: pd.Series,
        newer: pd.Series,
        expected: pd.Series,
        fill_method: str | None = None,
        is_prices: bool = False,
        check_exact: bool = True,
) -> None:
    """Exercise equivalent Series and one-column DataFrame inputs.

    Args:
        older: Older provider history.
        newer: Newer provider history whose name defines the output label.
        expected: Independently constructed expected result.
        fill_method: Missing-return policy passed to ``bfill_timeseries``.
        is_prices: Whether inputs and expected values are price levels.
        check_exact: Whether pandas comparisons require bitwise-equal values.
    """
    original_older = older.copy(deep=True)
    original_newer = newer.copy(deep=True)
    newer_frame = newer.to_frame()
    older_frame = older.to_frame()
    older_frame.columns = newer_frame.columns.copy()
    original_older_frame = older_frame.copy(deep=True)
    original_newer_frame = newer_frame.copy(deep=True)

    series_result = bfill_timeseries(
        df_newer=newer,
        df_older=older,
        freq="D",
        fill_method=fill_method,
        is_prices=is_prices,
    )
    frame_result = bfill_timeseries(
        df_newer=newer_frame,
        df_older=older_frame,
        freq="D",
        fill_method=fill_method,
        is_prices=is_prices,
    )

    assert isinstance(series_result, pd.Series)
    assert isinstance(frame_result, pd.DataFrame)
    pd.testing.assert_series_equal(
        series_result,
        expected,
        check_exact=check_exact,
        rtol=0.0,
        atol=_TOLERANCE,
    )
    pd.testing.assert_frame_equal(
        frame_result,
        expected.to_frame(),
        check_exact=check_exact,
        rtol=0.0,
        atol=_TOLERANCE,
    )
    pd.testing.assert_series_equal(older, original_older, check_exact=True)
    pd.testing.assert_series_equal(newer, original_newer, check_exact=True)
    pd.testing.assert_frame_equal(older_frame, original_older_frame, check_exact=True)
    pd.testing.assert_frame_equal(newer_frame, original_newer_frame, check_exact=True)


# =============================================================================
# Short-history frequency boundaries
# =============================================================================

@pytest.mark.parametrize("fill_method", [None, "to_zero", "ffill"])
def test_bfill_timeseries_preserves_single_return_observation(
        fill_method: str | None,
) -> None:
    """Return one observation without requiring an inferable frequency."""
    older = pd.Series([0.10], index=_DAILY_DATES[[0]], name="Older provider")
    newer = pd.Series([0.20], index=_DAILY_DATES[[0]], name=_ASSET_NAME)
    expected = pd.Series([0.20], index=_DAILY_DATES[:1], name=_ASSET_NAME)

    _assert_series_and_frame_results(
        older=older,
        newer=newer,
        expected=expected,
        fill_method=fill_method,
    )


@pytest.mark.parametrize(
    ("fill_method", "inserted_return"),
    [
        (None, np.nan),
        ("to_zero", 0.00),
        ("ffill", 0.10),
    ],
)
def test_bfill_timeseries_applies_fill_policy_to_two_observation_return_grid(
        fill_method: str | None,
        inserted_return: float,
) -> None:
    """Expand two observations without calling pandas frequency inference.

    January 2 is absent from the supplied histories. Its independently expected return is
    missing, zero, or the January 1 return according to the selected valid fill policy.
    """
    older = pd.Series([0.10], index=_DAILY_DATES[[0]], name="Older provider")
    newer = pd.Series([0.20], index=_DAILY_DATES[[2]], name=_ASSET_NAME)
    expected = pd.Series(
        [0.10, inserted_return, 0.20],
        index=_DAILY_DATES[:3],
        name=_ASSET_NAME,
    )

    _assert_series_and_frame_results(
        older=older,
        newer=newer,
        expected=expected,
        fill_method=fill_method,
    )


# =============================================================================
# Return-grid fill policies
# =============================================================================

@pytest.mark.parametrize(
    ("fill_method", "expected_values"),
    [
        (None, (0.10, np.nan, 0.20, np.nan, 0.30)),
        ("to_zero", (0.10, 0.00, 0.20, 0.00, 0.30)),
        ("ffill", (0.10, 0.10, 0.20, 0.20, 0.30)),
    ],
)
def test_bfill_timeseries_applies_fill_policy_to_gapped_return_grid(
        fill_method: str | None,
        expected_values: tuple[float, ...],
) -> None:
    """Give each valid fill policy distinct values on dates inserted into the grid."""
    older = pd.Series(
        [0.10, 0.20],
        index=_DAILY_DATES[[0, 2]],
        name="Older provider",
    )
    newer = pd.Series([0.30], index=_DAILY_DATES[[4]], name=_ASSET_NAME)
    expected = pd.Series(
        expected_values,
        index=_DAILY_DATES[:5],
        name=_ASSET_NAME,
    )

    _assert_series_and_frame_results(
        older=older,
        newer=newer,
        expected=expected,
        fill_method=fill_method,
    )


def test_bfill_timeseries_none_preserves_existing_and_inserted_missing_returns() -> None:
    """Keep both a supplied return gap and a newly inserted grid date missing."""
    older = pd.Series(
        [0.10, np.nan, 0.20],
        index=_DAILY_DATES[:3],
        name="Older provider",
    )
    newer = pd.Series([0.30], index=_DAILY_DATES[[4]], name=_ASSET_NAME)
    expected = pd.Series(
        [0.10, np.nan, 0.20, np.nan, 0.30],
        index=_DAILY_DATES[:5],
        name=_ASSET_NAME,
    )

    _assert_series_and_frame_results(
        older=older,
        newer=newer,
        expected=expected,
        fill_method=None,
    )


def test_bfill_timeseries_to_zero_preserves_dataframe_ragged_starts() -> None:
    """Fill inserted returns only after each DataFrame column begins.

    Asset B has no history before January 3, so January 1-2 remain missing. Thereafter the
    inserted January 4 return is zero. Asset A begins on January 1 and receives zeros on both
    inserted dates. Reversed newer columns also pin the output-order contract.
    """
    older = pd.DataFrame(
        {
            "Asset A": [0.10, 0.20],
            "Asset B": [np.nan, 0.40],
        },
        index=_DAILY_DATES[[0, 2]],
    )
    newer = pd.DataFrame(
        {
            "Asset B": [0.50],
            "Asset A": [0.30],
        },
        index=_DAILY_DATES[[4]],
    )
    original_older = older.copy(deep=True)
    original_newer = newer.copy(deep=True)
    expected = pd.DataFrame(
        {
            "Asset B": [np.nan, np.nan, 0.40, 0.00, 0.50],
            "Asset A": [0.10, 0.00, 0.20, 0.00, 0.30],
        },
        index=_DAILY_DATES[:5],
    )

    actual = bfill_timeseries(
        df_newer=newer,
        df_older=older,
        freq="D",
        fill_method="to_zero",
    )

    assert isinstance(actual, pd.DataFrame)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    pd.testing.assert_frame_equal(older, original_older, check_exact=True)
    pd.testing.assert_frame_equal(newer, original_newer, check_exact=True)


def test_bfill_timeseries_to_zero_preserves_all_missing_column() -> None:
    """Leave a column missing when it has no observation from either provider.

    The ``to_zero`` policy begins only after a column's first observed return. Asset B therefore
    remains missing across the expanded grid rather than acquiring a zero on its final date.
    """
    older = pd.DataFrame(
        {
            "Asset A": [0.10, 0.20],
            "Asset B": [np.nan, np.nan],
        },
        index=_DAILY_DATES[[0, 2]],
    )
    newer = pd.DataFrame(
        {
            "Asset A": [0.30],
            "Asset B": [np.nan],
        },
        index=_DAILY_DATES[[4]],
    )
    expected = pd.DataFrame(
        {
            "Asset A": [0.10, 0.00, 0.20, 0.00, 0.30],
            "Asset B": [np.nan, np.nan, np.nan, np.nan, np.nan],
        },
        index=_DAILY_DATES[:5],
    )

    actual = bfill_timeseries(
        df_newer=newer,
        df_older=older,
        freq="D",
        fill_method="to_zero",
    )

    assert isinstance(actual, pd.DataFrame)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


# =============================================================================
# Existing price-level convention
# =============================================================================

def test_bfill_timeseries_forward_fills_price_levels_on_expanded_grid() -> None:
    """Preserve price levels between observations when expanding their date grid.

    The supplied prices rise by 10% at every observation. Missing calendar dates retain the
    preceding level, while the newer provider's overlapping January 5 level and January 7 level
    remain anchored at 121 and 133.1.
    """
    older = pd.Series(
        [100.0, 110.0, 121.0],
        index=_DAILY_DATES[[0, 2, 4]],
        name="Older provider",
    )
    newer = pd.Series(
        [121.0, 133.1],
        index=_DAILY_DATES[[4, 6]],
        name=_ASSET_NAME,
    )
    expected = pd.Series(
        [100.0, 100.0, 110.0, 110.0, 121.0, 121.0, 133.1],
        index=_DAILY_DATES,
        name=_ASSET_NAME,
    )

    _assert_series_and_frame_results(
        older=older,
        newer=newer,
        expected=expected,
        is_prices=True,
        check_exact=False,
    )

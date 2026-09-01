"""Regression tests for duplicate-date selection in append overlap diagnostics.

``append_time_series`` treats each provider as a date-to-value mapping after stable chronological
normalization: when one provider supplies a date repeatedly, its last supplied row is the selected
observation. The returned splice already follows that rule, so its optional numerical diagnostic
must compare the same selected rows rather than average superseded duplicates.

The deterministic matrix places duplicates in the older provider, newer provider, and both at
once. A mixed DataFrame combines a finite column, a column whose selected value is missing, and an
all-missing column; equivalent named Series establish shape parity. Ordinary and nullable floating
storage, an empty-newer interaction, exact output and diagnostic values, warnings-as-errors,
metadata, and ownership keep this correction tied to the existing keep-last contract.
"""

import warnings
from typing import Literal

import numpy as np
import pandas as pd
import pytest

from qis.perfstats.timeseries_bfill import append_time_series


# =============================================================================
# Shared deterministic schema and expectations
# =============================================================================

DuplicateSide = Literal["older", "newer", "both"]

_ALL_MISSING = "all_missing"
_FINITE = "finite"
_SELECTED_MISSING = "selected_missing"
_COLUMNS = (_FINITE, _SELECTED_MISSING, _ALL_MISSING)
_DIAGNOSTIC_COLUMNS = (_SELECTED_MISSING, _FINITE, _ALL_MISSING)

_DATES = pd.to_datetime(("2024-01-01", "2024-01-03", "2024-01-04"))
_NEWER_INDEX_NAME = "Newer date"
_NEWER_COLUMNS_NAME = "Newer asset"
_NEWER_SERIES_NAME = "New asset"
_OLDER_SERIES_NAME = "Old asset"


def _as_float_frame(frame: pd.DataFrame, nullable: bool) -> pd.DataFrame:
    """Convert a fresh frame to ordinary or nullable floating storage.

    Args:
        frame: DataFrame whose labels and values are preserved.
        nullable: Use pandas nullable ``Float64`` when true.

    Returns:
        Converted DataFrame independent of the supplied object.
    """
    dtype = pd.Float64Dtype() if nullable else np.dtype("float64")
    return frame.astype(dtype)


def _as_float_series(series: pd.Series, nullable: bool) -> pd.Series:
    """Convert a fresh series to ordinary or nullable floating storage.

    Args:
        series: Series whose labels and values are preserved.
        nullable: Use pandas nullable ``Float64`` when true.

    Returns:
        Converted Series independent of the supplied object.
    """
    dtype = pd.Float64Dtype() if nullable else np.dtype("float64")
    return series.astype(dtype)


def _make_duplicate_inputs(
    duplicate_side: DuplicateSide,
    nullable: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create overlapping providers with deliberately non-maximal last duplicates.

    The finite duplicate values decrease from 300 to 30 in the older provider and from 1000 to
    100 in the newer provider. Selecting 30 and 100 therefore proves stable keep-last behavior
    rather than accidental maximum selection. The selected-missing column makes a discarded
    finite row distinguishable from the retained missing row.

    Args:
        duplicate_side: Provider or providers containing repeated overlap dates.
        nullable: Use pandas nullable ``Float64`` when true.

    Returns:
        Newer and older mixed-panel providers with distinct public axis metadata.
    """
    older_is_duplicated = duplicate_side in {"older", "both"}
    newer_is_duplicated = duplicate_side in {"newer", "both"}

    older = pd.DataFrame(
        {
            _FINITE: (1.0, 300.0, 30.0) if older_is_duplicated else (1.0, 30.0),
            _SELECTED_MISSING: ((1.0, 50.0, np.nan) if older_is_duplicated else (1.0, 50.0)),
            _ALL_MISSING: (np.nan,) * (3 if older_is_duplicated else 2),
        },
        index=(
            pd.DatetimeIndex((_DATES[0], _DATES[1], _DATES[1]), name="Older date")
            if older_is_duplicated
            else pd.DatetimeIndex(_DATES[:2], name="Older date")
        ),
    ).rename_axis("Older asset", axis="columns")
    newer = pd.DataFrame(
        {
            _FINITE: (1000.0, 100.0, 40.0) if newer_is_duplicated else (100.0, 40.0),
            _SELECTED_MISSING: ((150.0, np.nan, 50.0) if newer_is_duplicated else (15.0, 50.0)),
            _ALL_MISSING: (np.nan,) * (3 if newer_is_duplicated else 2),
        },
        index=(
            pd.DatetimeIndex((_DATES[1], _DATES[1], _DATES[2]), name=_NEWER_INDEX_NAME)
            if newer_is_duplicated
            else pd.DatetimeIndex(_DATES[1:], name=_NEWER_INDEX_NAME)
        ),
    ).rename_axis(_NEWER_COLUMNS_NAME, axis="columns")
    return _as_float_frame(newer, nullable), _as_float_frame(older, nullable)


def _expected_frame(duplicate_side: DuplicateSide, nullable: bool) -> pd.DataFrame:
    """Return the independently selected newer-precedence mixed-panel splice.

    Args:
        duplicate_side: Provider or providers containing repeated overlap dates.
        nullable: Use pandas nullable ``Float64`` when true.

    Returns:
        Three-date result after stable keep-last selection within each provider.
    """
    newer_is_duplicated = duplicate_side in {"newer", "both"}
    expected = pd.DataFrame(
        {
            _FINITE: (1.0, 100.0, 40.0),
            _SELECTED_MISSING: (1.0, np.nan if newer_is_duplicated else 15.0, 50.0),
            _ALL_MISSING: (np.nan, np.nan, np.nan),
        },
        index=pd.DatetimeIndex(_DATES, name=_NEWER_INDEX_NAME),
    ).rename_axis(_NEWER_COLUMNS_NAME, axis="columns")
    return _as_float_frame(expected, nullable)


def _expected_diff(nullable: bool) -> pd.Series:
    """Return literal diagnostics from selected older and newer overlap rows.

    Args:
        nullable: Use pandas nullable ``Float64`` when true.

    Returns:
        Ordered missing, finite, and all-missing diagnostics. The finite difference is
        ``abs(30 - 100) = 70``; the other two values are undefined.
    """
    expected = pd.Series(
        (np.nan, 70.0, np.nan),
        index=pd.Index(_DIAGNOSTIC_COLUMNS, name=_NEWER_COLUMNS_NAME),
    )
    return _as_float_series(expected, nullable)


def _append_frames(
    newer: pd.DataFrame,
    older: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Append frames while enforcing warnings and caller ownership.

    Args:
        newer: Newer mixed-panel provider.
        older: Older mixed-panel provider.

    Returns:
        Appended frame and ordered overlap diagnostic.
    """
    original_newer = newer.copy(deep=True)
    original_older = older.copy(deep=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual, actual_diff = append_time_series(
            df_newer=newer,
            df_older=older,
            numerical_check_columns=list(_DIAGNOSTIC_COLUMNS),
        )

    assert isinstance(actual, pd.DataFrame)
    assert actual is not newer
    assert actual is not older
    pd.testing.assert_frame_equal(newer, original_newer, check_exact=True)
    pd.testing.assert_frame_equal(older, original_older, check_exact=True)
    return actual, actual_diff


# =============================================================================
# Mixed-panel diagnostic selection
# =============================================================================


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "Float64"))
@pytest.mark.parametrize("duplicate_side", ("older", "newer", "both"))
def test_append_time_series_diagnostic_uses_selected_duplicate_rows(
    duplicate_side: DuplicateSide,
    nullable: bool,
) -> None:
    """Calculate diagnostics from the same keep-last rows used by the returned splice.

    Args:
        duplicate_side: Provider or providers containing repeated overlap dates.
        nullable: Exercise ordinary and pandas nullable floating-point storage.
    """
    newer, older = _make_duplicate_inputs(duplicate_side, nullable)

    actual, actual_diff = _append_frames(newer, older)

    pd.testing.assert_frame_equal(
        actual,
        _expected_frame(duplicate_side, nullable),
        check_exact=True,
        check_freq=False,
    )
    assert isinstance(actual_diff, pd.Series)
    pd.testing.assert_series_equal(actual_diff, _expected_diff(nullable), check_exact=True)


# =============================================================================
# Named Series parity
# =============================================================================


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "Float64"))
@pytest.mark.parametrize("duplicate_side", ("older", "newer", "both"))
def test_append_time_series_duplicate_diagnostic_matches_named_series(
    duplicate_side: DuplicateSide,
    nullable: bool,
) -> None:
    """Apply identical selected-row arithmetic to the equivalent named Series.

    Args:
        duplicate_side: Provider or providers containing repeated overlap dates.
        nullable: Exercise ordinary and pandas nullable floating-point storage.
    """
    newer, older = _make_duplicate_inputs(duplicate_side, nullable)
    newer_series = newer.loc[:, _FINITE]
    older_series = older.loc[:, _FINITE]
    assert isinstance(newer_series, pd.Series)
    assert isinstance(older_series, pd.Series)
    newer_series = newer_series.rename(_NEWER_SERIES_NAME)
    older_series = older_series.rename(_OLDER_SERIES_NAME)
    original_newer = newer_series.copy(deep=True)
    original_older = older_series.copy(deep=True)
    expected = _as_float_series(
        pd.Series(
            (1.0, 100.0, 40.0),
            index=pd.DatetimeIndex(_DATES, name=_NEWER_INDEX_NAME),
            name=_NEWER_SERIES_NAME,
        ),
        nullable,
    )
    expected_diff = _as_float_series(
        pd.Series((70.0,), index=pd.Index((_NEWER_SERIES_NAME,))),
        nullable,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual, actual_diff = append_time_series(
            df_newer=newer_series,
            df_older=older_series,
            numerical_check_columns=[_NEWER_SERIES_NAME],
        )

    assert isinstance(actual, pd.Series)
    assert isinstance(actual_diff, pd.Series)
    pd.testing.assert_series_equal(actual, expected, check_exact=True, check_freq=False)
    pd.testing.assert_series_equal(actual_diff, expected_diff, check_exact=True)
    pd.testing.assert_series_equal(newer_series, original_newer, check_exact=True)
    pd.testing.assert_series_equal(older_series, original_older, check_exact=True)


# =============================================================================
# Empty-newer interaction
# =============================================================================


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "Float64"))
def test_append_time_series_empty_newer_uses_selected_older_duplicates(nullable: bool) -> None:
    """Apply keep-last provider normalization before filling an empty newer declaration.

    Args:
        nullable: Exercise ordinary and pandas nullable floating-point storage.
    """
    newer = pd.DataFrame(columns=_COLUMNS, index=pd.DatetimeIndex([], name=_NEWER_INDEX_NAME))
    newer = newer.rename_axis(_NEWER_COLUMNS_NAME, axis="columns")
    older = pd.DataFrame(
        {
            _FINITE: (1.0, 20.0, 2.0),
            _SELECTED_MISSING: (1.0, 10.0, np.nan),
            _ALL_MISSING: (np.nan, np.nan, np.nan),
        },
        index=pd.DatetimeIndex((_DATES[0], _DATES[1], _DATES[1]), name="Older date"),
    ).rename_axis("Older asset", axis="columns")
    newer = _as_float_frame(newer, nullable)
    older = _as_float_frame(older, nullable)
    expected = pd.DataFrame(
        {
            _FINITE: (1.0, 2.0),
            _SELECTED_MISSING: (1.0, np.nan),
            _ALL_MISSING: (np.nan, np.nan),
        },
        index=pd.DatetimeIndex(_DATES[:2], name=_NEWER_INDEX_NAME),
    ).rename_axis(_NEWER_COLUMNS_NAME, axis="columns")
    expected = _as_float_frame(expected, nullable)

    actual, actual_diff = _append_frames(newer, older)

    pd.testing.assert_frame_equal(actual, expected, check_exact=True, check_freq=False)
    assert actual_diff is None

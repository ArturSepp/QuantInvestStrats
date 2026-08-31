"""Regression tests for calendar indexes in time-series reconstruction paths.

``bfill_timeseries`` constructs a requested calendar grid, and
``interpolate_infrequent_returns`` subtracts timestamps to parameterize a Brownian bridge. Both
operations therefore require nonempty providers to use ``DatetimeIndex`` objects without ``NaT``.
They should reject invalid grids at their public boundary instead of silently discarding values or
leaking incidental pandas errors.

Empty providers remain schema declarations and need no date-axis validation. The valid mixed-panel
control combines timezone-aware, unsorted, duplicate, nullable, finite, and all-missing states to
confirm that the guard preserves established provider selection, values, metadata, warnings,
caller ownership, and return shape.
"""

import warnings
from typing import Literal

import numpy as np
import pandas as pd
import pytest

from qis.perfstats.returns import to_returns
from qis.perfstats.timeseries_bfill import bfill_timeseries, interpolate_infrequent_returns


# =============================================================================
# Shared deterministic fixtures and comparison helpers
# =============================================================================

_DATES = pd.date_range("2024-01-01", periods=5, freq="D")

_PanelShape = Literal["series", "frame"]
_InvalidIndex = Literal["non-date", "NaT"]


def _make_panel(
    values: tuple[float, ...],
    index: pd.Index,
    shape: _PanelShape,
    *,
    name: str,
) -> pd.Series | pd.DataFrame:
    """Create one Series or single-column DataFrame provider.

    Args:
        values: Literal observations assigned in order.
        index: Provider labels.
        shape: Return a Series or DataFrame.
        name: Series name or DataFrame column label.

    Returns:
        Fresh provider with ordinary floating storage.
    """
    series = pd.Series(values, index=index, name=name, dtype=np.float64)
    if shape == "series":
        return series
    return series.to_frame()


def _assert_panel_equal(
    actual: pd.Series | pd.DataFrame,
    expected: pd.Series | pd.DataFrame,
) -> None:
    """Assert complete shape-specific equality.

    Args:
        actual: Public result or caller-owned provider.
        expected: Independently constructed result or provider snapshot.
    """
    if isinstance(actual, pd.Series):
        assert isinstance(expected, pd.Series)
        pd.testing.assert_series_equal(actual, expected, check_exact=True, check_freq=False)
    else:
        assert isinstance(expected, pd.DataFrame)
        pd.testing.assert_frame_equal(actual, expected, check_exact=True, check_freq=False)


def _invalid_index(kind: _InvalidIndex) -> pd.Index:
    """Return one unsupported nonempty index for a calendar operation.

    Args:
        kind: Construct non-date labels or a date index containing ``NaT``.

    Returns:
        Two-row invalid index.
    """
    if kind == "non-date":
        return pd.RangeIndex(2, name="row")
    return pd.DatetimeIndex((_DATES[0], pd.NaT), name="date")


def _expected_error(argument_name: str, kind: _InvalidIndex) -> tuple[type[Exception], str]:
    """Return the deterministic public exception contract.

    Args:
        argument_name: Public parameter carrying the invalid index.
        kind: Invalid index category.

    Returns:
        Exact exception class and anchored message pattern.
    """
    if kind == "non-date":
        return TypeError, rf"^{argument_name} must use a DatetimeIndex for calendar operations$"
    return ValueError, rf"^{argument_name} index must not contain NaT$"


# =============================================================================
# Backfill calendar-index failures
# =============================================================================


@pytest.mark.parametrize("shape", ("series", "frame"))
@pytest.mark.parametrize("argument_name", ("df_newer", "df_older"))
@pytest.mark.parametrize("invalid_kind", ("non-date", "NaT"))
def test_bfill_timeseries_rejects_invalid_nonempty_provider_index(
    invalid_kind: _InvalidIndex,
    argument_name: str,
    shape: _PanelShape,
) -> None:
    """Reject each invalid provider before calendar construction can lose its values.

    Args:
        invalid_kind: Non-date or undefined date boundary under test.
        argument_name: Newer or older provider receiving that boundary.
        shape: Public pandas container under test.
    """
    newer_index: pd.Index = pd.DatetimeIndex(_DATES[3:], name="date")
    older_index: pd.Index = pd.DatetimeIndex(_DATES[:2], name="date")
    if argument_name == "df_newer":
        newer_index = _invalid_index(invalid_kind)
    else:
        older_index = _invalid_index(invalid_kind)
    newer = _make_panel((0.40, 0.50), newer_index, shape, name="Asset")
    older = _make_panel((0.10, 0.20), older_index, shape, name="Asset")
    original_newer = newer.copy(deep=True)
    original_older = older.copy(deep=True)
    exception_type, message = _expected_error(argument_name, invalid_kind)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(exception_type, match=message):
            bfill_timeseries(df_newer=newer, df_older=older, freq="D")

    _assert_panel_equal(newer, original_newer)
    _assert_panel_equal(older, original_older)


# =============================================================================
# Empty-provider and valid mixed-panel controls
# =============================================================================


@pytest.mark.parametrize("shape", ("series", "frame"))
@pytest.mark.parametrize("empty_argument", ("df_newer", "df_older"))
@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "Float64"))
def test_bfill_timeseries_preserves_empty_non_date_provider_schema(
    nullable: bool,
    empty_argument: str,
    shape: _PanelShape,
) -> None:
    """Skip date validation for an empty provider while preserving the available history.

    Args:
        nullable: Use ordinary or pandas nullable floating storage.
        empty_argument: Newer or older provider declared without observations.
        shape: Public pandas container under test.
    """
    dtype = pd.Float64Dtype() if nullable else np.dtype("float64")
    available = pd.Series((0.10, 0.20), index=_DATES[:2], dtype=dtype, name="Asset")
    empty = pd.Series([], index=pd.RangeIndex(0, name="row"), dtype=dtype, name="Asset")
    available_panel: pd.Series | pd.DataFrame = (
        available if shape == "series" else available.to_frame()
    )
    empty_panel: pd.Series | pd.DataFrame = empty if shape == "series" else empty.to_frame()
    if empty_argument == "df_newer":
        newer, older = empty_panel, available_panel
    else:
        newer, older = available_panel, empty_panel
    original_newer = newer.copy(deep=True)
    original_older = older.copy(deep=True)
    expected = available if shape == "series" else available.to_frame()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = bfill_timeseries(df_newer=newer, df_older=older, freq="D")

    _assert_panel_equal(actual, expected)
    _assert_panel_equal(newer, original_newer)
    _assert_panel_equal(older, original_older)


def test_bfill_timeseries_preserves_valid_timezone_aware_mixed_panel() -> None:
    """Keep stable duplicate selection and ragged nullable values on a valid date grid.

    Older rows arrive as January 2, January 1, and January 2, so stable chronological repair
    retains the final January 2 value of 25%. Newer rows arrive as January 4 then January 3 and
    supply the literal 40% and 30% tail. The all-missing neighbor remains undefined throughout.
    """
    timezone = "America/New_York"
    older_index = pd.DatetimeIndex(
        ("2024-01-02", "2024-01-01", "2024-01-02"),
        tz=timezone,
        name="date",
    )
    newer_index = pd.DatetimeIndex(
        ("2024-01-04", "2024-01-03"),
        tz=timezone,
        name="date",
    )
    older = pd.DataFrame(
        {"finite": (0.20, 0.10, 0.25), "all_missing": (pd.NA, pd.NA, pd.NA)},
        index=older_index,
        dtype="Float64",
    )
    newer = pd.DataFrame(
        {"finite": (0.40, 0.30), "all_missing": (pd.NA, pd.NA)},
        index=newer_index,
        dtype="Float64",
    )
    original_newer = newer.copy(deep=True)
    original_older = older.copy(deep=True)
    expected = pd.DataFrame(
        {
            "finite": (0.10, 0.25, 0.30, 0.40),
            "all_missing": (pd.NA, pd.NA, pd.NA, pd.NA),
        },
        index=pd.date_range("2024-01-01", periods=4, freq="D", tz=timezone, name="date"),
        dtype="Float64",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = bfill_timeseries(df_newer=newer, df_older=older, freq="D")

    assert isinstance(actual, pd.DataFrame)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    pd.testing.assert_frame_equal(newer, original_newer, check_exact=True)
    pd.testing.assert_frame_equal(older, original_older, check_exact=True)


# =============================================================================
# Non-calendar return conversion control
# =============================================================================


def test_to_returns_preserves_non_date_index_without_resampling() -> None:
    """Keep arbitrary row labels valid when return conversion requests no calendar operation.

    Prices ``[100, 110, 121]`` independently imply relative returns ``[missing, 10%, 10%]``.
    The calendar guard must not reject or relabel this supported no-resampling calculation.
    """
    prices = pd.Series((100.0, 110.0, 121.0), index=pd.RangeIndex(3, name="row"), name="Asset")
    original = prices.copy(deep=True)
    expected = pd.Series((np.nan, 0.10, 0.10), index=prices.index, name=prices.name)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = to_returns(prices, freq=None, ffill_nans=False)

    assert isinstance(actual, pd.Series)
    pd.testing.assert_series_equal(
        actual,
        expected,
        check_exact=False,
        rtol=1.0e-12,
        atol=0.0,
    )
    pd.testing.assert_series_equal(prices, original, check_exact=True)


# =============================================================================
# Brownian-bridge calendar-index failures
# =============================================================================


@pytest.mark.parametrize("infrequent_shape", ("series", "frame"))
@pytest.mark.parametrize("argument_name", ("infrequent_returns", "pivot_returns"))
@pytest.mark.parametrize("invalid_kind", ("non-date", "NaT"))
def test_interpolate_infrequent_returns_rejects_invalid_nonempty_index(
    invalid_kind: _InvalidIndex,
    argument_name: str,
    infrequent_shape: _PanelShape,
) -> None:
    """Reject unsupported bridge grids before timestamp subtraction or slicing.

    Args:
        invalid_kind: Non-date or undefined date boundary under test.
        argument_name: Reported or pivot input receiving that boundary.
        infrequent_shape: Series or DataFrame reported-return shape.
    """
    infrequent_index: pd.Index = pd.DatetimeIndex(_DATES[[0, 4]], name="date")
    pivot_index: pd.Index = pd.DatetimeIndex(_DATES, name="date")
    if argument_name == "infrequent_returns":
        infrequent_index = _invalid_index(invalid_kind)
    else:
        pivot_index = (
            pd.RangeIndex(5, name="row")
            if invalid_kind == "non-date"
            else pd.DatetimeIndex(
                (_DATES[0], _DATES[1], pd.NaT, _DATES[3], _DATES[4]),
                name="date",
            )
        )
    infrequent_series = pd.Series((0.02, 0.03), index=infrequent_index, name="Private")
    infrequent: pd.Series | pd.DataFrame = (
        infrequent_series if infrequent_shape == "series" else infrequent_series.to_frame()
    )
    pivot = pd.Series((0.01, -0.02, 0.015, -0.005, 0.025), index=pivot_index, name="Market")
    original_infrequent = infrequent.copy(deep=True)
    original_pivot = pivot.copy(deep=True)
    exception_type, message = _expected_error(argument_name, invalid_kind)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(exception_type, match=message):
            interpolate_infrequent_returns(
                infrequent_returns=infrequent,
                pivot_returns=pivot,
                span=3,
                annualization_factor=365,
                vol_adjustment=1.0,
            )

    _assert_panel_equal(infrequent, original_infrequent)
    pd.testing.assert_series_equal(pivot, original_pivot, check_exact=True)

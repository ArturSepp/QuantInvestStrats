"""Regression tests for calendar-index validation in ``df_asfreq``.

Calendar resampling requires timestamp labels because it constructs a date schedule and carries
observations to scheduled dates. Non-date labels remain valid when no resampling is requested,
and empty objects remain useful schema declarations. These tests distinguish those pass-through
contracts from nonempty calendar operations, which should reject unsupported indexes before
pandas exposes an incidental attribute or schedule-generation error.

The deterministic valid control combines an unsorted timezone-aware index, a missing nullable
value, and literal daily expectations. It therefore confirms that validation changes only invalid
input failures while preserving chronological repair, timezone metadata, nullable storage,
caller ownership, and resampled values.
"""

import warnings
from typing import Literal

import pandas as pd
import pytest

from qis.utils.df_freq import df_asfreq


# =============================================================================
# Shared deterministic fixtures
# =============================================================================

_NON_DATE_ERROR = r"^df must use a DatetimeIndex for calendar operations$"
_NAT_ERROR = r"^df index must not contain NaT$"

_PanelShape = Literal["series", "frame"]


def _make_panel(index: pd.Index, shape: _PanelShape) -> pd.Series | pd.DataFrame:
    """Create one ordinary floating panel with the requested public shape.

    Args:
        index: Labels assigned to the two deterministic observations.
        shape: Return a Series or one-column DataFrame.

    Returns:
        Fresh panel containing the literal values one and two.
    """
    series = pd.Series((1.0, 2.0), index=index, name="Asset")
    if shape == "series":
        return series
    return series.to_frame()


def _assert_panel_unchanged(
    actual: pd.Series | pd.DataFrame,
    expected: pd.Series | pd.DataFrame,
) -> None:
    """Assert that a rejected or accepted call preserves its caller-owned input.

    Args:
        actual: Panel supplied to ``df_asfreq``.
        expected: Independent snapshot taken before the call.
    """
    if isinstance(actual, pd.Series):
        assert isinstance(expected, pd.Series)
        pd.testing.assert_series_equal(actual, expected, check_exact=True)
    else:
        assert isinstance(expected, pd.DataFrame)
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)


# =============================================================================
# Invalid nonempty calendar indexes
# =============================================================================


@pytest.mark.parametrize("shape", ("series", "frame"))
def test_df_asfreq_rejects_non_date_index_before_calendar_resampling(
    shape: _PanelShape,
) -> None:
    """Name the invalid public argument instead of exposing ``RangeIndex.tz``.

    Args:
        shape: Public pandas container under test.
    """
    panel = _make_panel(pd.RangeIndex(2, name="row"), shape)
    original = panel.copy(deep=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(TypeError, match=_NON_DATE_ERROR):
            df_asfreq(panel, freq="D")

    _assert_panel_unchanged(panel, original)


@pytest.mark.parametrize("shape", ("series", "frame"))
def test_df_asfreq_rejects_nat_before_calendar_resampling(shape: _PanelShape) -> None:
    """Reject an undefined calendar boundary before constructing a date schedule.

    Args:
        shape: Public pandas container under test.
    """
    index = pd.DatetimeIndex(("2024-01-01", pd.NaT), name="date")
    panel = _make_panel(index, shape)
    original = panel.copy(deep=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=_NAT_ERROR):
            df_asfreq(panel, freq="D")

    _assert_panel_unchanged(panel, original)


# =============================================================================
# Supported pass-through and valid calendar controls
# =============================================================================


@pytest.mark.parametrize("shape", ("series", "frame"))
def test_df_asfreq_preserves_non_date_index_when_frequency_is_none(shape: _PanelShape) -> None:
    """Keep arbitrary labels valid when the caller requests no calendar operation.

    Args:
        shape: Public pandas container under test.
    """
    panel = _make_panel(pd.Index(("first", "second"), name="row"), shape)
    original = panel.copy(deep=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = df_asfreq(panel, freq=None)

    _assert_panel_unchanged(actual, original)
    _assert_panel_unchanged(panel, original)


@pytest.mark.parametrize("shape", ("series", "frame"))
def test_df_asfreq_preserves_empty_non_date_schema(shape: _PanelShape) -> None:
    """Allow an empty declaration because no observation enters a calendar schedule.

    Args:
        shape: Public pandas container under test.
    """
    series = pd.Series([], index=pd.RangeIndex(0, name="row"), dtype="Float64", name="Asset")
    panel: pd.Series | pd.DataFrame = series if shape == "series" else series.to_frame()
    original = panel.copy(deep=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = df_asfreq(panel, freq="D")

    _assert_panel_unchanged(actual, original)
    _assert_panel_unchanged(panel, original)


@pytest.mark.parametrize("shape", ("series", "frame"))
def test_df_asfreq_preserves_valid_timezone_aware_nullable_values(shape: _PanelShape) -> None:
    """Sort and resample a valid nullable panel without changing its numerical contract.

    The supplied rows arrive as January 3, January 1, and January 2. After chronological repair,
    the already-daily panel retains its supplied missing January 2 observation, giving the
    independently expected sequence ``[1, missing, 3]``.

    Args:
        shape: Public pandas container under test.
    """
    index = pd.DatetimeIndex(
        ("2024-01-03", "2024-01-01", "2024-01-02"),
        tz="UTC",
        name="date",
    )
    series = pd.Series((3.0, 1.0, pd.NA), index=index, dtype="Float64", name="Asset")
    panel: pd.Series | pd.DataFrame = series if shape == "series" else series.to_frame()
    original = panel.copy(deep=True)
    expected_series = pd.Series(
        (1.0, pd.NA, 3.0),
        index=pd.DatetimeIndex(
            ("2024-01-01", "2024-01-02", "2024-01-03"),
            tz="UTC",
            name="date",
        ),
        dtype="Float64",
        name="Asset",
    )
    expected: pd.Series | pd.DataFrame = (
        expected_series if shape == "series" else expected_series.to_frame()
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = df_asfreq(
            panel,
            freq="D",
            include_start_date=True,
            include_end_date=True,
        )

    _assert_panel_unchanged(actual, expected)
    _assert_panel_unchanged(panel, original)

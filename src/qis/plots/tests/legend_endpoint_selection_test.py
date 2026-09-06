"""Characterize endpoint selection across every endpoint-bearing legend mode.

``LegendStats`` names describe the fields and their display order, but the historical selection
sample differs by mode. ``LAST``, ``AVG_LAST``, and ``AVG_STD_LAST`` use the value at the final
index. Explicit ``NONNAN`` modes use the final observed value, while composite modes select both
endpoints from the same missing-filtered sample used for their other statistics. ``NONZERO``
composites additionally remove exact zeros before selecting endpoints.

The deterministic boundaries distinguish complete, leading-missing, interior-missing,
trailing-missing, and all-missing histories. Exact text for every ``FIRST``/``LAST`` member,
ordinary and nullable storage, a mixed panel, Series/DataFrame parity, warnings-as-errors, public
canvas rendering, labels and order, figure cleanup, and caller ownership preserve the established
contract without changing numerical behavior or enum values.
"""

import warnings
from typing import Protocol, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# qis
import qis.plots.time_series as time_series_module
import qis.plots.utils as plot_utils_module
from qis.plots.utils import LegendStats


class _PlotUtilsModuleProtocol(Protocol):
    """Typed test-side interface for the shared legend builder."""

    def get_legend_lines(
        self,
        data: pd.DataFrame | pd.Series,
        *,
        legend_stats: LegendStats,
        var_format: str,
        nan_display: float,
    ) -> list[str]:
        """Build exact text for one endpoint-bearing legend mode.

        Args:
            data: Series or DataFrame summarized in the legend.
            legend_stats: Endpoint-bearing legend mode under test.
            var_format: Explicit numerical display format.
            nan_display: Scalar displayed for an unavailable statistic.

        Returns:
            Legend entries in input-column order.
        """
        raise NotImplementedError


class _TimeSeriesModuleProtocol(Protocol):
    """Typed test-side interface for the public time-series plot."""

    def plot_time_series(
        self,
        df: pd.DataFrame | pd.Series,
        *,
        x_date_freq: None,
        legend_stats: LegendStats,
        var_format: str,
        ax: Axes,
    ) -> Figure | None:
        """Render a time series with endpoint statistics in its legend.

        Args:
            df: Series or DataFrame plotted by date.
            x_date_freq: Disabled date-axis frequency formatting.
            legend_stats: Endpoint-bearing legend mode under test.
            var_format: Explicit numerical display format.
            ax: Matplotlib axis receiving the plot.

        Returns:
            Created figure, or ``None`` when the caller supplies an axis.
        """
        raise NotImplementedError


_PLOT_UTILS = cast(_PlotUtilsModuleProtocol, plot_utils_module)
_TIME_SERIES = cast(_TimeSeriesModuleProtocol, time_series_module)


# =============================================================================
# Shared deterministic fixtures and independently specified expectations
# =============================================================================

_DATES = pd.date_range("2024-01-31", periods=4, freq="ME")

_ALL_MISSING = "All Missing"
_ASSET = "Asset"
_COMPLETE = "Complete"
_INTERIOR_MISSING = "Interior Missing"
_LEADING_MISSING = "Leading Missing"
_TRAILING_MISSING = "Trailing Missing"

_ALL_MISSING_VALUES = (None, None, None, None)
_COMPLETE_VALUES = (1.0, 2.0, 3.0, 4.0)
_INTERIOR_MISSING_VALUES = (1.0, None, 3.0, 4.0)
_LEADING_MISSING_VALUES = (None, 2.0, 3.0, 4.0)
_TRAILING_MISSING_VALUES = (1.0, 2.0, 3.0, None)
_ZERO_ENDPOINT_VALUES = (0.0, 1.0, 2.0, 0.0)

_NAN_DISPLAY = -99.0
_VAR_FORMAT = "{:.1f}"

_TRAILING_EXPECTATIONS: tuple[tuple[LegendStats, str], ...] = (
    (LegendStats.LAST, "Asset: last=-99.0"),
    (LegendStats.AVG_LAST, "Asset: avg=2.0, last=-99.0"),
    (LegendStats.AVG_STD_LAST, "Asset: avg=2.0, std=1.0, last=-99.0"),
    (LegendStats.AVG_NONNAN_LAST, "Asset: avg=2.0, last=3.0"),
    (LegendStats.NONZERO_AVG_LAST, "Asset: avg=2.0, last=3.0"),
    (LegendStats.NONZERO_AVG_STD_LAST, "Asset: avg=2.0, std=1.0, last=3.0"),
    (LegendStats.MEDIAN_NONNAN_LAST, "Asset: median=2.0, last=3.0"),
    (
        LegendStats.AVG_MEDIAN_STD_NONNAN_LAST,
        "Asset: avg=2.0, median=2.0, std=1.0, last=3.0",
    ),
    (LegendStats.AVG_LAST_SCORE, "Asset: avg=2.0, last=3.0, last score=100%"),
    (
        LegendStats.AVG_STD_LAST_SCORE,
        "Asset: avg=2.0, std=1.0,  last=3.0, last score=100%",
    ),
    (LegendStats.FIRST_LAST, "Asset: first=1.0, last=3.0"),
    (LegendStats.FIRST_LAST_NON_ZERO, "Asset: first=1.0, last=3.0"),
    (LegendStats.FIRST_AVG_LAST, "Asset: first=1.0, avg=2.0, last=3.0"),
    (LegendStats.FIRST_MEDIAN_LAST, "Asset: first=1.0, median=2.0, last=3.0"),
    (LegendStats.FIRST_AVG_LAST_SHORT, "Asset: [1.0, 2.0, 3.0]"),
    (LegendStats.MISSING_AVG_LAST, "Asset: missing%=25.00%, avg=2.0, last=3.0"),
    (LegendStats.LAST_NONNAN, "Asset = 3.0"),
    (LegendStats.AVG_MIN_MAX_LAST, "Asset: avg=2.0, min=1.0, max=3.0, last=3.0"),
    (LegendStats.FIRST_MIN_MAX_LAST, "Asset: first=1.0, min=1.0, max=3.0, last=3.0"),
)

_LEADING_FIRST_EXPECTATIONS: tuple[tuple[LegendStats, str], ...] = (
    (LegendStats.FIRST_LAST, "Asset: first=2.0, last=4.0"),
    (LegendStats.FIRST_LAST_NON_ZERO, "Asset: first=2.0, last=4.0"),
    (LegendStats.FIRST_AVG_LAST, "Asset: first=2.0, avg=3.0, last=4.0"),
    (LegendStats.FIRST_MEDIAN_LAST, "Asset: first=2.0, median=3.0, last=4.0"),
    (LegendStats.FIRST_AVG_LAST_SHORT, "Asset: [2.0, 3.0, 4.0]"),
    (LegendStats.FIRST_MIN_MAX_LAST, "Asset: first=2.0, min=2.0, max=4.0, last=4.0"),
)

_NONZERO_EXPECTATIONS: tuple[tuple[LegendStats, str], ...] = (
    (LegendStats.NONZERO_AVG_LAST, "Asset: avg=1.5, last=2.0"),
    (LegendStats.NONZERO_AVG_STD_LAST, "Asset: avg=1.5, std=0.7, last=2.0"),
    (LegendStats.FIRST_LAST_NON_ZERO, "Asset: first=1.0, last=2.0"),
)

_MIXED_EXPECTATIONS: tuple[tuple[LegendStats, tuple[str, ...]], ...] = (
    (
        LegendStats.LAST,
        (
            "Complete: last=4.0",
            "Leading Missing: last=4.0",
            "Interior Missing: last=4.0",
            "Trailing Missing: last=-99.0",
            "All Missing: last=-99.0",
        ),
    ),
    (
        LegendStats.LAST_NONNAN,
        (
            "Complete = 4.0",
            "Leading Missing = 4.0",
            "Interior Missing = 4.0",
            "Trailing Missing = 3.0",
            "All Missing = -99.0",
        ),
    ),
    (
        LegendStats.FIRST_AVG_LAST,
        (
            "Complete: first=1.0, avg=2.5, last=4.0",
            "Leading Missing: first=2.0, avg=3.0, last=4.0",
            "Interior Missing: first=1.0, avg=2.7, last=4.0",
            "Trailing Missing: first=1.0, avg=2.0, last=3.0",
            "All Missing: first=-99.0, avg=-99.0, last=-99.0",
        ),
    ),
)

_PUBLIC_RENDER_EXPECTATIONS: tuple[tuple[LegendStats, str], ...] = (
    (LegendStats.LAST, "Asset: last=nan"),
    (LegendStats.FIRST_AVG_LAST, "Asset: first=1.0, avg=2.0, last=3.0"),
)


def _series(
    values: tuple[float | None, ...],
    *,
    name: str = _ASSET,
    nullable: bool,
) -> pd.Series:
    """Create one boundary history with ordinary or nullable floating storage.

    Args:
        values: Four observations, using ``None`` at missing positions.
        name: Series label propagated into the legend.
        nullable: Whether to use pandas nullable ``Float64`` storage.

    Returns:
        Named Series indexed by the shared monthly dates.
    """
    missing = pd.NA if nullable else np.nan
    normalized = tuple(missing if value is None else value for value in values)
    if nullable:
        return pd.Series(normalized, index=_DATES, name=name, dtype=pd.Float64Dtype())
    return pd.Series(normalized, index=_DATES, name=name, dtype=float)


def _mixed_panel(*, nullable: bool) -> pd.DataFrame:
    """Create every materially different missing-data state in one ordered panel.

    Args:
        nullable: Whether every column uses pandas nullable ``Float64`` storage.

    Returns:
        Complete, leading-, interior-, trailing-, and all-missing columns in contract order.
    """
    return pd.DataFrame(
        {
            _COMPLETE: _series(_COMPLETE_VALUES, name=_COMPLETE, nullable=nullable),
            _LEADING_MISSING: _series(
                _LEADING_MISSING_VALUES,
                name=_LEADING_MISSING,
                nullable=nullable,
            ),
            _INTERIOR_MISSING: _series(
                _INTERIOR_MISSING_VALUES,
                name=_INTERIOR_MISSING,
                nullable=nullable,
            ),
            _TRAILING_MISSING: _series(
                _TRAILING_MISSING_VALUES,
                name=_TRAILING_MISSING,
                nullable=nullable,
            ),
            _ALL_MISSING: _series(
                _ALL_MISSING_VALUES,
                name=_ALL_MISSING,
                nullable=nullable,
            ),
        },
        index=_DATES,
    )


def _legend_lines(
    data: pd.DataFrame | pd.Series,
    legend_stats: LegendStats,
) -> tuple[str, ...]:
    """Build endpoint legend text while treating every warning as a failure.

    Args:
        data: Series or DataFrame summarized in the legend.
        legend_stats: Endpoint-bearing legend mode under test.

    Returns:
        Immutable legend text in input-column order.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return tuple(
            _PLOT_UTILS.get_legend_lines(
                data,
                legend_stats=legend_stats,
                var_format=_VAR_FORMAT,
                nan_display=_NAN_DISPLAY,
            )
        )


def _axis_legend_text(ax: Axes) -> tuple[str, ...]:
    """Read exact legend entries from a rendered public plot.

    Args:
        ax: Matplotlib axis containing the completed plot.

    Returns:
        Legend labels in display order.
    """
    legend = ax.get_legend()
    assert legend is not None
    return tuple(text.get_text() for text in legend.get_texts())


# =============================================================================
# Complete endpoint-mode inventory
# =============================================================================


def test_get_legend_lines_endpoint_inventory_is_fully_characterized() -> None:
    """Require every enum member containing ``FIRST`` or ``LAST`` to have an expectation."""
    expected_modes = tuple(mode for mode, _ in _TRAILING_EXPECTATIONS)
    actual_modes = tuple(
        mode for mode in LegendStats if "FIRST" in mode.name or "LAST" in mode.name
    )

    assert actual_modes == expected_modes


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
@pytest.mark.parametrize(
    ("legend_stats", "expected"),
    _TRAILING_EXPECTATIONS,
    ids=tuple(mode.name.lower() for mode, _ in _TRAILING_EXPECTATIONS),
)
def test_get_legend_lines_selects_documented_trailing_endpoint(
    legend_stats: LegendStats,
    expected: str,
    nullable: bool,
) -> None:
    """Distinguish final-index selection from final-observed sample selection.

    Args:
        legend_stats: Endpoint-bearing mode under test.
        expected: Independently specified text for ``[1, 2, 3, missing]``.
        nullable: Whether the Series uses nullable ``Float64`` storage.
    """
    values = _series(_TRAILING_MISSING_VALUES, nullable=nullable)
    original = values.copy()

    assert _legend_lines(values, legend_stats) == (expected,)
    pd.testing.assert_series_equal(values, original)


# =============================================================================
# First-selected-observation contract
# =============================================================================


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
@pytest.mark.parametrize(
    ("legend_stats", "expected"),
    _LEADING_FIRST_EXPECTATIONS,
    ids=tuple(mode.name.lower() for mode, _ in _LEADING_FIRST_EXPECTATIONS),
)
def test_get_legend_lines_selects_first_observed_composite_value(
    legend_stats: LegendStats,
    expected: str,
    nullable: bool,
) -> None:
    """Select the first observed value after a leading gap in every ``FIRST`` mode.

    Args:
        legend_stats: First-bearing composite mode under test.
        expected: Independently specified text for ``[missing, 2, 3, 4]``.
        nullable: Whether the Series uses nullable ``Float64`` storage.
    """
    values = _series(_LEADING_MISSING_VALUES, nullable=nullable)

    assert _legend_lines(values, legend_stats) == (expected,)


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
@pytest.mark.parametrize(
    ("legend_stats", "expected"),
    _NONZERO_EXPECTATIONS,
    ids=tuple(mode.name.lower() for mode, _ in _NONZERO_EXPECTATIONS),
)
def test_get_legend_lines_selects_nonzero_sample_endpoints(
    legend_stats: LegendStats,
    expected: str,
    nullable: bool,
) -> None:
    """Remove exact-zero endpoints before selecting ``NONZERO`` summaries.

    Args:
        legend_stats: Exact-zero-filtered composite mode under test.
        expected: Independently specified text for ``[0, 1, 2, 0]``.
        nullable: Whether the Series uses nullable ``Float64`` storage.
    """
    values = _series(_ZERO_ENDPOINT_VALUES, nullable=nullable)

    assert _legend_lines(values, legend_stats) == (expected,)


# =============================================================================
# Mixed-panel and shape consistency
# =============================================================================


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
@pytest.mark.parametrize(
    ("legend_stats", "expected"),
    _MIXED_EXPECTATIONS,
    ids=tuple(mode.name.lower() for mode, _ in _MIXED_EXPECTATIONS),
)
def test_get_legend_lines_preserves_endpoint_states_in_one_mixed_panel(
    legend_stats: LegendStats,
    expected: tuple[str, ...],
    nullable: bool,
) -> None:
    """Preserve every missing-data state and column order in one vectorized call.

    Args:
        legend_stats: Representative mode for one endpoint-selection family.
        expected: Exact entries for every independently constructed column state.
        nullable: Whether the panel uses nullable ``Float64`` storage.
    """
    values = _mixed_panel(nullable=nullable)
    original = values.copy()

    assert _legend_lines(values, legend_stats) == expected
    pd.testing.assert_frame_equal(values, original)


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
@pytest.mark.parametrize(
    ("legend_stats", "expected"),
    _TRAILING_EXPECTATIONS,
    ids=tuple(mode.name.lower() for mode, _ in _TRAILING_EXPECTATIONS),
)
def test_get_legend_lines_endpoint_series_matches_one_column_dataframe(
    legend_stats: LegendStats,
    expected: str,
    nullable: bool,
) -> None:
    """Return identical endpoint text for a named Series and one-column DataFrame.

    Args:
        legend_stats: Endpoint-bearing mode under test.
        expected: Independently specified trailing-gap legend entry.
        nullable: Whether both inputs use nullable ``Float64`` storage.
    """
    series = _series(_TRAILING_MISSING_VALUES, nullable=nullable)
    frame = series.to_frame()

    series_result = _legend_lines(series, legend_stats)
    frame_result = _legend_lines(frame, legend_stats)

    assert series_result == (expected,)
    assert frame_result == (expected,)
    assert series_result == frame_result


# =============================================================================
# Public rendered-plot contract
# =============================================================================


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
@pytest.mark.parametrize(
    ("legend_stats", "expected"),
    _PUBLIC_RENDER_EXPECTATIONS,
    ids=tuple(mode.name.lower() for mode, _ in _PUBLIC_RENDER_EXPECTATIONS),
)
def test_plot_time_series_renders_indexed_and_observed_endpoints(
    legend_stats: LegendStats,
    expected: str,
    nullable: bool,
) -> None:
    """Render representative indexed and selected-sample endpoint contracts.

    Args:
        legend_stats: Representative endpoint-selection mode under test.
        expected: Independently specified public legend entry.
        nullable: Whether the plotted Series uses nullable ``Float64`` storage.
    """
    values = _series(_TRAILING_MISSING_VALUES, nullable=nullable)
    original = values.copy()
    figure, axis = plt.subplots()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _TIME_SERIES.plot_time_series(
                values,
                x_date_freq=None,
                legend_stats=legend_stats,
                var_format=_VAR_FORMAT,
                ax=axis,
            )
            figure.canvas.draw()

        assert _axis_legend_text(axis) == (expected,)
        pd.testing.assert_series_equal(values, original)
    finally:
        plt.close(figure)

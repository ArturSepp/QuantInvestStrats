"""Regression coverage for all-zero nonzero-endpoint legends.

``LegendStats.FIRST_LAST_NON_ZERO`` treats exact zeros as unavailable positions and displays the
first and last observations that remain. A history containing no nonzero observations therefore
has two undefined endpoints, just as an all-missing history does; it must not abort an otherwise
valid mixed panel.

One ordered fixture combines ordinary and nullable all-zero columns with leading, trailing, and
interior zeros, an all-missing history, a singleton nonzero history, and an ordinary signed
history. Exact helper and public-plot labels, custom and native missing displays,
Series/DataFrame consistency, warnings-as-errors, forced canvas rendering, column order, caller
ownership, and figure cleanup protect the narrow endpoint-selection contract.
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
import qis.plots.stackplot as stackplot_module
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
        var_format: str | None,
        nan_display: float,
    ) -> list[str]:
        """Return legend lines for the selected nonzero endpoints.

        Args:
            data: Series or DataFrame summarized in the legend.
            legend_stats: Public legend mode under test.
            var_format: Explicit Python format string, or native scalar formatting.
            nan_display: Scalar used when no selected endpoint exists.

        Returns:
            Legend lines in input-column order.
        """
        raise NotImplementedError


class _StackplotModuleProtocol(Protocol):
    """Typed test-side interface for the public stacked plot."""

    def plot_stack(
        self,
        df: pd.DataFrame,
        *,
        x_date_freq: None,
        legend_stats: LegendStats,
        var_format: str,
        ax: Axes,
    ) -> Figure | None:
        """Render a stacked plot with nonzero endpoint labels.

        Args:
            df: Ordinary floating-point panel to stack.
            x_date_freq: Disabled date-axis formatting for the focused test.
            legend_stats: Public legend mode under test.
            var_format: Explicit endpoint format.
            ax: Matplotlib axis receiving the plot.

        Returns:
            The created figure, or None when the caller supplies ``ax``.
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
        """Render time series with nonzero endpoint labels.

        Args:
            df: Series or DataFrame plotted by date.
            x_date_freq: Disabled date-axis formatting for the focused test.
            legend_stats: Public legend mode under test.
            var_format: Explicit endpoint format.
            ax: Matplotlib axis receiving the plot.

        Returns:
            The created figure, or None when the caller supplies ``ax``.
        """
        raise NotImplementedError


_PLOT_UTILS = cast(_PlotUtilsModuleProtocol, plot_utils_module)
_STACKPLOT = cast(_StackplotModuleProtocol, stackplot_module)
_TIME_SERIES = cast(_TimeSeriesModuleProtocol, time_series_module)


# =============================================================================
# Shared deterministic fixtures and independent expectations
# =============================================================================

_DATES = pd.date_range("2024-01-31", periods=6, freq="ME")

_FLOAT_ALL_ZERO = "Float All Zero"
_NULLABLE_ALL_ZERO = "Nullable All Zero"
_EDGE_ZERO = "Edge Zero"
_INTERIOR_ZERO = "Interior Zero"
_ALL_MISSING = "All Missing"
_SINGLETON_NONZERO = "Singleton Nonzero"
_SIGNED = "Signed"

_FORMATTED_EXPECTED = (
    "Float All Zero: first=nan, last=nan",
    "Nullable All Zero: first=nan, last=nan",
    "Edge Zero: first=2.00, last=3.00",
    "Interior Zero: first=1.00, last=4.00",
    "All Missing: first=nan, last=nan",
    "Singleton Nonzero: first=5.00, last=5.00",
    "Signed: first=-2.00, last=3.00",
)

_CUSTOM_MISSING_EXPECTED = (
    "Float All Zero: first=-99.00, last=-99.00",
    "Nullable All Zero: first=-99.00, last=-99.00",
    "Edge Zero: first=2.00, last=3.00",
    "Interior Zero: first=1.00, last=4.00",
    "All Missing: first=-99.00, last=-99.00",
    "Singleton Nonzero: first=5.00, last=5.00",
    "Signed: first=-2.00, last=3.00",
)

_NATIVE_EXPECTED = (
    "Float All Zero: first=nan, last=nan",
    "Nullable All Zero: first=nan, last=nan",
    "Edge Zero: first=2.0, last=3.0",
    "Interior Zero: first=1.0, last=4.0",
    "All Missing: first=nan, last=nan",
    "Singleton Nonzero: first=5.0, last=5.0",
    "Signed: first=-2.0, last=3.0",
)


def _mixed_values() -> pd.DataFrame:
    """Create every material zero-selection state in one ordered panel.

    Returns:
        Six-date panel retaining ordinary and nullable floating-point columns.
    """
    values = pd.DataFrame(index=_DATES)
    values[_FLOAT_ALL_ZERO] = pd.Series((0.0,) * 6, index=_DATES, dtype=float)
    values[_NULLABLE_ALL_ZERO] = pd.Series((0.0,) * 6, index=_DATES, dtype=pd.Float64Dtype())
    values[_EDGE_ZERO] = pd.Series((0.0, 2.0, 0.0, 3.0, 0.0, 0.0), index=_DATES, dtype=float)
    values[_INTERIOR_ZERO] = pd.Series((1.0, 0.0, 2.0, 0.0, 3.0, 4.0), index=_DATES, dtype=float)
    values[_ALL_MISSING] = pd.Series((pd.NA,) * 6, index=_DATES, dtype=pd.Float64Dtype())
    values[_SINGLETON_NONZERO] = pd.Series(
        (0.0, pd.NA, 5.0, 0.0, pd.NA, 0.0), index=_DATES, dtype=pd.Float64Dtype()
    )
    values[_SIGNED] = pd.Series((-2.0, -1.0, 0.0, 1.0, 2.0, 3.0), index=_DATES, dtype=float)
    return values


def _legend_lines(
    data: pd.DataFrame | pd.Series,
    *,
    var_format: str | None,
    nan_display: float = np.nan,
) -> tuple[str, ...]:
    """Build exact legend text while treating every warning as a failure.

    Args:
        data: Series or DataFrame summarized in the legend.
        var_format: Explicit Python format string, or native scalar formatting.
        nan_display: Scalar used when no selected endpoint exists.

    Returns:
        Immutable legend lines in input-column order.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return tuple(
            _PLOT_UTILS.get_legend_lines(
                data,
                legend_stats=LegendStats.FIRST_LAST_NON_ZERO,
                var_format=var_format,
                nan_display=nan_display,
            )
        )


def _axis_legend_text(axis: Axes) -> tuple[str, ...]:
    """Read exact legend labels from a rendered axis.

    Args:
        axis: Axis containing a completed public plot.

    Returns:
        Legend labels in display order.
    """
    legend = axis.get_legend()
    assert legend is not None
    return tuple(text.get_text() for text in legend.get_texts())


# =============================================================================
# Shared-helper mixed-panel contract
# =============================================================================


def test_get_legend_lines_handles_all_zero_columns_in_mixed_panel() -> None:
    """Return undefined endpoints without aborting valid neighboring histories."""
    values = _mixed_values()
    original = values.copy()

    actual = _legend_lines(values, var_format="{:.2f}")

    assert actual == _FORMATTED_EXPECTED
    pd.testing.assert_frame_equal(values, original)


@pytest.mark.parametrize(
    ("var_format", "nan_display", "expected"),
    (
        ("{:.2f}", -99.0, _CUSTOM_MISSING_EXPECTED),
        (None, np.nan, _NATIVE_EXPECTED),
    ),
    ids=("custom-missing", "native-format"),
)
def test_get_legend_lines_preserves_missing_and_format_contracts(
    var_format: str | None,
    nan_display: float,
    expected: tuple[str, ...],
) -> None:
    """Keep configured missing values and native formatting after zero selection.

    Args:
        var_format: Explicit endpoint format, or native scalar formatting.
        nan_display: Scalar used when no selected endpoint exists.
        expected: Independently specified text for the complete mixed panel.
    """
    assert (
        _legend_lines(_mixed_values(), var_format=var_format, nan_display=nan_display) == expected
    )


# =============================================================================
# Named Series/DataFrame consistency
# =============================================================================


@pytest.mark.parametrize("column", (_FLOAT_ALL_ZERO, _NULLABLE_ALL_ZERO))
def test_get_legend_lines_all_zero_series_matches_one_column_frame(column: str) -> None:
    """Return the same undefined endpoints for ordinary and nullable pandas shapes.

    Args:
        column: Ordinary or nullable all-zero fixture column.
    """
    selected = _mixed_values()[column]
    assert isinstance(selected, pd.Series)

    series_result = _legend_lines(selected, var_format="{:.2f}")
    frame_result = _legend_lines(selected.to_frame(), var_format="{:.2f}")

    expected = (f"{column}: first=nan, last=nan",)
    assert series_result == expected
    assert frame_result == expected


# =============================================================================
# Public rendering contracts
# =============================================================================


def test_plot_time_series_renders_all_zero_endpoints_in_mixed_panel() -> None:
    """Render the complete mixed endpoint contract through the public line plot."""
    values = _mixed_values()
    original = values.copy()
    figure, axis = plt.subplots()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _TIME_SERIES.plot_time_series(
                values,
                x_date_freq=None,
                legend_stats=LegendStats.FIRST_LAST_NON_ZERO,
                var_format="{:.2f}",
                ax=axis,
            )
            figure.canvas.draw()

        assert _axis_legend_text(axis) == _FORMATTED_EXPECTED
        pd.testing.assert_frame_equal(values, original)
    finally:
        plt.close(figure)


def test_plot_stack_renders_all_zero_endpoints_with_total_title() -> None:
    """Render the mode's ordinary stack-plot labels and established total title."""
    values = _mixed_values().loc[:, [_FLOAT_ALL_ZERO, _EDGE_ZERO]]
    assert isinstance(values, pd.DataFrame)
    original = values.copy()
    figure, axis = plt.subplots()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _STACKPLOT.plot_stack(
                values,
                x_date_freq=None,
                legend_stats=LegendStats.FIRST_LAST_NON_ZERO,
                var_format="{:.2f}",
                ax=axis,
            )
            figure.canvas.draw()

        assert _axis_legend_text(axis) == (
            "Float All Zero: first=nan, last=nan",
            "Edge Zero: first=2.00, last=3.00",
        )
        legend = axis.get_legend()
        assert legend is not None
        assert legend.get_title().get_text() == "Total: last=0.00"
        pd.testing.assert_frame_equal(values, original)
    finally:
        plt.close(figure)

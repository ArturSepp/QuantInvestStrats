"""Regression coverage for nullable floating-point stack-plot inputs.

Matplotlib's stacked-area renderer requires a real numeric array, while a transposed pandas
``Float64`` frame exposes an object array containing Python floats and ``pd.NA``. The public
``plot_stack()`` boundary must translate nullable missing values to ``np.nan`` before area
rendering without changing values, labels, stacking geometry, legends, or caller-owned data.

One ordered mixed panel combines ordinary and nullable finite data, ragged and all-missing
histories, exact zeros, and signed values. Ordinary/nullable artist equivalence, the already
working stacked-bar control, warnings-as-errors, forced canvas rendering, exact legend text,
caller ownership, and figure cleanup protect the narrow renderer-normalization contract.
"""

import warnings
from typing import Protocol, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

# qis
import qis.plots.stackplot as stackplot_module
from qis.plots.utils import LegendStats


class _StackplotModuleProtocol(Protocol):
    """Typed test-side interface for the public stacked plot."""

    def plot_stack(
        self,
        df: pd.DataFrame,
        *,
        use_bar_plot: bool,
        x_date_freq: None,
        legend_stats: LegendStats,
        var_format: str,
        ax: Axes,
    ) -> Figure | None:
        """Render a stacked plot with deterministic legend labels.

        Args:
            df: Numeric panel to stack.
            use_bar_plot: Use bars instead of stacked areas when true.
            x_date_freq: Disabled date-axis relabeling for this focused contract.
            legend_stats: Public legend mode under test.
            var_format: Explicit numeric format used by the expected labels.
            ax: Matplotlib axis receiving the plot.

        Returns:
            The created figure, or None when the caller supplies ``ax``.
        """
        raise NotImplementedError


_STACKPLOT = cast(_StackplotModuleProtocol, stackplot_module)


# =============================================================================
# Shared deterministic fixtures and independent expectations
# =============================================================================

_ALL_MISSING = "All Missing"
_ALL_ZERO = "All Zero"
_DATES = pd.date_range("2024-01-31", periods=5, freq="ME")
_FINITE = "Finite"
_NULLABLE_FINITE = "Nullable Finite"
_RAGGED = "Ragged"
_SIGNED = "Signed"

_EXPECTED_LEGEND = (
    "Finite: first=0.100, last=0.500",
    "Nullable Finite: first=0.500, last=0.100",
    "Ragged: first=0.200, last=0.400",
    "All Missing: first=nan, last=nan",
    "All Zero: first=0.000, last=0.000",
    "Signed: first=-0.200, last=0.200",
)


def _ordinary_mixed_panel() -> pd.DataFrame:
    """Return the literal ordinary-float reference panel."""
    return pd.DataFrame(
        {
            _FINITE: (0.1, 0.2, 0.3, 0.4, 0.5),
            _NULLABLE_FINITE: (0.5, 0.4, 0.3, 0.2, 0.1),
            _RAGGED: (np.nan, 0.2, 0.3, 0.4, np.nan),
            _ALL_MISSING: (np.nan, np.nan, np.nan, np.nan, np.nan),
            _ALL_ZERO: (0.0, 0.0, 0.0, 0.0, 0.0),
            _SIGNED: (-0.2, 0.0, 0.3, -0.1, 0.2),
        },
        index=_DATES,
        dtype=float,
    )


def _mixed_nullable_panel() -> pd.DataFrame:
    """Return every material column state with both ordinary and nullable storage."""
    values = _ordinary_mixed_panel()
    nullable_columns = {
        column: pd.Float64Dtype()
        for column in (_NULLABLE_FINITE, _RAGGED, _ALL_MISSING, _ALL_ZERO, _SIGNED)
    }
    return values.astype(nullable_columns)


def _render_stack(values: pd.DataFrame, *, use_bar_plot: bool) -> tuple[Figure, Axes]:
    """Render one stack plot under warnings-as-errors while protecting caller ownership.

    Args:
        values: Numeric panel to render.
        use_bar_plot: Use the stacked-bar path instead of the stacked-area path.

    Returns:
        Open figure and axis for exact artist inspection. The caller must close the figure.
    """
    original = values.copy()
    figure, axis = plt.subplots()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = _STACKPLOT.plot_stack(
                values,
                use_bar_plot=use_bar_plot,
                x_date_freq=None,
                legend_stats=LegendStats.FIRST_LAST,
                var_format="{:.3f}",
                ax=axis,
            )
            figure.canvas.draw()
        assert result is None
        pd.testing.assert_frame_equal(values, original)
    except Exception:
        plt.close(figure)
        raise
    return figure, axis


def _legend_text(axis: Axes) -> tuple[str, ...]:
    """Return legend labels in their rendered column order.

    Args:
        axis: Rendered stack-plot axis.

    Returns:
        Exact legend text for every plotted column.
    """
    legend = axis.get_legend()
    assert legend is not None
    return tuple(text.get_text() for text in legend.get_texts())


def _assert_area_paths_equal(actual: Axes, expected: Axes) -> None:
    """Compare every stacked-area polygon without relying on raster output.

    Args:
        actual: Axis rendered from mixed nullable storage.
        expected: Axis rendered from the ordinary-float reference.
    """
    assert len(actual.collections) == len(expected.collections)
    for actual_collection, expected_collection in zip(actual.collections, expected.collections):
        actual_paths = actual_collection.get_paths()
        expected_paths = expected_collection.get_paths()
        assert len(actual_paths) == len(expected_paths)
        for actual_path, expected_path in zip(actual_paths, expected_paths):
            actual_vertices = np.asarray(actual_path.vertices, dtype=float)
            expected_vertices = np.asarray(expected_path.vertices, dtype=float)
            np.testing.assert_allclose(
                actual_vertices,
                expected_vertices,
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )


def _assert_bar_geometry_equal(actual: Axes, expected: Axes) -> None:
    """Compare every stacked-bar rectangle in drawing order.

    Args:
        actual: Axis rendered from mixed nullable storage.
        expected: Axis rendered from the ordinary-float reference.
    """
    actual_rectangles = [patch for patch in actual.patches if isinstance(patch, Rectangle)]
    expected_rectangles = [patch for patch in expected.patches if isinstance(patch, Rectangle)]
    assert len(actual_rectangles) == len(actual.patches)
    assert len(expected_rectangles) == len(expected.patches)
    actual_geometry = np.asarray(
        [
            (patch.get_x(), patch.get_y(), patch.get_width(), patch.get_height())
            for patch in actual_rectangles
        ],
        dtype=float,
    )
    expected_geometry = np.asarray(
        [
            (patch.get_x(), patch.get_y(), patch.get_width(), patch.get_height())
            for patch in expected_rectangles
        ],
        dtype=float,
    )
    np.testing.assert_allclose(
        actual_geometry,
        expected_geometry,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )


# =============================================================================
# Nullable renderer-boundary regressions
# =============================================================================


def test_plot_stack_area_renders_finite_nullable_float64_panel() -> None:
    """Render finite nullable values instead of passing an object array to Matplotlib."""
    values = _ordinary_mixed_panel().loc[:, [_FINITE, _NULLABLE_FINITE]]
    assert isinstance(values, pd.DataFrame)
    values = values.astype(pd.Float64Dtype())
    figure, axis = _render_stack(values, use_bar_plot=False)
    try:
        assert len(axis.collections) == 2
        assert _legend_text(axis) == _EXPECTED_LEGEND[:2]
    finally:
        plt.close(figure)


@pytest.mark.parametrize("use_bar_plot", (False, True), ids=("area", "bar"))
def test_plot_stack_matches_ordinary_float_for_mixed_nullable_boundaries(
    use_bar_plot: bool,
) -> None:
    """Preserve complete stacking geometry across every material nullable column state.

    Args:
        use_bar_plot: Exercise stacked-area normalization and the unchanged stacked-bar control.
    """
    actual_values = _mixed_nullable_panel()
    expected_values = _ordinary_mixed_panel()
    actual_figure, actual_axis = _render_stack(actual_values, use_bar_plot=use_bar_plot)
    expected_figure, expected_axis = _render_stack(expected_values, use_bar_plot=use_bar_plot)

    try:
        if use_bar_plot:
            _assert_bar_geometry_equal(actual_axis, expected_axis)
        else:
            _assert_area_paths_equal(actual_axis, expected_axis)
        assert _legend_text(actual_axis) == _EXPECTED_LEGEND
        assert _legend_text(actual_axis) == _legend_text(expected_axis)
    finally:
        plt.close(actual_figure)
        plt.close(expected_figure)

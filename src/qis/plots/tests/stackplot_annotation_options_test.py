"""Regression coverage for stack-plot annotations and caller-owned colors.

Stacked areas and bars expose different Matplotlib artist types, but both public renderers support
the same mean and cumulative annotation options. Equivalent ordinary and nullable missing data
must use the same observed-sample means, while all-missing columns remain undefined. Rendering a
total line must not modify the caller's palette.

One mixed panel covers finite, partially missing, zero, signed, and all-missing columns. The tests
cross both renderers, both annotation modes, and both missing-data representations; inspect exact
annotation and legend colors; force a canvas draw under warnings-as-errors; verify caller
ownership; and close every figure.
"""

import warnings
from typing import Protocol, cast

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.container import BarContainer
from matplotlib.figure import Figure

# qis
import qis.plots.stackplot as stackplot_module


class _StackplotModuleProtocol(Protocol):
    """Typed test-side interface for stack-plot options."""

    def plot_stack(
        self,
        df: pd.DataFrame,
        *,
        use_bar_plot: bool,
        add_mean_levels: bool,
        add_cum_levels: bool,
        add_total_line: bool,
        colors: list[str],
        x_date_freq: None,
        var_format: str,
        skip_y_axis: bool,
        ax: Axes,
    ) -> Figure | None:
        """Render a stack plot with the focused public options.

        Args:
            df: Numeric panel to stack.
            use_bar_plot: Use bars instead of stacked areas when true.
            add_mean_levels: Annotate each observed-sample column mean.
            add_cum_levels: Annotate cumulative observed-sample column means.
            add_total_line: Draw the per-row total as a black line.
            colors: One caller-owned color per input column.
            x_date_freq: Disabled date-axis relabeling for deterministic inspection.
            var_format: Explicit annotation format.
            skip_y_axis: Whether to hide the numerical y-axis labels.
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
_COLORS: tuple[str, ...] = (
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
)
_ALL_MISSING_FIRST_CUMULATIVE_LABELS = (
    f"{_ALL_MISSING}=nan",
    "Finite=nan",
    "Missing=nan",
    "Zero=nan",
    "Signed=nan",
    "Total",
)
_CUMULATIVE_LABELS = (
    "Finite=0.200",
    "Missing=0.500",
    "Zero=0.500",
    "Signed=0.400",
    f"{_ALL_MISSING}=nan",
    "Total",
)
_MEAN_LABELS = (
    "Finite=0.200",
    "Missing=0.300",
    "Zero=0.000",
    "Signed=-0.100",
    f"{_ALL_MISSING}=nan",
    "Avg",
)


def _ordinary_mixed_panel() -> pd.DataFrame:
    """Return literal values spanning every material annotation reduction state."""
    return pd.DataFrame(
        {
            "Finite": (0.1, 0.2, 0.3),
            "Missing": (0.2, np.nan, 0.4),
            "Zero": (0.0, 0.0, 0.0),
            "Signed": (-0.4, -0.1, 0.2),
            _ALL_MISSING: (np.nan, np.nan, np.nan),
        },
        index=pd.date_range("2024-01-31", periods=3, freq="ME"),
        dtype=float,
    )


def _mixed_panel(*, nullable: bool, all_missing_first: bool = False) -> pd.DataFrame:
    """Return equivalent ordinary or nullable storage for the mixed panel.

    Args:
        nullable: Convert every column to pandas nullable floating storage when true.
        all_missing_first: Put the undefined component before finite cumulative state when true.

    Returns:
        Mixed panel in the selected missing-data representation.
    """
    values = _ordinary_mixed_panel()
    if all_missing_first:
        values = values.loc[:, [_ALL_MISSING, "Finite", "Missing", "Zero", "Signed"]]
    return values.astype(pd.Float64Dtype()) if nullable else values


def _annotation_text(axis: Axes) -> tuple[str, ...]:
    """Return exact annotation text in column order, including the mode label."""
    return tuple(text.get_text() for text in axis.texts)


def _display_colors(axis: Axes) -> tuple[str, ...]:
    """Return the rendered per-column annotation colors as normalized hex values."""
    return tuple(mcolors.to_hex(text.get_color()) for text in axis.texts[:-1])


def _legend_colors(axis: Axes) -> tuple[str, ...]:
    """Return the rendered per-column legend colors as normalized hex values."""
    legend = axis.get_legend()
    assert legend is not None
    return tuple(mcolors.to_hex(line.get_color()) for line in legend.get_lines())


def _assert_artist_colors(axis: Axes, *, use_bar_plot: bool) -> None:
    """Assert actual stack artist colors for the selected renderer.

    Args:
        axis: Rendered stack-plot axis.
        use_bar_plot: Read bar containers rather than area collections when true.
    """
    handles = axis.get_legend_handles_labels()[0]
    if use_bar_plot:
        bar_handles = [handle for handle in handles if isinstance(handle, BarContainer)]
        assert len(bar_handles) == len(handles)
        actual_colors = [
            mcolors.to_rgba(handle.patches[0].get_facecolor()) for handle in bar_handles
        ]
    else:
        area_handles = [handle for handle in handles if isinstance(handle, PolyCollection)]
        assert len(area_handles) == len(handles)
        actual_colors = [
            cast(tuple[float, float, float, float], handle.get_facecolor()[0])
            for handle in area_handles
        ]

    expected_colors = [mcolors.to_rgba(color) for color in _COLORS]
    np.testing.assert_allclose(
        np.asarray(actual_colors, dtype=float),
        np.asarray(expected_colors, dtype=float),
        rtol=0.0,
        atol=0.0,
    )


def _render_annotations(
    *,
    nullable: bool,
    use_bar_plot: bool,
    cumulative: bool,
    add_total_line: bool,
    all_missing_first: bool = False,
) -> tuple[Figure, Axes]:
    """Render one annotation case and assert caller ownership.

    Args:
        nullable: Use pandas nullable floating storage when true.
        use_bar_plot: Use the stacked-bar renderer when true.
        cumulative: Select cumulative rather than individual mean annotations.
        add_total_line: Draw the per-row total line when true.
        all_missing_first: Put the undefined component before finite cumulative state when true.

    Returns:
        Open figure and axis for exact artist inspection. The caller must close the figure.
    """
    values = _mixed_panel(nullable=nullable, all_missing_first=all_missing_first)
    original_values = values.copy(deep=True)
    colors = list(_COLORS)
    original_colors = list(colors)
    figure, axis = plt.subplots()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = _STACKPLOT.plot_stack(
                values,
                use_bar_plot=use_bar_plot,
                add_mean_levels=not cumulative,
                add_cum_levels=cumulative,
                add_total_line=add_total_line,
                colors=colors,
                x_date_freq=None,
                var_format="{:.3f}",
                skip_y_axis=False,
                ax=axis,
            )
            figure.canvas.draw()
        assert result is None
        pd.testing.assert_frame_equal(values, original_values)
        assert colors == original_colors
    except Exception:
        plt.close(figure)
        raise
    return figure, axis


# =============================================================================
# Renderer, reduction, and option-interaction regressions
# =============================================================================


@pytest.mark.parametrize("nullable", (False, True), ids=("ordinary", "nullable"))
@pytest.mark.parametrize("use_bar_plot", (False, True), ids=("area", "bar"))
@pytest.mark.parametrize("cumulative", (False, True), ids=("mean", "cumulative"))
def test_plot_stack_annotations_match_observed_means_across_supported_paths(
    nullable: bool,
    use_bar_plot: bool,
    cumulative: bool,
) -> None:
    """Render identical observed-sample labels for every renderer and missing representation.

    Args:
        nullable: Use pandas nullable floating storage when true.
        use_bar_plot: Use the stacked-bar renderer when true.
        cumulative: Select cumulative rather than individual mean annotations.
    """
    figure, axis = _render_annotations(
        nullable=nullable,
        use_bar_plot=use_bar_plot,
        cumulative=cumulative,
        add_total_line=False,
    )
    try:
        expected_labels = _CUMULATIVE_LABELS if cumulative else _MEAN_LABELS
        expected_colors = tuple(mcolors.to_hex(color) for color in _COLORS)
        assert _annotation_text(axis) == expected_labels
        assert _display_colors(axis) == expected_colors
        _assert_artist_colors(axis, use_bar_plot=use_bar_plot)
        assert _legend_colors(axis) == expected_colors
    finally:
        plt.close(figure)


@pytest.mark.parametrize("nullable", (False, True), ids=("ordinary", "nullable"))
@pytest.mark.parametrize("use_bar_plot", (False, True), ids=("area", "bar"))
def test_plot_stack_cumulative_annotations_keep_undefined_component_order(
    nullable: bool,
    use_bar_plot: bool,
) -> None:
    """Propagate an early undefined component without representation-specific failures.

    Args:
        nullable: Use pandas nullable floating storage when true.
        use_bar_plot: Use the stacked-bar renderer when true.
    """
    figure, axis = _render_annotations(
        nullable=nullable,
        use_bar_plot=use_bar_plot,
        cumulative=True,
        add_total_line=False,
        all_missing_first=True,
    )
    try:
        expected_colors = tuple(mcolors.to_hex(color) for color in _COLORS)
        assert _annotation_text(axis) == _ALL_MISSING_FIRST_CUMULATIVE_LABELS
        assert _display_colors(axis) == expected_colors
        _assert_artist_colors(axis, use_bar_plot=use_bar_plot)
        assert _legend_colors(axis) == expected_colors
    finally:
        plt.close(figure)


@pytest.mark.parametrize("use_bar_plot", (False, True), ids=("area", "bar"))
def test_plot_stack_total_line_preserves_caller_colors_with_annotations(
    use_bar_plot: bool,
) -> None:
    """Keep the caller's palette unchanged when total and annotation options interact.

    Args:
        use_bar_plot: Use the stacked-bar renderer when true.
    """
    figure, axis = _render_annotations(
        nullable=False,
        use_bar_plot=use_bar_plot,
        cumulative=False,
        add_total_line=True,
    )
    try:
        expected_colors = tuple(mcolors.to_hex(color) for color in _COLORS)
        assert _annotation_text(axis) == _MEAN_LABELS
        assert _display_colors(axis) == expected_colors
        _assert_artist_colors(axis, use_bar_plot=use_bar_plot)
        assert _legend_colors(axis) == expected_colors
        total_lines = [line for line in axis.lines if line.get_linestyle() == "-"]
        assert len(total_lines) == 1
        assert mcolors.to_hex(total_lines[0].get_color()) == "#000000"
    finally:
        plt.close(figure)

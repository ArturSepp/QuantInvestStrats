"""Regression tests for infinity validation before descriptive plot rendering.

QIS descriptive tables treat positive and negative infinity as invalid observed data. Public
plots that display those tables must apply the same contract before a plotting dependency can
emit warnings, add partial artists to a caller-owned axis, or allocate an internal figure that
the caller cannot close after an exception.

The primary fixture combines finite, positive-infinity, negative-infinity, both-sign, and
all-missing columns in one ordered panel. Ordinary ``float64`` and nullable ``Float64`` variants
exercise the same boundary. Independent controls contain only finite and missing observations,
which remain valid and must still render through the complete Matplotlib canvas.
"""

from typing import Literal, Protocol, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import qis.plots.histogram as histogram_module
import qis.plots.qqplot as qqplot_module
import qis.plots.time_series as time_series_module
from qis.perfstats.desc_table import DescTableType
from qis.plots.histogram import PdfType


pytestmark = pytest.mark.filterwarnings("error")


class _HistogramModuleProtocol(Protocol):
    """Typed test-side interface for the histogram entry point."""

    def plot_histogram(
        self,
        df: pd.DataFrame | pd.Series,
        *,
        pdf_type: PdfType,
        desc_table_type: DescTableType | None,
        ax: Axes | None = None,
    ) -> Figure | None:
        """Draw a histogram with optional descriptive statistics."""
        raise NotImplementedError


class _QqplotModuleProtocol(Protocol):
    """Typed test-side interface for the Q-Q entry point."""

    def plot_qq(
        self,
        df: pd.DataFrame | pd.Series,
        *,
        desc_table_type: DescTableType,
        ax: Axes | None = None,
    ) -> Figure | None:
        """Draw a normal Q-Q plot with optional descriptive statistics."""
        raise NotImplementedError


class _TimeSeriesModuleProtocol(Protocol):
    """Typed test-side interface for the time-series entry point."""

    def plot_time_series(
        self,
        df: pd.DataFrame | pd.Series,
        *,
        desc_table_type: DescTableType,
        x_date_freq: str | None,
        ax: Axes | None = None,
    ) -> Figure | None:
        """Draw a time series with optional descriptive statistics."""
        raise NotImplementedError


_HISTOGRAM = cast(_HistogramModuleProtocol, histogram_module)
_QQPLOT = cast(_QqplotModuleProtocol, qqplot_module)
_TIME_SERIES = cast(_TimeSeriesModuleProtocol, time_series_module)


# =============================================================================
# Shared deterministic fixtures and plot dispatch
# =============================================================================

_DATES = pd.date_range("2024-01-31", periods=24, freq="ME", name="Date")
_FINITE_VALUES = tuple(np.linspace(-0.20, 0.30, len(_DATES)))

_PlotCase = Literal[
    "qq",
    "time_series",
    "histogram_kde",
    "histogram_kde_norm",
    "histogram_histogram",
    "histogram_truncated",
    "histogram_kde_with_histogram",
]

_PLOT_CASES: tuple[_PlotCase, ...] = (
    "qq",
    "time_series",
    "histogram_kde",
    "histogram_kde_norm",
    "histogram_histogram",
    "histogram_truncated",
    "histogram_kde_with_histogram",
)

_HISTOGRAM_TYPES: dict[_PlotCase, PdfType] = {
    "histogram_kde": PdfType.KDE,
    "histogram_kde_norm": PdfType.KDE_NORM,
    "histogram_histogram": PdfType.HISTOGRAM,
    "histogram_truncated": PdfType.TRUNCETED_PDF,
    "histogram_kde_with_histogram": PdfType.KDE_WITH_HISTOGRAM,
}


def _call_plot(
    plot_case: _PlotCase,
    data: pd.DataFrame | pd.Series,
    *,
    ax: Axes | None,
) -> Figure | None:
    """Call one statistics-enabled plotting path.

    Args:
        plot_case: Plotting path, including the selected histogram PDF mode.
        data: Named Series or ordered DataFrame supplied by the caller.
        ax: Optional caller-owned Matplotlib axis.

    Returns:
        Internally created figure, or ``None`` when ``ax`` is supplied.
    """
    if plot_case == "qq":
        return _QQPLOT.plot_qq(data, desc_table_type=DescTableType.SHORT, ax=ax)
    if plot_case == "time_series":
        return _TIME_SERIES.plot_time_series(
            data,
            desc_table_type=DescTableType.SHORT,
            x_date_freq=None,
            ax=ax,
        )
    return _HISTOGRAM.plot_histogram(
        data,
        pdf_type=_HISTOGRAM_TYPES[plot_case],
        desc_table_type=DescTableType.SHORT,
        ax=ax,
    )


def _mixed_infinite_panel(*, nullable: bool) -> pd.DataFrame:
    """Create every materially different infinity state in one ordered panel.

    Args:
        nullable: Store columns as pandas nullable ``Float64`` when true.

    Returns:
        Finite, signed-infinity, both-sign, and all-missing columns.
    """
    finite = np.asarray(_FINITE_VALUES, dtype=float)
    positive = finite.copy()
    negative = finite.copy()
    both = finite.copy()
    positive[4] = np.inf
    negative[6] = -np.inf
    both[8] = -np.inf
    both[15] = np.inf
    panel = pd.DataFrame(
        {
            "Finite": finite,
            "Positive infinity": positive,
            "Negative infinity": negative,
            "Both infinities": both,
            "All missing": np.full(len(_DATES), np.nan),
        },
        index=_DATES,
    )
    if nullable:
        panel = panel.astype(pd.Float64Dtype())
    return panel


def _named_infinite_series(*, nullable: bool) -> pd.Series:
    """Create a named Series containing one positive infinity.

    Args:
        nullable: Store values as pandas nullable ``Float64`` when true.

    Returns:
        Named 24-date Series with finite neighbors around infinity.
    """
    values = np.asarray(_FINITE_VALUES, dtype=float)
    values[9] = np.inf
    if nullable:
        return pd.Series(
            pd.array(values, dtype=pd.Float64Dtype()),
            index=_DATES,
            name="Sample",
        )
    return pd.Series(values, index=_DATES, name="Sample")


def _valid_series(*, nullable: bool, with_missing: bool) -> pd.Series:
    """Create a finite control with an optional missing observation.

    Args:
        nullable: Store values as pandas nullable ``Float64`` when true.
        with_missing: Replace one interior observation with missing data.

    Returns:
        Named finite or partially missing Series.
    """
    values = np.asarray(_FINITE_VALUES, dtype=float)
    if with_missing:
        values[9] = np.nan
    if nullable:
        return pd.Series(
            pd.array(values, dtype=pd.Float64Dtype()),
            index=_DATES,
            name="Sample",
        )
    return pd.Series(values, index=_DATES, name="Sample")


def _axes_state(
    ax: Axes,
) -> tuple[
    int,
    int,
    int,
    int,
    int,
    int,
    tuple[float, float],
    tuple[float, float],
]:
    """Capture public artist counts and limits for mutation checks.

    Args:
        ax: Caller-owned axis to inspect.

    Returns:
        Counts for major artist collections followed by x and y limits.
    """
    return (
        len(ax.lines),
        len(ax.collections),
        len(ax.patches),
        len(ax.texts),
        len(ax.tables),
        len(ax.containers),
        ax.get_xlim(),
        ax.get_ylim(),
    )


# =============================================================================
# Invalid descriptive inputs fail before rendering
# =============================================================================


@pytest.mark.parametrize("nullable", (False, True))
@pytest.mark.parametrize("plot_case", _PLOT_CASES)
def test_descriptive_plots_reject_mixed_infinities_before_axis_mutation(
    plot_case: _PlotCase,
    nullable: bool,
) -> None:
    """Apply one exact infinity error before warnings or partial caller-owned artists.

    Args:
        plot_case: Statistics-enabled public plotting path.
        nullable: Whether the mixed panel uses nullable floating storage.
    """
    data = _mixed_infinite_panel(nullable=nullable)
    original = data.copy(deep=True)
    fig, ax = plt.subplots()
    ax.plot((0.0, 1.0), (0.0, 1.0), label="Existing artist")
    before = _axes_state(ax)

    try:
        with pytest.raises(ValueError, match="^data contains infinite values$"):
            _call_plot(plot_case, data, ax=ax)
        assert _axes_state(ax) == before
        pd.testing.assert_frame_equal(data, original)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("nullable", (False, True))
@pytest.mark.parametrize("plot_case", ("qq", "time_series", "histogram_histogram"))
def test_descriptive_plots_reject_series_infinity_before_axis_mutation(
    plot_case: _PlotCase,
    nullable: bool,
) -> None:
    """Apply the DataFrame infinity contract consistently to a named Series.

    Args:
        plot_case: Representative public plotting path.
        nullable: Whether the Series uses nullable floating storage.
    """
    data = _named_infinite_series(nullable=nullable)
    original = data.copy(deep=True)
    fig, ax = plt.subplots()
    before = _axes_state(ax)

    try:
        with pytest.raises(ValueError, match="^data contains infinite values$"):
            _call_plot(plot_case, data, ax=ax)
        assert _axes_state(ax) == before
        pd.testing.assert_series_equal(data, original)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("nullable", (False, True))
@pytest.mark.parametrize("plot_case", ("qq", "time_series", "histogram_histogram"))
def test_descriptive_plots_reject_infinity_before_allocating_a_figure(
    plot_case: _PlotCase,
    nullable: bool,
) -> None:
    """Avoid leaking an internally created figure when descriptive validation fails.

    Args:
        plot_case: Representative public plotting path.
        nullable: Whether the Series uses nullable floating storage.
    """
    data = _named_infinite_series(nullable=nullable)
    original = data.copy(deep=True)
    existing_figures = set(plt.get_fignums())

    try:
        with pytest.raises(ValueError, match="^data contains infinite values$"):
            _call_plot(plot_case, data, ax=None)
        assert set(plt.get_fignums()) == existing_figures
        pd.testing.assert_series_equal(data, original)
    finally:
        for figure_number in set(plt.get_fignums()) - existing_figures:
            plt.close(figure_number)


# =============================================================================
# Finite and missing descriptive inputs remain renderable
# =============================================================================


@pytest.mark.parametrize("nullable", (False, True))
@pytest.mark.parametrize("with_missing", (False, True))
@pytest.mark.parametrize("plot_case", ("qq", "time_series", "histogram_histogram"))
def test_descriptive_plots_preserve_valid_rendering_and_caller_ownership(
    plot_case: _PlotCase,
    with_missing: bool,
    nullable: bool,
) -> None:
    """Render valid controls through the canvas without changing caller data.

    Args:
        plot_case: Representative public plotting path.
        with_missing: Whether the valid control includes an interior missing value.
        nullable: Whether the Series uses nullable floating storage.
    """
    data = _valid_series(nullable=nullable, with_missing=with_missing)
    original = data.copy(deep=True)
    fig, ax = plt.subplots()

    try:
        result = _call_plot(plot_case, data, ax=ax)
        fig.canvas.draw()

        assert result is None
        legend = ax.get_legend()
        assert legend is not None
        assert any("Sample" in text.get_text() for text in legend.get_texts())
        pd.testing.assert_series_equal(data, original)
    finally:
        plt.close(fig)

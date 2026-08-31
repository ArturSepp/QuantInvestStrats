"""Regression coverage for unformatted statistic legends.

Public line plots accept ``None`` for their numerical format so Matplotlib can retain its native
tick formatter. Statistic legends share that argument, so they should render values with native
scalar text instead of assuming that every caller supplied a format string. Explicit formats and
the specialized formats for t-statistics, moments, scores, and data-quality ratios remain separate
display contracts.

The exact reference sample ``[0.0, 2.0, 4.0]`` has mean and sample standard deviation ``2.0``,
median ``2.0``, signed mean t-statistic ``sqrt(3)``, a zero share of ``1 / 3``, and a final-value
percentile rank of 100%. Representative modes assert those independently calculated public labels.
Every ``LegendStats`` member also compares ``None`` with Python's native ``'{}'`` format on the
same finite sample. Public Series/DataFrame plots, ordinary and nullable storage, mixed missing
histories, warning behavior, figure cleanup, label order, and caller ownership are covered.
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
import qis as qis_module
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
    ) -> list[str]:
        """Return formatted legend lines for the selected statistics.

        Args:
            data: Series or DataFrame summarized in the legend.
            legend_stats: Statistics displayed beside each label.
            var_format: Explicit Python format string, or no explicit format.

        Returns:
            One legend line per input column.
        """
        raise NotImplementedError


class _QisPublicProtocol(Protocol):
    """Typed test-side interface for the two public plotting entry points."""

    def plot_time_series(
        self,
        df: pd.DataFrame | pd.Series,
        *,
        x_date_freq: None,
        var_format: str | None,
        ax: Axes,
    ) -> Figure | None:
        """Plot a dated Series or DataFrame.

        Args:
            df: Values plotted by date.
            x_date_freq: Disabled date-axis formatting for the focused test.
            var_format: Explicit Python format string, or no explicit format.
            ax: Matplotlib axis receiving the plot.

        Returns:
            The created figure, or None when the caller supplies ``ax``.
        """
        raise NotImplementedError

    def plot_line(
        self,
        df: pd.DataFrame | pd.Series,
        *,
        legend_stats: LegendStats,
        yvar_format: str | None,
        ax: Axes,
    ) -> Figure | None:
        """Plot a Series or DataFrame with a statistic legend.

        Args:
            df: Values plotted against their index.
            legend_stats: Statistics displayed beside each label.
            yvar_format: Explicit Python format string, or no explicit format.
            ax: Matplotlib axis receiving the plot.

        Returns:
            The created figure, or None when the caller supplies ``ax``.
        """
        raise NotImplementedError


_PLOT_UTILS = cast(_PlotUtilsModuleProtocol, plot_utils_module)
_QIS = cast(_QisPublicProtocol, qis_module)


# =============================================================================
# Shared deterministic fixtures and independently calculated expectations
# =============================================================================

_DATES = pd.date_range("2024-01-31", periods=3, freq="ME")
_ASSET = "Asset"
_COMPLETE = "Complete"
_RAGGED = "Ragged"
_ALL_MISSING = "All Missing"

_COMPLETE_VALUES = (0.0, 2.0, 4.0)
_EXPECTED_MIXED_AVG_LAST = (
    "Complete: avg=2.0, last=4.0",
    "Ragged: avg=3.0, last=4.0",
    "All Missing: avg=nan, last=nan",
)
_REPRESENTATIVE_EXPECTATIONS: tuple[tuple[LegendStats, str], ...] = (
    (LegendStats.NONE, "Asset"),
    (LegendStats.LAST, "Asset: last=4.0"),
    (LegendStats.AVG, "Asset: avg=2.0"),
    (LegendStats.AVG_STD_LAST, "Asset: avg=2.0, std=2.0, last=4.0"),
    (
        LegendStats.AVG_STD_SKEW_KURT,
        "Asset: avg=2.0, std=2.0, skew=0.00, kurtosis=-1.50",
    ),
    (LegendStats.FIRST_AVG_LAST_SHORT, "Asset: [0.0, 2.0, 4.0]"),
    (LegendStats.AVG_LAST_SCORE, "Asset: avg=2.0, last=4.0, last score=100%"),
    (
        LegendStats.AVG_STD_MISSING_ZERO,
        "Asset: avg=2.0, std=2.0, missing%=0.00%, zeros%=33.33%",
    ),
    (LegendStats.TSTAT, "Asset: t-stat=1.73"),
)


def _complete_series(*, nullable: bool) -> pd.Series:
    """Create the exact three-observation reference sample.

    Args:
        nullable: Whether to use pandas nullable ``Float64`` storage.

    Returns:
        Named Series containing ``[0.0, 2.0, 4.0]``.
    """
    dtype = pd.Float64Dtype() if nullable else float
    return pd.Series(_COMPLETE_VALUES, index=_DATES, name=_ASSET, dtype=dtype)


def _mixed_panel(*, nullable: bool) -> pd.DataFrame:
    """Create complete, ragged, and all-missing histories in one panel.

    Args:
        nullable: Whether to use pandas nullable ``Float64`` storage.

    Returns:
        Three-column panel with materially different missing-data states.
    """
    missing = pd.NA if nullable else np.nan
    dtype = pd.Float64Dtype() if nullable else float
    return pd.DataFrame(
        {
            _COMPLETE: _COMPLETE_VALUES,
            _RAGGED: (missing, 2.0, 4.0),
            _ALL_MISSING: (missing, missing, missing),
        },
        index=_DATES,
        dtype=dtype,
    )


def _legend_lines(
    data: pd.DataFrame | pd.Series,
    legend_stats: LegendStats,
    var_format: str | None,
) -> tuple[str, ...]:
    """Build legend lines while treating every warning as a failure.

    Args:
        data: Series or DataFrame summarized in the legend.
        legend_stats: Statistics displayed beside each label.
        var_format: Explicit Python format string, or no explicit format.

    Returns:
        Immutable legend text in input-column order.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return tuple(
            _PLOT_UTILS.get_legend_lines(
                data,
                legend_stats=legend_stats,
                var_format=var_format,
            )
        )


def _axis_legend_text(ax: Axes) -> tuple[str, ...]:
    """Read the public legend text from a plotted axis.

    Args:
        ax: Axis containing the completed plot.

    Returns:
        Legend labels in display order.
    """
    legend = ax.get_legend()
    assert legend is not None
    return tuple(text.get_text() for text in legend.get_texts())


# =============================================================================
# Shared-helper contract
# =============================================================================


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
@pytest.mark.parametrize(
    "legend_stats",
    tuple(LegendStats),
    ids=tuple(mode.name.lower() for mode in LegendStats),
)
def test_none_format_matches_native_text_for_every_legend_mode(
    legend_stats: LegendStats,
    nullable: bool,
) -> None:
    """Make ``None`` equivalent to Python's native scalar format in every mode.

    Args:
        legend_stats: Public legend mode under test.
        nullable: Whether the Series uses nullable ``Float64`` storage.
    """
    data = _complete_series(nullable=nullable)
    original = data.copy()

    actual = _legend_lines(data, legend_stats, None)
    expected = _legend_lines(data, legend_stats, "{}")

    assert actual == expected
    pd.testing.assert_series_equal(data, original)


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
@pytest.mark.parametrize(
    ("legend_stats", "expected"),
    _REPRESENTATIVE_EXPECTATIONS,
    ids=tuple(mode.name.lower() for mode, _ in _REPRESENTATIVE_EXPECTATIONS),
)
def test_none_format_returns_independently_calculated_representative_text(
    legend_stats: LegendStats,
    expected: str,
    nullable: bool,
) -> None:
    """Preserve exact native and specialized text across formatting families.

    Args:
        legend_stats: Representative public legend mode under test.
        expected: Independently calculated legend text for the reference sample.
        nullable: Whether the Series uses nullable ``Float64`` storage.
    """
    data = _complete_series(nullable=nullable)

    assert _legend_lines(data, legend_stats, None) == (expected,)


# =============================================================================
# Public plotting entry points
# =============================================================================


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
def test_plot_time_series_uses_native_text_for_default_statistic_legend(nullable: bool) -> None:
    """Render the default ``AVG_LAST`` legend when no number format is requested.

    Args:
        nullable: Whether the mixed panel uses nullable ``Float64`` storage.
    """
    data = _mixed_panel(nullable=nullable)
    original = data.copy()
    fig, ax = plt.subplots()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _QIS.plot_time_series(data, x_date_freq=None, var_format=None, ax=ax)

        assert _axis_legend_text(ax) == _EXPECTED_MIXED_AVG_LAST
        pd.testing.assert_frame_equal(data, original)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
def test_plot_line_uses_native_text_for_statistic_legend(nullable: bool) -> None:
    """Render a statistic legend through the second optional-format public plot.

    Args:
        nullable: Whether the mixed panel uses nullable ``Float64`` storage.
    """
    data = _mixed_panel(nullable=nullable)
    original = data.copy()
    fig, ax = plt.subplots()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _QIS.plot_line(
                data,
                legend_stats=LegendStats.AVG_LAST,
                yvar_format=None,
                ax=ax,
            )

        assert _axis_legend_text(ax) == _EXPECTED_MIXED_AVG_LAST
        pd.testing.assert_frame_equal(data, original)
    finally:
        plt.close(fig)

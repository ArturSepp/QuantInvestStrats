"""Regression coverage for public legend t-statistic conventions and boundaries.

``LegendStats.TSTAT`` and ``LegendStats.AVG_STD_TSTAT`` promise a t-statistic of the sample mean,
so both public plotting paths must use the signed mean divided by its standard error. The statistic
depends on each column's non-missing sample count, while zero-volatility, one-observation, and
all-missing histories remain undefined without warnings. One deliberately ordered panel combines
every relevant state so vectorized or shared legend logic cannot let one column affect another.
Matched ordinary and nullable floating inputs, named Series/DataFrame consistency, exact legend
text, and caller ownership complete the public contract.
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
from qis.plots.utils import LegendStats


class _TimeSeriesModuleProtocol(Protocol):
    """Typed test-side interface for the public plotting function exercised below."""

    def plot_time_series(
        self,
        df: pd.DataFrame | pd.Series,
        *,
        x_date_freq: None,
        legend_stats: LegendStats,
        var_format: str,
        ax: Axes,
    ) -> Figure | None:
        """Plot a time series with the selected legend statistics.

        Args:
            df: Series or DataFrame supplied to the public plot.
            x_date_freq: Disabled date-axis formatting for this focused legend test.
            legend_stats: Summary-statistic mode displayed in the legend.
            var_format: Display format for mean and standard-deviation values.
            ax: Matplotlib axis receiving the plot.

        Returns:
            The created figure, or None when the caller supplies ``ax``.
        """
        raise NotImplementedError


_TIME_SERIES_MODULE = cast(_TimeSeriesModuleProtocol, time_series_module)


# =============================================================================
# Shared deterministic fixtures and independent expectations
# =============================================================================

_DATES = pd.date_range("2024-01-31", periods=6, freq="ME")

_POSITIVE_VARIABLE = "Positive Variable"
_RAGGED_POSITIVE_VARIABLE = "Ragged Positive Variable"
_NEGATIVE_VARIABLE = "Negative Variable"
_ZERO_MEAN_VARIABLE = "Zero Mean Variable"
_POSITIVE_CONSTANT = "Positive Constant"
_ZERO_CONSTANT = "Zero Constant"
_NEGATIVE_CONSTANT = "Negative Constant"
_RAGGED_POSITIVE_CONSTANT = "Ragged Positive Constant"
_ONE_OBSERVATION = "One Observation"
_ALL_MISSING = "All Missing"

_TSTAT_LINES = (
    "Positive Variable: t-stat=3.87",
    "Ragged Positive Variable: t-stat=3.46",
    "Negative Variable: t-stat=-3.87",
    "Zero Mean Variable: t-stat=0.00",
    "Positive Constant: t-stat=nan",
    "Zero Constant: t-stat=nan",
    "Negative Constant: t-stat=nan",
    "Ragged Positive Constant: t-stat=nan",
    "One Observation: t-stat=nan",
    "All Missing: t-stat=nan",
)

_AVG_STD_TSTAT_LINES = (
    "Positive Variable: avg=2.50, std=1.29, t-stat=3.87",
    "Ragged Positive Variable: avg=2.00, std=1.00, t-stat=3.46",
    "Negative Variable: avg=-2.50, std=1.29, t-stat=-3.87",
    "Zero Mean Variable: avg=0.00, std=1.15, t-stat=0.00",
    "Positive Constant: avg=2.00, std=0.00, t-stat=nan",
    "Zero Constant: avg=0.00, std=0.00, t-stat=nan",
    "Negative Constant: avg=-2.00, std=0.00, t-stat=nan",
    "Ragged Positive Constant: avg=2.00, std=0.00, t-stat=nan",
    "One Observation: avg=1.00, std=nan, t-stat=nan",
    "All Missing: avg=nan, std=nan, t-stat=nan",
)


def _mixed_values(*, nullable: bool) -> pd.DataFrame:
    """Create all t-statistic column states in deliberate reporting order.

    The complete positive sample has mean 2.5, sample standard deviation ``sqrt(5 / 3)``, and
    t-statistic ``sqrt(15)``. The three-point ragged sample has mean 2, sample standard deviation
    1, and t-statistic ``2 * sqrt(3)``. Its negative mirror and the zero-mean variable establish
    signed behavior, while constant and undersized samples establish undefined boundaries.

    Args:
        nullable: Whether to store values as pandas nullable ``Float64``/``pd.NA``.

    Returns:
        Six-date panel containing every materially different legend state.
    """
    values = pd.DataFrame(
        {
            _POSITIVE_VARIABLE: (1.0, 2.0, 3.0, 4.0, np.nan, np.nan),
            _RAGGED_POSITIVE_VARIABLE: (1.0, np.nan, 2.0, np.nan, 3.0, np.nan),
            _NEGATIVE_VARIABLE: (-1.0, -2.0, -3.0, -4.0, np.nan, np.nan),
            _ZERO_MEAN_VARIABLE: (-1.0, 1.0, -1.0, 1.0, np.nan, np.nan),
            _POSITIVE_CONSTANT: (2.0, 2.0, 2.0, 2.0, np.nan, np.nan),
            _ZERO_CONSTANT: (0.0, 0.0, 0.0, 0.0, np.nan, np.nan),
            _NEGATIVE_CONSTANT: (-2.0, -2.0, -2.0, -2.0, np.nan, np.nan),
            _RAGGED_POSITIVE_CONSTANT: (2.0, np.nan, 2.0, np.nan, 2.0, np.nan),
            _ONE_OBSERVATION: (np.nan, np.nan, 1.0, np.nan, np.nan, np.nan),
            _ALL_MISSING: (np.nan,) * len(_DATES),
        },
        index=_DATES,
    )
    if nullable:
        return values.astype(pd.Float64Dtype())
    return values


def _legend_lines(data: pd.DataFrame | pd.Series, legend_stats: LegendStats) -> tuple[str, ...]:
    """Draw through the public plotting API and return its exact legend text.

    Args:
        data: Time series whose summary statistics are displayed.
        legend_stats: Public legend mode under test.

    Returns:
        Legend entries in input-column order.
    """
    figure, axis = plt.subplots()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _TIME_SERIES_MODULE.plot_time_series(
                data,
                x_date_freq=None,
                legend_stats=legend_stats,
                var_format="{:.2f}",
                ax=axis,
            )
        legend = axis.get_legend()
        assert legend is not None
        return tuple(text.get_text() for text in legend.get_texts())
    finally:
        plt.close(figure)


# =============================================================================
# Mixed-panel numerical and undefined-value contract
# =============================================================================


@pytest.mark.parametrize(
    ("legend_stats", "expected"),
    (
        (LegendStats.TSTAT, _TSTAT_LINES),
        (LegendStats.AVG_STD_TSTAT, _AVG_STD_TSTAT_LINES),
    ),
)
@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
def test_plot_time_series_applies_signed_sample_mean_tstats_without_warnings(
    legend_stats: LegendStats,
    expected: tuple[str, ...],
    nullable: bool,
) -> None:
    """Display exact signed statistics and undefined boundaries in one public call.

    Args:
        legend_stats: T-statistic-only or mean/standard-deviation/t-statistic legend mode.
        expected: Independently specified legend entries for that mode.
        nullable: Whether the input uses nullable ``Float64``/``pd.NA`` storage.
    """
    values = _mixed_values(nullable=nullable)
    original_values = values.copy(deep=True)

    actual = _legend_lines(values, legend_stats)

    assert actual == expected
    pd.testing.assert_frame_equal(values, original_values)


# =============================================================================
# Named Series/DataFrame consistency
# =============================================================================


@pytest.mark.parametrize(
    ("legend_stats", "expected"),
    (
        (LegendStats.TSTAT, (_TSTAT_LINES[0],)),
        (LegendStats.AVG_STD_TSTAT, (_AVG_STD_TSTAT_LINES[0],)),
    ),
)
@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
def test_plot_time_series_tstat_named_series_matches_dataframe(
    legend_stats: LegendStats,
    expected: tuple[str, ...],
    nullable: bool,
) -> None:
    """Return identical legend text for equivalent named Series and one-column frames.

    Args:
        legend_stats: T-statistic-only or mean/standard-deviation/t-statistic legend mode.
        expected: Independently specified one-entry legend text.
        nullable: Whether the input uses nullable ``Float64``/``pd.NA`` storage.
    """
    series = pd.Series(
        (1.0, 2.0, 3.0, 4.0, np.nan, np.nan),
        index=_DATES,
        name=_POSITIVE_VARIABLE,
    )
    if nullable:
        series = series.astype(pd.Float64Dtype())
    frame = series.to_frame()
    original_series = series.copy(deep=True)
    original_frame = frame.copy(deep=True)

    frame_result = _legend_lines(frame, legend_stats)
    series_result = _legend_lines(series, legend_stats)

    assert frame_result == expected
    assert series_result == expected
    assert series_result == frame_result
    pd.testing.assert_frame_equal(frame, original_frame)
    pd.testing.assert_series_equal(series, original_series)

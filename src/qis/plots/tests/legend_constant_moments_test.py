"""Regression coverage for stable statistic-legend moments at narrow spreads.

``LegendStats.AVG_STD_SKEW_KURT`` reports defined means and sample standard deviations alongside
standardized moments. A finite constant has its stated level and zero sample spread, but skewness
and excess kurtosis are undefined because their normalization divides by that zero spread. The
public legend should display those established values without invoking warning-producing SciPy
reducers for an ineligible sample.

One deliberately ordered panel combines positive, zero, negative, and ragged constants with a
finite varying neighbor and a symmetric two-level sample formed from ``1.0`` and its next
representable float. Translating that near-degenerate sample to zero cannot change its biased
skewness of zero or biased excess kurtosis of negative two. The ordinary neighbor ``[1, 2, 3, 4,
5]`` has mean three, sample standard deviation ``sqrt(5 / 2)``, biased skewness zero, and biased
excess kurtosis ``(6.8 / 2**2) - 3 = -1.3``. Ordinary and nullable storage, warnings-as-errors,
named Series/DataFrame consistency, exact labels and order, figure cleanup, and caller ownership
complete the public boundary contract.
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

_DATES = pd.date_range("2024-01-31", periods=24, freq="ME")

_NEGATIVE_CONSTANT = "Negative Constant"
_NEAR_DEGENERATE = "Near Degenerate"
_POSITIVE_CONSTANT = "Positive Constant"
_RAGGED_CONSTANT = "Ragged Constant"
_VARIABLE = "Variable"
_ZERO_BASED = "Zero Based"
_ZERO_CONSTANT = "Zero Constant"

_MOMENT_LINES = (
    "Positive Constant: avg=2.000, std=0.000, skew=nan, kurtosis=nan",
    "Zero Constant: avg=0.000, std=0.000, skew=nan, kurtosis=nan",
    "Negative Constant: avg=-2.000, std=0.000, skew=nan, kurtosis=nan",
    "Ragged Constant: avg=3.000, std=0.000, skew=nan, kurtosis=nan",
    "Near Degenerate: avg=1.000, std=0.000, skew=0.00, kurtosis=-2.00",
    "Zero Based: avg=0.000, std=0.000, skew=0.00, kurtosis=-2.00",
    "Variable: avg=3.000, std=1.581, skew=0.00, kurtosis=-1.30",
)


def _mixed_values(*, nullable: bool) -> pd.DataFrame:
    """Create constant and near-degenerate moment states with varying controls.

    The ragged constant confirms that missing observations are removed before exact variation is
    assessed. Positive, zero, and negative constants ensure the guard depends on spread rather
    than level. Offset and zero-based copies of one symmetric two-level sample establish
    translation stability, while the ordinary varying neighbor protects the established SciPy
    calculation.

    Args:
        nullable: Whether to store values as pandas nullable ``Float64``/``pd.NA``.

    Returns:
        Twenty-four-date panel in the same deliberate order as the expected legend entries.
    """
    near_upper = float(np.nextafter(1.0, np.inf))
    near_width = near_upper - 1.0
    values = pd.DataFrame(
        {
            _POSITIVE_CONSTANT: (2.0,) * 24,
            _ZERO_CONSTANT: (0.0,) * 24,
            _NEGATIVE_CONSTANT: (-2.0,) * 24,
            _RAGGED_CONSTANT: (np.nan, 3.0, np.nan, 3.0, 3.0) + (np.nan,) * 19,
            _NEAR_DEGENERATE: (1.0, near_upper) * 12,
            _ZERO_BASED: (0.0, near_width) * 12,
            _VARIABLE: (1.0, 2.0, 3.0, 4.0, 5.0) + (np.nan,) * 19,
        },
        index=_DATES,
    )
    if nullable:
        return values.astype(pd.Float64Dtype())
    return values


def _legend_lines(data: pd.DataFrame | pd.Series) -> tuple[str, ...]:
    """Draw through the public plotting API and return warning-free legend text.

    Args:
        data: Series or DataFrame whose moment statistics are displayed.

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
                legend_stats=LegendStats.AVG_STD_SKEW_KURT,
                var_format="{:.3f}",
                ax=axis,
            )
        legend = axis.get_legend()
        assert legend is not None
        return tuple(text.get_text() for text in legend.get_texts())
    finally:
        plt.close(figure)


# =============================================================================
# Mixed-panel narrow-spread moment contract
# =============================================================================


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
def test_plot_time_series_stabilizes_narrow_spread_moments_without_warnings(
    nullable: bool,
) -> None:
    """Stabilize near-degenerate moments while exact-constant moments remain undefined.

    Args:
        nullable: Whether the input uses nullable ``Float64``/``pd.NA`` storage.
    """
    values = _mixed_values(nullable=nullable)
    original_values = values.copy(deep=True)

    actual = _legend_lines(values)

    assert actual == _MOMENT_LINES
    pd.testing.assert_frame_equal(values, original_values)


# =============================================================================
# Named Series/DataFrame consistency
# =============================================================================


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
def test_constant_moment_named_series_matches_dataframe(nullable: bool) -> None:
    """Return identical exact text for equivalent named Series and one-column frames.

    Args:
        nullable: Whether both inputs use nullable ``Float64`` storage.
    """
    selected = _mixed_values(nullable=nullable)[_POSITIVE_CONSTANT]
    assert isinstance(selected, pd.Series)
    series = selected
    frame = series.to_frame()
    original_series = series.copy(deep=True)
    original_frame = frame.copy(deep=True)
    expected = (_MOMENT_LINES[0],)

    frame_result = _legend_lines(frame)
    series_result = _legend_lines(series)

    assert frame_result == expected
    assert series_result == expected
    assert series_result == frame_result
    pd.testing.assert_frame_equal(frame, original_frame)
    pd.testing.assert_series_equal(series, original_series)

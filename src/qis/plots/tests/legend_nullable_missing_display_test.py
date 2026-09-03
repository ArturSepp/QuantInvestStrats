"""Regression coverage for nullable missing values in indexed legend endpoints.

``LegendStats.LAST``, ``LegendStats.AVG_LAST``, and ``LegendStats.AVG_STD_LAST`` deliberately
display the value at the final index rather than the last observed value. Equivalent missing
endpoints must nevertheless use the configured numeric missing display for both ordinary
``float64``/``np.nan`` and nullable ``Float64``/``pd.NA`` storage.

One ordered mixed panel combines finite, terminal-missing, and all-missing histories in both
storage representations. Exact public legend text, a non-default ``nan_display`` control,
Series/DataFrame consistency, warnings-as-errors, labels and order, and caller ownership protect
the display contract without changing endpoint selection or numerical statistics.
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


class _TimeSeriesModuleProtocol(Protocol):
    """Typed test-side interface for the public plotting function."""

    def plot_time_series(
        self,
        df: pd.DataFrame | pd.Series,
        *,
        x_date_freq: None,
        legend_stats: LegendStats,
        var_format: str,
        ax: Axes,
    ) -> Figure | None:
        """Plot a time series with the selected indexed-endpoint legend.

        Args:
            df: Series or DataFrame supplied to the public plot.
            x_date_freq: Disabled date-axis formatting for the focused test.
            legend_stats: Indexed-endpoint legend mode under test.
            var_format: Numeric display format for legend values.
            ax: Matplotlib axis receiving the plot.

        Returns:
            The created figure, or None when the caller supplies ``ax``.
        """
        raise NotImplementedError


class _PlotUtilsModuleProtocol(Protocol):
    """Typed test-side interface for the internal legend-text helper."""

    def get_legend_lines(
        self,
        data: pd.DataFrame | pd.Series,
        legend_stats: LegendStats,
        var_format: str,
        nan_display: float,
    ) -> list[str]:
        """Build legend entries with an explicit missing-value display.

        Args:
            data: Series or DataFrame summarized in the legend.
            legend_stats: Indexed-endpoint legend mode under test.
            var_format: Numeric display format for legend values.
            nan_display: Scalar substituted for an undefined or missing value.

        Returns:
            Legend entries in input-column order.
        """
        raise NotImplementedError


_PLOT_UTILS_MODULE = cast(_PlotUtilsModuleProtocol, plot_utils_module)
_TIME_SERIES_MODULE = cast(_TimeSeriesModuleProtocol, time_series_module)


# =============================================================================
# Shared deterministic fixtures and independent expectations
# =============================================================================

_FLOAT_ALL_MISSING = "float all missing"
_FLOAT_FINITE = "float finite"
_FLOAT_TERMINAL_MISSING = "float terminal missing"
_NULLABLE_ALL_MISSING = "nullable all missing"
_NULLABLE_FINITE = "nullable finite"
_NULLABLE_TERMINAL_MISSING = "nullable terminal missing"

_DATES = pd.date_range("2024-01-31", periods=4, freq="ME")
_TERMINAL_MISSING_POSITION = 3

_DEFAULT_EXPECTATIONS: tuple[tuple[LegendStats, tuple[str, ...]], ...] = (
    (
        LegendStats.LAST,
        (
            "float finite: last=4.00",
            "float terminal missing: last=nan",
            "nullable finite: last=4.00",
            "nullable terminal missing: last=nan",
            "float all missing: last=nan",
            "nullable all missing: last=nan",
        ),
    ),
    (
        LegendStats.AVG_LAST,
        (
            "float finite: avg=2.50, last=4.00",
            "float terminal missing: avg=2.00, last=nan",
            "nullable finite: avg=2.50, last=4.00",
            "nullable terminal missing: avg=2.00, last=nan",
            "float all missing: avg=nan, last=nan",
            "nullable all missing: avg=nan, last=nan",
        ),
    ),
    (
        LegendStats.AVG_STD_LAST,
        (
            "float finite: avg=2.50, std=1.29, last=4.00",
            "float terminal missing: avg=2.00, std=1.00, last=nan",
            "nullable finite: avg=2.50, std=1.29, last=4.00",
            "nullable terminal missing: avg=2.00, std=1.00, last=nan",
            "float all missing: avg=nan, std=nan, last=nan",
            "nullable all missing: avg=nan, std=nan, last=nan",
        ),
    ),
)

_CUSTOM_NAN_DISPLAY = -99.0
_CUSTOM_EXPECTATIONS: tuple[tuple[LegendStats, tuple[str, ...]], ...] = (
    (
        LegendStats.LAST,
        (
            "float finite: last=4.00",
            "float terminal missing: last=-99.00",
            "nullable finite: last=4.00",
            "nullable terminal missing: last=-99.00",
            "float all missing: last=-99.00",
            "nullable all missing: last=-99.00",
        ),
    ),
    (
        LegendStats.AVG_LAST,
        (
            "float finite: avg=2.50, last=4.00",
            "float terminal missing: avg=2.00, last=-99.00",
            "nullable finite: avg=2.50, last=4.00",
            "nullable terminal missing: avg=2.00, last=-99.00",
            "float all missing: avg=-99.00, last=-99.00",
            "nullable all missing: avg=-99.00, last=-99.00",
        ),
    ),
    (
        LegendStats.AVG_STD_LAST,
        (
            "float finite: avg=2.50, std=1.29, last=4.00",
            "float terminal missing: avg=2.00, std=1.00, last=-99.00",
            "nullable finite: avg=2.50, std=1.29, last=4.00",
            "nullable terminal missing: avg=2.00, std=1.00, last=-99.00",
            "float all missing: avg=-99.00, std=-99.00, last=-99.00",
            "nullable all missing: avg=-99.00, std=-99.00, last=-99.00",
        ),
    ),
)


def _mixed_values() -> pd.DataFrame:
    """Create equivalent finite and missing histories with two floating dtypes.

    Returns:
        Four-date panel in the same deliberate order as the exact expected legend entries.
    """
    values = pd.DataFrame(index=_DATES)
    values[_FLOAT_FINITE] = pd.Series((1.0, 2.0, 3.0, 4.0), index=_DATES, dtype="float64")
    values[_FLOAT_TERMINAL_MISSING] = pd.Series(
        (1.0, 2.0, 3.0, np.nan), index=_DATES, dtype="float64"
    )
    values[_NULLABLE_FINITE] = pd.Series(
        (1.0, 2.0, 3.0, 4.0), index=_DATES, dtype=pd.Float64Dtype()
    )
    values[_NULLABLE_TERMINAL_MISSING] = pd.Series(
        (1.0, 2.0, 3.0, pd.NA), index=_DATES, dtype=pd.Float64Dtype()
    )
    values[_FLOAT_ALL_MISSING] = pd.Series((np.nan,) * len(_DATES), index=_DATES, dtype="float64")
    values[_NULLABLE_ALL_MISSING] = pd.Series(
        (pd.NA,) * len(_DATES), index=_DATES, dtype=pd.Float64Dtype()
    )
    return values


def _legend_lines(data: pd.DataFrame | pd.Series, legend_stats: LegendStats) -> tuple[str, ...]:
    """Draw through the public plotting API and return exact legend text.

    Args:
        data: Time series whose indexed endpoint is displayed.
        legend_stats: Indexed-endpoint legend mode under test.

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
# Public mixed-panel display contract
# =============================================================================


@pytest.mark.parametrize(("legend_stats", "expected"), _DEFAULT_EXPECTATIONS)
def test_plot_time_series_normalizes_nullable_missing_indexed_endpoints(
    legend_stats: LegendStats,
    expected: tuple[str, ...],
) -> None:
    """Render equivalent missing endpoints identically without changing their selection.

    Args:
        legend_stats: Direct indexed-endpoint mode under test.
        expected: Independently specified text for every dtype and missing-data state.
    """
    values = _mixed_values()
    original_values = values.copy()

    actual = _legend_lines(values, legend_stats)

    assert actual == expected
    pd.testing.assert_frame_equal(values, original_values)


# =============================================================================
# Named Series/DataFrame consistency
# =============================================================================


@pytest.mark.parametrize(("legend_stats", "expected"), _DEFAULT_EXPECTATIONS)
def test_plot_time_series_nullable_terminal_missing_series_matches_dataframe(
    legend_stats: LegendStats,
    expected: tuple[str, ...],
) -> None:
    """Return identical missing-endpoint text for a named Series and one-column frame.

    Args:
        legend_stats: Direct indexed-endpoint mode under test.
        expected: Mixed-panel entries containing the independent one-column expectation.
    """
    selected = _mixed_values()[_NULLABLE_TERMINAL_MISSING]
    assert isinstance(selected, pd.Series)
    series = selected
    frame = series.to_frame()
    original_series = series.copy()
    original_frame = frame.copy()

    expected_line = (expected[_TERMINAL_MISSING_POSITION],)
    series_result = _legend_lines(series, legend_stats)
    frame_result = _legend_lines(frame, legend_stats)

    assert series_result == expected_line
    assert frame_result == expected_line
    assert series_result == frame_result
    pd.testing.assert_series_equal(series, original_series)
    pd.testing.assert_frame_equal(frame, original_frame)


# =============================================================================
# Configured missing-value display
# =============================================================================


@pytest.mark.parametrize(("legend_stats", "expected"), _CUSTOM_EXPECTATIONS)
def test_get_legend_lines_applies_nan_display_to_indexed_missing_endpoints(
    legend_stats: LegendStats,
    expected: tuple[str, ...],
) -> None:
    """Honor ``nan_display`` without replacing a missing endpoint by an observation.

    Args:
        legend_stats: Direct indexed-endpoint mode under test.
        expected: Exact entries using the independently selected ``-99`` missing marker.
    """
    values = _mixed_values()
    original_values = values.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = _PLOT_UTILS_MODULE.get_legend_lines(
            values,
            legend_stats=legend_stats,
            var_format="{:.2f}",
            nan_display=_CUSTOM_NAN_DISPLAY,
        )

    assert tuple(actual) == expected
    pd.testing.assert_frame_equal(values, original_values)

"""Regression coverage for sample-standard-deviation legend boundaries.

Every ``LegendStats`` member containing ``STD`` promises a sample standard deviation with
``ddof=1`` after that mode's documented selection. A selected sample with fewer than two finite
observations therefore has an undefined spread, while two or more observations use the familiar
sample denominator. The calculation must reach both boundaries without relying on NumPy warning
filters.

One deliberately ordered panel combines all-missing, singleton, two-observation, ragged,
complete, one-nonzero, and two-nonzero histories. Exact public legend text is specified
independently for all six affected modes under ordinary and nullable floating storage. Equivalent
named Series establish shape parity, while already-correct t-statistic, moment, and missing-ratio
modes protect the adjacent accepted contracts.
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
            var_format: Display format for numerical statistics.
            ax: Matplotlib axis receiving the plot.

        Returns:
            The created figure, or None when the caller supplies ``ax``.
        """
        raise NotImplementedError


_TIME_SERIES_MODULE = cast(_TimeSeriesModuleProtocol, time_series_module)


# =============================================================================
# Shared deterministic fixtures and independently specified expectations
# =============================================================================

_ALL_MISSING = "All Missing"
_COMPLETE = "Complete"
_ONE = "One"
_ONE_NONZERO = "One Nonzero"
_RAGGED = "Ragged"
_TWO = "Two"
_TWO_NONZERO = "Two Nonzero"

_DATES = pd.date_range("2024-01-31", periods=5, freq="ME")

_AVG_STD_LINES = (
    "All Missing: avg=nan, std=nan",
    "One: avg=5.000, std=nan",
    "Two: avg=2.000, std=1.414",
    "Ragged: avg=2.333, std=1.528",
    "Complete: avg=3.000, std=1.581",
    "One Nonzero: avg=1.000, std=2.236",
    "Two Nonzero: avg=0.800, std=1.304",
)

_AVG_STD_LAST_LINES = (
    "All Missing: avg=nan, std=nan, last=nan",
    "One: avg=5.000, std=nan, last=5.000",
    "Two: avg=2.000, std=1.414, last=3.000",
    "Ragged: avg=2.333, std=1.528, last=2.000",
    "Complete: avg=3.000, std=1.581, last=5.000",
    "One Nonzero: avg=1.000, std=2.236, last=5.000",
    "Two Nonzero: avg=0.800, std=1.304, last=3.000",
)

_NONZERO_AVG_STD_LAST_LINES = (
    "All Missing: avg=nan, std=nan, last=nan",
    "One: avg=5.000, std=nan, last=5.000",
    "Two: avg=2.000, std=1.414, last=3.000",
    "Ragged: avg=2.333, std=1.528, last=2.000",
    "Complete: avg=3.000, std=1.581, last=5.000",
    "One Nonzero: avg=5.000, std=nan, last=5.000",
    "Two Nonzero: avg=2.000, std=1.414, last=3.000",
)

_AVG_MEDIAN_STD_NONNAN_LAST_LINES = (
    "All Missing: avg=nan, median=nan, std=nan, last=nan",
    "One: avg=5.000, median=5.000, std=nan, last=5.000",
    "Two: avg=2.000, median=2.000, std=1.414, last=3.000",
    "Ragged: avg=2.333, median=2.000, std=1.528, last=2.000",
    "Complete: avg=3.000, median=3.000, std=1.581, last=5.000",
    "One Nonzero: avg=1.000, median=0.000, std=2.236, last=5.000",
    "Two Nonzero: avg=0.800, median=0.000, std=1.304, last=3.000",
)

_AVG_STD_LAST_SCORE_LINES = (
    "All Missing: avg=nan, std=nan,  last=nan, last score=nan%",
    "One: avg=5.000, std=nan,  last=5.000, last score=100%",
    "Two: avg=2.000, std=1.414,  last=3.000, last score=100%",
    "Ragged: avg=2.333, std=1.528,  last=2.000, last score=67%",
    "Complete: avg=3.000, std=1.581,  last=5.000, last score=100%",
    "One Nonzero: avg=1.000, std=2.236,  last=5.000, last score=100%",
    "Two Nonzero: avg=0.800, std=1.304,  last=3.000, last score=100%",
)

_AVG_STD_MISSING_ZERO_LINES = (
    "All Missing: avg=nan, std=nan, missing%=100.00%, zeros%=nan%",
    "One: avg=5.000, std=nan, missing%=0.00%, zeros%=0.00%",
    "Two: avg=2.000, std=1.414, missing%=0.00%, zeros%=0.00%",
    "Ragged: avg=2.333, std=1.528, missing%=25.00%, zeros%=0.00%",
    "Complete: avg=3.000, std=1.581, missing%=0.00%, zeros%=0.00%",
    "One Nonzero: avg=1.000, std=2.236, missing%=0.00%, zeros%=80.00%",
    "Two Nonzero: avg=0.800, std=1.304, missing%=0.00%, zeros%=60.00%",
)

_TARGET_EXPECTATIONS: tuple[tuple[LegendStats, tuple[str, ...]], ...] = (
    (LegendStats.AVG_STD, _AVG_STD_LINES),
    (LegendStats.AVG_STD_LAST, _AVG_STD_LAST_LINES),
    (LegendStats.NONZERO_AVG_STD_LAST, _NONZERO_AVG_STD_LAST_LINES),
    (LegendStats.AVG_MEDIAN_STD_NONNAN_LAST, _AVG_MEDIAN_STD_NONNAN_LAST_LINES),
    (LegendStats.AVG_STD_LAST_SCORE, _AVG_STD_LAST_SCORE_LINES),
    (LegendStats.AVG_STD_MISSING_ZERO, _AVG_STD_MISSING_ZERO_LINES),
)

_INTERACTION_EXPECTATIONS: tuple[tuple[LegendStats, str], ...] = (
    (LegendStats.AVG_STD_TSTAT, "Complete: avg=3.000, std=1.581, t-stat=4.24"),
    (
        LegendStats.AVG_STD_SKEW_KURT,
        "Complete: avg=3.000, std=1.581, skew=0.00, kurtosis=-1.30",
    ),
    (LegendStats.MISSING_AVG_LAST, "Complete: missing%=0.00%, avg=3.000, last=5.000"),
)


def _mixed_values(*, nullable: bool) -> pd.DataFrame:
    """Create every materially different standard-deviation sample state.

    Singleton and two-observation histories end in finite values so this regression isolates
    sample spread from the separate nullable endpoint-format policy. The ragged history ends below
    its maximum to preserve nontrivial last-value and percentile-score behavior. Zero-heavy
    histories distinguish whole-column sampling from ``NONZERO`` selection.

    Args:
        nullable: Whether to store values as pandas nullable ``Float64``/``pd.NA``.

    Returns:
        Five-date panel in the same deliberate order as every expected legend tuple.
    """
    values = pd.DataFrame(
        {
            _ALL_MISSING: (np.nan, np.nan, np.nan, np.nan, np.nan),
            _ONE: (np.nan, np.nan, np.nan, np.nan, 5.0),
            _TWO: (np.nan, np.nan, np.nan, 1.0, 3.0),
            _RAGGED: (np.nan, 1.0, np.nan, 4.0, 2.0),
            _COMPLETE: (1.0, 2.0, 3.0, 4.0, 5.0),
            _ONE_NONZERO: (0.0, 0.0, 0.0, 0.0, 5.0),
            _TWO_NONZERO: (0.0, 1.0, 0.0, 0.0, 3.0),
        },
        index=_DATES,
    )
    if nullable:
        return values.astype(pd.Float64Dtype())
    return values


def _legend_lines(data: pd.DataFrame | pd.Series, legend_stats: LegendStats) -> tuple[str, ...]:
    """Draw through the public plotting API and return exact warning-free legend text.

    Args:
        data: Time series whose statistics are displayed.
        legend_stats: Summary-statistic mode under test.

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
                var_format="{:.3f}",
                ax=axis,
            )
        legend = axis.get_legend()
        assert legend is not None
        return tuple(text.get_text() for text in legend.get_texts())
    finally:
        plt.close(figure)


# =============================================================================
# Mixed-panel sample-standard-deviation contract
# =============================================================================


@pytest.mark.parametrize(
    ("legend_stats", "expected"),
    _TARGET_EXPECTATIONS,
    ids=tuple(legend_stats.name.lower() for legend_stats, _ in _TARGET_EXPECTATIONS),
)
@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
def test_legend_standard_deviation_uses_warning_free_selected_sample(
    legend_stats: LegendStats,
    expected: tuple[str, ...],
    nullable: bool,
) -> None:
    """Apply one sample-spread boundary across all affected public legend modes.

    Args:
        legend_stats: Standard-deviation legend mode under test.
        expected: Independently specified text for the complete mixed panel.
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


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
def test_legend_standard_deviation_named_series_matches_dataframe(nullable: bool) -> None:
    """Return identical exact statistics for equivalent named Series and one-column frames.

    Args:
        nullable: Whether the input uses nullable ``Float64``/``pd.NA`` storage.
    """
    selected = _mixed_values(nullable=nullable)[_TWO]
    assert isinstance(selected, pd.Series)
    series = selected
    frame = series.to_frame()
    original_series = series.copy(deep=True)
    original_frame = frame.copy(deep=True)

    for legend_stats, mixed_expected in _TARGET_EXPECTATIONS:
        expected = (mixed_expected[2],)
        frame_result = _legend_lines(frame, legend_stats)
        series_result = _legend_lines(series, legend_stats)

        assert frame_result == expected
        assert series_result == expected
        assert series_result == frame_result

    pd.testing.assert_frame_equal(frame, original_frame)
    pd.testing.assert_series_equal(series, original_series)


# =============================================================================
# Adjacent accepted-contract controls
# =============================================================================


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
def test_legend_standard_deviation_preserves_adjacent_modes(nullable: bool) -> None:
    """Preserve accepted t-statistic, moment, and missing-ratio legend behavior.

    Args:
        nullable: Whether the input uses nullable ``Float64``/``pd.NA`` storage.
    """
    complete = _mixed_values(nullable=nullable)[_COMPLETE]
    assert isinstance(complete, pd.Series)
    original_complete = complete.copy(deep=True)

    for legend_stats, expected in _INTERACTION_EXPECTATIONS:
        assert _legend_lines(complete, legend_stats) == (expected,)

    pd.testing.assert_series_equal(complete, original_complete)

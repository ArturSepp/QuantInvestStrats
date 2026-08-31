"""Regression coverage for missing and zero ratios in public legend diagnostics.

``LegendStats.MISSING_AVG_LAST`` reports post-inception missing coverage, while
``LegendStats.AVG_STD_MISSING_ZERO`` reports the same missing coverage together with the share of
near-zero observations. The denominator begins at each column's first observation so leading
pre-inception gaps remain outside both ratios. An all-missing history has complete missing
coverage but no observed window in which a zero ratio can be defined.

One deliberately ordered panel distinguishes complete, leading-ragged, interior-missing,
zero-heavy, mixed missing/zero, and all-missing histories. Independently specified percentages,
ordinary and nullable floating storage, public Series/DataFrame consistency, warnings-as-errors,
exact legend text, and caller ownership keep the diagnostic labels tied to the values they claim
to display.
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
from qis.utils.df_ops import compute_nans_zeros_ratio_after_first_non_nan


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
        """Plot a time series with the selected legend diagnostics.

        Args:
            df: Series or DataFrame supplied to the public plot.
            x_date_freq: Disabled date-axis formatting for this focused legend test.
            legend_stats: Missing-data diagnostic mode displayed in the legend.
            var_format: Display format for level statistics.
            ax: Matplotlib axis receiving the plot.

        Returns:
            The created figure, or None when the caller supplies ``ax``.
        """
        raise NotImplementedError


_TIME_SERIES_MODULE = cast(_TimeSeriesModuleProtocol, time_series_module)


# =============================================================================
# Shared deterministic fixtures and independent expectations
# =============================================================================

_ALL_MISSING = "All Missing"
_COMPLETE = "Complete"
_INTERIOR_MISSING = "Interior Missing"
_LEADING_GAPS = "Leading Gaps"
_MIXED = "Mixed"
_ZEROS = "Zeros"

_DATES = pd.date_range("2024-01-31", periods=6, freq="ME")

_AVG_STD_MISSING_ZERO_LINES = (
    "Complete: avg=3.50, std=1.87, missing%=0.00%, zeros%=0.00%",
    "Leading Gaps: avg=2.50, std=1.29, missing%=0.00%, zeros%=0.00%",
    "Interior Missing: avg=2.50, std=1.29, missing%=33.33%, zeros%=0.00%",
    "Zeros: avg=1.67, std=1.63, missing%=0.00%, zeros%=33.33%",
    "Mixed: avg=0.75, std=0.96, missing%=20.00%, zeros%=40.00%",
    "All Missing: avg=nan, std=nan, missing%=100.00%, zeros%=nan%",
)

_MISSING_AVG_LAST_LINES = (
    "Complete: missing%=0.00%, avg=3.50, last=6.00",
    "Leading Gaps: missing%=0.00%, avg=2.50, last=4.00",
    "Interior Missing: missing%=33.33%, avg=2.50, last=4.00",
    "Zeros: missing%=0.00%, avg=1.67, last=4.00",
    "Mixed: missing%=20.00%, avg=0.75, last=0.00",
    "All Missing: missing%=100.00%, avg=nan, last=nan",
)

_EXPECTED_MISSING_RATIOS = np.asarray((0.0, 0.0, 2.0 / 6.0, 0.0, 1.0 / 5.0, 1.0))
_EXPECTED_ZERO_RATIOS = np.asarray((0.0, 0.0, 0.0, 2.0 / 6.0, 2.0 / 5.0, np.nan))


def _mixed_values(*, nullable: bool) -> pd.DataFrame:
    """Create every materially different post-inception diagnostic state.

    The leading-ragged history confirms that the denominator starts at its first observation.
    Interior missing values and zeros are separated so a mislabeled ratio cannot satisfy both
    controls, while the mixed history independently establishes denominators of five.

    Args:
        nullable: Whether to store values as pandas nullable ``Float64``/``pd.NA``.

    Returns:
        Six-date panel in the same deliberate order as the expected legend entries.
    """
    values = pd.DataFrame(
        {
            _COMPLETE: (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            _LEADING_GAPS: (np.nan, np.nan, 1.0, 2.0, 3.0, 4.0),
            _INTERIOR_MISSING: (1.0, np.nan, np.nan, 2.0, 3.0, 4.0),
            _ZEROS: (1.0, 0.0, 0.0, 2.0, 3.0, 4.0),
            _MIXED: (np.nan, 1.0, 0.0, np.nan, 2.0, 0.0),
            _ALL_MISSING: (np.nan,) * len(_DATES),
        },
        index=_DATES,
    )
    if nullable:
        return values.astype(pd.Float64Dtype())
    return values


def _legend_lines(data: pd.DataFrame | pd.Series, legend_stats: LegendStats) -> tuple[str, ...]:
    """Draw through the public plotting API and return exact legend text.

    Args:
        data: Time series whose diagnostics are displayed.
        legend_stats: Missing-data legend mode under test.

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
# Missing-versus-zero diagnostic identity
# =============================================================================


@pytest.mark.parametrize(
    ("legend_stats", "expected"),
    (
        (LegendStats.AVG_STD_MISSING_ZERO, _AVG_STD_MISSING_ZERO_LINES[:-1]),
        (LegendStats.MISSING_AVG_LAST, _MISSING_AVG_LAST_LINES[:-1]),
    ),
)
@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
def test_plot_time_series_distinguishes_missing_and_zero_ratios(
    legend_stats: LegendStats,
    expected: tuple[str, ...],
    nullable: bool,
) -> None:
    """Attach each diagnostic label to its independently calculated ratio.

    Args:
        legend_stats: Composite or missing-only diagnostic mode.
        expected: Exact entries for every history with an observed starting point.
        nullable: Whether the input uses nullable ``Float64``/``pd.NA`` storage.
    """
    values = _mixed_values(nullable=nullable).drop(columns=_ALL_MISSING)
    original_values = values.copy(deep=True)

    actual = _legend_lines(values, legend_stats)

    assert actual == expected
    pd.testing.assert_frame_equal(values, original_values)


# =============================================================================
# All-missing mixed-panel boundary
# =============================================================================


@pytest.mark.parametrize(
    ("legend_stats", "expected"),
    (
        (LegendStats.AVG_STD_MISSING_ZERO, _AVG_STD_MISSING_ZERO_LINES),
        (LegendStats.MISSING_AVG_LAST, _MISSING_AVG_LAST_LINES),
    ),
)
@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
def test_plot_time_series_reports_all_missing_diagnostics_without_errors(
    legend_stats: LegendStats,
    expected: tuple[str, ...],
    nullable: bool,
) -> None:
    """Keep finite neighbors intact while reporting an all-missing history.

    Args:
        legend_stats: Composite or missing-only diagnostic mode.
        expected: Exact entries including complete missing coverage and undefined zeros.
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
        (LegendStats.AVG_STD_MISSING_ZERO, (_AVG_STD_MISSING_ZERO_LINES[2],)),
        (LegendStats.MISSING_AVG_LAST, (_MISSING_AVG_LAST_LINES[2],)),
    ),
)
@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
def test_missing_ratio_named_series_matches_dataframe(
    legend_stats: LegendStats,
    expected: tuple[str, ...],
    nullable: bool,
) -> None:
    """Return identical diagnostics for equivalent Series and one-column frames.

    Args:
        legend_stats: Composite or missing-only diagnostic mode.
        expected: Independently specified one-entry legend text.
        nullable: Whether the input uses nullable ``Float64``/``pd.NA`` storage.
    """
    selected = _mixed_values(nullable=nullable)[_INTERIOR_MISSING]
    assert isinstance(selected, pd.Series)
    series = selected
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


# =============================================================================
# Ratio-helper contract
# =============================================================================


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
def test_ratio_helper_uses_post_inception_windows_and_defines_all_missing(nullable: bool) -> None:
    """Return independent missing and zero ratios without changing the input panel.

    Args:
        nullable: Whether the input uses nullable ``Float64``/``pd.NA`` storage.
    """
    values = _mixed_values(nullable=nullable)
    original_values = values.copy(deep=True)

    missing_ratios, zero_ratios = compute_nans_zeros_ratio_after_first_non_nan(values)

    np.testing.assert_allclose(missing_ratios, _EXPECTED_MISSING_RATIOS)
    np.testing.assert_allclose(zero_ratios, _EXPECTED_ZERO_RATIOS, equal_nan=True)
    pd.testing.assert_frame_equal(values, original_values)

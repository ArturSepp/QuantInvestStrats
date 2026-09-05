"""Regression coverage for near-degenerate legend spread and t-statistic precision.

Sample standard deviation is translation invariant, but reducing uncentered binary64 levels can
lose precision when their represented spread is only a few ULPs. Every ``LegendStats`` mode that
displays sample spread, and both modes that use it as the signed sample-mean t-statistic
denominator, must follow the translation-stable convention already established for descriptive
tables.

The mixed panel combines several offsets and represented widths with asymmetric, ragged,
constant, ordinary-scale, and all-missing controls. Expected spreads were calculated with
80-digit ``Decimal`` arithmetic over the exact stored binary64 observations. Expected
t-statistics retain the established binary64 sample mean and divide it by the independently
calculated sample standard error. Ordinary and nullable storage, all affected legend modes,
Series/DataFrame consistency, caller ownership, warning handling, and an actual canvas render are
covered without deriving an expected result from the production reduction.
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


class _PlotUtilsModuleProtocol(Protocol):
    """Typed test-side interface for the shared legend helper exercised below."""

    def get_legend_lines(
        self,
        data: pd.DataFrame | pd.Series,
        legend_stats: LegendStats,
        var_format: str,
        tstat_format: str,
    ) -> list[str]:
        """Return legend text for the requested statistics."""
        raise NotImplementedError


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
        """Plot a time series with the requested legend statistics."""
        raise NotImplementedError


_PLOT_UTILS_MODULE = cast(_PlotUtilsModuleProtocol, plot_utils_module)
_TIME_SERIES_MODULE = cast(_TimeSeriesModuleProtocol, time_series_module)


# =============================================================================
# Shared deterministic fixtures and independent expectations
# =============================================================================

_DATES = pd.date_range("2024-01-31", periods=24, freq="ME")

_POSITIVE_ONE_ULP = "Positive One ULP"
_NEGATIVE_THREE_ULP = "Negative Three ULP"
_LARGE_ONE_ULP = "Large One ULP"
_POSITIVE_THREE_ULP = "Positive Three ULP"
_POSITIVE_SEVEN_ULP = "Positive Seven ULP"
_ASYMMETRIC_NEAR = "Asymmetric Near"
_RAGGED_ONE_ULP = "Ragged One ULP"
_EXACT_CONSTANT = "Exact Constant"
_REGULAR_CENTERED = "Regular Centered"
_ALL_MISSING = "All Missing"

_ASSETS = (
    _POSITIVE_ONE_ULP,
    _NEGATIVE_THREE_ULP,
    _LARGE_ONE_ULP,
    _POSITIVE_THREE_ULP,
    _POSITIVE_SEVEN_ULP,
    _ASYMMETRIC_NEAR,
    _RAGGED_ONE_ULP,
    _EXACT_CONSTANT,
    _REGULAR_CENTERED,
    _ALL_MISSING,
)

_EXPECTED_SPREADS = (
    "1.13410152037e-16",
    "1.70115228056e-16",
    "7.61082646929e-09",
    "2.53491443479e-24",
    "5.91480034784e-24",
    "5.34406885838e-24",
    "1.13906478925e-16",
    "0",
    "7.07106781187",
    "nan",
)

_EXPECTED_TSTATS = (
    "4.31970101226e+16",
    "-2.87980067484e+16",
    "6.4368561093e+16",
    "1.93260151835e+16",
    "8.28257793579e+15",
    "9.16713391124e+15",
    "3.92614713158e+16",
    "nan",
    "0",
    "nan",
)

_SPREAD_MODES = (
    LegendStats.AVG_STD,
    LegendStats.AVG_STD_SKEW_KURT,
    LegendStats.AVG_STD_LAST,
    LegendStats.AVG_STD_TSTAT,
    LegendStats.NONZERO_AVG_STD_LAST,
    LegendStats.AVG_MEDIAN_STD_NONNAN_LAST,
    LegendStats.AVG_STD_LAST_SCORE,
    LegendStats.AVG_STD_MISSING_ZERO,
)

_TSTAT_MODES = (LegendStats.TSTAT, LegendStats.AVG_STD_TSTAT)

_VALUE_FORMAT = "{:.12g}"


def _next_float(value: float, steps: int) -> float:
    """Move a finite value upward by an exact number of representable steps.

    Args:
        value: Starting floating-point value.
        steps: Number of calls to ``np.nextafter`` toward positive infinity.

    Returns:
        Representable value exactly ``steps`` ULP transitions above the start.
    """
    result = value
    for _ in range(steps):
        result = float(np.nextafter(result, np.inf))
    return result


def _mixed_samples(*, nullable: bool) -> pd.DataFrame:
    """Create the mixed precision, selection, and undefined-value panel.

    Args:
        nullable: Whether every column uses pandas nullable ``Float64`` storage.

    Returns:
        Twenty-four-row panel containing every material legend-spread state.
    """
    positive_pair = (1.0, _next_float(1.0, 1))
    negative_pair = (-1.0, _next_float(-1.0, 3))
    large_pair = (1.0e8, _next_float(1.0e8, 1))
    positive_three_pair = (1.0e-8, _next_float(1.0e-8, 3))
    positive_seven_pair = (1.0e-8, _next_float(1.0e-8, 7))
    asymmetric_pattern = tuple(_next_float(1.0e-8, step) for step in (0, 1, 3, 4, 7, 9))

    samples = pd.DataFrame(
        {
            _POSITIVE_ONE_ULP: positive_pair * 12,
            _NEGATIVE_THREE_ULP: negative_pair * 12,
            _LARGE_ONE_ULP: large_pair * 12,
            _POSITIVE_THREE_ULP: positive_three_pair * 12,
            _POSITIVE_SEVEN_ULP: positive_seven_pair * 12,
            _ASYMMETRIC_NEAR: asymmetric_pattern * 4,
            _RAGGED_ONE_ULP: (np.nan,) * 4 + positive_pair * 10,
            _EXACT_CONSTANT: (2.0,) * 24,
            _REGULAR_CENTERED: tuple(float(value) - 11.5 for value in range(24)),
            _ALL_MISSING: (np.nan,) * 24,
        },
        index=_DATES,
    )
    if nullable:
        return samples.astype(pd.Float64Dtype())
    return samples


def _legend_lines(data: pd.DataFrame | pd.Series, legend_stats: LegendStats) -> tuple[str, ...]:
    """Return exact shared-helper text while treating every warning as a failure.

    Args:
        data: Series or DataFrame supplied to the shared legend helper.
        legend_stats: Public statistic mode under test.

    Returns:
        Legend entries in input-column order.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return tuple(
            _PLOT_UTILS_MODULE.get_legend_lines(
                data=data,
                legend_stats=legend_stats,
                var_format=_VALUE_FORMAT,
                tstat_format=_VALUE_FORMAT,
            )
        )


def _extract_statistic(lines: tuple[str, ...], marker: str) -> tuple[str, ...]:
    """Extract one exactly formatted statistic from every legend entry.

    Args:
        lines: Complete legend entries in input-column order.
        marker: Field marker immediately preceding the requested value.

    Returns:
        Requested values as their exact public display strings.
    """
    values: list[str] = []
    for line in lines:
        _, separator, suffix = line.partition(marker)
        assert separator == marker
        values.append(suffix.split(",", maxsplit=1)[0])
    return tuple(values)


def _assert_labels_in_input_order(lines: tuple[str, ...]) -> None:
    """Assert that statistic calculation preserves the mixed panel's labels and order."""
    assert tuple(line.split(":", maxsplit=1)[0] for line in lines) == _ASSETS


# =============================================================================
# Mixed-panel translation-stable spread and t-statistic contract
# =============================================================================


@pytest.mark.parametrize(
    "legend_stats",
    _SPREAD_MODES,
    ids=tuple(mode.name.lower() for mode in _SPREAD_MODES),
)
@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
def test_get_legend_lines_stabilizes_near_degenerate_sample_spread(
    legend_stats: LegendStats,
    nullable: bool,
) -> None:
    """Display Decimal-referenced spread for every affected legend mode and column state.

    Args:
        legend_stats: Sample-standard-deviation legend mode under test.
        nullable: Whether the panel uses nullable ``Float64``/``pd.NA`` storage.
    """
    samples = _mixed_samples(nullable=nullable)
    original = samples.copy()

    lines = _legend_lines(samples, legend_stats)

    _assert_labels_in_input_order(lines)
    assert _extract_statistic(lines, "std=") == _EXPECTED_SPREADS
    pd.testing.assert_frame_equal(samples, original)


@pytest.mark.parametrize(
    "legend_stats",
    _TSTAT_MODES,
    ids=tuple(mode.name.lower() for mode in _TSTAT_MODES),
)
@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
def test_get_legend_lines_uses_stabilized_spread_for_sample_mean_tstat(
    legend_stats: LegendStats,
    nullable: bool,
) -> None:
    """Use the stable sample spread as the signed t-statistic denominator.

    The independently specified t-statistics retain the existing binary64 sample mean and divide
    by the Decimal-referenced sample standard error. Exact constants and all-missing histories
    remain undefined, while the centered zero-mean control remains exactly zero.

    Args:
        legend_stats: T-statistic-only or composite statistic mode under test.
        nullable: Whether the panel uses nullable ``Float64``/``pd.NA`` storage.
    """
    samples = _mixed_samples(nullable=nullable)
    original = samples.copy()

    lines = _legend_lines(samples, legend_stats)

    _assert_labels_in_input_order(lines)
    assert _extract_statistic(lines, "t-stat=") == _EXPECTED_TSTATS
    pd.testing.assert_frame_equal(samples, original)


# =============================================================================
# Shape parity and rendered public plotting interaction
# =============================================================================


@pytest.mark.parametrize(
    "legend_stats",
    (LegendStats.AVG_STD, LegendStats.AVG_STD_TSTAT),
    ids=("spread", "spread-and-tstat"),
)
@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
def test_get_legend_lines_near_degenerate_named_series_matches_dataframe(
    legend_stats: LegendStats,
    nullable: bool,
) -> None:
    """Return identical precision-sensitive text for equivalent Series and DataFrames.

    Args:
        legend_stats: Representative spread-only or spread-and-t-statistic mode.
        nullable: Whether the input uses nullable ``Float64`` storage.
    """
    selected = _mixed_samples(nullable=nullable)[_POSITIVE_ONE_ULP]
    assert isinstance(selected, pd.Series)
    series = selected
    frame = series.to_frame()

    series_lines = _legend_lines(series, legend_stats)
    frame_lines = _legend_lines(frame, legend_stats)

    assert series_lines == frame_lines
    assert _extract_statistic(series_lines, "std=") == (_EXPECTED_SPREADS[0],)
    if legend_stats == LegendStats.AVG_STD_TSTAT:
        assert _extract_statistic(series_lines, "t-stat=") == (_EXPECTED_TSTATS[0],)


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
def test_plot_time_series_renders_translation_stable_legend_spread(nullable: bool) -> None:
    """Render a public legend using the stable spread without warnings or input mutation.

    Args:
        nullable: Whether the plotted mixed panel uses nullable ``Float64`` storage.
    """
    samples = _mixed_samples(nullable=nullable)
    original = samples.copy()
    figure, axis = plt.subplots()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _TIME_SERIES_MODULE.plot_time_series(
                samples,
                x_date_freq=None,
                legend_stats=LegendStats.AVG_STD,
                var_format=_VALUE_FORMAT,
                ax=axis,
            )
            figure.canvas.draw()

        legend = axis.get_legend()
        assert legend is not None
        lines = tuple(text.get_text() for text in legend.get_texts())
        _assert_labels_in_input_order(lines)
        assert _extract_statistic(lines, "std=") == _EXPECTED_SPREADS
        pd.testing.assert_frame_equal(samples, original)
    finally:
        plt.close(figure)

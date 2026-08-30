"""Regression coverage for missing realized-volatility windows and regimes.

``compute_sampled_vols`` reports one annualized estimate per requested window, while
``BenchmarkVolsQuantilesRegime`` assigns only finite estimates to quantile regimes. An unavailable
window should therefore remain missing without NumPy reduction warnings, and an entirely
unavailable benchmark should fail with a descriptive classifier error before quantile binning.
Isolated missing windows must not invalidate a later usable benchmark history.

The fixtures exercise ordinary and nullable pandas representations through public APIs. The mixed
panel places a fully observed history, an all-missing history, and a history with only one finite
January return in the same vectorized call. Healthy expected volatility is calculated directly
from squared return deviations and the business-day annualization factor; no QIS volatility or
classification helper constructs the references.
"""

from typing import Callable, Dict, Protocol, cast
import warnings

import numpy as np
import pandas as pd
import pytest

# qis
from qis.perfstats.regime_classifier import BenchmarkVolsQuantilesRegime
from qis.perfstats.returns import compute_sampled_vols


class _VolatilityClassifierProtocol(Protocol):
    """Typed test-side interface for the public volatility classifier."""

    def compute_sampled_returns_with_regime_id(
        self,
        *,
        prices: pd.DataFrame,
        benchmark: str,
        include_start_date: bool,
        include_end_date: bool,
    ) -> pd.DataFrame:
        """Return sampled returns with their volatility regime classification."""
        raise NotImplementedError

    def get_regime_ids_colors(self) -> Dict[str, str]:
        """Return the ordered regime-ID-to-color mapping."""
        raise NotImplementedError


# =============================================================================
# Shared deterministic fixtures
# =============================================================================

_BENCHMARK = "Benchmark"
_HEALTHY = "Healthy"
_ALL_MISSING = "All Missing"
_LATE_START = "Late Start"
_RAGGED_ASSET = "Ragged Asset"
_NUM_BUCKETS = 4

_BUSINESS_PERIODS_PER_YEAR = 252.0
_ALL_MISSING_DATES = pd.bdate_range("2022-01-03", "2022-12-30")
_ALL_MISSING_VOL_DATES = pd.DatetimeIndex(
    (
        "2022-01-31",
        "2022-02-28",
        "2022-03-31",
        "2022-04-30",
        "2022-05-31",
        "2022-06-30",
        "2022-07-31",
        "2022-08-31",
        "2022-09-30",
        "2022-10-31",
        "2022-11-30",
        "2022-12-30",
    )
)
_MIXED_DATES = pd.bdate_range("2022-01-03", "2022-03-31")
_MIXED_VOL_DATES = pd.DatetimeIndex(("2022-01-31", "2022-02-28", "2022-03-31"))
_MIXED_VOL_WINDOWS = (
    (pd.Timestamp("2022-01-03"), pd.Timestamp("2022-01-31")),
    (pd.Timestamp("2022-01-31"), pd.Timestamp("2022-02-28")),
    (pd.Timestamp("2022-02-28"), pd.Timestamp("2022-03-31")),
)
_RAGGED_DATES = pd.bdate_range("2022-01-03", "2024-12-31")

_EXPECTED_REGIME_IDS = (
    "Benchmark vol<5%",
    "Benchmark vol=(5%, 6%]",
    "Benchmark vol=(6%, 8%]",
    "Benchmark vol>8%",
)
_EXPECTED_REGIME_COLORS = ("#a50026", "#fdbf6f", "#b7e075", "#006837")
_EXPECTED_REGIME_COUNTS = (9, 8, 8, 8)


def _as_nullable(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert a fixture to pandas nullable floating storage.

    Args:
        frame: Ordinary floating fixture.

    Returns:
        Equivalent nullable ``Float64`` fixture.
    """
    return frame.astype(pd.Float64Dtype())


def _as_ordinary(frame: pd.DataFrame) -> pd.DataFrame:
    """Retain a fixture as ordinary NumPy floating storage.

    Args:
        frame: Ordinary floating fixture.

    Returns:
        Equivalent ``float64`` fixture.
    """
    return frame.astype(np.float64)


_DTYPE_CONVERTERS: tuple[Callable[[pd.DataFrame], pd.DataFrame], ...] = (
    _as_ordinary,
    _as_nullable,
)


def _all_missing_panel() -> pd.DataFrame:
    """Create a one-column benchmark with no observed prices.

    Returns:
        All-missing benchmark over twelve complete volatility windows.
    """
    return pd.DataFrame({_BENCHMARK: np.nan}, index=_ALL_MISSING_DATES)


def _mixed_volatility_panel() -> pd.DataFrame:
    """Create healthy, all-missing, and one-return window states together.

    The healthy price path alternates one-percent simple returns. The late-starting column has two
    adjacent January observations, yielding exactly one finite January return; forward filling
    produces zero returns thereafter. The overlapping February window retains that January 31
    return, while March is the sufficiently observed all-zero control.

    Returns:
        Three-column price panel spanning three monthly volatility windows.
    """
    healthy_returns = np.where(np.arange(len(_MIXED_DATES)) % 2 == 0, 0.01, -0.01)
    healthy_prices = 100.0 * np.cumprod(1.0 + healthy_returns)
    late_start_prices = np.full(len(_MIXED_DATES), np.nan, dtype=float)
    late_start_prices[_MIXED_DATES.get_loc("2022-01-28")] = 100.0
    late_start_prices[_MIXED_DATES.get_loc("2022-01-31")] = 101.0
    return pd.DataFrame(
        {
            _HEALTHY: healthy_prices,
            _ALL_MISSING: np.nan,
            _LATE_START: late_start_prices,
        },
        index=_MIXED_DATES,
    )


def _ragged_usable_panel() -> pd.DataFrame:
    """Create a late-starting benchmark followed by 33 usable volatility windows.

    Alternating signs keep monthly return means near zero while the absolute amplitude increases
    by 1.5 basis points per month. Direct ranking therefore leaves regime counts of 9, 8, 8, and 8
    after the first three benchmark months are made unavailable.

    Returns:
        Three-year benchmark and dependent-asset price panel.
    """
    month_numbers = np.asarray(
        [(date.year - 2022) * 12 + date.month for date in _RAGGED_DATES],
        dtype=float,
    )
    amplitudes = 0.001 + 0.00015 * month_numbers
    signs = np.where(np.arange(len(_RAGGED_DATES)) % 2 == 0, 1.0, -1.0)
    benchmark_returns = amplitudes * signs
    benchmark_prices = 100.0 * np.cumprod(1.0 + benchmark_returns)
    asset_prices = 80.0 * np.cumprod(1.0 + 0.5 * benchmark_returns)
    benchmark_prices[_RAGGED_DATES < pd.Timestamp("2022-04-01")] = np.nan
    asset_prices[:40] = np.nan
    return pd.DataFrame(
        {_BENCHMARK: benchmark_prices, _RAGGED_ASSET: asset_prices},
        index=_RAGGED_DATES,
    )


def _annualized_sample_std(values: np.ndarray) -> float:
    """Calculate annualized sample volatility independently from QIS.

    Args:
        values: Finite daily simple returns in one monthly window.

    Returns:
        Sample standard deviation multiplied by ``sqrt(252)``.
    """
    mean = float(np.sum(values) / len(values))
    squared_deviations = np.sum(np.square(values - mean))
    sample_variance = float(squared_deviations / (len(values) - 1))
    return float(np.sqrt(sample_variance * _BUSINESS_PERIODS_PER_YEAR))


def _expected_mixed_vols() -> pd.DataFrame:
    """Construct the mixed-panel volatility reference from literal returns.

    Returns:
        Expected healthy, unavailable, and late-starting monthly volatility values.
    """
    healthy_returns = np.where(np.arange(len(_MIXED_DATES)) % 2 == 0, 0.01, -0.01).astype(float)
    healthy_returns[0] = np.nan
    expected_healthy: list[float] = []
    for start, end in _MIXED_VOL_WINDOWS:
        # Included start dates make adjacent monthly windows share the prior month-end return.
        in_window = (_MIXED_DATES >= start) & (_MIXED_DATES <= end)
        window_values = healthy_returns[in_window]
        expected_healthy.append(_annualized_sample_std(window_values[np.isfinite(window_values)]))
    late_start_returns = np.full(len(_MIXED_DATES), np.nan, dtype=float)
    late_start_returns[_MIXED_DATES >= pd.Timestamp("2022-01-31")] = 0.0
    late_start_returns[_MIXED_DATES.get_loc("2022-01-31")] = 0.01
    expected_late_start: list[float] = []
    for start, end in _MIXED_VOL_WINDOWS:
        in_window = (_MIXED_DATES >= start) & (_MIXED_DATES <= end)
        window_values = late_start_returns[in_window]
        finite_values = window_values[np.isfinite(window_values)]
        expected_late_start.append(
            _annualized_sample_std(finite_values) if len(finite_values) >= 2 else np.nan
        )
    return pd.DataFrame(
        {
            _HEALTHY: expected_healthy,
            _ALL_MISSING: [np.nan, np.nan, np.nan],
            _LATE_START: expected_late_start,
        },
        index=_MIXED_VOL_DATES,
    )


# =============================================================================
# Missing sampled-volatility windows
# =============================================================================


@pytest.mark.parametrize("convert", _DTYPE_CONVERTERS, ids=("ordinary", "nullable"))
def test_compute_sampled_vols_preserves_all_missing_series_and_frame(
    convert: Callable[[pd.DataFrame], pd.DataFrame],
) -> None:
    """Return twelve missing estimates without warnings for either pandas shape.

    Args:
        convert: Ordinary or nullable floating fixture conversion.
    """
    prices = convert(_all_missing_panel())
    original_prices = prices.copy()
    series_prices = prices[_BENCHMARK]
    assert isinstance(series_prices, pd.Series)
    expected = pd.Series(np.nan, index=_ALL_MISSING_VOL_DATES, name=_BENCHMARK)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual_series = compute_sampled_vols(
            prices=series_prices,
            freq_vol="ME",
            include_start_date=True,
            include_end_date=True,
        )
        actual_frame = compute_sampled_vols(
            prices=prices,
            freq_vol="ME",
            include_start_date=True,
            include_end_date=True,
        )

    assert isinstance(actual_series, pd.Series)
    assert isinstance(actual_frame, pd.DataFrame)
    pd.testing.assert_series_equal(actual_series, expected)
    pd.testing.assert_frame_equal(actual_frame, expected.to_frame())
    pd.testing.assert_frame_equal(prices, original_prices)


@pytest.mark.parametrize("convert", _DTYPE_CONVERTERS, ids=("ordinary", "nullable"))
def test_compute_sampled_vols_preserves_mixed_window_states(
    convert: Callable[[pd.DataFrame], pd.DataFrame],
) -> None:
    """Estimate eligible columns while unavailable neighbors remain missing and quiet.

    Healthy expected values use the independent sample-variance formula above. The all-missing
    column remains missing in every month; the one-return January window is also missing. The
    overlapping February window includes that boundary return, while the sufficiently observed
    all-zero March control equals zero.

    Args:
        convert: Ordinary or nullable floating fixture conversion.
    """
    prices = convert(_mixed_volatility_panel())
    original_prices = prices.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = compute_sampled_vols(
            prices=prices,
            freq_vol="ME",
            include_start_date=True,
            include_end_date=True,
        )

    assert isinstance(actual, pd.DataFrame)
    pd.testing.assert_frame_equal(actual, _expected_mixed_vols(), rtol=1.0e-12, atol=1.0e-12)
    pd.testing.assert_frame_equal(prices, original_prices)


# =============================================================================
# Missing volatility-regime benchmarks
# =============================================================================


@pytest.mark.parametrize("convert", _DTYPE_CONVERTERS, ids=("ordinary", "nullable"))
def test_benchmark_vols_quantiles_regime_rejects_all_missing_benchmark(
    convert: Callable[[pd.DataFrame], pd.DataFrame],
) -> None:
    """Reject an unusable benchmark descriptively before quantile classification.

    Args:
        convert: Ordinary or nullable floating fixture conversion.
    """
    benchmark = convert(_all_missing_panel())
    asset_prices = pd.Series(
        100.0 * np.cumprod(1.0 + np.full(len(_ALL_MISSING_DATES), 0.001)),
        index=_ALL_MISSING_DATES,
        name=_RAGGED_ASSET,
    )
    prices = benchmark.join(asset_prices)
    original_prices = prices.copy()
    classifier = cast(
        _VolatilityClassifierProtocol,
        BenchmarkVolsQuantilesRegime(freq="ME", q=_NUM_BUCKETS),
    )
    expected_message = (
        "Volatility regime benchmark 'Benchmark' has no finite volatility observations for q=4."
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError) as exc_info:
            classifier.compute_sampled_returns_with_regime_id(
                prices=prices,
                benchmark=_BENCHMARK,
                include_start_date=True,
                include_end_date=True,
            )

    assert str(exc_info.value) == expected_message
    pd.testing.assert_frame_equal(prices, original_prices)


@pytest.mark.parametrize("convert", _DTYPE_CONVERTERS, ids=("ordinary", "nullable"))
def test_benchmark_vols_quantiles_regime_preserves_ragged_usable_benchmark(
    convert: Callable[[pd.DataFrame], pd.DataFrame],
) -> None:
    """Ignore unavailable early windows while classifying every later finite window.

    The initial sampled return and first three monthly volatility windows remain unclassified.
    The 33 later windows occupy the independently ranked groups of 9, 8, 8, and 8 without warnings
    or changes to labels, colors, column order, or caller data.

    Args:
        convert: Ordinary or nullable floating fixture conversion.
    """
    prices = convert(_ragged_usable_panel())
    original_prices = prices.copy()
    classifier = cast(
        _VolatilityClassifierProtocol,
        BenchmarkVolsQuantilesRegime(freq="ME", q=_NUM_BUCKETS),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        classified = classifier.compute_sampled_returns_with_regime_id(
            prices=prices,
            benchmark=_BENCHMARK,
            include_start_date=True,
            include_end_date=True,
        )

    actual_regimes = classified["regime"]
    assert isinstance(actual_regimes, pd.Series)
    assert classified.columns.tolist() == [_BENCHMARK, _RAGGED_ASSET, "regime"]
    assert actual_regimes.isna().tolist() == [True] * 4 + [False] * 33
    assert actual_regimes.value_counts(sort=False).to_dict() == dict(
        zip(_EXPECTED_REGIME_IDS, _EXPECTED_REGIME_COUNTS)
    )
    assert classifier.get_regime_ids_colors() == dict(
        zip(_EXPECTED_REGIME_IDS, _EXPECTED_REGIME_COLORS)
    )
    pd.testing.assert_frame_equal(prices, original_prices)

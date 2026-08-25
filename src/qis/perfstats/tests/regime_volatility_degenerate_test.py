"""Regression coverage for invalid and unattainable volatility-quantile regimes.

``BenchmarkVolsQuantilesRegime`` promises one ordered regime and color per requested volatility
bucket. A request must first provide a positive integer bucket count; that contract cannot then be
satisfied when repeated quantile edges, tied volatility samples, or an insufficient sample leave
one or more buckets empty. These tests construct each boundary through public price inputs and
require descriptive failure contracts instead of either a bare pandas exception or apparently
valid metadata containing unobserved regimes.

Expected bucket counts follow directly from the deterministic monthly rank structures: constant
prices form no quantile intervals, the tied fixture occupies three quartile bands, two monthly
samples occupy two bands, and the healthy control has 36 strictly increasing volatility samples
that divide into four groups of nine. No QIS classification helper constructs these references.
"""

from typing import Any, Callable, Dict, Protocol, cast
import warnings

import numpy as np
import pandas as pd
import pytest

# qis
from qis.perfstats.regime_classifier import BenchmarkVolsQuantilesRegime


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
_ASSET = "Ragged Asset"
_NUM_BUCKETS = 4

_HEALTHY_DATES = pd.bdate_range("2022-01-03", "2024-12-31")
_HEALTHY_REGIME_IDS = (
    "Benchmark vol<4%",
    "Benchmark vol=(4%, 6%]",
    "Benchmark vol=(6%, 8%]",
    "Benchmark vol>8%",
)
_HEALTHY_REGIME_COLORS = ("#a50026", "#fdbf6f", "#b7e075", "#006837")


def _price_panel(dates: pd.DatetimeIndex, benchmark_returns: np.ndarray) -> pd.DataFrame:
    """Create benchmark prices and a ragged dependent asset from known returns.

    Args:
        dates: Business-day observation dates matching ``benchmark_returns``.
        benchmark_returns: Deterministic simple returns used to construct prices.

    Returns:
        Two-column price panel with a complete benchmark and late-starting dependent asset.
    """
    benchmark_prices = 100.0 * np.cumprod(1.0 + benchmark_returns)
    asset_prices = 80.0 * np.cumprod(1.0 + 0.5 * benchmark_returns)
    ragged_observations = min(40, len(asset_prices) // 3)
    asset_prices[:ragged_observations] = np.nan
    return pd.DataFrame(
        {
            _BENCHMARK: benchmark_prices,
            _ASSET: asset_prices,
        },
        index=dates,
    )


def _constant_volatility_prices() -> pd.DataFrame:
    """Create 24 monthly samples whose realized volatility is exactly zero.

    Returns:
        Constant benchmark and ragged dependent-asset prices over two years.
    """
    dates = pd.bdate_range("2022-01-03", "2023-12-29")
    return _price_panel(dates, np.zeros(len(dates), dtype=float))


def _insufficient_volatility_prices() -> pd.DataFrame:
    """Create only two distinct monthly volatility observations.

    Returns:
        Two-month price panel, which cannot occupy four requested quantile regimes.
    """
    dates = pd.bdate_range("2022-01-03", "2022-02-28")
    month_numbers = np.asarray([date.month for date in dates], dtype=float)
    amplitudes = 0.001 + 0.00015 * month_numbers
    signs = np.where(np.arange(len(dates)) % 2 == 0, 1.0, -1.0)
    return _price_panel(dates, amplitudes * signs)


def _tied_volatility_prices() -> pd.DataFrame:
    """Create quartile edges that publish one unobserved regime on accepted main.

    The first six months have zero returns. The final six alternate one-percent returns; their
    small month-length differences create six positive realized-volatility values. Direct rank
    counting places six observations in the first quartile, none in the second, and three in each
    upper quartile, so only three of four requested bands are occupied.

    Returns:
        One-year price panel with tied lower-half volatility samples.
    """
    dates = pd.bdate_range("2022-01-03", "2022-12-30")
    month_numbers = np.asarray([date.month for date in dates])
    signs = np.where(np.arange(len(dates)) % 2 == 0, 1.0, -1.0)
    benchmark_returns = np.where(month_numbers <= 6, 0.0, 0.01 * signs)
    return _price_panel(dates, benchmark_returns)


def _healthy_volatility_prices() -> pd.DataFrame:
    """Create 36 strictly increasing monthly volatility samples.

    Alternating daily signs keep monthly means near zero while the absolute amplitude increases
    by 1.5 basis points each month. Direct rank counting therefore assigns nine samples to each
    quartile. The dependent asset begins late so its missing history cannot affect the benchmark
    regime partition.

    Returns:
        Three-year benchmark and ragged dependent-asset price panel.
    """
    month_numbers = np.asarray(
        [(date.year - 2022) * 12 + date.month for date in _HEALTHY_DATES],
        dtype=float,
    )
    amplitudes = 0.001 + 0.00015 * month_numbers
    signs = np.where(np.arange(len(_HEALTHY_DATES)) % 2 == 0, 1.0, -1.0)
    return _price_panel(_HEALTHY_DATES, amplitudes * signs)


# =============================================================================
# Bucket-count request validation
# =============================================================================


@pytest.mark.parametrize(
    "invalid_q",
    (
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(True, id="true"),
        pytest.param(False, id="false"),
        pytest.param(np.bool_(True), id="numpy-boolean"),
        pytest.param(1.5, id="non-integral-float"),
        pytest.param(4.0, id="integral-float"),
        pytest.param("4", id="string"),
        pytest.param(None, id="none"),
    ),
)
def test_volatility_quantiles_reject_invalid_bucket_counts(invalid_q: object) -> None:
    """Reject values that cannot unambiguously specify a positive bucket count.

    Args:
        invalid_q: Non-positive, boolean, or non-integral constructor request.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError) as exc_info:
            BenchmarkVolsQuantilesRegime(q=cast(Any, invalid_q))

    assert str(exc_info.value) == f"q must be a positive integer, got {invalid_q!r}"


@pytest.mark.parametrize(
    ("q", "expected_counts", "expected_colors"),
    (
        pytest.param(
            1,
            {"Benchmark vol<inf%": 36},
            {"Benchmark vol<inf%": "#006837"},
            id="one-bucket",
        ),
        pytest.param(
            np.int64(2),
            {"Benchmark vol<6%": 18, "Benchmark vol>6%": 18},
            {"Benchmark vol<6%": "#a50026", "Benchmark vol>6%": "#006837"},
            id="numpy-integer",
        ),
    ),
)
def test_volatility_quantiles_preserve_valid_integer_bucket_counts(
    q: object, expected_counts: Dict[str, int], expected_colors: Dict[str, str]
) -> None:
    """Accept positive integer requests and retain exact healthy classifications.

    Args:
        q: Python or NumPy positive integer bucket count.
        expected_counts: Independently ranked observations in each expected regime.
        expected_colors: Exact ordered regime-to-color mapping for the request.
    """
    prices = _healthy_volatility_prices()
    original_prices = prices.copy(deep=True)
    classifier = cast(
        _VolatilityClassifierProtocol,
        BenchmarkVolsQuantilesRegime(freq="ME", q=cast(Any, q)),
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
    assert actual_regimes.value_counts(sort=False).to_dict() == expected_counts
    assert classifier.get_regime_ids_colors() == expected_colors
    assert actual_regimes.isna().tolist() == [True] + [False] * 36
    pd.testing.assert_frame_equal(prices, original_prices)


# =============================================================================
# Degenerate quantile-band diagnostics
# =============================================================================


@pytest.mark.parametrize(
    ("prices_factory", "expected_nonempty_bands"),
    (
        pytest.param(_constant_volatility_prices, 0, id="constant"),
        pytest.param(_tied_volatility_prices, 3, id="tied"),
        pytest.param(_insufficient_volatility_prices, 2, id="insufficient"),
    ),
)
def test_volatility_quantiles_reject_unattainable_bands(
    prices_factory: Callable[[], pd.DataFrame], expected_nonempty_bands: int
) -> None:
    """Reject samples that cannot populate every requested volatility regime.

    Args:
        prices_factory: Deterministic public price-panel constructor.
        expected_nonempty_bands: Independently counted occupied quantile bands.
    """
    prices = prices_factory()
    original_prices = prices.copy(deep=True)
    classifier = cast(
        _VolatilityClassifierProtocol,
        BenchmarkVolsQuantilesRegime(freq="ME", q=_NUM_BUCKETS),
    )
    expected_message = (
        f"Volatility regime benchmark '{_BENCHMARK}' is degenerate for q={_NUM_BUCKETS}: "
        f"only {expected_nonempty_bands} of {_NUM_BUCKETS} quantile bands are non-empty"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=expected_message) as exc_info:
            classifier.compute_sampled_returns_with_regime_id(
                prices=prices,
                benchmark=_BENCHMARK,
                include_start_date=True,
                include_end_date=True,
            )

    assert "edges=[" in str(exc_info.value)
    assert "Use fewer buckets or a longer or more variable benchmark history." in str(
        exc_info.value
    )
    pd.testing.assert_frame_equal(prices, original_prices)


# =============================================================================
# Healthy classification preservation
# =============================================================================


def test_volatility_quantiles_preserve_healthy_ragged_panel() -> None:
    """Preserve four complete regimes when every requested band is attainable.

    The 36 benchmark volatility samples occupy four groups of nine. The first sampled return and
    the ragged asset's early history remain missing, while benchmark classification, ordered IDs,
    exact colors, labels, and caller ownership remain unchanged.
    """
    prices = _healthy_volatility_prices()
    original_prices = prices.copy(deep=True)
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
    assert classified.columns.tolist() == [_BENCHMARK, _ASSET, "regime"]
    assert actual_regimes.isna().tolist() == [True] + [False] * 36
    assert actual_regimes.value_counts(sort=False).to_dict() == {
        regime_id: 9 for regime_id in _HEALTHY_REGIME_IDS
    }
    assert classifier.get_regime_ids_colors() == dict(
        zip(_HEALTHY_REGIME_IDS, _HEALTHY_REGIME_COLORS)
    )
    assert classified[_ASSET].isna().sum() > classified[_BENCHMARK].isna().sum()
    pd.testing.assert_frame_equal(prices, original_prices)

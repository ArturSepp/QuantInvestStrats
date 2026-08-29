"""Regression coverage for the complete two-state benchmark-sign contract.

``BenchmarkReturnsPositiveNegativeRegime`` advertises exactly two ordered regimes: the first for
negative benchmark returns and the second for zero or positive returns. Both regimes must remain
in the categorical schema even when one or neither is observed, because downstream aggregation
requests the advertised order. An explicit custom mapping must likewise contain exactly two
ordered name/color entries so classification metadata cannot diverge from the two sign states.

The deterministic five-date panels below contain a complete asset, a late-starting ragged asset,
and positive-only, negative-only, mixed, or all-missing benchmarks. Expected returns, labels,
conditional means, contributions, and frequencies are specified directly from the price paths;
no second QIS classification or aggregation path constructs the references. The matrix also
covers ordinary and nullable all-missing benchmarks, zero as positive, warning behavior,
Series/DataFrame consistency, custom metadata, and caller ownership.
"""

from dataclasses import dataclass
from typing import Dict, Protocol, cast
import warnings

import numpy as np
import pandas as pd
import pytest

# qis
from qis.perfstats.regime_classifier import (
    BenchmarkReturnsPositiveNegativeRegime,
    compute_regime_avg,
)


class _SignClassifierProtocol(Protocol):
    """Typed test-side interface for the public sign classifier."""

    def compute_sampled_returns_with_regime_id(
        self,
        *,
        prices: pd.DataFrame | pd.Series,
        benchmark: str,
        include_start_date: bool,
        include_end_date: bool,
    ) -> pd.DataFrame:
        """Return sampled returns with their benchmark-sign classifications."""
        raise NotImplementedError

    def get_regime_ids_colors(self) -> Dict[str, str]:
        """Return the ordered regime-ID-to-color mapping."""
        raise NotImplementedError


# =============================================================================
# Shared deterministic cases and independent references
# =============================================================================

_DATES = pd.date_range("2024-01-01", periods=5, freq="D")

_COMPLETE = "Complete Asset"
_RAGGED = "Ragged Asset"
_BENCHMARK = "Benchmark"
_REGIME = "regime"

_REGIME_IDS = ("Negative", "Positive")
_DEFAULT_COLORS = {"Negative": "#FA8072", "Positive": "#006400"}
_CUSTOM_COLORS = {"Down": "#aa0000", "Up": "#00aa00"}

_COMPLETE_PRICES = (80.0, 80.8, 82.416, 84.88848, 88.2840192)
_COMPLETE_RETURNS = (np.nan, 0.01, 0.02, 0.03, 0.04)
_RAGGED_PRICES = (np.nan, np.nan, 50.0, 55.0, 49.5)
_RAGGED_RETURNS = (np.nan, np.nan, np.nan, 0.10, -0.10)

_ALL_MISSING = (np.nan, np.nan, np.nan, np.nan, np.nan)
_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class _SignCase:
    """Independent price, return, classification, and aggregation expectations."""

    benchmark_prices: tuple[float, ...]
    benchmark_returns: tuple[float, ...]
    regime_values: tuple[object, ...]
    frequencies: tuple[float, float]
    negative_means: tuple[float, float, float]
    positive_means: tuple[float, float, float]


_POSITIVE_ONLY = _SignCase(
    benchmark_prices=(100.0, 110.0, 121.0, 133.1, 146.41),
    benchmark_returns=(np.nan, 0.10, 0.10, 0.10, 0.10),
    regime_values=(np.nan, "Positive", "Positive", "Positive", "Positive"),
    frequencies=(0.0, 1.0),
    negative_means=(np.nan, np.nan, np.nan),
    positive_means=(0.025, 0.0, 0.10),
)
_NEGATIVE_ONLY = _SignCase(
    benchmark_prices=(100.0, 90.0, 81.0, 72.9, 65.61),
    benchmark_returns=(np.nan, -0.10, -0.10, -0.10, -0.10),
    regime_values=(np.nan, "Negative", "Negative", "Negative", "Negative"),
    frequencies=(1.0, 0.0),
    negative_means=(0.025, 0.0, -0.10),
    positive_means=(np.nan, np.nan, np.nan),
)
_MIXED_WITH_ZERO = _SignCase(
    benchmark_prices=(100.0, 90.0, 90.0, 99.0, 89.1),
    benchmark_returns=(np.nan, -0.10, 0.0, 0.10, -0.10),
    regime_values=(np.nan, "Negative", "Positive", "Positive", "Negative"),
    frequencies=(0.5, 0.5),
    negative_means=(0.025, -0.10, -0.10),
    positive_means=(0.025, 0.10, 0.05),
)
_ALL_MISSING_ORDINARY = _SignCase(
    benchmark_prices=_ALL_MISSING,
    benchmark_returns=_ALL_MISSING,
    regime_values=_ALL_MISSING,
    frequencies=(0.0, 0.0),
    negative_means=(np.nan, np.nan, np.nan),
    positive_means=(np.nan, np.nan, np.nan),
)


def _classifier(regime_ids_colors: Dict[str, str] | None = None) -> _SignClassifierProtocol:
    """Create the daily sign classifier behind a fully typed interface.

    Args:
        regime_ids_colors: Optional ordered custom regime-name and color mapping.

    Returns:
        Daily positive/negative benchmark-return classifier.
    """
    if regime_ids_colors is None:
        classifier = BenchmarkReturnsPositiveNegativeRegime(freq="D")
    else:
        classifier = BenchmarkReturnsPositiveNegativeRegime(
            freq="D",
            regime_ids_colors=regime_ids_colors,
        )
    return cast(_SignClassifierProtocol, classifier)


def _price_panel(case: _SignCase) -> pd.DataFrame:
    """Create one complete/ragged/benchmark price panel.

    Args:
        case: Deterministic benchmark price path.

    Returns:
        Three-column daily price panel for public classification.
    """
    benchmark = pd.Series(case.benchmark_prices, index=_DATES, name=_BENCHMARK, dtype=float)
    return pd.concat(
        (
            pd.Series(_COMPLETE_PRICES, index=_DATES, name=_COMPLETE),
            pd.Series(_RAGGED_PRICES, index=_DATES, name=_RAGGED),
            benchmark,
        ),
        axis=1,
    )


def _expected_returns(case: _SignCase) -> pd.DataFrame:
    """Construct the exact adjacent-price return reference.

    Args:
        case: Independently specified benchmark-return path.

    Returns:
        Expected complete, ragged, and benchmark simple returns.
    """
    return pd.DataFrame(
        {
            _COMPLETE: _COMPLETE_RETURNS,
            _RAGGED: _RAGGED_RETURNS,
            _BENCHMARK: case.benchmark_returns,
        },
        index=_DATES,
    )


def _expected_regimes(
    values: tuple[object, ...], regime_ids: tuple[str, str] = _REGIME_IDS
) -> pd.Series:
    """Construct the complete ordered categorical regime reference.

    Args:
        values: Expected labels and missing placements.
        regime_ids: Ordered negative/nonnegative regime IDs.

    Returns:
        Categorical Series retaining both configured categories.
    """
    return pd.Series(
        pd.Categorical(values, categories=regime_ids, ordered=True),
        index=_DATES,
        name=_REGIME,
    )


def _categorical_regime_index(regime_ids: tuple[str, str] = _REGIME_IDS) -> pd.CategoricalIndex:
    """Construct the ordered categorical index produced by grouping.

    Args:
        regime_ids: Ordered negative/nonnegative regime IDs.

    Returns:
        Complete two-state categorical regime index.
    """
    return pd.CategoricalIndex(
        regime_ids,
        categories=regime_ids,
        ordered=True,
        name=_REGIME,
    )


def _expected_aggregates(case: _SignCase) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Construct direct conditional-mean, contribution, and frequency references.

    Args:
        case: Explicit per-regime expectations for one benchmark path.

    Returns:
        Expected means, arithmetic contributions, and shared frequencies.
    """
    regime_index = _categorical_regime_index()
    frequencies = pd.Series(case.frequencies, index=regime_index, dtype=float)
    means = pd.DataFrame(
        {
            "Negative": case.negative_means,
            "Positive": case.positive_means,
        },
        index=pd.Index((_COMPLETE, _RAGGED, _BENCHMARK)),
    )
    means.columns = regime_index
    contributions = means.multiply(frequencies, axis="columns")
    return means, contributions, frequencies


# =============================================================================
# Complete sign-regime schema
# =============================================================================


@pytest.mark.parametrize(
    "case",
    (
        pytest.param(_POSITIVE_ONLY, id="positive-only"),
        pytest.param(_NEGATIVE_ONLY, id="negative-only"),
        pytest.param(_MIXED_WITH_ZERO, id="mixed-with-zero"),
        pytest.param(_ALL_MISSING_ORDINARY, id="all-missing-ordinary"),
    ),
)
def test_sign_regime_retains_both_categories_for_every_observed_state(case: _SignCase) -> None:
    """Retain both regimes through classification and mixed-panel aggregation.

    Args:
        case: Positive-only, negative-only, mixed, or all-missing benchmark case.

    The complete and ragged assets share the benchmark-defined frequencies while retaining their
    own available-observation means. Unobserved regimes remain present with zero frequency and
    undefined statistics; zero benchmark returns enter the second, nonnegative regime.
    """
    prices = _price_panel(case)
    original_prices = prices.copy(deep=True)
    classifier = _classifier()
    expected_means, expected_contributions, expected_frequencies = _expected_aggregates(case)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        classified = classifier.compute_sampled_returns_with_regime_id(
            prices=prices,
            benchmark=_BENCHMARK,
            include_start_date=True,
            include_end_date=True,
        )
        actual_means, actual_contributions, actual_frequencies = compute_regime_avg(
            sampled_returns_with_regime_id=classified,
            freq="YE",
            is_report_pa_returns=False,
            regime_ids=list(_REGIME_IDS),
        )

    pd.testing.assert_frame_equal(
        classified.drop(columns=_REGIME),
        _expected_returns(case),
        check_dtype=False,
        check_exact=False,
        rtol=0.0,
        atol=_TOLERANCE,
    )
    pd.testing.assert_series_equal(classified[_REGIME], _expected_regimes(case.regime_values))
    pd.testing.assert_frame_equal(
        actual_means,
        expected_means,
        check_dtype=False,
        check_exact=False,
        rtol=0.0,
        atol=_TOLERANCE,
    )
    pd.testing.assert_frame_equal(
        actual_contributions,
        expected_contributions,
        check_dtype=False,
        check_exact=False,
        rtol=0.0,
        atol=_TOLERANCE,
    )
    pd.testing.assert_series_equal(actual_frequencies, expected_frequencies)
    assert classifier.get_regime_ids_colors() == _DEFAULT_COLORS
    pd.testing.assert_frame_equal(prices, original_prices, check_exact=True)


def test_sign_regime_retains_both_categories_for_nullable_all_missing_series() -> None:
    """Retain both zero-frequency regimes for a nullable all-missing benchmark Series.

    A nullable benchmark exercises the accepted Series conversion path without depending on the
    separate nullable multi-column conversion work. With no classified dates, both advertised
    regime frequencies are zero and both conditional benchmark means remain undefined.
    """
    prices = pd.Series(
        [pd.NA] * len(_DATES),
        index=_DATES,
        name=_BENCHMARK,
        dtype=pd.Float64Dtype(),
    )
    original_prices = prices.copy(deep=True)
    classifier = _classifier()
    regime_index = _categorical_regime_index()
    expected_means = pd.DataFrame(
        ((np.nan, np.nan),),
        index=pd.Index((_BENCHMARK,)),
        columns=regime_index,
    ).astype(pd.Float64Dtype())
    expected_frequencies = pd.Series((0.0, 0.0), index=regime_index, dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        classified = classifier.compute_sampled_returns_with_regime_id(
            prices=prices,
            benchmark=_BENCHMARK,
            include_start_date=True,
            include_end_date=True,
        )
        actual_means, actual_contributions, actual_frequencies = compute_regime_avg(
            sampled_returns_with_regime_id=classified,
            freq="YE",
            is_report_pa_returns=False,
            regime_ids=list(_REGIME_IDS),
        )

    pd.testing.assert_series_equal(classified[_REGIME], _expected_regimes(_ALL_MISSING))
    pd.testing.assert_frame_equal(actual_means, expected_means, check_dtype=False)
    pd.testing.assert_frame_equal(actual_contributions, expected_means, check_dtype=False)
    pd.testing.assert_series_equal(actual_frequencies, expected_frequencies)
    pd.testing.assert_series_equal(prices, original_prices, check_exact=True)


# =============================================================================
# Mapping cardinality and custom metadata
# =============================================================================


@pytest.mark.parametrize(
    "mapping",
    (
        pytest.param({}, id="empty"),
        pytest.param({"Down": "#aa0000"}, id="one-entry"),
        pytest.param(
            {"Down": "#aa0000", "Flat": "#aaaa00", "Up": "#00aa00"},
            id="three-entries",
        ),
    ),
)
def test_sign_regime_rejects_mapping_without_exactly_two_entries(mapping: Dict[str, str]) -> None:
    """Reject explicit metadata that cannot describe exactly two sign states.

    Args:
        mapping: Empty, one-entry, or three-entry ordered mapping.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError) as exc_info:
            BenchmarkReturnsPositiveNegativeRegime(regime_ids_colors=mapping)

    assert str(exc_info.value) == (
        "Positive/negative regimes require exactly 2 regime labels and colors; "
        f"received {len(mapping)}"
    )


def test_sign_regime_preserves_valid_custom_mapping_for_series_and_frame() -> None:
    """Use both custom entries in order for equivalent Series and DataFrame inputs.

    The mixed benchmark returns are negative, zero, positive, and negative. The first supplied
    ID therefore appears twice, while the second appears for zero and positive returns. Both
    outputs retain the complete ordered categorical schema and leave the first return missing.
    """
    prices = cast(pd.Series, _price_panel(_MIXED_WITH_ZERO)[_BENCHMARK])
    price_frame = prices.to_frame()
    original_prices = prices.copy(deep=True)
    original_frame = price_frame.copy(deep=True)
    expected_regimes = _expected_regimes(
        (np.nan, "Down", "Up", "Up", "Down"),
        regime_ids=("Down", "Up"),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        series_classifier = _classifier(_CUSTOM_COLORS)
        series_result = series_classifier.compute_sampled_returns_with_regime_id(
            prices=prices,
            benchmark=_BENCHMARK,
            include_start_date=True,
            include_end_date=True,
        )
        frame_classifier = _classifier(_CUSTOM_COLORS)
        frame_result = frame_classifier.compute_sampled_returns_with_regime_id(
            prices=price_frame,
            benchmark=_BENCHMARK,
            include_start_date=True,
            include_end_date=True,
        )

    pd.testing.assert_frame_equal(series_result, frame_result)
    pd.testing.assert_series_equal(series_result[_REGIME], expected_regimes)
    assert series_classifier.get_regime_ids_colors() == _CUSTOM_COLORS
    assert frame_classifier.get_regime_ids_colors() == _CUSTOM_COLORS
    pd.testing.assert_series_equal(prices, original_prices, check_exact=True)
    pd.testing.assert_frame_equal(price_frame, original_frame, check_exact=True)

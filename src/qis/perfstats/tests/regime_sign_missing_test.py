"""Regression tests for missing benchmark returns in sign-based regimes.

``BenchmarkReturnsPositiveNegativeRegime`` partitions dates by the sign of the benchmark return:
finite negative returns are ``Negative``, while zero and finite positive returns are ``Positive``.
A missing return has no sign and therefore must retain a missing regime ID rather than entering
either bucket.

The deterministic daily fixture contains leading and interior missing benchmark prices. Direct
adjacent-price arithmetic produces four missing returns, one negative return, one positive return,
and one zero return. Expected values are constructed without another QIS return or classification
path, and the tests also pin shared regime frequencies, Series/DataFrame consistency, labels,
column order, and caller ownership.
"""

from typing import Protocol, cast

import numpy as np
import pandas as pd

from qis.perfstats.regime_classifier import (
    BenchmarkReturnsPositiveNegativeRegime,
    compute_mean_freq_regimes,
)


# =============================================================================
# Shared deterministic fixture and independent reference
# =============================================================================

_DATES = pd.date_range("2024-01-01", periods=7, freq="D")

_ASSET_NAME = "Strategy"
_BENCHMARK_NAME = "Reference Index"
_REGIME_COLUMN = "regime"

_ASSET_PRICES = (50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0)
_BENCHMARK_PRICES = (np.nan, 100.0, 90.0, np.nan, 99.0, 108.9, 108.9)

_TOLERANCE = 1.0e-12


class _SignClassifierProtocol(Protocol):
    """Typed test-side interface for the public classifier method."""

    def compute_sampled_returns_with_regime_id(
            self,
            *,
            prices: pd.DataFrame | pd.Series,
            benchmark: str,
            include_start_date: bool,
            include_end_date: bool,
    ) -> pd.DataFrame:
        """Return classified daily returns for the supplied prices."""
        raise NotImplementedError


def _price_panel() -> pd.DataFrame:
    """Create the daily two-column price panel used by the regression.

    Returns:
        Price panel with a complete asset and a benchmark containing leading and interior gaps.
    """
    return pd.DataFrame(
        {
            _ASSET_NAME: _ASSET_PRICES,
            _BENCHMARK_NAME: _BENCHMARK_PRICES,
        },
        index=_DATES,
    )


def _expected_classified_returns() -> pd.DataFrame:
    """Construct expected returns and regimes directly from adjacent prices.

    The interior missing benchmark price makes both adjacent return intervals undefined. The
    remaining finite benchmark returns are ``90 / 100 - 1 = -10%``,
    ``108.9 / 99 - 1 = 10%``, and ``108.9 / 108.9 - 1 = 0%``.

    Returns:
        Independently calculated return panel with the expected sign-regime column.
    """
    return pd.DataFrame(
        {
            _ASSET_NAME: (
                np.nan,
                51.0 / 50.0 - 1.0,
                52.0 / 51.0 - 1.0,
                53.0 / 52.0 - 1.0,
                54.0 / 53.0 - 1.0,
                55.0 / 54.0 - 1.0,
                56.0 / 55.0 - 1.0,
            ),
            _BENCHMARK_NAME: (
                np.nan,
                np.nan,
                90.0 / 100.0 - 1.0,
                np.nan,
                np.nan,
                108.9 / 99.0 - 1.0,
                108.9 / 108.9 - 1.0,
            ),
            _REGIME_COLUMN: (
                np.nan,
                np.nan,
                "Negative",
                np.nan,
                np.nan,
                "Positive",
                "Positive",
            ),
        },
        index=_DATES,
    )


def _classifier(regime_ids_colors: dict[str, str] | None = None) -> _SignClassifierProtocol:
    """Create the daily sign classifier behind a fully typed test interface.

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


# =============================================================================
# Missing-regime classification and shared frequencies
# =============================================================================

def test_positive_negative_regime_preserves_missing_benchmark_returns() -> None:
    """Leave missing benchmark returns unclassified and exclude them from frequencies.

    Four of seven benchmark returns are missing and therefore have no sign. Of the three
    classified dates, one is negative and two are nonnegative, so the independently counted
    frequencies are exactly ``Negative = 1/3`` and ``Positive = 2/3``. The complete output,
    including asset returns, labels, column order, and the caller-owned input, is asserted.
    """
    prices = _price_panel()
    original_prices = prices.copy(deep=True)
    expected = _expected_classified_returns()
    expected_frequencies = pd.Series(
        (1.0 / 3.0, 2.0 / 3.0),
        index=pd.Index(("Negative", "Positive"), name=_REGIME_COLUMN),
    )

    actual = _classifier().compute_sampled_returns_with_regime_id(
        prices=prices,
        benchmark=_BENCHMARK_NAME,
        include_start_date=True,
        include_end_date=True,
    )
    _, actual_frequencies = compute_mean_freq_regimes(actual)

    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_dtype=False,
        check_exact=False,
        rtol=0.0,
        atol=_TOLERANCE,
    )
    pd.testing.assert_series_equal(
        actual_frequencies,
        expected_frequencies,
        check_exact=False,
        rtol=0.0,
        atol=_TOLERANCE,
    )
    pd.testing.assert_frame_equal(prices, original_prices, check_exact=True)


def test_positive_negative_regime_preserves_custom_names_and_missing_mask() -> None:
    """Apply custom sign labels without assigning missing returns to either regime.

    Custom IDs must replace the finite default labels in their supplied order while the four
    independently identified missing benchmark returns remain unclassified. The complete return
    panel and caller-owned prices are asserted so custom metadata cannot alter numerical values.
    """
    prices = _price_panel()
    original_prices = prices.copy(deep=True)
    expected = _expected_classified_returns()
    expected[_REGIME_COLUMN] = (
        np.nan,
        np.nan,
        "Down",
        np.nan,
        np.nan,
        "Up",
        "Up",
    )

    actual = _classifier(
        regime_ids_colors={"Down": "salmon", "Up": "darkgreen"},
    ).compute_sampled_returns_with_regime_id(
        prices=prices,
        benchmark=_BENCHMARK_NAME,
        include_start_date=True,
        include_end_date=True,
    )

    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_dtype=False,
        check_exact=False,
        rtol=0.0,
        atol=_TOLERANCE,
    )
    pd.testing.assert_frame_equal(prices, original_prices, check_exact=True)


# =============================================================================
# Series and DataFrame consistency
# =============================================================================

def test_positive_negative_regime_preserves_series_shape_name_and_input() -> None:
    """Match one-column DataFrame behavior for a named benchmark Series.

    The Series path must return the benchmark and regime columns on the original daily index,
    preserve every missing regime from the independent reference, and leave both equivalent
    caller inputs unchanged.
    """
    benchmark = _price_panel()[_BENCHMARK_NAME]
    benchmark_frame = benchmark.to_frame()
    original_benchmark = benchmark.copy(deep=True)
    original_frame = benchmark_frame.copy(deep=True)
    expected = _expected_classified_returns().loc[:, [_BENCHMARK_NAME, _REGIME_COLUMN]]

    series_result = _classifier().compute_sampled_returns_with_regime_id(
        prices=benchmark,
        benchmark=_BENCHMARK_NAME,
        include_start_date=True,
        include_end_date=True,
    )
    frame_result = _classifier().compute_sampled_returns_with_regime_id(
        prices=benchmark_frame,
        benchmark=_BENCHMARK_NAME,
        include_start_date=True,
        include_end_date=True,
    )

    pd.testing.assert_frame_equal(
        series_result,
        expected,
        check_dtype=False,
        check_exact=False,
        rtol=0.0,
        atol=_TOLERANCE,
    )
    pd.testing.assert_frame_equal(series_result, frame_result)
    pd.testing.assert_series_equal(benchmark, original_benchmark, check_exact=True)
    pd.testing.assert_frame_equal(benchmark_frame, original_frame, check_exact=True)


def test_positive_negative_regime_supports_nullable_benchmark_missing_values() -> None:
    """Classify observed nullable returns without evaluating ``pd.NA`` as a boolean."""
    benchmark = pd.Series(
        _BENCHMARK_PRICES,
        index=_DATES,
        name=_BENCHMARK_NAME,
        dtype="Float64",
    )
    original_benchmark = benchmark.copy(deep=True)
    expected_regimes = pd.Series(
        (np.nan, np.nan, "Negative", np.nan, np.nan, "Positive", "Positive"),
        index=_DATES,
        name=_REGIME_COLUMN,
        dtype=object,
    )

    actual = _classifier().compute_sampled_returns_with_regime_id(
        prices=benchmark,
        benchmark=_BENCHMARK_NAME,
        include_start_date=True,
        include_end_date=True,
    )

    pd.testing.assert_series_equal(actual[_REGIME_COLUMN], expected_regimes)
    pd.testing.assert_series_equal(benchmark, original_benchmark, check_exact=True)

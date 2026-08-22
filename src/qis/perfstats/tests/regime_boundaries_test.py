"""Regression coverage for shared regime frequencies with ragged asset histories.

A regime is determined by the benchmark observation for a date, not by whether any particular
asset happens to have a return on that date. Consequently, every asset in a panel must use the
same regime frequencies, and those frequencies must not change when asset columns are reordered.

The deterministic fixtures below exercise both the public quantile-classifier path and the core
aggregation functions. Expected frequencies are counted directly from known regime labels;
conditional means and arithmetic contributions are calculated explicitly rather than through a
second QIS performance path. An unused categorical regime and an empty sample pin the output
labels, order, and pandas return type at the boundaries.
"""

from typing import Protocol, cast

import numpy as np
import pandas as pd

# qis
from qis.perfstats.regime_classifier import (
    BenchmarkReturnsQuantilesRegime,
    compute_mean_freq_regimes,
    compute_regime_avg,
)


class _QuantileClassifierProtocol(Protocol):
    """Typed test-side interface for the classifier method exercised below."""

    def compute_sampled_returns_with_regime_id(
            self,
            *,
            prices: pd.DataFrame,
            benchmark: str,
            include_start_date: bool,
            include_end_date: bool) -> pd.DataFrame:
        """Return classified periodic returns for the supplied price panel."""
        ...


# =============================================================================
# Shared deterministic fixtures
# =============================================================================

_CLASSIFICATION_DATES = pd.date_range('2024-01-01', periods=9, freq='D')

_BENCHMARK_RETURNS = np.array((-0.08, -0.06, -0.04, -0.02, 0.02, 0.04, 0.06, 0.08))
_DIRECT_REGIME_IDS = ('Bear', 'Normal', 'Bull', 'Crisis')
_QUANTILE_EDGES = np.array((0.0, 0.25, 0.75, 1.0))
_QUANTILE_REGIME_IDS = ('Bear', 'Normal', 'Bull')

_TOLERANCE = 1e-12


def _categorical_regime_index(regime_ids: tuple[str, ...]) -> pd.CategoricalIndex:
    """Create the ordered index produced by categorical regime grouping.

    Args:
        regime_ids: Regime labels in their required reporting order.

    Returns:
        Ordered categorical index named for the regime column.
    """
    return pd.CategoricalIndex(
        regime_ids,
        categories=regime_ids,
        ordered=True,
        name='regime',
    )


def _direct_ragged_regime_panel() -> pd.DataFrame:
    """Create returns with benchmark-defined regimes and asset-specific missingness.

    The six observations contain two Bear, two Normal, and two Bull dates. ``Ragged Asset`` has
    no return on the second Normal date or either Bull date, while ``Complete Asset`` has all six
    returns. ``Crisis`` is an ordered but unobserved category.

    Returns:
        Return panel with an ordered categorical regime column.
    """
    regimes = pd.Categorical(
        ('Bear', 'Bear', 'Normal', 'Normal', 'Bull', 'Bull'),
        categories=_DIRECT_REGIME_IDS,
        ordered=True,
    )
    return pd.DataFrame(
        {
            'Ragged Asset': (0.10, 0.20, 0.30, np.nan, np.nan, np.nan),
            'Complete Asset': (0.01, 0.03, 0.02, 0.04, -0.01, 0.05),
            'regime': regimes,
        }
    )


def _quantile_prices(column_order: tuple[str, str]) -> pd.DataFrame:
    """Create daily prices with a complete benchmark and late-starting asset.

    Args:
        column_order: Requested order of ``Ragged Asset`` and ``Benchmark``.

    Returns:
        Price panel in the requested order. The benchmark has eight known returns, while the
        asset begins only for the final five price observations.
    """
    benchmark_prices = pd.Series(
        np.concatenate(
            (
                np.array((100.0,)),
                100.0 * np.cumprod(1.0 + _BENCHMARK_RETURNS),
            )
        ),
        index=_CLASSIFICATION_DATES,
        name='Benchmark',
    )
    ragged_prices = pd.Series(
        (np.nan, np.nan, np.nan, np.nan, 50.0, 51.0, 52.0, 53.0, 54.0),
        index=_CLASSIFICATION_DATES,
        name='Ragged Asset',
    )
    prices = pd.concat((ragged_prices, benchmark_prices), axis=1)
    return prices.loc[:, list(column_order)]


def _regime_frequency_series(
        values: tuple[float, ...],
        regime_ids: tuple[str, ...]) -> pd.Series:
    """Create a labeled expected frequency Series.

    Args:
        values: Independently counted frequency for each regime.
        regime_ids: Regime labels in the corresponding reporting order.

    Returns:
        Float Series with the categorical regime index used by grouped results.
    """
    return pd.Series(
        values,
        index=_categorical_regime_index(regime_ids),
        dtype=float,
    )


# =============================================================================
# Public quantile-classifier frequency contract
# =============================================================================

def test_compute_mean_freq_regimes_is_independent_of_asset_column_order() -> None:
    """Count benchmark regimes independently of the first asset's missing history.

    The benchmark returns have two observations in the bottom quartile, four in the middle 50%,
    and two in the top quartile. Their independently counted frequencies are therefore
    ``[0.25, 0.50, 0.25]``. The ragged asset has no return in the Bear bucket, so counting its
    non-missing values would incorrectly produce ``[0.0, 0.5, 0.5]`` when it is the first column.

    Running both column orders proves that the shared benchmark partition—not asset placement or
    missingness—governs the result. The classifier must also leave each caller-owned price panel
    unchanged.
    """
    classifier = cast(
        _QuantileClassifierProtocol,
        BenchmarkReturnsQuantilesRegime(freq='D', q=_QUANTILE_EDGES),
    )
    expected_frequencies = _regime_frequency_series(
        (0.25, 0.50, 0.25),
        _QUANTILE_REGIME_IDS,
    )

    for column_order in (
            ('Ragged Asset', 'Benchmark'),
            ('Benchmark', 'Ragged Asset')):
        prices = _quantile_prices(column_order)
        original_prices = prices.copy(deep=True)

        classified_returns = classifier.compute_sampled_returns_with_regime_id(
            prices=prices,
            benchmark='Benchmark',
            include_start_date=False,
            include_end_date=False,
        )
        _, actual_frequencies = compute_mean_freq_regimes(classified_returns)

        pd.testing.assert_series_equal(
            actual_frequencies,
            expected_frequencies,
            check_names=False,
        )
        pd.testing.assert_frame_equal(prices, original_prices)


# =============================================================================
# Ragged conditional means and arithmetic contributions
# =============================================================================

def test_compute_regime_avg_uses_shared_frequencies_with_ragged_returns() -> None:
    """Weight every asset's conditional mean by the benchmark regime frequency.

    Each observed regime occupies two of six dates, so Bear, Normal, and Bull each receive weight
    ``1 / 3`` and the unused Crisis category receives zero. The ragged asset's Bear mean is 15%
    and Normal mean is 30%; the complete asset's means are 2%, 3%, and 2%. With annualization
    fixed at one by ``freq='YE'`` and geometric reporting disabled, the direct contributions are
    the corresponding means multiplied by ``1 / 3``.

    Missing conditional means remain missing even when a regime has zero frequency. Labels,
    order, and the caller-owned input are part of the regression contract.
    """
    sampled_returns = _direct_ragged_regime_panel()
    original_returns = sampled_returns.copy(deep=True)
    regime_columns = _categorical_regime_index(_DIRECT_REGIME_IDS)
    expected_frequencies = _regime_frequency_series(
        (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0),
        _DIRECT_REGIME_IDS,
    )
    expected_means = pd.DataFrame(
        {
            'Bear': (0.15, 0.02),
            'Normal': (0.30, 0.03),
            'Bull': (np.nan, 0.02),
            'Crisis': (np.nan, np.nan),
        },
        index=('Ragged Asset', 'Complete Asset'),
    )
    expected_means.columns = regime_columns
    expected_contributions = pd.DataFrame(
        {
            'Bear': (0.05, 0.02 / 3.0),
            'Normal': (0.10, 0.01),
            'Bull': (np.nan, 0.02 / 3.0),
            'Crisis': (np.nan, np.nan),
        },
        index=('Ragged Asset', 'Complete Asset'),
    )
    expected_contributions.columns = regime_columns

    actual_means, actual_contributions, actual_frequencies = compute_regime_avg(
        sampled_returns_with_regime_id=sampled_returns,
        freq='YE',
        is_report_pa_returns=False,
        regime_ids=list(_DIRECT_REGIME_IDS),
    )

    pd.testing.assert_frame_equal(
        actual_means,
        expected_means,
        check_exact=False,
        rtol=0.0,
        atol=_TOLERANCE,
    )
    pd.testing.assert_frame_equal(
        actual_contributions,
        expected_contributions,
        check_exact=False,
        rtol=0.0,
        atol=_TOLERANCE,
    )
    pd.testing.assert_series_equal(
        actual_frequencies,
        expected_frequencies,
        check_names=False,
    )
    pd.testing.assert_frame_equal(sampled_returns, original_returns)


# =============================================================================
# Empty-sample output contract
# =============================================================================

def test_compute_mean_freq_regimes_returns_labeled_zeros_for_empty_sample() -> None:
    """Preserve a labeled pandas frequency Series when no observations exist.

    An empty categorical input still defines the complete ordered regime schema. Its conditional
    means are missing for every regime and its independently counted frequencies are four zeros.
    Returning an unlabeled NumPy array would discard the regime names and violate the function's
    annotated pandas return type, so this boundary asserts the complete labeled output.
    """
    regime_index = _categorical_regime_index(_DIRECT_REGIME_IDS)
    sampled_returns = pd.DataFrame(
        {
            'Asset': pd.Series(dtype=float),
            'regime': pd.Series(
                pd.Categorical([], categories=_DIRECT_REGIME_IDS, ordered=True)
            ),
        }
    )
    original_returns = sampled_returns.copy(deep=True)
    expected_means = pd.DataFrame({'Asset': (np.nan,) * 4}, index=regime_index)
    expected_frequencies = _regime_frequency_series(
        (0.0, 0.0, 0.0, 0.0),
        _DIRECT_REGIME_IDS,
    )

    actual_means, actual_frequencies = compute_mean_freq_regimes(sampled_returns)

    assert isinstance(actual_frequencies, pd.Series)
    pd.testing.assert_frame_equal(actual_means, expected_means)
    pd.testing.assert_series_equal(
        actual_frequencies,
        expected_frequencies,
        check_names=False,
    )
    pd.testing.assert_frame_equal(sampled_returns, original_returns)

"""Regression coverage for quantile-regime bucket IDs, colors, and report columns.

Quantile classifiers expose their bucket metadata through ``RegimeClassifier`` so plotting and
conditional-performance consumers can use the same ordered regimes that classification created.
Every successfully formed bucket therefore needs exactly one ID and one valid color.

The return fixtures use eight independently specified daily returns, making four equal-count
quartiles and the expected missing first observation explicit. The volatility fixture increases
the daily return amplitude once per calendar month; its 36 classified monthly samples consequently
occupy four ordered rank groups of nine. Tests cover semantic three-bucket compatibility, generic
four-bucket labels, explicit mappings, volatility metadata, fresh report construction, labels,
missing placement, warnings, and caller ownership without using another QIS path as a reference.
"""

from typing import Dict, Protocol, Tuple, Union, cast
import warnings

import matplotlib.colors as mpl_colors
import numpy as np
import pandas as pd
import pytest

# qis
from qis.perfstats.config import PerfParams, RegimeData
from qis.perfstats.regime_classifier import (
    BenchmarkReturnsQuantilesRegime,
    BenchmarkVolsQuantilesRegime,
)


class _QuantileClassifierProtocol(Protocol):
    """Typed test-side interface shared by both quantile classifiers."""

    def class_data_to_colors(self, regime_data: pd.Series) -> pd.Series:
        """Map classified regime IDs to plotting colors."""
        raise NotImplementedError

    def compute_sampled_returns_with_regime_id(
            self,
            *,
            prices: Union[pd.DataFrame, pd.Series],
            benchmark: str,
            include_start_date: bool,
            include_end_date: bool) -> pd.DataFrame:
        """Return sampled returns with their ordered regime classification."""
        raise NotImplementedError

    def get_regime_ids(self) -> list[str]:
        """Return ordered regime IDs."""
        raise NotImplementedError

    def get_regime_ids_colors(self) -> Dict[str, str]:
        """Return the ordered regime-ID-to-color mapping."""
        raise NotImplementedError

    def get_regime_colors(self) -> list[Tuple[float, float, float, float]]:
        """Return the existing ordered RGBA regime colors."""
        raise NotImplementedError


class _VolatilityReportProtocol(_QuantileClassifierProtocol, Protocol):
    """Typed test-side interface for volatility-regime report construction."""

    def compute_regimes_pa_perf_table(
            self,
            *,
            prices: pd.DataFrame,
            benchmark: str,
            perf_params: PerfParams) -> Tuple[pd.DataFrame, Dict[RegimeData, pd.DataFrame]]:
        """Return the conditional performance table and its component tables."""
        raise NotImplementedError


# =============================================================================
# Shared deterministic fixtures
# =============================================================================

_RETURN_DATES = pd.date_range('2024-01-01', periods=9, freq='D')
_RETURN_VALUES = np.array((-0.08, -0.06, -0.04, -0.02, 0.02, 0.04, 0.06, 0.08))

_FOUR_QUANTILE_EDGES = np.array((0.0, 0.25, 0.50, 0.75, 1.0))
_FOUR_RETURN_COLORS = ('#a50026', '#fdbf6f', '#b7e075', '#006837')
_FOUR_RETURN_IDS = ('Q1', 'Q2', 'Q3', 'Q4')
_THREE_RETURN_IDS = ('Bear', 'Normal', 'Bull')

_VOLATILITY_DATES = pd.bdate_range('2022-01-03', '2024-12-31')
_VOLATILITY_IDS = (
    'Benchmark vol<4%',
    'Benchmark vol=(4%, 6%]',
    'Benchmark vol=(6%, 8%]',
    'Benchmark vol>8%',
)


def _return_prices() -> pd.Series:
    """Create prices from eight known daily simple returns.

    Returns:
        Benchmark price Series whose ordered returns form four equal-count quartiles.
    """
    return pd.Series(
        np.concatenate(
            (np.array((100.0,)), 100.0 * np.cumprod(1.0 + _RETURN_VALUES))
        ),
        index=_RETURN_DATES,
        name='Benchmark',
    )


def _volatility_prices() -> pd.DataFrame:
    """Create a two-asset panel with strictly increasing monthly volatility.

    Alternating daily return signs keep each monthly mean near zero, while the absolute amplitude
    increases by 1.5 basis points per month. The resulting 36 classified monthly volatility
    samples increase monotonically, so direct rank counting assigns nine samples to each quartile.

    Returns:
        Complete business-day prices for the benchmark and a half-amplitude asset.
    """
    month_numbers = np.array(
        [
            (date.year - 2022) * 12 + date.month
            for date in _VOLATILITY_DATES
        ],
        dtype=float,
    )
    amplitudes = 0.001 + 0.00015 * month_numbers
    signs = np.where(np.arange(len(_VOLATILITY_DATES)) % 2 == 0, 1.0, -1.0)
    benchmark_returns = amplitudes * signs
    return pd.DataFrame(
        {
            'Benchmark': 100.0 * np.cumprod(1.0 + benchmark_returns),
            'Asset': 80.0 * np.cumprod(1.0 + 0.5 * benchmark_returns),
        },
        index=_VOLATILITY_DATES,
    )


def _assert_four_return_regimes(
        classified: pd.DataFrame,
        classifier: _QuantileClassifierProtocol) -> None:
    """Assert the complete generic four-bucket return contract.

    Args:
        classified: Return panel produced from the deterministic return fixture.
        classifier: Classifier that produced the panel and its metadata.
    """
    expected_observed = ['Q1', 'Q1', 'Q2', 'Q2', 'Q3', 'Q3', 'Q4', 'Q4']
    actual_regimes = classified['regime']

    assert list(actual_regimes.cat.categories) == list(_FOUR_RETURN_IDS)
    assert actual_regimes.isna().tolist() == [True] + [False] * 8
    assert actual_regimes.dropna().tolist() == expected_observed
    assert classifier.get_regime_ids() == list(_FOUR_RETURN_IDS)
    assert classifier.get_regime_ids_colors() == dict(
        zip(_FOUR_RETURN_IDS, _FOUR_RETURN_COLORS)
    )


# =============================================================================
# Return-quantile cardinality and compatibility
# =============================================================================

def test_benchmark_returns_quantiles_regime_preserves_three_semantic_regimes() -> None:
    """Preserve the established Bear/Normal/Bull behavior for three buckets.

    Eight ordered observations divided into three equal-frequency buckets place three returns in
    Bear, two in Normal, and three in Bull. The first price has no return and remains unclassified.
    The established semantic labels, order, and colors are compatibility requirements.
    """
    prices = _return_prices()
    classifier = cast(
        _QuantileClassifierProtocol,
        BenchmarkReturnsQuantilesRegime(freq='D', q=3),
    )

    classified = classifier.compute_sampled_returns_with_regime_id(
        prices=prices,
        benchmark='Benchmark',
        include_start_date=False,
        include_end_date=False,
    )

    assert list(classified['regime'].cat.categories) == list(_THREE_RETURN_IDS)
    assert classified['regime'].value_counts(sort=False).tolist() == [3, 2, 3]
    assert classifier.get_regime_ids() == list(_THREE_RETURN_IDS)
    assert classifier.get_regime_ids_colors() == {
        'Bear': '#FA8072',
        'Normal': '#9ACD32',
        'Bull': '#006400',
    }


@pytest.mark.parametrize('as_frame', (False, True), ids=('series', 'dataframe'))
@pytest.mark.parametrize('q', (4, _FOUR_QUANTILE_EDGES), ids=('integer', 'explicit-edges'))
def test_benchmark_returns_quantiles_regime_generates_one_id_and_color_per_bucket(
        as_frame: bool,
        q: Union[int, np.ndarray]) -> None:
    """Generate four ordered generic labels and colors for four return buckets.

    The independently specified returns contain two observations per quartile. Both supported
    pandas input shapes and both public quantile specifications must return the same categorical
    labels, leave the missing first return unclassified, emit no warning, and preserve the input.

    Args:
        as_frame: Whether to supply the benchmark as a one-column DataFrame.
        q: Integer bucket count or the equivalent explicit quantile boundaries.
    """
    price_series = _return_prices()
    prices: Union[pd.Series, pd.DataFrame] = (
        price_series.to_frame() if as_frame else price_series
    )
    original_prices = prices.copy(deep=True)
    classifier = cast(
        _QuantileClassifierProtocol,
        BenchmarkReturnsQuantilesRegime(freq='D', q=q),
    )

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        classified = classifier.compute_sampled_returns_with_regime_id(
            prices=prices,
            benchmark='Benchmark',
            include_start_date=False,
            include_end_date=False,
        )

    _assert_four_return_regimes(classified, classifier)
    if isinstance(prices, pd.DataFrame):
        assert isinstance(original_prices, pd.DataFrame)
        pd.testing.assert_frame_equal(prices, original_prices)
    else:
        assert isinstance(original_prices, pd.Series)
        pd.testing.assert_series_equal(prices, original_prices)


def test_benchmark_returns_quantiles_regime_preserves_explicit_mapping() -> None:
    """Use caller-supplied labels, order, and colors when cardinality matches.

    The explicit mapping contains exactly four entries for four buckets. Classification should
    assign two returns to each entry in insertion order without modifying either caller-owned
    prices or mapping.
    """
    prices = _return_prices()
    original_prices = prices.copy(deep=True)
    mapping = {
        'Severe': '#8b0000',
        'Weak': '#f08080',
        'Firm': '#90ee90',
        'Strong': '#006400',
    }
    original_mapping = mapping.copy()
    classifier = cast(
        _QuantileClassifierProtocol,
        BenchmarkReturnsQuantilesRegime(
            freq='D',
            q=4,
            regime_ids_colors=mapping,
        ),
    )

    classified = classifier.compute_sampled_returns_with_regime_id(
        prices=prices,
        benchmark='Benchmark',
        include_start_date=False,
        include_end_date=False,
    )

    assert list(classified['regime'].cat.categories) == list(mapping)
    assert classified['regime'].value_counts(sort=False).tolist() == [2, 2, 2, 2]
    assert classifier.get_regime_ids_colors() == mapping
    assert mapping == original_mapping
    pd.testing.assert_series_equal(prices, original_prices)


def test_benchmark_returns_quantiles_regime_rejects_mapping_cardinality_mismatch() -> None:
    """Reject a label mapping that cannot describe every requested bucket.

    Four quantile buckets and two labels have no valid one-to-one interpretation. The classifier
    should report both cardinalities directly instead of leaking pandas' incidental bin-label
    exception later during classification.
    """
    with pytest.raises(
            ValueError,
            match='4 quantile buckets require 4 regime labels and colors; received 2'):
        BenchmarkReturnsQuantilesRegime(
            freq='D',
            q=4,
            regime_ids_colors={'Low': '#8b0000', 'High': '#006400'},
        )


# =============================================================================
# Volatility-quantile metadata and report construction
# =============================================================================

def test_benchmark_vols_quantiles_regime_exposes_classified_ids_and_colors() -> None:
    """Expose one valid color for each ordered volatility regime.

    The 36 monotonically ranked monthly volatility samples divide into four groups of nine. Their
    independently expected threshold labels must be the mapping keys in classification order.
    Classified observations must map to those colors rather than neutral white, while the initial
    missing return remains white. The caller-owned price panel must remain unchanged.
    """
    prices = _volatility_prices()
    original_prices = prices.copy(deep=True)
    classifier = cast(
        _QuantileClassifierProtocol,
        BenchmarkVolsQuantilesRegime(freq='ME', q=4),
    )

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        classified = classifier.compute_sampled_returns_with_regime_id(
            prices=prices,
            benchmark='Benchmark',
            include_start_date=True,
            include_end_date=True,
        )

    actual_regimes = classified['regime']
    assert isinstance(actual_regimes, pd.Series)
    expected_observed = np.repeat(_VOLATILITY_IDS, 9).tolist()
    assert actual_regimes.isna().tolist() == [True] + [False] * 36
    assert actual_regimes.dropna().tolist() == expected_observed
    assert classifier.get_regime_ids() == list(_VOLATILITY_IDS)

    regime_ids_colors = classifier.get_regime_ids_colors()
    assert list(regime_ids_colors) == list(_VOLATILITY_IDS)
    assert all(mpl_colors.is_color_like(color) for color in regime_ids_colors.values())
    rgba_colors = classifier.get_regime_colors()
    assert len(rgba_colors) == 4
    assert [mpl_colors.to_hex(color) for color in rgba_colors] == list(
        regime_ids_colors.values()
    )

    mapped_colors = classifier.class_data_to_colors(actual_regimes)
    assert mapped_colors.iloc[0] == '#FFFFFF'
    assert (mapped_colors.iloc[1:] != '#FFFFFF').all()
    for regime_id, color in regime_ids_colors.items():
        regime_colors = np.asarray(mapped_colors[actual_regimes == regime_id], dtype=str)
        assert np.all(regime_colors == color)

    pd.testing.assert_frame_equal(prices, original_prices)


def test_benchmark_vols_quantiles_regime_builds_fresh_four_regime_report() -> None:
    """Build all four report columns without priming classifier metadata first.

    A fresh classifier has no data-derived threshold labels until it classifies the supplied
    prices. Report construction must defer the ordered-ID lookup until after that step. Direct
    rank counting gives four groups of nine, so the regime-average component has exactly the four
    independently expected threshold columns in ascending-volatility order.
    """
    prices = _volatility_prices()
    original_prices = prices.copy(deep=True)
    classifier = cast(
        _VolatilityReportProtocol,
        BenchmarkVolsQuantilesRegime(freq='ME', q=4),
    )

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        table, regime_data = classifier.compute_regimes_pa_perf_table(
            prices=prices,
            benchmark='Benchmark',
            perf_params=PerfParams(),
        )

    expected_average_columns = [f'{regime_id} Average' for regime_id in _VOLATILITY_IDS]
    assert table.index.tolist() == ['Benchmark', 'Asset']
    assert regime_data[RegimeData.REGIME_AVG].columns.tolist() == expected_average_columns
    assert classifier.get_regime_ids() == list(_VOLATILITY_IDS)
    pd.testing.assert_frame_equal(prices, original_prices)

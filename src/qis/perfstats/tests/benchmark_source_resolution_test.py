"""Regression tests for benchmark-source resolution across public performance tables.

The benchmark-aware risk-adjusted and regime-table entry points accept the same three source
modes: an in-panel benchmark name, a standalone benchmark Series, or both an explicit name and a
Series. Those modes must resolve to the same price panel and selected benchmark before either
entry point performs its distinct numerical calculation.

The deterministic fixture uses thirteen quarter-end asset observations and a benchmark Series
observed one day before each quarter end. One explicit missing benchmark observation makes the
forward-filled values independently visible. Tests compare each standalone-Series call with the
equivalent manually assembled in-panel call, then assert values, labels, ordering, nullable
storage at the shared boundary, named calendar preservation, source precedence, errors, and
caller ownership.
"""

from typing import Literal, Protocol, cast

import pandas as pd
import pytest

import qis.perfstats.perf_stats as perf_stats_module
import qis.perfstats.regime_classifier as regime_classifier_module
from qis.perfstats.config import PerfParams


class _PerfStatsModuleProtocol(Protocol):
    """Typed test-side interface for the benchmark-aware performance table."""

    def resolve_benchmark_source(
        self,
        *,
        prices: pd.DataFrame,
        benchmark: str | None,
        benchmark_price: object | None,
    ) -> tuple[pd.DataFrame, str]:
        """Return the normalized benchmark panel and selected label."""
        raise NotImplementedError

    def compute_ra_perf_table_with_benchmark(
        self,
        *,
        prices: pd.DataFrame,
        benchmark: str | None = None,
        benchmark_price: pd.Series | None = None,
        perf_params: PerfParams | None = None,
    ) -> pd.DataFrame:
        """Return a risk-adjusted table for one resolved benchmark source."""
        raise NotImplementedError


class _RegimeClassifierModuleProtocol(Protocol):
    """Typed test-side interface for the benchmark-aware regime table."""

    def compute_bnb_regimes_pa_perf_table(
        self,
        *,
        prices: pd.DataFrame,
        benchmark: str | None = None,
        benchmark_price: pd.Series | None = None,
        freq: str = "QE",
        perf_params: PerfParams | None = None,
    ) -> pd.DataFrame:
        """Return a regime table for one resolved benchmark source."""
        raise NotImplementedError


_PERF_STATS = cast(_PerfStatsModuleProtocol, perf_stats_module)
_REGIME_CLASSIFIER = cast(_RegimeClassifierModuleProtocol, regime_classifier_module)


# =============================================================================
# Shared deterministic fixtures and independent alignment
# =============================================================================

_DATES = pd.date_range("2021-03-31", periods=13, freq="QE", name="Date")
_SOURCE_DATES = _DATES - pd.Timedelta(days=1)

_ASSET_NAME = "Asset"
_EXPLICIT_BENCHMARK_NAME = "ExplicitName"
_SERIES_BENCHMARK_NAME = "SeriesName"

_ASSET_PRICES = (80.0, 82.0, 79.0, 84.0, 83.0, 88.0, 86.0, 92.0, 91.0, 97.0, 95.0, 101.0, 99.0)
_SOURCE_BENCHMARK_PRICES = (
    100.0,
    90.0,
    95.0,
    85.0,
    float("nan"),
    98.0,
    110.0,
    105.0,
    120.0,
    118.0,
    135.0,
    130.0,
    150.0,
)
_ALIGNED_BENCHMARK_PRICES = (
    100.0,
    90.0,
    95.0,
    85.0,
    85.0,
    98.0,
    110.0,
    105.0,
    120.0,
    118.0,
    135.0,
    130.0,
    150.0,
)
_IGNORED_BENCHMARK_PRICES = (
    200.0,
    180.0,
    210.0,
    170.0,
    220.0,
    160.0,
    230.0,
    150.0,
    240.0,
    140.0,
    250.0,
    130.0,
    260.0,
)

_PERF_PARAMS = PerfParams(freq="QE")

_TableKind = Literal["risk_adjusted", "regime"]


def _asset_prices() -> pd.DataFrame:
    """Create the caller-owned asset panel.

    Returns:
        Complete quarterly asset prices with a named calendar.
    """
    return pd.DataFrame({_ASSET_NAME: _ASSET_PRICES}, index=_DATES)


def _external_benchmark(
    *,
    name: str,
    values: tuple[float, ...] = _SOURCE_BENCHMARK_PRICES,
    index: pd.DatetimeIndex = _SOURCE_DATES,
) -> pd.Series:
    """Create a standalone benchmark Series.

    Args:
        name: Series label presented by the external source.
        values: Benchmark observations to store.
        index: Observation dates for the external source.

    Returns:
        Benchmark Series with caller-owned metadata.
    """
    return pd.Series(values, index=index, name=name, dtype=float)


def _expected_panel(benchmark_name: str = _EXPLICIT_BENCHMARK_NAME) -> pd.DataFrame:
    """Construct the aligned in-panel benchmark without using QIS code.

    The source observation one day before each quarter end carries onto that quarter end. The
    missing fifth source observation is then filled from the fourth value, producing 85.0 at the
    fifth quarter end.

    Args:
        benchmark_name: Label assigned to the independently aligned benchmark.

    Returns:
        Benchmark-first quarterly panel with exact expected values.
    """
    aligned_benchmark = pd.Series(
        _ALIGNED_BENCHMARK_PRICES,
        index=_DATES,
        name=benchmark_name,
        dtype=float,
    )
    return pd.concat((aligned_benchmark, _asset_prices()), axis=1)


def _compute_table(
    table_kind: _TableKind,
    *,
    prices: pd.DataFrame,
    benchmark: str | None = None,
    benchmark_price: pd.Series | None = None,
) -> pd.DataFrame:
    """Call one public benchmark-aware table through a typed interface.

    Args:
        table_kind: Public entry point to exercise.
        prices: Caller-owned asset price panel.
        benchmark: Optional selected benchmark label.
        benchmark_price: Optional standalone benchmark Series.

    Returns:
        Public risk-adjusted or regime performance table.
    """
    if table_kind == "risk_adjusted":
        return _PERF_STATS.compute_ra_perf_table_with_benchmark(
            prices=prices,
            benchmark=benchmark,
            benchmark_price=benchmark_price,
            perf_params=_PERF_PARAMS,
        )
    return _REGIME_CLASSIFIER.compute_bnb_regimes_pa_perf_table(
        prices=prices,
        benchmark=benchmark,
        benchmark_price=benchmark_price,
        freq="QE",
        perf_params=_PERF_PARAMS,
    )


# =============================================================================
# Shared valid-source resolution contract
# =============================================================================


def test_resolve_benchmark_source_preserves_nullable_alignment_and_metadata() -> None:
    """Preserve exact nullable values, labels, calendar, order, and caller ownership."""
    prices = _asset_prices()
    benchmark_price = pd.Series(
        _SOURCE_BENCHMARK_PRICES,
        index=_SOURCE_DATES,
        name=_SERIES_BENCHMARK_NAME,
        dtype="Float64",
    )
    original_prices = prices.copy()
    original_benchmark = benchmark_price.copy()
    expected_benchmark = pd.Series(
        _ALIGNED_BENCHMARK_PRICES,
        index=_DATES,
        name=_EXPLICIT_BENCHMARK_NAME,
        dtype="Float64",
    )
    expected = pd.concat((expected_benchmark, prices), axis=1)

    actual, resolved_name = _PERF_STATS.resolve_benchmark_source(
        prices=prices,
        benchmark=_EXPLICIT_BENCHMARK_NAME,
        benchmark_price=benchmark_price,
    )

    assert resolved_name == _EXPLICIT_BENCHMARK_NAME
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.columns.tolist() == [_EXPLICIT_BENCHMARK_NAME, _ASSET_NAME]
    assert actual.index.name == "Date"
    assert actual[_EXPLICIT_BENCHMARK_NAME].dtype == pd.Float64Dtype()
    pd.testing.assert_frame_equal(prices, original_prices)
    pd.testing.assert_series_equal(benchmark_price, original_benchmark)


def test_resolve_benchmark_source_rejects_wrong_source_type() -> None:
    """Preserve explicit source-type validation before downstream pandas operations."""
    with pytest.raises(ValueError, match="benchmark_price must be pd.Series"):
        _PERF_STATS.resolve_benchmark_source(
            prices=_asset_prices(),
            benchmark=None,
            benchmark_price=[100.0, 101.0],
        )


@pytest.mark.parametrize("table_kind", ("risk_adjusted", "regime"))
def test_benchmark_tables_preserve_explicit_name_over_series_name(
    table_kind: _TableKind,
) -> None:
    """Give an explicit benchmark name precedence over the standalone Series name.

    The manually assembled name-only call is the public interaction control. Its benchmark path,
    source calendar and aligned missing observation are specified independently, so exact frame
    equality also proves that normalization does not alter numerical calculations.
    """
    prices = _asset_prices()
    benchmark_price = _external_benchmark(name=_SERIES_BENCHMARK_NAME)
    original_prices = prices.copy()
    original_benchmark = benchmark_price.copy()

    expected = _compute_table(
        table_kind,
        prices=_expected_panel(),
        benchmark=_EXPLICIT_BENCHMARK_NAME,
    )
    actual = _compute_table(
        table_kind,
        prices=prices,
        benchmark=_EXPLICIT_BENCHMARK_NAME,
        benchmark_price=benchmark_price,
    )

    pd.testing.assert_frame_equal(actual, expected)
    assert actual.index.tolist() == [_EXPLICIT_BENCHMARK_NAME, _ASSET_NAME]
    pd.testing.assert_frame_equal(prices, original_prices)
    pd.testing.assert_series_equal(benchmark_price, original_benchmark)


@pytest.mark.parametrize("table_kind", ("risk_adjusted", "regime"))
def test_benchmark_tables_trust_existing_series_named_column(
    table_kind: _TableKind,
) -> None:
    """Keep existing in-panel data when a Series-only source has the same name.

    The external path is deliberately different from the in-panel benchmark. Exact equality with
    the name-only control therefore proves that resolution neither duplicates nor replaces the
    existing column.
    """
    benchmark_name = "Benchmark"
    prices = _expected_panel(benchmark_name=benchmark_name)
    benchmark_price = _external_benchmark(
        name=benchmark_name,
        values=_IGNORED_BENCHMARK_PRICES,
        index=_DATES,
    )
    original_prices = prices.copy()
    original_benchmark = benchmark_price.copy()

    expected = _compute_table(table_kind, prices=prices, benchmark=benchmark_name)
    actual = _compute_table(
        table_kind,
        prices=prices,
        benchmark_price=benchmark_price,
    )

    pd.testing.assert_frame_equal(actual, expected)
    assert actual.index.tolist() == [benchmark_name, _ASSET_NAME]
    pd.testing.assert_frame_equal(prices, original_prices)
    pd.testing.assert_series_equal(benchmark_price, original_benchmark)


@pytest.mark.parametrize("table_kind", ("risk_adjusted", "regime"))
def test_benchmark_tables_preserve_other_valid_source_modes(
    table_kind: _TableKind,
) -> None:
    """Preserve Series-only augmentation and explicit existing-column precedence."""
    benchmark_name = "Benchmark"
    prices = _asset_prices()
    benchmark_price = _external_benchmark(name=benchmark_name)
    expected_panel = _expected_panel(benchmark_name=benchmark_name)

    expected = _compute_table(table_kind, prices=expected_panel, benchmark=benchmark_name)
    series_only = _compute_table(
        table_kind,
        prices=prices,
        benchmark_price=benchmark_price,
    )
    explicit_existing = _compute_table(
        table_kind,
        prices=expected_panel,
        benchmark=benchmark_name,
        benchmark_price=_external_benchmark(
            name="IgnoredSeriesName",
            values=_IGNORED_BENCHMARK_PRICES,
            index=_DATES,
        ),
    )

    pd.testing.assert_frame_equal(series_only, expected)
    pd.testing.assert_frame_equal(explicit_existing, expected)


@pytest.mark.parametrize("table_kind", ("risk_adjusted", "regime"))
def test_benchmark_tables_reject_absent_sources(table_kind: _TableKind) -> None:
    """Preserve a meaningful error when neither benchmark source is supplied."""
    with pytest.raises(ValueError, match="either benchmark"):
        _compute_table(table_kind, prices=_asset_prices())


@pytest.mark.parametrize("table_kind", ("risk_adjusted", "regime"))
def test_benchmark_tables_reject_missing_named_source(table_kind: _TableKind) -> None:
    """Preserve a meaningful error when the selected in-panel column is absent."""
    with pytest.raises(ValueError, match="MissingBenchmark is not in"):
        _compute_table(
            table_kind,
            prices=_asset_prices(),
            benchmark="MissingBenchmark",
        )

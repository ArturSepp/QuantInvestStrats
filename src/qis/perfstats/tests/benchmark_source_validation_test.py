"""Regression tests for benchmark-source labels and selected-column uniqueness.

The benchmark-aware performance and regime tables share one source resolver. A standalone
benchmark Series therefore needs a usable label when no explicit name is supplied, and the
resolved label must identify at most one existing price column before either numerical path
begins. These tests exercise that common boundary so invalid input raises the same deterministic
error instead of leaking downstream pandas, regression, or classifier exceptions.

The deterministic panel contains thirteen quarter-end observations, which is enough for both
public calculations. Valid calls are compared with independently assembled in-panel controls.
The matrix also preserves explicit-name precedence, nullable floating storage, calendar metadata,
and caller ownership while rejecting ambiguous duplicate benchmark columns.
"""

from collections.abc import Hashable
from typing import Literal, Protocol, cast

import pandas as pd
import pytest

import qis.perfstats.perf_stats as perf_stats_module
import qis.perfstats.regime_classifier as regime_classifier_module
from qis.perfstats.config import PerfParams


class _PerfStatsModuleProtocol(Protocol):
    """Typed test-side interface for benchmark-source resolution and performance tables."""

    def resolve_benchmark_source(
        self,
        *,
        prices: pd.DataFrame,
        benchmark: str | None,
        benchmark_price: object | None,
    ) -> tuple[pd.DataFrame, str]:
        """Return the normalized price panel and selected benchmark label."""
        raise NotImplementedError

    def compute_ra_perf_table_with_benchmark(
        self,
        *,
        prices: pd.DataFrame,
        benchmark: str | None = None,
        benchmark_price: pd.Series | None = None,
        perf_params: PerfParams | None = None,
    ) -> pd.DataFrame:
        """Return a risk-adjusted table for one benchmark source."""
        raise NotImplementedError


class _RegimeClassifierModuleProtocol(Protocol):
    """Typed test-side interface for benchmark-aware regime tables."""

    def compute_bnb_regimes_pa_perf_table(
        self,
        *,
        prices: pd.DataFrame,
        benchmark: str | None = None,
        benchmark_price: pd.Series | None = None,
        freq: str = "QE",
        perf_params: PerfParams | None = None,
    ) -> pd.DataFrame:
        """Return a regime table for one benchmark source."""
        raise NotImplementedError


_PERF_STATS = cast(_PerfStatsModuleProtocol, perf_stats_module)
_REGIME_CLASSIFIER = cast(_RegimeClassifierModuleProtocol, regime_classifier_module)

pytestmark = pytest.mark.filterwarnings("error")


# =============================================================================
# Shared deterministic fixtures and typed dispatch
# =============================================================================

_DATES = pd.date_range("2021-03-31", periods=13, freq="QE", name="Date")
_SOURCE_DATES = _DATES - pd.Timedelta(days=1)

_ASSET_NAME = "Asset"
_BENCHMARK_NAME = "Benchmark"

_ASSET_PRICES = (80.0, 82.0, 79.0, 84.0, 83.0, 88.0, 86.0, 92.0, 91.0, 97.0, 95.0, 101.0, 99.0)
_BENCHMARK_PRICES = (
    100.0,
    90.0,
    95.0,
    85.0,
    92.0,
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

_INFERRED_NAME_ERROR = (
    "benchmark_price must have a non-empty string name when benchmark is not supplied"
)
_EXPLICIT_NAME_ERROR = "benchmark must be a non-empty string"
_DUPLICATE_ERROR = "benchmark 'Benchmark' must identify one unique prices column; found 2"
_PERF_PARAMS = PerfParams(freq="QE")

_SourceMode = Literal["both", "name_only", "series_only"]
_TableKind = Literal["regime", "risk_adjusted"]


def _asset_prices() -> pd.DataFrame:
    """Create the caller-owned asset panel.

    Returns:
        Complete quarterly asset prices with a named calendar.
    """
    return pd.DataFrame({_ASSET_NAME: _ASSET_PRICES}, index=_DATES)


def _benchmark_price(
    *,
    name: Hashable | None,
    nullable: bool = False,
    values: tuple[float, ...] = _BENCHMARK_PRICES,
) -> pd.Series:
    """Create a standalone benchmark observed one day before each asset date.

    Args:
        name: Raw Series label presented to the resolver.
        nullable: Whether to store values with pandas nullable floating dtype.
        values: Deterministic benchmark levels.

    Returns:
        Caller-owned benchmark Series with the requested name and dtype.
    """
    dtype: object = pd.Float64Dtype() if nullable else float
    return pd.Series(values, index=_SOURCE_DATES, name=name, dtype=dtype)


def _benchmark_panel(*, duplicate: bool = False) -> pd.DataFrame:
    """Independently assemble an in-panel benchmark control.

    Args:
        duplicate: Whether to repeat the selected benchmark label.

    Returns:
        Benchmark-first price panel with one or two selected columns.
    """
    benchmark = pd.Series(_BENCHMARK_PRICES, index=_DATES, name=_BENCHMARK_NAME)
    panel = pd.concat((benchmark, _asset_prices()), axis=1)
    if duplicate:
        second_benchmark = pd.Series(
            _IGNORED_BENCHMARK_PRICES,
            index=_DATES,
            name=_BENCHMARK_NAME,
        )
        panel = pd.concat((benchmark, second_benchmark, _asset_prices()), axis=1)
    return panel


def _compute_table(
    table_kind: _TableKind,
    *,
    prices: pd.DataFrame,
    benchmark: str | None = None,
    benchmark_price: pd.Series | None = None,
) -> pd.DataFrame:
    """Call one benchmark-aware public entry point.

    Args:
        table_kind: Public calculation to exercise.
        prices: Caller-owned price panel.
        benchmark: Optional explicit benchmark label.
        benchmark_price: Optional standalone benchmark Series.

    Returns:
        Risk-adjusted or regime-conditional performance table.
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


def _duplicate_source(source_mode: _SourceMode) -> tuple[str | None, pd.Series | None]:
    """Return inputs for one duplicate-column source mode.

    Args:
        source_mode: Name-only, Series-only, or combined source specification.

    Returns:
        Explicit label and optional standalone Series for the selected mode.
    """
    if source_mode == "name_only":
        return _BENCHMARK_NAME, None
    benchmark_price = _benchmark_price(name=_BENCHMARK_NAME)
    if source_mode == "series_only":
        return None, benchmark_price
    return _BENCHMARK_NAME, benchmark_price


# =============================================================================
# Resolved-label validation
# =============================================================================


@pytest.mark.parametrize("invalid_name", (None, "", 7, ("Benchmark", 1)))
def test_resolver_rejects_invalid_inferred_benchmark_names(
    invalid_name: Hashable | None,
) -> None:
    """Reject every unusable inferred label with one exact boundary error."""
    with pytest.raises(ValueError, match=f"^{_INFERRED_NAME_ERROR}$"):
        _PERF_STATS.resolve_benchmark_source(
            prices=_asset_prices(),
            benchmark=None,
            benchmark_price=_benchmark_price(name=invalid_name),
        )


@pytest.mark.parametrize("invalid_name", ("", 7, ("Benchmark", 1)))
def test_resolver_rejects_invalid_explicit_benchmark_names(
    invalid_name: Hashable,
) -> None:
    """Reject an unusable explicit label even when a valid Series is available."""
    with pytest.raises(ValueError, match=f"^{_EXPLICIT_NAME_ERROR}$"):
        _PERF_STATS.resolve_benchmark_source(
            prices=_asset_prices(),
            benchmark=cast(str, invalid_name),
            benchmark_price=_benchmark_price(name=_BENCHMARK_NAME),
        )


@pytest.mark.parametrize("table_kind", ("regime", "risk_adjusted"))
def test_public_tables_propagate_empty_explicit_name_error(
    table_kind: _TableKind,
) -> None:
    """Reject an empty explicit name before either public numerical calculation."""
    with pytest.raises(ValueError, match=f"^{_EXPLICIT_NAME_ERROR}$"):
        _compute_table(
            table_kind,
            prices=_asset_prices(),
            benchmark="",
            benchmark_price=_benchmark_price(name=_BENCHMARK_NAME),
        )


@pytest.mark.parametrize("table_kind", ("regime", "risk_adjusted"))
@pytest.mark.parametrize("invalid_name", (None, "", 7, ("Benchmark", 1)))
def test_public_tables_propagate_invalid_inferred_name_error(
    invalid_name: Hashable | None,
    table_kind: _TableKind,
) -> None:
    """Fail at the shared input boundary before either numerical calculation."""
    with pytest.raises(ValueError, match=f"^{_INFERRED_NAME_ERROR}$"):
        _compute_table(
            table_kind,
            prices=_asset_prices(),
            benchmark_price=_benchmark_price(name=invalid_name),
        )


@pytest.mark.parametrize("table_kind", ("regime", "risk_adjusted"))
@pytest.mark.parametrize("raw_name", (None, "", 7, ("Benchmark", 1)))
def test_explicit_name_overrides_invalid_raw_series_name(
    raw_name: Hashable | None,
    table_kind: _TableKind,
) -> None:
    """Preserve explicit-name precedence over irrelevant raw Series metadata.

    The expected result comes from a separately assembled name-only panel. Exact equality proves
    that label validation does not alter either valid numerical path.
    """
    prices = _asset_prices()
    benchmark_price = _benchmark_price(name=raw_name)
    original_prices = prices.copy(deep=True)
    original_benchmark = benchmark_price.copy(deep=True)

    expected = _compute_table(
        table_kind,
        prices=_benchmark_panel(),
        benchmark=_BENCHMARK_NAME,
    )
    actual = _compute_table(
        table_kind,
        prices=prices,
        benchmark=_BENCHMARK_NAME,
        benchmark_price=benchmark_price,
    )

    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(prices, original_prices)
    pd.testing.assert_series_equal(benchmark_price, original_benchmark)


@pytest.mark.parametrize("raw_name", (None, "", 7, ("Benchmark", 1)))
def test_resolver_explicit_name_preserves_nullable_series(
    raw_name: Hashable | None,
) -> None:
    """Preserve nullable storage and metadata when an explicit valid name is available."""
    prices = _asset_prices()
    benchmark_price = _benchmark_price(name=raw_name, nullable=True)
    original_prices = prices.copy(deep=True)
    original_benchmark = benchmark_price.copy(deep=True)
    expected_benchmark = pd.Series(
        _BENCHMARK_PRICES,
        index=_DATES,
        name=_BENCHMARK_NAME,
        dtype="Float64",
    )
    expected = pd.concat((expected_benchmark, prices), axis=1)

    actual, resolved_name = _PERF_STATS.resolve_benchmark_source(
        prices=prices,
        benchmark=_BENCHMARK_NAME,
        benchmark_price=benchmark_price,
    )

    assert resolved_name == _BENCHMARK_NAME
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.index.name == "Date"
    assert actual[_BENCHMARK_NAME].dtype == pd.Float64Dtype()
    pd.testing.assert_frame_equal(prices, original_prices)
    pd.testing.assert_series_equal(benchmark_price, original_benchmark)


# =============================================================================
# Selected-column cardinality
# =============================================================================


@pytest.mark.parametrize("source_mode", ("both", "name_only", "series_only"))
def test_resolver_rejects_duplicate_selected_benchmark_columns(
    source_mode: _SourceMode,
) -> None:
    """Reject two physical matches before choosing or replacing either column."""
    prices = _benchmark_panel(duplicate=True)
    original_prices = prices.copy(deep=True)
    benchmark, benchmark_price = _duplicate_source(source_mode)
    original_benchmark = None if benchmark_price is None else benchmark_price.copy(deep=True)

    with pytest.raises(ValueError, match=f"^{_DUPLICATE_ERROR}$"):
        _PERF_STATS.resolve_benchmark_source(
            prices=prices,
            benchmark=benchmark,
            benchmark_price=benchmark_price,
        )

    pd.testing.assert_frame_equal(prices, original_prices)
    if benchmark_price is not None and original_benchmark is not None:
        pd.testing.assert_series_equal(benchmark_price, original_benchmark)


@pytest.mark.parametrize("table_kind", ("regime", "risk_adjusted"))
@pytest.mark.parametrize("source_mode", ("both", "name_only", "series_only"))
def test_public_tables_propagate_duplicate_selected_column_error(
    source_mode: _SourceMode,
    table_kind: _TableKind,
) -> None:
    """Expose the same duplicate-label contract through both public entry points."""
    benchmark, benchmark_price = _duplicate_source(source_mode)

    with pytest.raises(ValueError, match=f"^{_DUPLICATE_ERROR}$"):
        _compute_table(
            table_kind,
            prices=_benchmark_panel(duplicate=True),
            benchmark=benchmark,
            benchmark_price=benchmark_price,
        )


@pytest.mark.parametrize("table_kind", ("regime", "risk_adjusted"))
def test_zero_and_one_selected_matches_preserve_valid_results(
    table_kind: _TableKind,
) -> None:
    """Preserve Series augmentation at zero matches and panel authority at one match."""
    expected_panel = _benchmark_panel()
    expected = _compute_table(
        table_kind,
        prices=expected_panel,
        benchmark=_BENCHMARK_NAME,
    )
    added = _compute_table(
        table_kind,
        prices=_asset_prices(),
        benchmark=_BENCHMARK_NAME,
        benchmark_price=_benchmark_price(name="Ignored"),
    )
    authoritative = _compute_table(
        table_kind,
        prices=expected_panel,
        benchmark=_BENCHMARK_NAME,
        benchmark_price=_benchmark_price(
            name="Ignored",
            values=_IGNORED_BENCHMARK_PRICES,
        ),
    )

    pd.testing.assert_frame_equal(added, expected)
    pd.testing.assert_frame_equal(authoritative, expected)


def test_resolver_preserves_nullable_zero_and_one_match_modes() -> None:
    """Preserve nullable augmentation and authoritative existing-column precedence."""
    benchmark_price = _benchmark_price(name="Ignored", nullable=True)
    expected_benchmark = pd.Series(
        _BENCHMARK_PRICES,
        index=_DATES,
        name=_BENCHMARK_NAME,
        dtype="Float64",
    )
    expected_added = pd.concat((expected_benchmark, _asset_prices()), axis=1)

    added, added_name = _PERF_STATS.resolve_benchmark_source(
        prices=_asset_prices(),
        benchmark=_BENCHMARK_NAME,
        benchmark_price=benchmark_price,
    )
    existing = _benchmark_panel()
    authoritative, authoritative_name = _PERF_STATS.resolve_benchmark_source(
        prices=existing,
        benchmark=_BENCHMARK_NAME,
        benchmark_price=_benchmark_price(
            name="Ignored",
            nullable=True,
            values=_IGNORED_BENCHMARK_PRICES,
        ),
    )

    assert added_name == authoritative_name == _BENCHMARK_NAME
    pd.testing.assert_frame_equal(added, expected_added)
    assert authoritative is existing

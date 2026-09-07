"""Regression coverage for benchmark removal from volatility-regime tables.

The volatility-quantile classifier shares the regime-table API with the return-quantile and
positive/negative classifiers. Its ``drop_benchmark`` request must reach the shared table builder:
only the final presentation table drops that row, while the component data, dynamically derived
regime metadata, unavailable early windows, and caller-owned prices remain unchanged.

The forwarding test separates delegation from numerical behavior and includes nullable pandas
storage. The end-to-end test uses an ordinary ragged panel whose monthly volatility rises over
time, giving four attainable regimes with independently fixed output-schema expectations.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from qis.perfstats.config import PerfParams, RegimeData
from qis.perfstats.regime_classifier import (
    BenchmarkVolsQuantilesRegime,
    RegimeClassifier,
)


# =============================================================================
# Shared deterministic fixtures
# =============================================================================

_ASSET = "Ragged Asset"
_BENCHMARK = "Benchmark"
_DATES = pd.bdate_range("2022-01-03", "2024-12-31")
_EXPECTED_REGIME_COLORS = ("#a50026", "#fdbf6f", "#b7e075", "#006837")
_EXPECTED_REGIME_IDS = (
    "Benchmark vol<5%",
    "Benchmark vol=(5%, 6%]",
    "Benchmark vol=(6%, 8%]",
    "Benchmark vol>8%",
)
_PERF_PARAMS = PerfParams(freq="ME")


def _price_panel() -> pd.DataFrame:
    """Create a ragged panel with four attainable monthly volatility regimes.

    Returns:
        Three years of benchmark and dependent-asset prices with unavailable early windows.
    """
    month_numbers = np.asarray(
        [(date.year - 2022) * 12 + date.month for date in _DATES],
        dtype=float,
    )
    amplitudes = 0.001 + 0.00015 * month_numbers
    signs = np.where(np.arange(len(_DATES)) % 2 == 0, 1.0, -1.0)
    benchmark_returns = amplitudes * signs
    benchmark_prices = 100.0 * np.cumprod(1.0 + benchmark_returns)
    asset_prices = 80.0 * np.cumprod(1.0 + 0.5 * benchmark_returns)
    benchmark_prices[_DATES < pd.Timestamp("2022-04-01")] = np.nan
    asset_prices[:40] = np.nan
    return pd.DataFrame(
        {_BENCHMARK: benchmark_prices, _ASSET: asset_prices},
        index=_DATES,
    )


# =============================================================================
# Delegation contract
# =============================================================================


@pytest.mark.parametrize(
    ("requested_drop", "expected_drop"),
    ((None, False), (False, False), (True, True)),
)
def test_volatility_regime_forwards_only_supported_table_options(
    monkeypatch: pytest.MonkeyPatch,
    requested_drop: bool | None,
    expected_drop: bool,
) -> None:
    """Forward the explicit benchmark option while tolerating unrelated compatibility kwargs."""
    calls: list[dict[str, Any]] = []
    marker_table = pd.DataFrame({"marker": [1.0]}, index=[_ASSET])

    def record_base_call(
        _self: RegimeClassifier,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, dict[RegimeData, pd.DataFrame]]:
        """Record one delegation to the shared table implementation."""
        calls.append(kwargs)
        return marker_table, {}

    monkeypatch.setattr(
        RegimeClassifier,
        "compute_regimes_pa_perf_table",
        record_base_call,
    )
    prices = _price_panel().astype(pd.Float64Dtype())
    classifier = BenchmarkVolsQuantilesRegime(freq="ME", q=4)
    call_kwargs: dict[str, Any] = {
        "prices": prices,
        "benchmark": _BENCHMARK,
        "perf_params": _PERF_PARAMS,
        "unused_option": "preserve tolerant compatibility behavior",
    }
    if requested_drop is not None:
        call_kwargs["drop_benchmark"] = requested_drop

    actual_table, actual_components = classifier.compute_regimes_pa_perf_table(**call_kwargs)

    assert actual_table is marker_table
    assert actual_components == {}
    assert len(calls) == 1
    assert calls[0]["prices"] is prices
    assert calls[0]["benchmark"] == _BENCHMARK
    assert calls[0]["perf_params"] is _PERF_PARAMS
    assert calls[0]["freq"] == "ME"
    assert calls[0]["drop_benchmark"] is expected_drop
    assert "unused_option" not in calls[0]


# =============================================================================
# Public numerical and schema contract
# =============================================================================


@pytest.mark.parametrize("drop_benchmark", (False, True))
def test_volatility_regime_drops_only_final_benchmark_row(drop_benchmark: bool) -> None:
    """Drop the final benchmark row without changing components, metadata, or caller data."""
    prices = _price_panel()
    original_prices = prices.copy()
    classifier = BenchmarkVolsQuantilesRegime(freq="ME", q=4)

    table, regime_components = classifier.compute_regimes_pa_perf_table(
        prices=prices,
        benchmark=_BENCHMARK,
        perf_params=_PERF_PARAMS,
        drop_benchmark=drop_benchmark,
    )

    expected_index = [_ASSET] if drop_benchmark else [_BENCHMARK, _ASSET]
    assert table.index.tolist() == expected_index
    assert regime_components
    for component in regime_components.values():
        assert component.index.tolist() == [_BENCHMARK, _ASSET]
    assert tuple(classifier.get_regime_ids()) == _EXPECTED_REGIME_IDS
    assert tuple(classifier.get_regime_ids_colors().values()) == _EXPECTED_REGIME_COLORS
    pd.testing.assert_frame_equal(prices, original_prices)

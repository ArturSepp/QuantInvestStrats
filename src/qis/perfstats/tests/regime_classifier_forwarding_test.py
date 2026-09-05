"""Regression coverage for caller-supplied regime-classifier forwarding.

``compute_bnb_regimes_pa_perf_table`` is a convenience wrapper with a quantile-classifier
fallback. Plotting and reporting callers also pass a fully configured ``regime_classifier``
through its compatibility keyword arguments. That supplied object must be authoritative: the
wrapper should resolve the benchmark source once, call the supplied classifier once, and return
its table without rebuilding the caller's classification policy.

The tests separate dispatch from numerical behavior. A recording classifier proves exact input
forwarding and prevents fallback construction. Real positive/negative and quantile classifiers
then establish parity with their direct public methods. The sign-regime averages are calculated
independently from twelve literal returns, while omitted and explicit-``None`` controls pin the
existing quantile fallback. Caller ownership and unrelated compatibility keywords are covered in
the same narrow matrix.
"""

from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
import pandas as pd
import pytest

import qis.perfstats.regime_classifier as regime_classifier_module
from qis.perfstats.config import PerfParams, RegimeData, ReturnTypes
from qis.perfstats.regime_classifier import (
    BenchmarkReturnsPositiveNegativeRegime,
    BenchmarkReturnsQuantilesRegime,
)


class _ComputeBnbTable(Protocol):
    """Typed test-side interface for the convenience wrapper's supported inputs."""

    def __call__(
        self,
        *,
        prices: pd.DataFrame,
        benchmark: str | None = None,
        benchmark_price: pd.Series | None = None,
        freq: str = "QE",
        return_type: ReturnTypes = ReturnTypes.RELATIVE,
        q: int | None = None,
        regime_ids_colors: dict[str, str] | None = None,
        perf_params: PerfParams | None = None,
        drop_benchmark: bool = False,
        regime_classifier: object | None = None,
        unused_option: str | None = None,
    ) -> pd.DataFrame:
        """Return one regime-conditional performance table."""
        raise NotImplementedError


class _ReportClassifier(Protocol):
    """Typed test-side interface shared by real and recording classifiers."""

    def compute_regimes_pa_perf_table(
        self,
        *,
        prices: pd.DataFrame,
        benchmark: str,
        perf_params: PerfParams,
        drop_benchmark: bool = False,
    ) -> tuple[pd.DataFrame, dict[RegimeData, pd.DataFrame]]:
        """Return a conditional table and its component tables."""
        raise NotImplementedError


_COMPUTE_BNB_TABLE = cast(
    _ComputeBnbTable,
    regime_classifier_module.compute_bnb_regimes_pa_perf_table,
)


# =============================================================================
# Shared deterministic fixtures and test doubles
# =============================================================================

_ASSET_NAME = "Asset"
_BENCHMARK_NAME = "Benchmark"
_DATES = pd.date_range("2021-03-31", periods=13, freq="QE", name="Date")

_ASSET_RETURNS = (-0.02, 0.10, -0.03, 0.08, -0.04, 0.12, -0.05, 0.14, -0.06, 0.16, -0.07, 0.18)
_BENCHMARK_RETURNS = (-0.10, 0.20, -0.05, 0.10, -0.20, 0.30, -0.08, 0.15, -0.12, 0.25, -0.06, 0.14)

_PERF_PARAMS = PerfParams(freq="QE")
_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class _RecordedCall:
    """Arguments received by one recording-classifier invocation."""

    benchmark: str
    drop_benchmark: bool
    perf_params: PerfParams
    prices: pd.DataFrame


class _RecordingClassifier:
    """Return a marker table while retaining every forwarded argument."""

    def __init__(self, output: pd.DataFrame) -> None:
        self.calls: list[_RecordedCall] = []
        self.output = output

    def compute_regimes_pa_perf_table(
        self,
        *,
        prices: pd.DataFrame,
        benchmark: str,
        perf_params: PerfParams,
        drop_benchmark: bool = False,
    ) -> tuple[pd.DataFrame, dict[RegimeData, pd.DataFrame]]:
        """Record the exact call and return the configured marker table."""
        self.calls.append(
            _RecordedCall(
                benchmark=benchmark,
                drop_benchmark=drop_benchmark,
                perf_params=perf_params,
                prices=prices,
            )
        )
        return self.output, {}


def _price_panel() -> pd.DataFrame:
    """Create prices from twelve literal simple-return observations.

    Returns:
        Quarterly benchmark and asset prices with a named calendar.
    """
    benchmark_growth = np.concatenate(((1.0,), np.cumprod(1.0 + np.asarray(_BENCHMARK_RETURNS))))
    asset_growth = np.concatenate(((1.0,), np.cumprod(1.0 + np.asarray(_ASSET_RETURNS))))
    return pd.DataFrame(
        {
            _BENCHMARK_NAME: 100.0 * benchmark_growth,
            _ASSET_NAME: 80.0 * asset_growth,
        },
        index=_DATES,
    )


def _expected_sign_averages() -> pd.DataFrame:
    """Calculate conditional means directly from the literal return tuples.

    The benchmark signs alternate, so the six even-positioned returns belong to ``Negative``
    and the six odd-positioned returns belong to ``Positive``. Their arithmetic means are
    ``-0.101666...`` and ``0.19`` for the benchmark and ``-0.045`` and ``0.13`` for the asset.

    Returns:
        Independently calculated two-regime average table.
    """
    benchmark_returns = np.asarray(_BENCHMARK_RETURNS)
    asset_returns = np.asarray(_ASSET_RETURNS)
    negative = benchmark_returns < 0.0
    return pd.DataFrame(
        {
            "Negative Average": (
                float(np.mean(benchmark_returns[negative])),
                float(np.mean(asset_returns[negative])),
            ),
            "Positive Average": (
                float(np.mean(benchmark_returns[~negative])),
                float(np.mean(asset_returns[~negative])),
            ),
        },
        index=(_BENCHMARK_NAME, _ASSET_NAME),
    )


# =============================================================================
# Supplied-classifier dispatch contract
# =============================================================================


def test_supplied_classifier_is_authoritative_after_benchmark_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Call only the supplied classifier with exact resolved inputs.

    The benchmark is supplied separately so this test observes the wrapper's normalized panel.
    A patched fallback constructor raises immediately if the wrapper ignores the supplied
    classifier. The unrelated keyword proves the established compatibility path remains tolerant.
    """
    full_prices = _price_panel()
    prices = full_prices[[_ASSET_NAME]]
    benchmark_price = full_prices[_BENCHMARK_NAME]
    assert isinstance(prices, pd.DataFrame)
    assert isinstance(benchmark_price, pd.Series)
    original_prices = prices.copy(deep=True)
    original_benchmark = benchmark_price.copy(deep=True)
    expected_prices = pd.concat((benchmark_price, prices), axis=1)
    marker_table = pd.DataFrame({"marker": (1.0,)}, index=("result",))
    classifier = _RecordingClassifier(marker_table)

    def _reject_fallback_construction(*args: object, **kwargs: object) -> None:
        raise AssertionError("the quantile fallback must not be constructed")

    monkeypatch.setattr(
        regime_classifier_module,
        "BenchmarkReturnsQuantilesRegime",
        _reject_fallback_construction,
    )

    actual = _COMPUTE_BNB_TABLE(
        prices=prices,
        benchmark_price=benchmark_price,
        perf_params=_PERF_PARAMS,
        drop_benchmark=True,
        regime_classifier=classifier,
        unused_option="preserve compatibility",
    )

    assert actual is marker_table
    assert len(classifier.calls) == 1
    recorded = classifier.calls[0]
    assert recorded.benchmark == _BENCHMARK_NAME
    assert recorded.drop_benchmark is True
    assert recorded.perf_params is _PERF_PARAMS
    pd.testing.assert_frame_equal(recorded.prices, expected_prices)
    pd.testing.assert_frame_equal(prices, original_prices)
    pd.testing.assert_series_equal(benchmark_price, original_benchmark)


@pytest.mark.parametrize("explicit_none", (False, True))
def test_missing_classifier_uses_configured_quantile_fallback(
    monkeypatch: pytest.MonkeyPatch,
    explicit_none: bool,
) -> None:
    """Retain fallback construction when the classifier is omitted or explicitly ``None``."""
    constructor_calls: list[dict[str, object]] = []
    marker_table = pd.DataFrame({"marker": (2.0,)}, index=("fallback",))
    fallback = _RecordingClassifier(marker_table)
    regime_ids_colors = {
        "Q1": "red",
        "Q2": "orange",
        "Q3": "green",
        "Q4": "blue",
    }

    def _record_fallback_construction(
        *,
        freq: str,
        return_type: ReturnTypes,
        q: int | None,
        regime_ids_colors: dict[str, str] | None,
    ) -> _RecordingClassifier:
        constructor_calls.append(
            {
                "freq": freq,
                "return_type": return_type,
                "q": q,
                "regime_ids_colors": regime_ids_colors,
            }
        )
        return fallback

    monkeypatch.setattr(
        regime_classifier_module,
        "BenchmarkReturnsQuantilesRegime",
        _record_fallback_construction,
    )

    if explicit_none:
        actual = _COMPUTE_BNB_TABLE(
            prices=_price_panel(),
            benchmark=_BENCHMARK_NAME,
            freq="ME",
            return_type=ReturnTypes.LOG,
            q=4,
            regime_ids_colors=regime_ids_colors,
            perf_params=_PERF_PARAMS,
            regime_classifier=None,
        )
    else:
        actual = _COMPUTE_BNB_TABLE(
            prices=_price_panel(),
            benchmark=_BENCHMARK_NAME,
            freq="ME",
            return_type=ReturnTypes.LOG,
            q=4,
            regime_ids_colors=regime_ids_colors,
            perf_params=_PERF_PARAMS,
        )

    assert actual is marker_table
    assert constructor_calls == [
        {
            "freq": "ME",
            "return_type": ReturnTypes.LOG,
            "q": 4,
            "regime_ids_colors": regime_ids_colors,
        }
    ]
    assert len(fallback.calls) == 1


# =============================================================================
# Real-classifier numerical and configuration contracts
# =============================================================================


def test_supplied_sign_classifier_matches_independent_conditional_means() -> None:
    """Honor a sign classifier and reproduce independently grouped regime means.

    Conflicting wrapper construction arguments are intentional: they must not replace the
    supplied classifier's quarterly relative-return policy or its two configured regime IDs.
    """
    prices = _price_panel()
    original_prices = prices.copy(deep=True)
    classifier = BenchmarkReturnsPositiveNegativeRegime(freq="QE")
    report_classifier: _ReportClassifier = classifier
    direct_table, direct_components = report_classifier.compute_regimes_pa_perf_table(
        prices=prices,
        benchmark=_BENCHMARK_NAME,
        perf_params=_PERF_PARAMS,
    )

    actual = _COMPUTE_BNB_TABLE(
        prices=prices,
        benchmark=_BENCHMARK_NAME,
        freq="ME",
        return_type=ReturnTypes.LOG,
        q=4,
        regime_ids_colors={
            "Ignored 1": "red",
            "Ignored 2": "orange",
            "Ignored 3": "green",
            "Ignored 4": "blue",
        },
        perf_params=_PERF_PARAMS,
        regime_classifier=classifier,
    )

    pd.testing.assert_frame_equal(actual, direct_table)
    pd.testing.assert_frame_equal(
        direct_components[RegimeData.REGIME_AVG],
        _expected_sign_averages(),
        check_exact=False,
        rtol=0.0,
        atol=_TOLERANCE,
    )
    pd.testing.assert_frame_equal(prices, original_prices)


def test_supplied_quantile_classifier_preserves_its_configuration() -> None:
    """Return the same table as a direct custom quantile-classifier invocation."""
    prices = _price_panel()
    original_prices = prices.copy(deep=True)
    classifier = BenchmarkReturnsQuantilesRegime(
        freq="QE",
        q=4,
        regime_ids_colors={
            "Low": "red",
            "Lower middle": "orange",
            "Upper middle": "green",
            "High": "blue",
        },
    )
    report_classifier: _ReportClassifier = classifier
    direct_table, _ = report_classifier.compute_regimes_pa_perf_table(
        prices=prices,
        benchmark=_BENCHMARK_NAME,
        perf_params=_PERF_PARAMS,
        drop_benchmark=True,
    )

    actual = _COMPUTE_BNB_TABLE(
        prices=prices,
        benchmark=_BENCHMARK_NAME,
        freq="ME",
        q=3,
        regime_ids_colors={"Ignored low": "red", "Ignored mid": "yellow", "Ignored high": "green"},
        perf_params=_PERF_PARAMS,
        drop_benchmark=True,
        regime_classifier=classifier,
    )

    pd.testing.assert_frame_equal(actual, direct_table)
    assert actual.index.tolist() == [_ASSET_NAME]
    assert actual.columns[:4].tolist() == [
        "Low Average",
        "Lower middle Average",
        "Upper middle Average",
        "High Average",
    ]
    assert classifier.freq == "QE"
    assert classifier.return_type is ReturnTypes.RELATIVE
    assert classifier.get_regime_ids() == ["Low", "Lower middle", "Upper middle", "High"]
    pd.testing.assert_frame_equal(prices, original_prices)

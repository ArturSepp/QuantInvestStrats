"""Regression tests for shared PyBloqs fallback ownership.

The strategy-benchmark module keeps its historical entry point while delegating construction to
the multi-strategy module. Exact forwarding and signal propagation matter because optional-report
callers must observe the same contract from either import path.
"""

import importlib
import inspect
import warnings

import pytest


def _import_fallback_modules():
    pytest.importorskip("pybloqs")
    canonical = importlib.import_module("qis.portfolio.reports.multi_strategy_factsheet_pybloqs")
    wrapper = importlib.import_module("qis.portfolio.reports.strategy_benchmark_factsheet_pybloqs")
    return canonical, wrapper


def test_generate_multi_portfolio_factsheet_delegates_with_exact_arguments(monkeypatch) -> None:
    """The legacy entry point must remain an exact same-signature forwarding boundary."""
    canonical, wrapper = _import_fallback_modules()
    assert inspect.signature(wrapper.generate_multi_portfolio_factsheet) == inspect.signature(
        canonical.generate_multi_portfolio_factsheet
    )

    result = object()
    calls = []

    def capture_call(**kwargs):
        calls.append(kwargs)
        return result

    monkeypatch.setattr(canonical, "generate_multi_portfolio_factsheet", capture_call)
    arguments = {
        "multi_portfolio_data": object(),
        "time_period": object(),
        "perf_params": object(),
        "regime_classifier": object(),
        "regime_benchmark": object(),
        "backtest_name": object(),
        "heatmap_freq": object(),
        "figsize": object(),
        "is_grouped": object(),
        "fontsize": object(),
        "extra_option": object(),
    }

    assert wrapper.generate_multi_portfolio_factsheet(**arguments) is result
    assert calls == [arguments]


def test_generate_multi_portfolio_factsheet_preserves_warnings_and_exceptions(monkeypatch) -> None:
    """Delegation must not swallow or replace canonical warnings and exceptions."""
    canonical, wrapper = _import_fallback_modules()
    expected_error = RuntimeError("sentinel error")

    def warn_and_raise(**kwargs):
        warnings.warn("sentinel warning", UserWarning, stacklevel=2)
        raise expected_error

    monkeypatch.setattr(canonical, "generate_multi_portfolio_factsheet", warn_and_raise)

    with pytest.warns(UserWarning, match="sentinel warning"):
        with pytest.raises(RuntimeError) as exc_info:
            wrapper.generate_multi_portfolio_factsheet(object())
    assert exc_info.value is expected_error

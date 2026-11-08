"""Alpha-only method selection: dense references, same samples and no silent fallback."""

import inspect
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import qis
from qis.datasets.synthetic import generate_synthetic_prices
from qis.perfstats.significance import InterceptInference as Method
from qis.portfolio.reports import _performance_inference as views
from qis.portfolio.reports.tests.paired_performance_inference_test import (
    compute, portfolio,
)
from qis.portfolio.reports.tests.multi_performance_inference_test import (
    calculate, holm_reference, multi,
)


@pytest.fixture(scope="module")
def panel():
    return generate_synthetic_prices(end="2025-12-31", apply_quirks=False)[
        ["SEQ_US", "SBD_TSY", "SCM_GLD"]].resample("ME").last()


def dense_reference(values, lags, prewhitened):
    """Independent dense Toeplitz quadratic form; no qis variance/AR helper."""
    nobs = len(values)
    residual = values - values.mean()
    rho = 0.0
    if prewhitened:
        rho = np.linalg.lstsq(residual[:-1, None], residual[1:], rcond=None)[0][0]
        residual = residual[1:] - rho * residual[:-1]
    distance = np.abs(np.arange(len(residual))[:, None] - np.arange(len(residual))[None, :])
    kernel = np.maximum(1.0 - distance / (lags + 1), 0.0)
    variance = residual @ kernel @ residual / (nobs * (nobs - 1) * (1 - rho) ** 2)
    return 12 * np.sqrt(variance), rho


@pytest.mark.parametrize("method", list(Method))
def test_same_sample_mean_and_uncertainty_match_dense_reference(panel, method):
    result = compute(panel, alpha_inference=method)
    for window, row in result.alpha_windows.iterrows():
        sample = result.alpha_returns.loc[row.start:row.end]
        assert sample.notna().all() and len(sample) == row.n_obs
        assert result.metadata["alpha"]["inference"] == method.name
        for alternative in (False, True):
            selected = method if not alternative else next(m for m in Method if m != method)
            se, rho = dense_reference(sample.to_numpy(), int(row.hac_lags),
                                      selected == Method.PREWHITENED_AR1)
            mean = 12 * sample.mean()
            prefix = "comparison_" if alternative else "hac_"
            assert row[prefix + "se"] == pytest.approx(se, rel=2e-11)
            assert row[prefix + "lower"] == pytest.approx(mean - norm.isf(.025) * se)
            assert row[prefix + "upper"] == pytest.approx(mean + norm.isf(.025) * se)
            assert row["comparison_pvalue" if alternative else "pvalue"] == pytest.approx(
                2 * norm.sf(abs(mean / se)), rel=2e-10)
            if selected == Method.PREWHITENED_AR1:
                assert row["comparison_rho" if alternative else "rho"] == pytest.approx(rho)
        assert row.alpha == pytest.approx(row.comparison_estimate)


def test_switch_exchanges_alpha_results_but_not_sharpe_samples_or_beta(panel):
    bartlett = compute(panel)
    prewhitened = compute(panel, alpha_inference=Method.PREWHITENED_AR1)
    for key in ("windows", "expanding", "rolling", "returns", "alpha_diagnostics"):
        pd.testing.assert_frame_equal(getattr(bartlett, key), getattr(prewhitened, key))
    pd.testing.assert_series_equal(bartlett.alpha_returns, prewhitened.alpha_returns)
    for key in ("start", "end", "n_obs", "hac_lags"):
        pd.testing.assert_series_equal(bartlett.alpha_windows[key], prewhitened.alpha_windows[key])
    np.testing.assert_allclose(bartlett.alpha_windows.hac_se,
                               prewhitened.alpha_windows.comparison_se)
    np.testing.assert_allclose(prewhitened.alpha_windows.hac_se,
                               bartlett.alpha_windows.comparison_se)
    for key in ("alpha_expanding", "alpha_rolling"):
        a, b = getattr(bartlett, key), getattr(prewhitened, key)
        np.testing.assert_allclose(a.ci_lower, b.comparison_lower, equal_nan=True)
        np.testing.assert_allclose(b.ci_lower, a.comparison_lower, equal_nan=True)


def test_single_alpha_uses_same_option_and_keeps_sharpe(panel):
    options = dict(nav=panel.SEQ_US, benchmark_prices=panel[["SBD_TSY"]],
                   instrument_prices=panel, weights=portfolio(panel, "SEQ_US").weights,
                   time_period=qis.TimePeriod("2005", "2025-12-31"), freq="ME",
                   freq_beta="ME", factor_beta_span=12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        b = views.compute_performance_inference(**options)
        p = views.compute_performance_inference(**options, alpha_inference=Method.PREWHITENED_AR1)
    pd.testing.assert_frame_equal(b.windows, p.windows)
    np.testing.assert_allclose(b.alpha_windows.comparison_se, p.alpha_windows.hac_se)
    row = p.alpha_windows.iloc[0]
    se, _ = dense_reference(p.alpha_returns.loc[row.start:row.end].to_numpy(),
                            int(row.hac_lags), True)
    assert row.hac_se == pytest.approx(se)


def test_failed_prewhitening_stays_unavailable_and_never_changes_primary(panel, monkeypatch):
    def invalid(*args, **kwargs):
        raise ValueError("unstable AR(1) filter")

    monkeypatch.setattr(views, "estimate_prewhitened_intercepts", invalid)
    for method in Method:
        result = compute(panel, alpha_inference=method)
        assert result.windows.sharpe.notna().all()
        if method == Method.BARTLETT_HAC:
            assert result.alpha_windows.hac_se.notna().all()
            assert result.alpha_windows.comparison_se.isna().all()
        else:
            assert result.alpha_windows.hac_se.isna().all()
            assert result.alpha_windows.comparison_se.notna().all()
        assert any("unstable AR(1) filter" in issue for issue in result.issues)
        assert any("no method substitution" in issue for issue in result.issues)


def test_identical_pair_retains_bartlett_neutrality_but_no_invented_ar_filter(panel):
    identical = portfolio(panel, "SEQ_US")
    result = compute(panel, strategy=identical, comparison=identical)
    assert (result.alpha_windows.pvalue == 1).all()
    assert (result.alpha_windows.hac_se == 0).all()
    assert result.alpha_windows.comparison_se.isna().all()


def test_primary_method_only_enters_unchanged_holm_family(panel):
    data = multi(panel)
    result = calculate(data, alpha_inference=Method.PREWHITENED_AR1)
    assert result.metadata["family_size"] == 12
    assert result.summary.comparison_pvalue.xs("sharpe", level="metric").isna().all()
    assert result.summary.comparison_pvalue.xs("alpha", level="metric").notna().all()
    np.testing.assert_allclose(result.summary.pvalue_holm, holm_reference(result.summary.pvalue))
    baseline = calculate(data)
    pd.testing.assert_frame_equal(
        result.summary.xs("sharpe", level="metric").drop(columns="pvalue_holm"),
        baseline.summary.xs("sharpe", level="metric").drop(columns="pvalue_holm"))
    for position, pair in result.pairs.items():
        assert pair.metadata["alpha"]["inference"] == Method.PREWHITENED_AR1.name
        np.testing.assert_allclose(result.summary.xs((position, "alpha"), level=(0, 1)).pvalue,
                                   pair.alpha_windows.pvalue)


def test_all_public_factsheets_keep_bartlett_default_and_reject_unsupported_method(panel):
    for function in (qis.generate_strategy_factsheet, qis.generate_strategy_benchmark_factsheet_plt,
                     qis.generate_multi_portfolio_factsheet):
        parameter = inspect.signature(function).parameters["alpha_inference"]
        assert parameter.default is Method.BARTLETT_HAC
        assert parameter.kind == inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(ValueError, match="InterceptInference"):
        compute(panel, alpha_inference="PREWHITENED_AR1")


def test_both_methods_receive_identical_sample_and_lags(panel, monkeypatch):
    calls = {"bartlett": [], "prewhitened": []}
    for name, original in (("bartlett", views.estimate_hac_regression),
                            ("prewhitened", views.estimate_prewhitened_intercepts)):
        def record(data, *args, _name=name, _original=original, **kwargs):
            calls[_name].append((data.copy(), kwargs["hac_lags"]))
            return _original(data, *args, **kwargs)
        monkeypatch.setattr(views, "estimate_hac_regression" if name == "bartlett" else
                            "estimate_prewhitened_intercepts", record)
    compute(panel)
    assert len(calls["bartlett"]) == len(calls["prewhitened"]) > 3
    for (b, bl), (p, pl) in zip(calls["bartlett"], calls["prewhitened"]):
        pd.testing.assert_series_equal(b, p)
        assert bl == pl


@pytest.mark.parametrize("method", list(Method))
def test_alpha_comparison_text_fits_its_table_cells(panel, method):
    data = compute(panel, alpha_inference=method)
    figure = views.plot_performance_inference(data, "Same-sample alpha comparison")
    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        table = figure.axes[3].tables[0]
        for cell in table.get_celld().values():
            outer = cell.get_window_extent(renderer)
            text = cell.get_text().get_window_extent(renderer)
            assert outer.x0 <= text.x0 < text.x1 <= outer.x1
            assert outer.y0 <= text.y0 < text.y1 <= outer.y1
    finally:
        plt.close(figure)

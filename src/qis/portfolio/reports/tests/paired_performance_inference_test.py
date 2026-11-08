"""Paired appendix: independent covariance references, eligibility and page-one invariance."""

import copy
from types import SimpleNamespace
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import qis
from qis.datasets.synthetic import generate_synthetic_prices
from qis.perfstats.significance import InterceptInference
from qis.portfolio.reports._paired_performance_inference import compute_paired_performance_inference
from qis.portfolio.reports._performance_inference import plot_performance_inference
from qis.portfolio.reports import strategy_benchmark_factsheet as report


@pytest.fixture(scope="module")
def panel():
    return generate_synthetic_prices(end="2025-12-31", apply_quirks=False)[
        ["SEQ_US", "SBD_TSY"]].resample("ME").last()


def portfolio(panel, asset):
    nav = panel[asset].rename(asset)
    return SimpleNamespace(ticker=asset, prices=panel,
                           weights=pd.DataFrame({asset: 1.0}, index=panel.index),
                           get_portfolio_nav=lambda: nav)


def compute(panel, **overrides):
    options = dict(strategy=portfolio(panel, "SEQ_US"), comparison=portfolio(panel, "SBD_TSY"),
                   benchmark_prices=panel, time_period=qis.TimePeriod("2005-01-01", "2025-12-31"),
                   freq="ME", freq_beta="ME", factor_beta_span=12)
    options.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return compute_paired_performance_inference(**options)


def dense_hac(values, lag, correction):
    centered = values - values.mean()
    distance = np.abs(np.subtract.outer(np.arange(len(values)), np.arange(len(values))))
    kernel = np.maximum(1 - distance / (lag + 1), 0)
    return np.sqrt(centered @ kernel @ centered / (len(values) * (len(values) - correction)))


def test_paired_sharpe_and_alpha_match_independent_dense_covariance(panel):
    rates = pd.Series(.03, index=panel.index)
    data = compute(panel, rates_data=rates)
    raw = panel / panel.shift(1) - 1
    cash = .03 * panel.index.to_series().diff().dt.days / 365
    expected = raw.subtract(cash, axis=0).iloc[1:]
    np.testing.assert_allclose(data.returns, expected, rtol=1e-12, atol=1e-14)
    values = expected.to_numpy()
    theta = values.mean(axis=0) / values.std(axis=0)
    z = (values - values.mean(axis=0)) / values.std(axis=0)
    scores = z - .5 * theta * (z ** 2 - 1)
    row = data.windows.iloc[0]
    paired_se = np.sqrt(12) * dense_hac(scores[:, 0] - scores[:, 1], int(row.hac_lags), 0)
    assert row.sharpe == pytest.approx(np.sqrt(12) * (theta[0] - theta[1]))
    assert row.hac_se == pytest.approx(paired_se)
    np.testing.assert_allclose([row.hac_lower, row.hac_upper],
                               row.sharpe + norm.isf(.025) * paired_se * np.array([-1, 1]))
    assert row.pvalue == pytest.approx(2 * norm.sf(abs(row.sharpe / paired_se)))
    independent_se = np.sqrt(12 * sum(dense_hac(scores[:, i], int(row.hac_lags), 0) ** 2
                                    for i in range(2)))
    assert abs(paired_se - independent_se) > .001  # covariance cannot be ignored
    diagnostic = data.alpha_diagnostics
    expected_alpha = (expected.SEQ_US - expected.SBD_TSY -
                      (diagnostic.Strategy_lagged_beta - diagnostic.Comparison_lagged_beta) *
                      expected.SEQ_US)
    np.testing.assert_allclose(data.alpha_returns, expected_alpha, equal_nan=True, atol=1e-14)
    alpha = expected_alpha.dropna().to_numpy()
    row = data.alpha_windows.iloc[0]
    reference_se = 12 * dense_hac(alpha, int(row.hac_lags), 1)
    assert row.alpha == pytest.approx(12 * alpha.mean())
    assert row.hac_se == pytest.approx(reference_se)
    assert row.pvalue == pytest.approx(2 * norm.sf(abs(row.alpha / reference_se)))
    np.testing.assert_allclose([row.hac_lower, row.hac_upper],
                               row.alpha + norm.isf(.025) * reference_se * np.array([-1, 1]))


def test_swap_reverses_sign_and_bounds_not_pvalues(panel):
    forward = compute(panel)
    reverse = compute(panel, strategy=portfolio(panel, "SBD_TSY"),
                      comparison=portfolio(panel, "SEQ_US"))
    for a, b, estimate in ((forward.windows, reverse.windows, "sharpe"),
                           (forward.alpha_windows, reverse.alpha_windows, "alpha")):
        np.testing.assert_allclose(a[estimate], -b[estimate], atol=1e-12)
        np.testing.assert_allclose(a.hac_lower, -b.hac_upper, atol=1e-12)
        np.testing.assert_allclose(a.pvalue, b.pvalue, atol=1e-12)


def test_identical_portfolios_are_exactly_zero_with_p_one(panel):
    one = portfolio(panel, "SEQ_US")
    data = compute(panel, strategy=one, comparison=one)
    for frame, estimate in ((data.windows, "sharpe"), (data.alpha_windows, "alpha")):
        assert (frame[[estimate, "hac_se", "hac_lower", "hac_upper"]] == 0).all().all()
        assert (frame.pvalue == 1).all()


def test_only_first_market_and_separate_frequency_with_measurement_boundary(panel):
    options = dict(time_period=qis.TimePeriod("2010-12-31", "2025-12-31"), freq_beta="QE")
    all_markets = compute(panel, **options)
    first = compute(panel, benchmark_prices=panel.iloc[:, :1], **options)
    pd.testing.assert_series_equal(all_markets.alpha_returns, first.alpha_returns)
    assert first.metadata["alpha"]["benchmark"] == "SEQ_US"
    assert first.metadata["af"] == 12 and first.metadata["alpha"]["af"] == 4
    assert first.alpha_returns.index[0] == pd.Timestamp("2011-03-31")
    assert pd.notna(first.alpha_returns.iloc[0])  # pre-report prices warm beta, not realised alpha
    assert first.alpha_windows.iloc[0].alpha == pytest.approx(4 * first.alpha_returns.mean())
    assert first.alpha_windows.loc["Recent 5 years", "n_obs"] == 20
    assert pd.isna(first.alpha_windows.loc["Recent 5 years", "alpha"])


def test_pairwise_trim_and_no_interior_condensation(panel):
    late = panel.copy()
    late.loc[:"2009-12-31", "SBD_TSY"] = np.nan
    data = compute(late)
    assert data.windows.iloc[0].start == pd.Timestamp("2010-02-28")
    late.loc["2015-06-30", "SBD_TSY"] = np.nan
    gap = compute(late)
    assert pd.isna(gap.windows.iloc[0].sharpe)
    assert pd.notna(gap.windows.loc["Recent 3 years", "sharpe"])
    assert any("interior missing" in issue for issue in gap.issues)


def test_both_holdings_must_be_known_and_warmed(panel):
    reference = compute(panel)
    assert reference.alpha_returns.iloc[:13].isna().all()
    assert reference.alpha_windows.iloc[0].start == reference.alpha_returns.first_valid_index()
    second = portfolio(panel, "SBD_TSY")
    second.weights.loc["2015-06-30"] = np.nan
    gap = compute(panel, comparison=second)
    assert pd.isna(gap.alpha_returns.loc["2015-07-31"])
    assert pd.isna(gap.alpha_windows.iloc[0].alpha)
    pd.testing.assert_frame_equal(gap.windows, reference.windows)
    second.weights = None
    missing = compute(panel, comparison=second)
    assert missing.alpha_returns.isna().all()
    assert missing.windows.sharpe.notna().all()
    assert any("holdings unavailable" in issue for issue in missing.issues)


def test_future_prices_holdings_and_returns_do_not_change_past_views(panel):
    original = compute(panel)
    changed = panel.copy()
    changed.loc["2020":] *= [1.8, .7]
    second = portfolio(changed, "SBD_TSY")
    second.weights.loc["2020":] *= 2
    modified = compute(changed, comparison=second)
    pd.testing.assert_series_equal(original.alpha_returns.loc[:"2019"],
                                   modified.alpha_returns.loc[:"2019"])
    for key in ("expanding", "rolling", "alpha_expanding", "alpha_rolling"):
        a, b = getattr(original, key), getattr(modified, key)
        pd.testing.assert_frame_equal(a.loc[a.date < "2020"], b.loc[b.date < "2020"])


@pytest.mark.parametrize(("years", "expected"), [(4, 3), (2, 2), (1, None)])
def test_short_histories_fallback_and_warn(panel, years, expected):
    data = compute(panel, time_period=qis.TimePeriod(
        panel.index[-1] - pd.DateOffset(years=years), panel.index[-1]))
    assert data.metadata["rolling_years"] == expected
    assert any("fallback" in issue or "panel unavailable" in issue for issue in data.issues)
    if years == 1:
        assert data.windows.sharpe.isna().all()


def test_incompatible_alpha_frequency_and_absent_market_preserve_sharpe(panel):
    fine = compute(panel, freq_beta="B")
    assert fine.alpha_returns.empty and fine.windows.sharpe.notna().all()
    assert any("sampling grid unavailable" in issue for issue in fine.issues)
    missing = compute(panel, benchmark_prices=None)
    assert missing.alpha_returns.isna().all() and missing.windows.sharpe.notna().all()
    fine_sharpe = compute(panel, freq="B")
    assert fine_sharpe.returns.empty and fine_sharpe.windows.sharpe.isna().all()
    assert fine_sharpe.alpha_windows.alpha.notna().all()
    assert any("Paired Sharpe: sampling grid unavailable" in issue for issue in fine_sharpe.issues)


def test_layout_links_and_curve_endpoints(panel):
    data = compute(panel)
    assert data.expanding.iloc[-1].estimate == pytest.approx(data.windows.iloc[0].sharpe)
    assert data.alpha_expanding.iloc[-1].estimate == pytest.approx(data.alpha_windows.iloc[0].alpha)
    figure = plot_performance_inference(data, "Strategy vs Comparison", figsize=(8.3, 11.7))
    try:
        figure.canvas.draw()
        assert len(figure.axes) == 6
        assert figure.axes[0].get_position().x1 < figure.axes[3].get_position().x0
        assert figure.axes[1].get_ylabel() == "Annualised ΔSharpe"
        assert figure.axes[4].get_ylabel() == "Annualised marginal alpha"
        assert len([text for text in figure.texts if text.get_url()]) == 3
        for text in figure.texts:
            bounds = text.get_window_extent(figure.canvas.get_renderer())
            assert 0 <= bounds.x0 < bounds.x1 <= figure.bbox.width
    finally:
        plt.close(figure)


@pytest.mark.parametrize("alpha_inference", list(InterceptInference))
def test_default_page_before_brinson_nondefault_indices_and_identical_page_one(
        monkeypatch, alpha_inference):
    prices = generate_synthetic_prices(end="2014-12-31", apply_quirks=False)[
        ["SEQ_US", "SBD_TSY"]]
    strategy = qis.backtest_model_portfolio(prices=prices, weights=np.array([.7, .3]), ticker="TAA")
    comparison = qis.backtest_model_portfolio(
        prices=prices, weights=np.array([.4, .6]), ticker="SAA")
    multi = qis.MultiPortfolioData([comparison, strategy], benchmark_prices=prices)
    before = copy.deepcopy(multi.navs)
    captured = []
    actual = report.compute_paired_performance_inference

    def capture(**kwargs):
        assert kwargs["alpha_inference"] is alpha_inference
        assert kwargs["strategy"] is strategy and kwargs["comparison"] is comparison
        result = actual(**kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(report, "compute_paired_performance_inference", capture)
    options = dict(strategy_idx=1, benchmark_idx=0, alpha_inference=alpha_inference,
                   time_period=qis.TimePeriod("2008", "2014-12-31"),
                   is_grouped=False, perf_params=qis.PerfParams(freq="ME"),
                   freq_beta="ME", factor_beta_span=12)
    figures = report.generate_strategy_benchmark_factsheet_plt(multi, **options)
    embedded_flags = []

    def embedded(**kwargs):
        assert kwargs["alpha_inference"] is alpha_inference
        embedded_flags.append(kwargs["add_performance_inference_sheet"])
        return []  # Isolate forwarding; full embedded reports have integration coverage.

    monkeypatch.setattr(report, "generate_strategy_factsheet", embedded)
    legacy = report.generate_strategy_benchmark_factsheet_plt(
        multi, add_performance_inference_sheet=False, add_strategy_factsheet=True, **options)
    try:
        assert len(figures) == 3 and len(legacy) == 2 and len(captured) == 1
        assert embedded_flags == [False, False]
        assert "Incremental performance" in figures[1].get_suptitle()
        assert "Brinson" in figures[2].get_suptitle()
        assert figures[1].get_suptitle().startswith("TAA vs SAA")
        for fig in (figures[0], legacy[0]):
            fig.canvas.draw()
        np.testing.assert_array_equal(np.asarray(figures[0].canvas.buffer_rgba()),
                                      np.asarray(legacy[0].canvas.buffer_rgba()))
        pd.testing.assert_frame_equal(multi.navs, before)
    finally:
        for fig in figures + legacy:
            plt.close(fig)

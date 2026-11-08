"""Multi-strategy paired inference: frozen family, independent Holm and unchanged first page."""

import copy
from types import SimpleNamespace
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import qis
from qis.datasets.synthetic import generate_synthetic_prices
from qis.perfstats.significance import InterceptInference
from qis.portfolio.reports import _multi_performance_inference as inference
from qis.portfolio.reports import multi_strategy_factsheet as report
from qis.portfolio.reports.tests.paired_performance_inference_test import portfolio, compute


@pytest.fixture(scope="module")
def panel():
    return generate_synthetic_prices(end="2025-12-31", apply_quirks=False)[
        ["SEQ_US", "SBD_TSY", "SCM_GLD"]].resample("ME").last()


def multi(panel, count=3):
    portfolios = [portfolio(panel, panel.columns[i % 3]) for i in range(count)]
    return SimpleNamespace(portfolio_datas=portfolios, benchmark_prices=panel,
                           get_navs=lambda: panel)


def calculate(data, **overrides):
    options = dict(freq="ME", freq_beta="ME", factor_beta_span=12)
    options.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return inference.compute_multi_performance_inference(data, **options)


def holm_reference(values):
    """Sorted step-down products and running maximum, without calling a correction library."""
    values = np.asarray(values)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    largest = 0.0
    for rank, index in enumerate(order):
        largest = max(largest, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, largest)
    return adjusted


def test_actual_pairs_first_market_and_independent_holm(panel):
    result = calculate(multi(panel))
    assert result.metadata["base_portfolio_id"] == 0
    assert result.metadata["portfolio_ids"] == [1, 2]
    assert result.metadata["family_size"] == 12
    assert result.metadata["family_members"] == tuple(result.summary.index)
    assert result.metadata["adjustment_status"] == "complete"
    np.testing.assert_allclose(result.summary.pvalue_holm,
                               holm_reference(result.summary.pvalue))
    for i in (1, 2):
        reference = compute(panel, strategy=portfolio(panel, panel.columns[i]),
                            comparison=portfolio(panel, "SEQ_US"),
                            benchmark_prices=panel.iloc[:, :1])
        pd.testing.assert_series_equal(result.pairs[i].alpha_returns, reference.alpha_returns)
        for metric, table in (("sharpe", reference.windows), ("alpha", reference.alpha_windows)):
            actual = result.summary.xs((i, metric), level=("portfolio_id", "metric"))
            np.testing.assert_allclose(actual.estimate, table[metric])
            np.testing.assert_allclose(actual.pvalue, table.pvalue)
            np.testing.assert_allclose(actual.ci_lower, table.hac_lower)
            assert result.pairs[i].metadata["alpha"]["benchmark"] == "SEQ_US"


def test_nondefault_base_and_identity_by_position(panel):
    data = multi(panel)
    # Duplicate labels cannot erase or merge hypotheses.
    for item in data.portfolio_datas:
        item.ticker = "Same label"
    result = calculate(data, base_portfolio_id=2)
    assert list(result.pairs) == [0, 1]
    assert result.summary.index.is_unique
    reference = compute(panel, strategy=portfolio(panel, "SEQ_US"),
                        comparison=portfolio(panel, "SCM_GLD"))
    np.testing.assert_allclose(result.summary.xs((0, "sharpe"), level=(0, 1)).estimate,
                               reference.windows.sharpe)


@pytest.mark.parametrize("base", [-1, 3, True, np.bool_(False), .5, "0"])
def test_invalid_base_rejected_before_figures(panel, base):
    before = plt.get_fignums()
    with pytest.raises(ValueError, match="base_portfolio_id"):
        report.generate_multi_portfolio_factsheet(multi(panel), base_portfolio_id=base)
    assert plt.get_fignums() == before


def test_pair_specific_samples_and_incomplete_family_do_not_shrink(panel):
    baseline = calculate(multi(panel))
    late = panel.copy()
    late.loc[:"2024-01-31", "SCM_GLD"] = np.nan
    data = multi(late)
    with pytest.warns(UserWarning, match="Holm unavailable.*family not reduced"):
        result = inference.compute_multi_performance_inference(
            data, freq="ME", freq_beta="ME", factor_beta_span=12)
    pd.testing.assert_frame_equal(result.pairs[1].windows, baseline.pairs[1].windows)
    assert result.summary.loc[(2, "sharpe", "Full history"), "n_obs"] < 24
    assert len(result.summary) == 12
    assert result.summary.pvalue_holm.isna().all()
    assert result.summary.loc[1].pvalue.notna().all()
    assert result.summary.loc[2].pvalue.isna().all()
    assert len(result.metadata["unavailable_family_members"]) == 6
    gap = panel.copy()
    gap.loc["2015-06-30", "SCM_GLD"] = np.nan
    missing = calculate(multi(gap))
    assert pd.isna(missing.summary.loc[(2, "sharpe", "Full history"), "estimate"])
    assert pd.notna(missing.summary.loc[(2, "sharpe", "Recent 3 years"), "estimate"])
    assert missing.summary.pvalue_holm.isna().all()


def test_single_portfolio_skips_and_identical_portfolios_are_zero(panel):
    with pytest.warns(UserWarning, match="No non-base portfolios"):
        single = inference.compute_multi_performance_inference(multi(panel, count=1))
    assert single.summary.empty and inference.plot_multi_performance_inference(single) == []
    data = multi(panel, count=2)
    data.portfolio_datas[1] = data.portfolio_datas[0]
    result = calculate(data)
    assert (result.summary.estimate == 0).all()
    assert (result.summary.pvalue_holm == 1).all()


def test_small_pvalues_do_not_display_as_exact_zero():
    assert inference._pvalue_label(.00001) == "<0.001"
    assert inference._pvalue_label(.001) == "0.001"
    assert inference._pvalue_label(np.nan) == "N/A"


@pytest.fixture(scope="module")
def pair_template(panel):
    return compute(panel)


def stub_pairs(monkeypatch, panel, pair_template, count):
    """Predetermined p-values exercise pagination, not another data simulation/calibration."""
    data = multi(panel, count=count)
    for i, item in enumerate(data.portfolio_datas):
        item.ticker = f"Variant {i} with a long display name"
    calls = []

    def paired(**kwargs):
        position = next(i for i, item in enumerate(data.portfolio_datas)
                        if item is kwargs["strategy"])
        assert kwargs["comparison"] is data.portfolio_datas[0]
        calls.append(position)
        result = copy.deepcopy(pair_template)
        result.issues = []
        for offset, frame in enumerate((result.windows, result.alpha_windows)):
            frame["pvalue"] = [.0001 * (6 * position + 3 * offset + j) for j in range(3)]
        return result

    monkeypatch.setattr(inference, "compute_paired_performance_inference", paired)
    return data, calls


@pytest.mark.parametrize("count,pages", [(7, 1), (8, 2), (14, 3)])
def test_pagination_whole_family_reference_and_text_bounds(
        panel, pair_template, monkeypatch, count, pages, tmp_path):
    data, calls = stub_pairs(monkeypatch, panel, pair_template, count)
    result = calculate(data)
    assert calls == list(range(1, count))
    assert result.metadata["family_size"] == (count - 1) * 6
    np.testing.assert_allclose(result.summary.pvalue_holm,
                               holm_reference(result.summary.pvalue))
    # Splitting by metric/window/page would yield different adjusted values.
    assert not np.allclose(result.summary.pvalue_holm.iloc[:6],
                           holm_reference(result.summary.pvalue.iloc[:6]))
    figures = inference.plot_multi_performance_inference(result)
    try:
        assert len(figures) == pages
        seen = []
        for i, figure in enumerate(figures):
            figure.canvas.draw()
            assert len(figure.axes) == 6
            assert len([text for text in figure.texts if text.get_url()]) == 3
            labels = [text.get_text() for text in figure.axes[0].texts if ": Variant" in
                      text.get_text()]
            seen.extend(int(text.split(":")[0]) for text in labels)
            for column, axis in enumerate(figure.axes):
                # Alpha has a third (alternate-method) line; six rows must still fit.
                stride = 5 if column % 2 else 4
                for start in range(1, len(axis.texts) - 3, stride):
                    row = axis.texts[start:start + stride]
                    pairs = [(row[0], row[1]), (row[2], row[3])]
                    if stride == 5:
                        pairs.append((row[3], row[4]))
                    for upper, lower in pairs:
                        a = upper.get_window_extent(figure.canvas.get_renderer())
                        b = lower.get_window_extent(figure.canvas.get_renderer())
                        assert a.y0 >= b.y1, (upper.get_text(), lower.get_text())
            for text in figure.findobj(plt.Text):
                if text.get_visible() and text.get_text():
                    bounds = text.get_window_extent(figure.canvas.get_renderer())
                    assert 0 <= bounds.x0 < bounds.x1 <= figure.bbox.width, text.get_text()
                    assert 0 <= bounds.y0 < bounds.y1 <= figure.bbox.height, text.get_text()
            figure.savefig(tmp_path / f"overview-{i}.pdf")
        assert seen == list(range(1, count))
        for metric in (0, 1):
            # Last three labels are the shared forest-axis ticks on every page/window.
            expected = [text.get_text() for text in figures[0].axes[metric].texts[-3:]]
            for figure in figures:
                for row in range(3):
                    labels = figure.axes[2 * row + metric].texts[-3:]
                    assert [text.get_text() for text in labels] == expected
    finally:
        for figure in figures:
            plt.close(figure)


def test_missing_one_declared_test_disables_entire_adjustment(panel, pair_template, monkeypatch):
    data, _ = stub_pairs(monkeypatch, panel, pair_template, 3)
    actual = inference.compute_paired_performance_inference

    def missing(**kwargs):
        result = actual(**kwargs)
        if kwargs["strategy"] is data.portfolio_datas[2]:
            result.alpha_windows.loc["Recent 3 years", "pvalue"] = np.nan
        return result

    monkeypatch.setattr(inference, "compute_paired_performance_inference", missing)
    result = calculate(data)
    assert len(result.summary) == 12 and result.summary.pvalue.notna().sum() == 11
    assert result.summary.pvalue_holm.isna().all()
    assert result.metadata["unavailable_family_members"] == ((2, "alpha", "Recent 3 years"),)


@pytest.mark.parametrize("alpha_inference", list(InterceptInference))
def test_default_order_unchanged_first_page_details_and_forwarding(monkeypatch, alpha_inference):
    prices = generate_synthetic_prices(end="2014-12-31", apply_quirks=False)[
        ["SEQ_US", "SBD_TSY"]]
    portfolios = [qis.backtest_model_portfolio(prices=prices, weights=np.array(weights),
                                              ticker=name)
                  for weights, name in (([.7, .3], "Base"), ([.4, .6], "Alternative"))]
    data = qis.MultiPortfolioData(portfolios, benchmark_prices=prices)
    before = copy.deepcopy(data.navs)
    options = dict(time_period=qis.TimePeriod("2008", "2014-12-31"),
                   alpha_inference=alpha_inference,
                   perf_params=qis.PerfParams(freq="ME"), freq_beta="ME", factor_beta_span=12)
    figures = report.generate_multi_portfolio_factsheet(data, **options)
    forwarded = []

    def embedded(**kwargs):
        assert kwargs["alpha_inference"] is alpha_inference
        forwarded.append(kwargs["add_performance_inference_sheet"])
        return []

    monkeypatch.setattr(report, "generate_strategy_factsheet", embedded)
    legacy = report.generate_multi_portfolio_factsheet(
        data, add_performance_inference_sheet=False, add_strategy_factsheets=True, **options)
    group_figure = plt.figure()
    group_figure.suptitle("Group sentinel")
    monkeypatch.setattr(data, "plot_group_exposures_and_pnl", lambda **kwargs: [group_figure])
    details = report.generate_multi_portfolio_factsheet(
        data, add_pairwise_inference_sheets=True, add_group_exposures_and_pnl=True, **options)
    try:
        assert len(figures) == 2 and len(legacy) == 1 and len(details) == 4
        assert forwarded == [False, False]
        assert "Incremental performance vs base" in figures[1].get_suptitle()
        assert "Alternative vs Base" in details[2].get_suptitle()
        primary = "PW" if alpha_inference == InterceptInference.PREWHITENED_AR1 else "B"
        assert any(f"Primary: {primary} |" in text.get_text() for text in details[2].texts)
        assert details[3] is group_figure
        for fig in (figures[0], legacy[0]):
            fig.canvas.draw()
        np.testing.assert_array_equal(np.asarray(figures[0].canvas.buffer_rgba()),
                                      np.asarray(legacy[0].canvas.buffer_rgba()))
        pd.testing.assert_frame_equal(data.navs, before)
    finally:
        for fig in figures + legacy + details:
            plt.close(fig)

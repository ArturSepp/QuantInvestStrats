"""Opt-in exposure coverage: independent beta reference, timing and missing-data contracts."""

import inspect
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import qis
from qis.datasets.synthetic import generate_synthetic_prices
from qis.portfolio.reports.alpha_coverage import HoldingsAlphaCoverage
from qis.portfolio.reports._holdings_alpha import compute_holdings_alpha
from qis.portfolio.reports.tests.holdings_alpha_test import independent_beta


@pytest.fixture(scope="module")
def sample():
    prices = generate_synthetic_prices(end="2025-12-31", apply_quirks=False)[
        ["SEQ_US", "SBD_TSY", "SCM_GLD"]].resample("ME").last()
    assets = prices[["SEQ_US", "SBD_TSY"]].copy()
    weights = assets * 0 + [0.8, -0.3]
    benchmark = prices[["SEQ_US"]].rename(columns={"SEQ_US": "Market"})
    returns = assets.pct_change(fill_method=None).mul(weights.shift()).sum(axis=1, skipna=False)
    assets.loc[:assets.index[70], "SBD_TSY"] = np.nan
    return dict(instrument_prices=assets, weights=weights, benchmark_prices=benchmark,
                portfolio_returns=returns.iloc[30:],
                benchmark_returns=benchmark.pct_change(fill_method=None).iloc[30:],
                freq_beta="ME", factor_beta_span=12), prices


def test_strict_none_and_zero_policy_keep_original_alpha(sample):
    options, _ = sample
    original = compute_holdings_alpha(**options)
    strict = compute_holdings_alpha(**options, alpha_coverage=None)
    zero = compute_holdings_alpha(**options, alpha_coverage=HoldingsAlphaCoverage(0, 0))
    pd.testing.assert_series_equal(original[0], strict[0])
    pd.testing.assert_series_equal(original[0], zero[0])
    assert zero[1].proxy_gross.eq(0).all() and zero[1].omitted_gross.eq(0).all()


def test_proxy_matches_independent_signed_beta_and_keeps_own_warmup(sample):
    options, prices = sample
    policy = HoldingsAlphaCoverage(proxy_map={"SBD_TSY": "Gold proxy"},
                                  proxy_prices=prices[["SCM_GLD"]].rename(
                                      columns={"SCM_GLD": "Gold proxy"}))
    alpha, diagnostic, meta, issues = compute_holdings_alpha(**options, alpha_coverage=policy)
    strict, strict_diagnostic, _, _ = compute_holdings_alpha(**options)
    expected = independent_beta(prices[["SEQ_US", "SCM_GLD"]],
                                options["weights"].set_axis(["SEQ_US", "SCM_GLD"], axis=1),
                                options["benchmark_prices"]).shift()
    # Young asset's first finite return is index 72; 13 observations end at 84; lag -> 85.
    transition = prices.index[85]
    earlier = diagnostic.index < transition
    np.testing.assert_allclose(diagnostic.loc[earlier, "lagged_beta"],
                               expected.reindex(diagnostic.index)[earlier], rtol=1e-12)
    np.testing.assert_allclose(alpha[earlier], options["portfolio_returns"][earlier]
                               - expected.reindex(alpha.index)[earlier]
                               * options["benchmark_returns"].iloc[:, 0][earlier], atol=1e-14)
    pd.testing.assert_series_equal(diagnostic.strict_alpha, strict, check_names=False)
    pd.testing.assert_series_equal(diagnostic.strict_complete_coverage,
                                  strict_diagnostic.complete_coverage, check_names=False)
    pd.testing.assert_series_equal(alpha.loc[transition:], strict.loc[transition:])
    assert diagnostic.loc[earlier, "proxy_gross"].eq(.3).all()
    assert diagnostic.loc[transition:, "proxy_gross"].eq(0).all()
    assert diagnostic.loc[earlier, "direct_gross"].eq(.8).all()
    assert meta["beta_warmup"] == 12 and meta["proxy_assisted"]
    assert any("proxy" in issue.lower() for issue in issues)


@pytest.mark.parametrize("weights,cap,eligible", [
    ([1e-6], 1e-5, True), ([1.001e-6], 1e-5, False),
    ([8e-7, -8e-7], 1e-6, False), ([5e-7, -5e-7], 1e-6, True),
])
def test_dust_uses_individual_and_gross_limits_without_renormalising(
        sample, weights, cap, eligible):
    options, _ = sample
    options = dict(options, instrument_prices=options["instrument_prices"][["SEQ_US"]],
                   weights=options["weights"][["SEQ_US"]].copy())
    for i, weight in enumerate(weights):
        options["weights"][f"Missing{i}"] = weight
    alpha, diagnostic, _, _ = compute_holdings_alpha(
        **options, alpha_coverage=HoldingsAlphaCoverage(total_weight_tolerance=cap))
    assert diagnostic.complete_coverage.all() == eligible
    if eligible:
        np.testing.assert_allclose(diagnostic.lagged_beta, .8, atol=1e-12)
        np.testing.assert_allclose(diagnostic.omitted_gross, np.abs(weights).sum())
        assert alpha.notna().all()
    else:
        assert alpha.isna().all()
    assert diagnostic.strict_alpha.isna().all()


def test_unknown_weights_cannot_be_excused_and_known_tiny_betas_are_retained(sample):
    options, prices = sample
    weights = options["weights"].copy()
    weights.SBD_TSY = 5e-7
    weights.loc[prices.index[40], "SBD_TSY"] = np.nan
    _, diagnostic, _, _ = compute_holdings_alpha(
        **dict(options, weights=weights), alpha_coverage=HoldingsAlphaCoverage())
    assert diagnostic.unknown_weights.loc[prices.index[41]]
    assert not diagnostic.complete_coverage.loc[prices.index[41]]
    assert pd.isna(diagnostic.total_gross.loc[prices.index[41]])
    assert diagnostic.omitted_gross.loc[prices.index[85]:].eq(0).all()


def test_proxy_needs_own_warmup_and_missing_dates_are_not_filled(sample):
    options, prices = sample
    proxies = prices[["SCM_GLD"]].copy()
    proxies.loc[:proxies.index[45]] = np.nan
    proxies.loc[proxies.index[65]] = np.nan
    policy = HoldingsAlphaCoverage(proxy_map={"SBD_TSY": "SCM_GLD"}, proxy_prices=proxies)
    alpha, diagnostic, _, _ = compute_holdings_alpha(**options, alpha_coverage=policy)
    assert alpha.loc[:prices.index[59]].isna().all()
    assert pd.notna(alpha.loc[prices.index[60]])
    assert alpha.loc[prices.index[66]:prices.index[67]].isna().all()
    assert diagnostic.uncovered_gross.loc[prices.index[66]] == .3


def test_proxy_inputs_do_not_leak_future_prices_or_change_sharpe_inputs(sample):
    options, prices = sample
    cut = prices.index[60]
    policies = []
    for multiplier in (1, 4):
        proxies = prices[["SCM_GLD"]].copy()
        proxies.loc[cut:] *= multiplier
        policies.append(HoldingsAlphaCoverage(proxy_map={"SBD_TSY": "SCM_GLD"},
                                             proxy_prices=proxies))
    first = compute_holdings_alpha(**options, alpha_coverage=policies[0])
    second = compute_holdings_alpha(**options, alpha_coverage=policies[1])
    pd.testing.assert_series_equal(first[0].loc[:cut], second[0].loc[:cut])
    pd.testing.assert_frame_equal(first[1].loc[:cut], second[1].loc[:cut])
    assert not np.isclose(first[0].loc[prices.index[61]], second[0].loc[prices.index[61]])


@pytest.mark.parametrize("kwargs", [
    {"weight_tolerance": -1}, {"weight_tolerance": np.nan},
    {"total_weight_tolerance": np.inf}, {"weight_tolerance": True},
    {"proxy_map": {"Asset": "Proxy"}}, {"proxy_map": {"Asset": 2}},
])
def test_bad_policy_fails_explicitly(kwargs):
    with pytest.raises(ValueError):
        HoldingsAlphaCoverage(**kwargs)


def test_bad_proxy_mapping_does_not_silently_revert_to_strict(sample):
    options, prices = sample
    policy = HoldingsAlphaCoverage(proxy_map={"Typo": "SCM_GLD"},
                                  proxy_prices=prices[["SCM_GLD"]])
    with pytest.raises(ValueError, match="absent"):
        compute_holdings_alpha(**options, alpha_coverage=policy)


def test_public_api_defaults_and_single_inference_metadata(sample):
    from qis.portfolio.reports._performance_inference import compute_performance_inference

    assert qis.HoldingsAlphaCoverage is HoldingsAlphaCoverage
    for function in (qis.generate_strategy_factsheet, qis.generate_strategy_benchmark_factsheet_plt,
                     qis.generate_multi_portfolio_factsheet):
        parameter = inspect.signature(function).parameters["alpha_coverage"]
        assert parameter.default is None and parameter.kind == inspect.Parameter.KEYWORD_ONLY
    options, prices = sample
    common = dict(nav=prices.SEQ_US, benchmark_prices=options["benchmark_prices"],
                  time_period=qis.TimePeriod(prices.index[30], prices.index[-1]), freq="ME",
                  instrument_prices=options["instrument_prices"], weights=options["weights"],
                  freq_beta="ME", factor_beta_span=12)
    policy = HoldingsAlphaCoverage(proxy_map={"SBD_TSY": "SCM_GLD"},
                                  proxy_prices=prices[["SCM_GLD"]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        strict = compute_performance_inference(**common)
        assisted = compute_performance_inference(**common, alpha_coverage=policy)
    pd.testing.assert_frame_equal(strict.windows, assisted.windows)
    assert assisted.alpha_windows.n_obs.iloc[0] > strict.alpha_windows.n_obs.iloc[0]
    assert assisted.metadata["alpha"]["proxy_assisted"]


def test_paired_and_multi_retain_same_family_and_covariance(sample):
    from qis.portfolio.reports.tests.multi_performance_inference_test import calculate, multi
    from qis.portfolio.reports.tests.paired_performance_inference_test import portfolio, compute

    _, prices = sample
    data = multi(prices)
    # NAV history is unchanged: only one portfolio's held-instrument beta history is young.
    data.portfolio_datas[1].prices = prices.copy()
    data.portfolio_datas[1].prices.loc[:prices.index[70], "SBD_TSY"] = np.nan
    policy = HoldingsAlphaCoverage(proxy_map={"SBD_TSY": "SCM_GLD"},
                                  proxy_prices=prices[["SCM_GLD"]])
    result = calculate(data, alpha_coverage=policy)
    strict = calculate(data)
    assert result.metadata["family_members"] == strict.metadata["family_members"]
    for position, pair in result.pairs.items():
        pd.testing.assert_frame_equal(pair.windows, strict.pairs[position].windows)
        d = pair.alpha_diagnostics
        expected = d.Strategy_alpha - d.Comparison_alpha
        pd.testing.assert_series_equal(pair.alpha_returns, expected, check_names=False)
        assert pair.metadata["alpha"]["portfolios"]["Comparison"]["proxy_map"] == {}
    assert result.pairs[1].metadata["alpha"]["proxy_assisted"]
    swapped = compute(prices, strategy=portfolio(prices, "SEQ_US"),
                      comparison=data.portfolio_datas[1], alpha_coverage=policy)
    np.testing.assert_allclose(result.pairs[1].alpha_returns, -swapped.alpha_returns,
                               equal_nan=True, atol=1e-14)


@pytest.mark.parametrize("kind", ["bad_column", "zero", "infinity", "duplicate", "unsorted"])
def test_invalid_proxy_history_rejected(sample, kind):
    _, prices = sample
    proxies = prices[["SCM_GLD"]].copy()
    if kind == "bad_column":
        proxies.columns = ["Wrong"]
    elif kind == "zero":
        proxies.iloc[10, 0] = 0
    elif kind == "infinity":
        proxies.iloc[10, 0] = np.inf
    elif kind == "duplicate":
        proxies = pd.concat([proxies.iloc[:1], proxies])
    else:
        proxies = proxies.iloc[::-1]
    with pytest.raises(ValueError):
        HoldingsAlphaCoverage(proxy_map={"SBD_TSY": "SCM_GLD"}, proxy_prices=proxies)


def test_audit_paginates_all_mappings_and_exclusions_without_overflow(sample, tmp_path):
    from qis.portfolio.reports._alpha_coverage_report import plot_alpha_coverage
    from qis.portfolio.reports._performance_inference import compute_performance_inference

    options, prices = sample
    policy = HoldingsAlphaCoverage(proxy_map={"SBD_TSY": "SCM_GLD"},
                                  proxy_prices=prices[["SCM_GLD"]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = compute_performance_inference(
            prices.SEQ_US, options["benchmark_prices"],
            qis.TimePeriod(prices.index[30], prices.index[-1]), "ME",
            instrument_prices=options["instrument_prices"], weights=options["weights"],
            freq_beta="ME", factor_beta_span=12, alpha_coverage=policy)
    # Stress layout only, not estimator inputs: many disjoint exclusions and long mapping IDs.
    result.alpha_returns.iloc[::3] = np.nan
    result.metadata["alpha"]["proxy_map"].update(
        {f"Long instrument identifier {i}": f"Explicit proxy identifier {i}" for i in range(30)})
    figures = plot_alpha_coverage(result, "Synthetic coverage audit")
    try:
        assert len(figures) > 2
        text = "\n".join(item.get_text() for figure in figures for item in figure.texts)
        for date in result.alpha_returns.index[::3]:
            assert date.strftime("%d%b%Y") in text
        for instrument in result.metadata["alpha"]["proxy_map"]:
            assert instrument in text
        for i, figure in enumerate(figures):
            figure.canvas.draw()
            for item in figure.texts:
                extent = item.get_window_extent(figure.canvas.get_renderer())
                assert extent.x0 >= 0 and extent.x1 <= figure.bbox.width
                assert extent.y0 >= 0 and extent.y1 <= figure.bbox.height
            figure.savefig(tmp_path / f"coverage-{i}.png")
    finally:
        for figure in figures:
            plt.close(figure)


@pytest.mark.parametrize("kind", ["single", "paired", "multi"])
def test_factsheets_forward_policy_keep_page_one_and_add_audit(kind, monkeypatch):
    from qis.portfolio.reports import strategy_factsheet as single
    from qis.portfolio.reports import strategy_benchmark_factsheet as paired
    from qis.portfolio.reports import multi_strategy_factsheet as multi

    prices = generate_synthetic_prices(end="2014-12-31", apply_quirks=False)[
        ["SEQ_US", "SBD_TSY"]]
    portfolios = [qis.backtest_model_portfolio(prices, weights=np.array(weights), ticker=ticker)
                  for weights, ticker in (([.6, .4], "TAA"), ([.4, .6], "SAA"))]
    data = qis.MultiPortfolioData(portfolio_datas=portfolios, benchmark_prices=prices[["SEQ_US"]])
    options = dict(time_period=qis.TimePeriod("2005", "2014-12-31"),
                   perf_params=qis.PerfParams(freq="ME"), freq_beta="QE", factor_beta_span=12)
    policy = HoldingsAlphaCoverage(proxy_map={"SBD_TSY": "Bond proxy"},
                                  proxy_prices=prices[["SBD_TSY"]].rename(
                                      columns={"SBD_TSY": "Bond proxy"}))
    captured, embedded_policies = [], []
    module, function = {"single": (single, single.generate_strategy_factsheet),
                        "paired": (paired, paired.generate_strategy_benchmark_factsheet_plt),
                        "multi": (multi, multi.generate_multi_portfolio_factsheet)}[kind]
    compute_name = {"single": "compute_performance_inference",
                    "paired": "compute_paired_performance_inference",
                    "multi": "compute_multi_performance_inference"}[kind]
    original = getattr(module, compute_name)

    def capture(*args, **kwargs):
        captured.append(kwargs.get("alpha_coverage"))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, compute_name, capture)
    if kind == "single":
        arguments = (portfolios[0],)
        options.update(benchmark_prices=prices[["SEQ_US"]],
                       add_current_position_var_risk_sheet=False)
    else:
        arguments = (data,)

        def embedded(**kwargs):
            embedded_policies.append(kwargs["alpha_coverage"])
            return []

        monkeypatch.setattr(module, "generate_strategy_factsheet", embedded)
        if kind == "paired":
            options.update(add_brinson_attribution=False, add_strategy_factsheet=True)
        else:
            options.update(add_group_exposures_and_pnl=False, add_strategy_factsheets=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        old = function(*arguments, **options)
        new = function(*arguments, **options, alpha_coverage=policy)
    try:
        assert captured == [None, policy]
        if kind != "single":
            assert embedded_policies[:2] == [None, None]
            assert len(embedded_policies) == 4
            assert all(item.proxy_map == policy.proxy_map for item in embedded_policies[2:])
        assert len(new) > len(old)
        assert any("coverage audit" in figure.get_suptitle() for figure in new)
        for figure in (old[0], new[0]):
            figure.canvas.draw()
        np.testing.assert_array_equal(np.asarray(old[0].canvas.buffer_rgba()),
                                       np.asarray(new[0].canvas.buffer_rgba()))
    finally:
        for figure in old + new:
            plt.close(figure)


"""Factsheet forwarding and rendered sleeve-accounting integration tests."""
import matplotlib.pyplot as plt
import pandas as pd
import pytest
import qis

from qis.portfolio.tests.brinson_wrapper_test import make_portfolios
from qis.portfolio.reports.strategy_factsheet import generate_strategy_factsheet
from qis.portfolio.reports.multi_strategy_factsheet import generate_multi_portfolio_factsheet
from qis.portfolio.reports.strategy_benchmark_factsheet import (
    generate_strategy_benchmark_factsheet_plt, generate_strategy_benchmark_active_perf_plt,
)


@pytest.mark.parametrize("kind,module,func", [
    ("strategy", "qis.portfolio.reports.strategy_factsheet", "generate_strategy_factsheet"),
    ("multi_strategy", "qis.portfolio.reports.multi_strategy_factsheet",
     "generate_multi_portfolio_factsheet"),
    ("strategy_benchmark", "qis.portfolio.reports.strategy_benchmark_factsheet",
     "generate_strategy_benchmark_factsheet_plt"),
])
def test_facade_forwards_separate_inputs(monkeypatch, kind, module, func):
    """The facade forwards the same Series and matrix without mutating either."""
    import importlib
    a, b = make_portfolios()
    matrix = pd.DataFrame({"PE": [.6], "Credit": [.4]}, index=["SEQ_US"])
    labels = a.group_data.combine_first(b.group_data)
    received = {}
    def capture(**kwargs):
        received.update(kwargs)
        return []
    monkeypatch.setattr(importlib.import_module(module), func, capture)
    data = a if kind == "strategy" else qis.MultiPortfolioData([a, b])
    result = qis.factsheet(data, kind=kind, benchmark_prices=b.nav.to_frame(),
                          group_data=labels, lookthrough_matrix=matrix)
    assert result == []
    assert received["group_data"] is labels
    assert received["lookthrough_matrix"] is matrix


@pytest.mark.parametrize("report", ["strategy", "multi", "benchmark", "active"])
def test_generators_apply_partition_to_calculations(monkeypatch, report):
    """Render all portfolio generators and observe fractional inputs at calculation boundaries."""
    a, b = make_portfolios(.001)
    matrix = pd.DataFrame({"PE": [.6], "Credit": [.4]}, index=["SEQ_US"])
    multi = qis.MultiPortfolioData([a, b], benchmark_prices=b.nav.to_frame())
    period = qis.TimePeriod("2020-12-31", "2021-06-30")
    options = qis.fetch_default_report_kwargs(
        time_period=period, reporting_frequency=qis.ReportingFrequency.MONTHLY,
        add_rates_data=False)
    seen = []
    original = qis.PortfolioData.get_weights
    def weights(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if kwargs.get("is_grouped") and kwargs.get("lookthrough_matrix") is not None:
            seen.append(result)
            assert "PE" in result and "Credit" in result
        return result
    monkeypatch.setattr(qis.PortfolioData, "get_weights", weights)
    before_a, before_b = a.group_data.copy(), b.group_data.copy()
    figs = []
    try:
        if report == "strategy":
            figs = generate_strategy_factsheet(
                a, b.nav.to_frame(), time_period=period, is_grouped=True,
                lookthrough_matrix=matrix, add_grouped_weights_sheet=True,
                add_grouped_exposures=True, add_grouped_cum_pnl=True, **options)
        elif report == "multi":
            figs = generate_multi_portfolio_factsheet(
                multi, time_period=period, lookthrough_matrix=matrix,
                add_group_exposures_and_pnl=True, **options)
        elif report == "benchmark":
            figs = generate_strategy_benchmark_factsheet_plt(
                multi, time_period=period, lookthrough_matrix=matrix,
                add_exposures_comp=True, add_exposures_pnl_attribution=True, **options)
        else:
            figs = generate_strategy_benchmark_active_perf_plt(
                multi, time_period=period, lookthrough_matrix=matrix, **options)
        assert figs
        figs = qis.to_flat_list(figs)
        for fig in figs:
            fig.canvas.draw()
        # Multi default page uses grouped NAV statistics; other routes have weight panels.
        if report != "multi":
            assert seen
        pd.testing.assert_series_equal(a.group_data, before_a)
        pd.testing.assert_series_equal(b.group_data, before_b)
    finally:
        for fig in figs:
            plt.close(fig)
        plt.close("all")




def test_tracking_error_report_keeps_its_three_grouping_grids_independent():
    """Weights, Brinson, Euler risk, standalone TRE and trading use their selected grid."""
    from qis.portfolio.tests.tracking_error_report_test import report_inputs
    from qis.portfolio.reports.strategy_benchmark_tre_factsheet import (
        weights_tracking_error_report_by_ac_subac,
    )
    universe, multi, _, model, _ = report_inputs.__wrapped__()
    asset = multi.portfolio_datas[0].weights.columns[0]
    ac = pd.DataFrame({"AC split": [.5], "AC remainder": [.5]}, index=[asset])
    sub = pd.DataFrame({"Sub split": [.3], "Sub remainder": [.7]}, index=[asset])
    trading = pd.DataFrame({"Trading split": [.8], "Trading remainder": [.2]}, index=[asset])
    figures = {}
    try:
        figures, tables = weights_tracking_error_report_by_ac_subac(
            multi, ac_group_data=universe.group_data,
            sub_ac_group_data=universe.group_data,
            turnover_groups=universe.group_data, covar_risk_model=model,
            lookthrough_matrix=ac, sub_ac_lookthrough_matrix=sub,
            turnover_lookthrough_matrix=trading, add_benchmarks_to_navs=False,
            add_titles=False, figsize=(6., 4.))
        assert "AC split" in tables["brinson_table_ac"].index
        assert "Sub split" in tables["brinson_table_subac"].index
        assert "AC split" in tables["ac_tracking_error"].columns
        assert "Trading split" in tables["ac_turnover"].columns
        assert "AC split" not in tables["ac_turnover"].columns
        expected = multi.compute_brinson_attribution(
            group_data=universe.group_data, lookthrough_matrix=ac)[0]
        pd.testing.assert_frame_equal(tables["brinson_table_ac"], expected)
    finally:
        for fig in figures.values():
            plt.close(fig)
        plt.close("all")


@pytest.mark.parametrize("module,name", [
    ("multi_strategy_factsheet_pybloqs", "generate_multi_portfolio_factsheet_with_pybloqs"),
    ("multi_strategy_factsheet_pybloqs", "generate_multi_portfolio_factsheet"),
    ("strategy_benchmark_factsheet_pybloqs", "generate_multi_portfolio_factsheet_with_pybloqs"),
    ("strategy_benchmark_factsheet_pybloqs", "generate_multi_portfolio_factsheet"),
])
def test_optional_total_only_reports_reject_fractional_grouping(module, name):
    """Optional total-only reports cannot silently ignore a supplied partition."""
    import importlib
    pytest.importorskip("pybloqs")
    func = getattr(importlib.import_module("qis.portfolio.reports." + module), name)
    with pytest.raises(ValueError, match="matplotlib factsheets"):
        func(None, lookthrough_matrix=pd.DataFrame({"Group": [1.]}, index=["fund"]))

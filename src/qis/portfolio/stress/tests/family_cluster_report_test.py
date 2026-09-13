
"""Report families, descriptive labels and cluster display regressions."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qis.portfolio.stress.analytics import run_portfolio_stress_test
from qis.portfolio.stress.reporting import StressReportConfig
from qis.portfolio.stress.scenarios import StressScenarios
from qis.portfolio.stress.tests.scenarios_test import grouped_portfolio
from qis.portfolio.stress.tests.cluster_contributions_test import _result


def test_family_contributors_sum_members_without_shock_allocation_weights(market):
    """Euler and exposure sums preserve the Credit family; bump splitting is separate."""
    p = grouped_portfolio(market)
    result = run_portfolio_stress_test(p, StressScenarios(pd.DataFrame({"Equity": [-.2]})))
    family = result.report_diagnostics["Reported factor groups"]
    risk = result.report_diagnostics["Holding reported factor Euler volatility"]
    np.testing.assert_allclose(family.loc["credit_family", "factor_beta"],
                               result.factor_betas[["Credit", "Credit EM"]].sum())
    atomic = result.report_diagnostics["Holding factor Euler volatility"]
    np.testing.assert_allclose(risk.credit_family, atomic[["Credit", "Credit EM"]].sum(axis=1))
    np.testing.assert_allclose(risk.sum(axis=0), family.euler_vol)
    assert "Credit EM" not in family.index


def test_cluster_top_contributor_uses_own_worst_scenario_and_full_nav(market):
    """The reported signed asset contribution belongs to the identified worst cluster scenario."""
    from qis.portfolio.stress._clusters import (
        compute_cluster_contributions, cluster_top_contributors)
    result = _result(market)
    groups = compute_cluster_contributions(result, {
        "ME": pd.Series([1, 2], index=["stock", "proxy"])})
    top = cluster_top_contributors(result, groups, displayed=True)
    valuation = result.valuations["conditional"].pnl
    for group in groups.summary.index:
        ids = groups.holdings.index[groups.holdings.cluster.eq(group)]
        worst = valuation[ids].sum(axis=1).idxmin()
        expected = valuation.loc[worst, ids].abs().idxmax()
        assert top.loc[group, "scenario"] == worst
        assert top.loc[group, "holding_id"] == expected
        np.testing.assert_allclose(top.loc[group, "nav_contribution"],
                                   valuation.loc[worst, expected]/100.)


def test_cluster_page_moves_headers_up_and_adds_shared_factor_bars(market):
    """Both numeric headers sit on top; the three lower panels share row order."""
    from qis.portfolio.stress._figures import _cluster_contribution_page, _beta_page
    result = _result(market)
    config = StressReportConfig()
    cluster = _cluster_contribution_page(result, config)
    beta = _beta_page(result, config)
    try:
        assert len(cluster.axes) == 5
        for ax in cluster.axes[:3]:
            assert all(t.get_position()[1] == 1 for t in ax.get_xticklabels())
        exposure, factor, total = cluster.axes[2:]
        assert all(ax.yaxis_inverted() for ax in (exposure, factor, total))
        assert [t.get_text() for t in factor.get_yticklabels()] == [
            t.get_text() for t in total.get_yticklabels()]
        strings = [cell.get_text().get_text() for ax in beta.axes
                   for table in ax.tables for cell in table.get_celld().values()]
        assert any(t.startswith("Portfolio | ") and t.endswith("m") for t in strings)
    finally:
        plt.close(cluster)
        plt.close(beta)


def test_descriptive_cluster_labels_are_optional_and_copied():
    """Caller cluster descriptions survive independently of their mutable mapping."""
    labels = {"ME-1": "Equity core"}
    config = StressReportConfig(
        cluster_memberships={"ME": pd.Series([1], index=["stock"])},
        cluster_linkages={"ME": np.empty((0, 4))}, cluster_cutoffs={"ME": .5},
        cluster_labels=labels)
    labels["ME-1"] = "Changed"
    assert config.cluster_labels["ME-1"] == "Equity core"


def test_top_three_keep_signed_contributions_and_missing_ranks(market):
    """Rank absolute P&L in each group's own worst case, with no repeated small-group assets."""
    from qis.portfolio.stress._clusters import (
        compute_cluster_contributions, cluster_top_contributors)
    result = _result(market)
    groups = compute_cluster_contributions(result, {
        "ME": pd.Series([1, 2], index=["stock", "proxy"])})
    top = cluster_top_contributors(result, groups, displayed=True)
    valuation = result.valuations["conditional"].pnl
    for group in [*groups.summary.index, "Portfolio"]:
        ids = (groups.holdings.index if group == "Portfolio" else
               groups.holdings.index[groups.holdings.cluster.eq(group)])
        worst = min(valuation.index, key=lambda scenario: sum(valuation.loc[scenario, ids]))
        ranked = sorted(ids, key=lambda asset: -abs(valuation.loc[worst, asset]))[:3]
        assert top.loc[group, "scenario"] == worst
        for rank in range(1, 4):
            suffix = "" if rank == 1 else f"_{rank}"
            if rank > len(ranked):
                assert pd.isna(top.loc[group, "holding_id" + suffix])
                assert pd.isna(top.loc[group, "nav_contribution" + suffix])
            else:
                asset = ranked[rank - 1]
                assert top.loc[group, "holding_id" + suffix] == asset
                np.testing.assert_allclose(top.loc[group, "nav_contribution" + suffix],
                                           valuation.loc[worst, asset] / 100.)


"""Independent checks for exact cluster P&L, exposure and Euler-risk aggregation."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from qis.portfolio.stress.analytics import run_portfolio_stress_test
from qis.portfolio.stress.instruments import InstrumentLeg, InstrumentType
from qis.portfolio.stress.portfolio import PortfolioHolding
from qis.portfolio.stress.scenarios import StressScenarios, ShockConvention


def _result(market, marks=(60., -20., 60.)):
    """Build signed funded holdings whose first two share the same fitted response."""
    holdings = [PortfolioHolding(name, name, mark, (
        InstrumentLeg(InstrumentType.DELTA_1, quote, -1. if mark < 0 else 1.),))
        for name, mark, quote in zip(("A", "B", "C"), marks,
                                    ("actual", "actual", "proxy_quote"))]
    portfolio = market(holdings, denominator=100.)
    request = StressScenarios(pd.DataFrame({"Equity": [-.2, .1]}, index=["down", "up"]),
                              convention=ShockConvention.SIMPLE)
    return run_portfolio_stress_test(portfolio, request)


def test_exact_cluster_pnl_and_euler_totals_keep_signed_shared_holdings(market):
    """Cadence-local IDs do not collide and cluster sums retain the full denominator."""
    from qis.portfolio.stress._clusters import compute_cluster_contributions
    result = _result(market)
    fitted = {"ME": pd.Series([1], index=["stock"]), "QE": pd.Series([1], index=["proxy"])}
    tables = compute_cluster_contributions(result, fitted)
    assert set(tables.summary.index) == {"ME-1", "QE-1"}
    assert tables.summary.loc["ME-1", "net_mtm"] == 40.
    assert tables.summary.loc["ME-1", "gross_mtm"] == 80.
    assert tables.summary.loc["ME-1", "holding_count"] == 2
    for key in ("requested", "conditional"):
        np.testing.assert_allclose(tables.scenario_pnl[key].loc["ME-1"],
            result.valuations[key].pnl[["A", "B"]].sum(axis=1))
        np.testing.assert_allclose(tables.scenario_nav[key].sum(),
            result.valuations[key].portfolio_pnl/100.)
    risk = result.response_risk_contributions
    np.testing.assert_allclose(tables.risk.loc["ME-1", ["Systematic", "Idiosyncratic"]],
        risk.loc["stock", ["mcte_systematic", "mcte_residual"]])
    np.testing.assert_allclose(tables.factor_exposures.sum(), result.factor_betas)
    assert tables.risk.Total.sum() == pytest.approx(result.risk.annual_factor_model_vol)


def test_unassigned_and_zero_net_shared_response_risk_are_not_dropped(market):
    """Offsetting shared holdings still count in gross exposure without invented residual risk."""
    from qis.portfolio.stress._clusters import compute_cluster_contributions, UNASSIGNED
    result = _result(market, (60., -60., 60.))
    tables = compute_cluster_contributions(result, {"ME": pd.Series([1], index=["stock"])})
    assert tables.summary.loc["ME-1", "gross_mtm"] == 120.
    assert tables.risk.loc["ME-1", "Idiosyncratic"] == pytest.approx(0.)
    assert UNASSIGNED in tables.summary.index
    assert tables.holdings.loc["C", "cluster"] == UNASSIGNED
    np.testing.assert_allclose(tables.risk.Total.sum(), result.risk.annual_factor_model_vol)


def test_ambiguous_cadence_memberships_are_rejected(market):
    """The same response cannot silently be counted in two fitted partitions."""
    from qis.portfolio.stress._clusters import compute_cluster_contributions
    result = _result(market)
    with pytest.raises(ValueError, match="only one cadence"):
        compute_cluster_contributions(result, {
            "ME": pd.Series([1], index=["stock"]), "QE": pd.Series([2], index=["stock"])})


def test_display_remainder_preserves_totals_and_special_groups(market):
    """A bounded display sums omitted groups; no cluster weight is renormalised."""
    from qis.portfolio.stress._clusters import (
        compute_cluster_contributions, display_cluster_table, UNASSIGNED, OTHER)
    result = _result(market)
    tables = compute_cluster_contributions(result, {})
    index = pd.Index([f"ME-{i}" for i in range(10)] + [UNASSIGNED], name="cluster")
    summary = pd.DataFrame({"gross_mtm": np.arange(11., 0., -1)}, index=index)
    display = pd.Series([*index[:6], *([OTHER]*4), UNASSIGNED], index=index)
    tables = replace(tables, summary=summary, display_groups=display)
    values = pd.DataFrame({"loss": np.arange(11.)-.5}, index=index)
    collapsed = display_cluster_table(values, tables)
    assert len(collapsed) == 8 and UNASSIGNED in collapsed.index
    assert collapsed.loc[OTHER, "loss"] == values.loc[index[6:10], "loss"].sum()
    assert collapsed.loss.sum() == values.loss.sum()


def test_negative_cluster_risk_matches_independent_dense_covariance(market):
    """A short correlated cluster retains its negative systematic Euler contribution."""
    from qis.portfolio.stress._clusters import compute_cluster_contributions
    result = _result(market, (60., 20., -10.))
    members = {"ME": pd.Series([1, 2], index=["stock", "proxy"])}
    tables = compute_cluster_contributions(result, members)
    beta = result.factor_loadings.to_numpy()
    covariance = beta @ result.factor_covariance.to_numpy() @ beta.T
    weights = result.response_exposures.to_numpy()/100.
    total_covariance = covariance + np.diag(result.residual_variances)
    vol = np.sqrt(weights @ total_covariance @ weights)
    expected = weights * (covariance @ weights) / vol
    assert tables.risk.loc["ME-2", "Systematic"] < 0.
    np.testing.assert_allclose(tables.risk.loc[["ME-1", "ME-2"], "Systematic"],
                               expected[[0, 2]])
    # A supplied total covariance can use another denominator: model Euler must not drift.
    changed = result.response_risk_contributions * .7
    aligned = compute_cluster_contributions(
        replace(result, response_risk_contributions=changed), members)
    pd.testing.assert_frame_equal(aligned.risk, tables.risk)


def test_multi_response_holding_keeps_exact_pnl_in_explicit_bucket(market):
    """Nonlinear or composite holdings spanning fitted clusters are not split arbitrarily."""
    from qis.portfolio.stress._clusters import compute_cluster_contributions, MULTIPLE
    holding = PortfolioHolding("combo", "Two responses", 100., (
        InstrumentLeg(InstrumentType.FUTURE, "actual", 1.),
        InstrumentLeg(InstrumentType.FUTURE, "proxy_quote", 1.)))
    result = run_portfolio_stress_test(market([holding], denominator=100.),
                                      StressScenarios(pd.DataFrame({"Equity": [-.2]})))
    tables = compute_cluster_contributions(result, {
        "ME": pd.Series([1, 2], index=["stock", "proxy"])})
    assert tables.holdings.loc["combo", "cluster"] == MULTIPLE
    np.testing.assert_allclose(tables.scenario_pnl["conditional"].loc[MULTIPLE],
                               result.valuations["conditional"].portfolio_pnl)


def test_many_fitted_clusters_choose_additive_display_remainder(market):
    """Ten distinct fitted clusters are capped without losing holdings or signed exposure."""
    from qis.portfolio.stress._clusters import compute_cluster_contributions, display_cluster_table
    from qis.portfolio.stress.instruments import Underlying, ResponseBasis
    from qis.portfolio.risk.risk_model import RiskModel
    date = pd.Timestamp("2026-08-31")
    ids = [f"response{i}" for i in range(10)]
    beta = pd.DataFrame({"Equity": np.linspace(.5, 1.5, 10)}, index=ids)
    cov = pd.DataFrame([[.04]], index=["Equity"], columns=["Equity"])
    residual = pd.Series(.01, index=ids)
    asset_cov = beta @ cov @ beta.T + np.diag(residual)
    model = RiskModel({date: asset_cov}, {date: beta}, {date: cov}, {date: residual})
    quotes = {key: Underlying(key, 100., "USD", key, ResponseBasis.REFERENCE) for key in ids}
    holdings = [PortfolioHolding(f"H{i}", key, float(i+1), (
        InstrumentLeg(InstrumentType.DELTA_1, key, 1.),)) for i, key in enumerate(ids)]
    result = run_portfolio_stress_test(
        market(holdings, denominator=100., model_override=model, underlyings=quotes),
        StressScenarios(pd.DataFrame({"Equity": [-.2]})))
    tables = compute_cluster_contributions(result, {"ME": pd.Series(range(10), index=ids)})
    displayed = display_cluster_table(tables.summary, tables)
    assert len(tables.summary) == 10 and len(displayed) == 8
    assert displayed.loc["Other clusters", "holding_count"] == 3
    assert displayed.holding_count.sum() == 10
    np.testing.assert_allclose(displayed.net_mtm.sum(), 55.)
    np.testing.assert_allclose(display_cluster_table(tables.factor_exposures, tables).sum(),
                               result.factor_betas)


def test_cluster_panels_share_the_same_visual_row_order(market):
    """Heatmaps and QIS horizontal bars show the same cluster at each vertical position."""
    import matplotlib.pyplot as plt
    from qis.portfolio.stress._figures import _cluster_contribution_page
    from qis.portfolio.stress.reporting import StressReportConfig
    result = _result(market)
    config = StressReportConfig(
        cluster_memberships={"ME": pd.Series([1, 2], index=["stock", "proxy"])},
        cluster_linkages={"ME": np.array([[0., 1., .7, 2.]])},
        cluster_cutoffs={"ME": .5})
    fig = _cluster_contribution_page(result, config)
    try:
        fig.canvas.draw()
        exposure, risk = fig.axes[2], fig.axes[4]
        assert [t.get_text() for t in risk.get_yticklabels()] == [
            t.get_text() for t in exposure.get_yticklabels()]
        assert risk.yaxis_inverted() == exposure.yaxis_inverted()
    finally:
        plt.close(fig)

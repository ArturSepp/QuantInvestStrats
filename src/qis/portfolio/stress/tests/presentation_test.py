"""Numerical and presentation contracts for the v0 stress-report layout."""

import numpy as np
import pandas as pd
import pytest

from qis.portfolio.stress.analytics import run_portfolio_stress_test
from qis.portfolio.stress.reporting import StressReportConfig
from qis.portfolio.stress.scenarios import ScenarioMode, ShockConvention, StressScenarios
from qis.portfolio.stress.tests.scenarios_test import grouped_portfolio


def test_euler_families_and_regressions_have_independent_references(market):
    """Family contributions add, whereas the Credit grid splits before log conversion."""
    p = grouped_portfolio(market)
    x = np.arange(-20, 21) / 100
    grid = StressScenarios(
        pd.DataFrame({"credit_family": x}, index=x),
        ScenarioMode.CONDITIONAL,
        ShockConvention.SIMPLE,
    )
    result = run_portfolio_stress_test(p, grid, factor_grids={"Credit": grid})
    tables = result.report_diagnostics
    beta = result.factor_betas.to_numpy()
    cov = result.factor_covariance.to_numpy()
    total = result.risk.annual_factor_model_vol
    expected = beta * (cov @ beta) / total
    np.testing.assert_allclose(tables["Factor Euler volatility"].euler_vol, expected)
    families = tables["Family Euler volatility"]
    assert families.loc["credit_family", "euler_vol"] == pytest.approx(
        tables["Factor Euler volatility"].loc[["Credit", "Credit EM"], "euler_vol"].sum()
    )
    assert families.euler_vol.sum() == pytest.approx(expected.sum())
    np.testing.assert_allclose(tables["Holding factor Euler volatility"].sum(), expected)
    risk = tables["Annualised portfolio risk"]
    assert risk.loc["Systematic", "euler_vol"] == pytest.approx(families.euler_vol.sum())
    assert risk.loc[["Systematic", "Idiosyncratic"], "euler_vol"].sum() == pytest.approx(total)
    ref = np.linalg.lstsq(
        np.column_stack([x, x * x]), result.grid_summaries["Credit"].portfolio_return, rcond=None
    )[0]
    np.testing.assert_allclose(tables["Grid quadratic regressions"].loc["Credit"], ref)
    np.testing.assert_allclose(
        result.grids["Credit"].factor_log_shocks["Credit"], np.log1p(x / 2), atol=1e-15
    )


def test_parser_appendix_is_optional_and_snapshotted():
    """A parser supplies the last-page table and explanations, or omits the page."""
    assert StressReportConfig().appendix_table is None
    table = pd.DataFrame({"Value": ["Parser-owned"]}, index=["Source"])
    config = StressReportConfig(
        appendix_table=table,
        appendix_title="Source validation",
        appendix_notes=("Confirm the supplied source.",),
    )
    table.iloc[0, 0] = "Mutated"
    assert config.appendix_table.iloc[0, 0] == "Parser-owned"


def test_v0_pages_show_regression_and_parser_content(market):
    """The common renderer exposes the requested titles, regression line and optional page."""
    import matplotlib.pyplot as plt
    from qis.portfolio.stress._figures import report_pages

    p = grouped_portfolio(market)
    x = np.array([-0.2, 0.0, 0.2])
    grid = StressScenarios(
        pd.DataFrame({"credit_family": x}, index=x),
        ScenarioMode.CONDITIONAL,
        ShockConvention.SIMPLE,
    )
    result = run_portfolio_stress_test(p, grid, factor_grids={"Credit": grid})
    config = StressReportConfig(
        model_name="MATF",
        appendix_title="Source validation",
        appendix_table=pd.DataFrame({"Status": ["Source confirmed"]}, index=["Account"]),
        appendix_notes=("Parser variable explanation",),
        write_workbook=False,
    )
    pages = list(report_pages(result, config))
    try:
        assert len(pages) == 10
        titles = [title for title, _ in pages]
        assert titles[3] == "Portfolio MATF exposures and risk"
        assert titles[6:9] == [
            "Estimated MATF loadings and explanatory power",
            "MATF asset cluster dendrograms",
            "MATF correlation and scenario construction",
        ]
        assert titles[-1] == "Source validation"
        assert any("Parser variable explanation" in text.get_text() for text in pages[-1][1].texts)
        curves = [line for line in pages[5][1].axes[0].lines if line.get_linestyle() == "--"]
        assert len(curves) == 1
        assert len(pages[5][1].axes[0].collections) >= 2
    finally:
        for _, figure in pages:
            plt.close(figure)


def test_overlapping_groups_keep_additive_atomic_risk(market):
    """Valid overlapping shock groups cannot double-count portfolio Euler risk."""
    from dataclasses import replace
    from qis.portfolio.risk.factor_groups import FactorGroupSpec

    p = grouped_portfolio(market)
    groups = dict(p.risk_model.factor_groups)
    groups["cross_family"] = FactorGroupSpec("cross_family", ("Equity", "Credit"))
    p = replace(p, risk_model=replace(p.risk_model, factor_groups=groups))
    result = run_portfolio_stress_test(p, StressScenarios(pd.DataFrame({"Equity": [-0.1]})))
    frame = result.report_diagnostics["Family Euler volatility"]
    assert set(frame.index) == set(result.factor_betas.index)
    assert frame.euler_vol.sum() == pytest.approx(
        result.report_diagnostics["Annualised portfolio risk"].loc["Systematic", "euler_vol"]
    )


def test_factor_panels_rank_euler_risk_instead_of_dollar_exposure(market):
    """A lower-beta, higher-volatility factor must precede a larger dollar exposure."""
    from dataclasses import replace
    import matplotlib.pyplot as plt
    from qis.portfolio.stress._figures import _contributor_page

    p = grouped_portfolio(market)
    model, date = p.risk_model, p.risk_date
    factors = model.factor_covar[date].index
    cov = pd.DataFrame(np.diag([0.01, 0.04, 0.25, 0.09]), index=factors, columns=factors)
    beta = model.factor_loadings[date]
    asset_cov = beta @ cov @ beta.T + np.diag(model.residual_vars[date])
    p = replace(p, risk_model=replace(model, factor_covar={date: cov}, covar={date: asset_cov}))
    result = run_portfolio_stress_test(p, StressScenarios(pd.DataFrame({"Equity": [-0.1]})))
    assert result.factor_exposures.Equity > result.factor_exposures.Credit
    fig = _contributor_page(result, StressReportConfig(model_name="MATF"))
    try:
        assert [ax.get_title().split("\n")[0] for ax in fig.axes] == [
            "Credit",
            "Credit EM",
            "Equity",
        ]
    finally:
        plt.close(fig)

@pytest.mark.parametrize("funded", [True, False])
def test_derivative_captions_preserve_denominator_meaning(market, funded):
    """A gross-asset derivative denominator must never be labelled debt-net NAV."""
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    from qis.portfolio.stress._figures import report_pages
    from qis.portfolio.stress.instruments import InstrumentLeg, InstrumentType
    from qis.portfolio.stress.portfolio import PortfolioHolding

    kind = InstrumentType.DELTA_1 if funded else InstrumentType.FUTURE
    holding = PortfolioHolding(
        "position", "Position", 100.0 if funded else 0.0,
        (InstrumentLeg(kind, "actual", 1.0),),
    )
    portfolio = market([holding])
    grid = StressScenarios(pd.DataFrame({"Equity": [-0.2, 0.0, 0.2]}, index=[-0.2, 0, 0.2]))
    result = run_portfolio_stress_test(portfolio, grid, factor_grids={"Equity": grid})
    pages = list(report_pages(result, StressReportConfig(write_workbook=False)))
    try:
        texts = "\n".join(
            item.get_text() for _, figure in pages for item in figure.findobj(Text)
        )
        if funded:
            assert "Portfolio P&L (% of NAV)" in texts
            assert "Dollar exposure = NAV x e_f" in texts
        else:
            assert "Portfolio P&L (% of reporting denominator)" in texts
            assert "NAV" not in texts
            assert "shared response" in texts
    finally:
        for _, figure in pages:
            plt.close(figure)

"""Numerical and presentation contracts for the v0 stress-report layout."""

import numpy as np
import pandas as pd
import pytest

from qis.portfolio.stress.analytics import StressTestConfig, run_portfolio_stress_test
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
    np.testing.assert_allclose(
        tables["Grid polynomial regressions"].loc["Credit", ["linear", "quadratic"]], ref
    )
    np.testing.assert_allclose(
        result.grids["Credit"].factor_log_shocks["Credit"], np.log1p(x / 2), atol=1e-15
    )


def test_parser_appendix_is_optional_and_snapshotted():
    """A parser supplies the coverage table and explanations, or omits that page."""
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
        report_name="Example account",
        model_label="MATF CUSTOM",
        appendix_title="Source validation",
        appendix_table=pd.DataFrame({"Status": ["Source confirmed"]}, index=["Account"]),
        appendix_notes=("Parser variable explanation",),
        write_workbook=False,
    )
    pages = list(report_pages(result, config))
    try:
        assert len(pages) == 13
        titles = [title for title, _ in pages]
        assert titles[3] == "Portfolio MATF exposures and risk"
        first_texts = pages[0][1].texts
        heading = next(text for text in first_texts
                       if text.get_text().startswith("Example account | Notional"))
        assert "Risk model MATF CUSTOM" in heading.get_text()
        assert f"{result.metadata['reporting_denominator']:,.0f}" in heading.get_text()
        section = next(text for text in first_texts
                       if text.get_text() == titles[0])
        assert heading.get_position()[1] > section.get_position()[1]
        assert not any("Example account | Notional" in text.get_text()
                       for text in pages[1][1].texts)
        assert titles[6:10] == [
            "MATF asset cluster dendrograms",
            "Cluster contributions to stress, factor exposures and risk",
            "Estimated MATF loadings and explanatory power",
            "MATF correlation and scenario construction",
        ]
        assert titles[10] == "MATF conditional shocks and covariance"
        assert titles[-2] == "Source validation"
        assert titles[-1] == "Notation and guide to the analysis"
        assert any("Parser variable explanation" in text.get_text() for text in pages[-2][1].texts)
        guide_texts = pages[-1][1].texts
        headings = [item.get_text() for item in guide_texts if item.get_gid() == "guide-heading"]
        assert len(headings) == 11
        assert [heading.split(".")[0] for heading in headings] == [str(i) for i in range(1, 12)]
        assert "MATF" in headings[3] and "MATF" in headings[8]
        assert min(item.get_fontsize() for item in guide_texts) >= 9
        pages[-1][1].canvas.draw()
        renderer = pages[-1][1].canvas.get_renderer()
        bounds = [item.get_window_extent(renderer) for item in guide_texts
                  if item.get_gid() in ("guide-heading", "guide-body")]
        assert min(box.y0 for box in bounds) / pages[-1][1].bbox.height > .055
        assert max(box.x1 for box in bounds) / pages[-1][1].bbox.width < .985
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
            "credit_family",
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


@pytest.mark.parametrize("funded, order", [(True, 2), (False, 2)])
@pytest.mark.parametrize("confidence", [0.8, 0.95])
def test_grid_polynomial_is_quadratic_for_every_payoff_type(market, funded, order, confidence):
    """Independent least squares agrees with quadratic fits for funded and option holdings."""
    import matplotlib.pyplot as plt
    from scipy.stats import t
    from qis.portfolio.stress._figures import _grid_page
    from qis.portfolio.stress.instruments import InstrumentLeg, InstrumentType
    from qis.portfolio.stress.portfolio import PortfolioHolding

    kind = InstrumentType.DELTA_1 if funded else InstrumentType.CALL
    leg = InstrumentLeg(kind, "proxy_quote", 1.0, strike=None if funded else 80.0)
    portfolio = market([PortfolioHolding("position", "Position", 80.0, (leg,))])
    x = np.linspace(-0.3, 0.3, 61)
    grid = StressScenarios(
        pd.DataFrame({"Equity": x}, index=x),
        ScenarioMode.CONDITIONAL, ShockConvention.SIMPLE,
    )
    result = run_portfolio_stress_test(
        portfolio, grid, factor_grids={"Equity": grid},
        config=StressTestConfig(confidence=confidence),
    )
    row = result.report_diagnostics["Grid polynomial regressions"].loc["Equity"]
    y = result.grid_summaries["Equity"].portfolio_return.to_numpy()
    design = np.column_stack([x ** power for power in range(1, order + 1)])
    reference = np.linalg.lstsq(design, y, rcond=None)[0]
    terms = ["linear", "quadratic", "cubic"][:order]
    np.testing.assert_allclose(row[terms], reference, rtol=1e-10, atol=1e-12)
    assert row["order"] == order
    assert row.cubic == 0.0
    assert row.r_squared == pytest.approx(1 - np.sum((y - design @ reference) ** 2) / (y @ y))
    assert "lower_1sigma" in result.grid_summaries["Equity"]
    band = result.report_diagnostics["Grid regression confidence bands"].loc["Equity"]
    residual_variance = np.sum((y - design @ reference) ** 2) / (len(x) - order)
    _, r = np.linalg.qr(design)
    mean_se = np.sqrt(residual_variance * np.sum((design @ np.linalg.inv(r)) ** 2, axis=1))
    width = t.ppf((1 + confidence) / 2, len(x) - order) * mean_se
    np.testing.assert_allclose(band["mean"], design @ reference, atol=1e-12)
    np.testing.assert_allclose(band.mean_se, mean_se, atol=1e-12)
    np.testing.assert_allclose(band.mean_ci_lower, design @ reference - width, atol=1e-12)
    np.testing.assert_allclose(band.mean_ci_upper, design @ reference + width, atol=1e-12)
    assert band.confidence.eq(confidence).all()
    assert band.df_resid.eq(len(x) - order).all()
    assert band.loc[0.0, "mean_se"] == 0.0
    figure = _grid_page(result, StressReportConfig())
    try:
        figure.canvas.draw()
        curve = next(line for line in figure.axes[0].lines if line.get_linestyle() == "--")
        np.testing.assert_allclose(curve.get_ydata(), design @ reference, atol=1e-12)
        assert curve.get_ydata()[30] == pytest.approx(0.0, abs=1e-12)
        legend = figure.axes[0].get_legend().get_texts()[0].get_text()
        assert "R^2" in legend
        assert "x^3" not in legend
        shading = next(collection for collection in figure.axes[0].collections
                       if collection.get_label() == "Conditional +/-2sigma")
        vertices = shading.get_paths()[0].vertices
        risk_band = result.grid_summaries["Equity"]
        for anchor, lower, upper in zip(x, risk_band.lower_2sigma, risk_band.upper_2sigma):
            ys = vertices[np.isclose(vertices[:, 0], anchor, atol=1e-14), 1]
            np.testing.assert_allclose([ys.min(), ys.max()], [lower, upper], atol=1e-12)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("anchors, identifiable", [([-0.2, 0.0, 0.2], True), ([0.0, 0.2], False)])
def test_quadratic_grid_requires_two_independent_regressors(market, anchors, identifiable):
    """A symmetric three-point grid identifies the fit; a single nonzero anchor does not."""
    from qis.portfolio.stress.instruments import InstrumentLeg, InstrumentType
    from qis.portfolio.stress.portfolio import PortfolioHolding

    portfolio = market([PortfolioHolding(
        "future", "Future", 0.0, (InstrumentLeg(InstrumentType.FUTURE, "actual", 1.0),),
    )])
    x = np.array(anchors)
    grid = StressScenarios(pd.DataFrame({"Equity": x}, index=x))
    result = run_portfolio_stress_test(portfolio, grid, factor_grids={"Equity": grid})
    assert result.report_diagnostics["Grid polynomial regressions"].empty is (not identifiable)


def test_zero_payoff_grid_has_zero_coefficients_and_undefined_r_squared(market):
    """A flat zero payoff must not be reported as a spurious perfect fit."""
    from qis.portfolio.stress.instruments import InstrumentLeg, InstrumentType
    from qis.portfolio.stress.portfolio import PortfolioHolding

    portfolio = market([PortfolioHolding(
        "zero", "Zero position", 0.0,
        (InstrumentLeg(InstrumentType.FUTURE, "actual", 0.0),),
    )])
    x = np.linspace(-0.2, 0.2, 9)
    grid = StressScenarios(pd.DataFrame({"Equity": x}, index=x))
    result = run_portfolio_stress_test(portfolio, grid, factor_grids={"Equity": grid})
    row = result.report_diagnostics["Grid polynomial regressions"].loc["Equity"]
    np.testing.assert_allclose(row[["linear", "quadratic", "cubic"]], 0.0)
    assert np.isnan(row.r_squared)


def test_two_point_quadratic_has_no_estimable_regression_confidence_band(market):
    """Two independent anchors identify coefficients but leave no error degrees of freedom."""
    portfolio = grouped_portfolio(market)
    x = np.array([0.1, 0.2])
    grid = StressScenarios(pd.DataFrame({"Equity": x}, index=x))
    result = run_portfolio_stress_test(portfolio, grid, factor_grids={"Equity": grid})
    assert not result.report_diagnostics["Grid polynomial regressions"].empty
    band = result.report_diagnostics["Grid regression confidence bands"].loc["Equity"]
    assert band.df_resid.eq(0).all()
    assert band[["mean_se", "mean_ci_lower", "mean_ci_upper"]].isna().all().all()


def test_six_selected_sensitivity_panels_preserve_caller_order(market):
    """Six supplied grids render row-major without dropping or reordering a curve."""
    import matplotlib.pyplot as plt
    from qis.portfolio.stress._figures import _grid_page

    p = grouped_portfolio(market)
    x = np.array([-.2, 0., .2])
    grid = StressScenarios(pd.DataFrame({"Equity": x}, index=x),
                           ScenarioMode.CONDITIONAL, ShockConvention.SIMPLE)
    keys = ("sixth", "first", "third", "second", "fifth", "fourth")
    result = run_portfolio_stress_test(p, grid, factor_grids={key: grid for key in keys})
    config = StressReportConfig(selected_grids=keys)
    fig = _grid_page(result, config)
    try:
        assert len(fig.axes) == 6
        assert len({round(ax.get_position().y0, 6) for ax in fig.axes}) == 2
        assert len({round(ax.get_position().x0, 6) for ax in fig.axes}) == 3
        for ax, key in zip(fig.axes, keys):
            line = next(line for line in ax.lines if line.get_linestyle() == "--")
            coeff = result.report_diagnostics["Grid polynomial regressions"].loc[key]
            np.testing.assert_allclose(line.get_ydata(), coeff.linear*x + coeff.quadratic*x*x)
        with pytest.raises(ValueError, match="six"):
            StressReportConfig(selected_grids=keys + ("seventh",))
    finally:
        plt.close(fig)


def test_no_active_factor_contributions_leave_sensitivity_slots_empty(market):
    """A zero-risk portfolio must not select arbitrary factors through empty defaults."""
    from dataclasses import replace
    import matplotlib.pyplot as plt
    from qis.portfolio.stress._figures import _grid_page
    p = grouped_portfolio(market)
    x = np.array([-.2, 0., .2])
    grid = StressScenarios(pd.DataFrame({"Equity": x}, index=x),
                           ScenarioMode.CONDITIONAL, ShockConvention.SIMPLE)
    result = run_portfolio_stress_test(p, grid, factor_grids={"Equity": grid})
    diagnostics = dict(result.report_diagnostics)
    diagnostics["Factor Euler volatility"] = diagnostics["Factor Euler volatility"] * 0.
    diagnostics["Reported factor groups"] = diagnostics["Reported factor groups"].copy()
    diagnostics["Reported factor groups"]["euler_vol"] = 0.
    result = replace(result, report_diagnostics=diagnostics)
    fig = _grid_page(result, StressReportConfig())
    try:
        assert all(not ax.axison for ax in fig.axes)
    finally:
        plt.close(fig)

"""Independent checks of scenario sensitivities and conditional Euler volatility bands."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from qis.portfolio.stress import (
    InstrumentLeg, InstrumentType, PortfolioHolding, ScenarioMode, ShockConvention,
    StressScenarios, StressTestConfig, run_portfolio_stress_test,
)


@pytest.mark.parametrize("kind", list(InstrumentType))
@pytest.mark.parametrize("currency", ["USD", "EUR"])
def test_scenario_jacobian_matches_full_payoff_differences(market, kind, currency):
    """Check changing units-times-spot and foreign future settlement FX without rebasing."""
    option = kind in (InstrumentType.CALL, InstrumentType.PUT)
    strike = 80.0 if kind is InstrumentType.CALL else 140.0 if option else None
    holding = PortfolioHolding("h", "Holding", 110.0,
                               (InstrumentLeg(kind, "actual", 3.0, strike=strike),))
    p = market([holding], currency=currency)
    beta = p.risk_model.factor_loadings[p.risk_date]
    x = pd.Series([0.12, -0.08, 0.03, -0.02], index=beta.columns)
    actual = p.response_jacobian(x) @ beta
    for factor in beta.columns:
        bump = x * 0.0
        bump[factor] = 1e-6
        reference = (p.get_mtm(x + bump) - p.get_mtm(x - bump)) / 2e-6
        np.testing.assert_allclose(actual[factor], reference, rtol=2e-8, atol=2e-7)
    pd.testing.assert_frame_equal(p.response_jacobian(x * 0), p.response_jacobian())


@pytest.mark.parametrize("kind", list(InstrumentType))
def test_conditional_band_reconciles_to_euler_and_joint_covariance(market, kind):
    """Condition joint portfolio/anchor covariance independently at every grid point."""
    option = kind in (InstrumentType.CALL, InstrumentType.PUT)
    p = market([PortfolioHolding("h", "Holding", 100.0,
                (InstrumentLeg(kind, "actual", 2.0, strike=100.0 if option else None),))])
    x = np.array([-0.2, 0.0, 0.2])
    request = StressScenarios(pd.DataFrame({"Credit": x, "Credit EM": x / 2}, index=x),
                             ScenarioMode.CONDITIONAL, ShockConvention.SIMPLE)
    result = run_portfolio_stress_test(p, request, factor_grids={"credit": request})
    summary = result.grid_summaries["credit"]
    model, date = p.risk_model, p.risk_date
    cov, betas = model.factor_covar[date], model.factor_loadings[date]
    anchors = ["Credit", "Credit EM"]
    for anchor, shocks in result.grids["credit"].factor_log_shocks.iterrows():
        w = p.response_jacobian(shocks).sum() / p.reporting_denominator
        exposure = betas.T @ w
        variance = float(exposure @ cov @ exposure + (w*w) @ model.residual_vars[date])
        cross = exposure @ cov.loc[:, anchors]
        reduction = float(cross @ np.linalg.solve(cov.loc[anchors, anchors], cross))
        expected = np.sqrt(max(0.0, variance-reduction) / 12)
        row = summary.loc[anchor]
        assert row.conditional_vol_horizon == pytest.approx(expected, abs=1e-12)
        assert row.lower_1sigma == pytest.approx(row.portfolio_return-expected)
        assert row.upper_2sigma == pytest.approx(row.portfolio_return+2*expected)
    euler = result.report_diagnostics["Grid conditional factor Euler"].loc["credit"]
    np.testing.assert_allclose(euler.drop(columns="Total").sum(axis=1), euler.Total, atol=1e-12)
    np.testing.assert_allclose(euler.Total, summary.conditional_vol_horizon, atol=1e-12)
    np.testing.assert_allclose(euler[anchors], 0.0, atol=1e-12)
    assert not np.isclose(summary.conditional_vol_horizon.iloc[0],
                          summary.conditional_vol_horizon.iloc[-1])


def test_band_policy_and_horizon_are_explicit(market):
    """Disabling bands and independent grids stays unbanded; horizon scales by square root."""
    p = market([PortfolioHolding("h", "Holding", 100.0,
                (InstrumentLeg(InstrumentType.DELTA_1, "actual", 1.0),))])
    grid = StressScenarios(pd.DataFrame({"Equity": [-0.2, 0.0, 0.2]}, index=[-0.2, 0., 0.2]),
                           ScenarioMode.CONDITIONAL, ShockConvention.SIMPLE)
    monthly = run_portfolio_stress_test(p, grid, factor_grids={"equity": grid})
    quarterly = run_portfolio_stress_test(p, grid, factor_grids={"equity": grid},
                                        config=StressTestConfig(horizon_years=0.25))
    np.testing.assert_allclose(quarterly.grid_summaries["equity"].conditional_vol_horizon,
                               monthly.grid_summaries["equity"].conditional_vol_horizon*np.sqrt(3))
    disabled = run_portfolio_stress_test(p, grid, factor_grids={"equity": grid},
                                       config=StressTestConfig(ordinary_asset_bands=False))
    independent = run_portfolio_stress_test(p, grid,
                      factor_grids={"equity": replace(grid, mode=ScenarioMode.INDEPENDENT)})
    for result in [disabled, independent]:
        assert "lower_1sigma" not in result.grid_summaries["equity"]


def test_offsetting_shared_responses_have_zero_bands(market):
    """Identical offsetting exposures net before systematic and shared residual risk."""
    holdings = [PortfolioHolding(str(q), "Future", 0.,
                (InstrumentLeg(InstrumentType.FUTURE, "actual", q),)) for q in [-1., 1.]]
    p = market(holdings)
    grid = StressScenarios(pd.DataFrame({"Equity": [-0.1, 0., 0.1]}, index=[-0.1, 0., 0.1]),
                           ScenarioMode.CONDITIONAL, ShockConvention.SIMPLE)
    result = run_portfolio_stress_test(p, grid, factor_grids={"equity": grid})
    assert result.grid_summaries["equity"].conditional_vol_horizon.eq(0).all()
    assert result.report_diagnostics["Grid conditional factor Euler"].eq(0).all().all()


def test_negative_euler_is_retained_and_family_sums_reconcile(market):
    """Hedging factors reduce the band; absolute contributions must not replace signed ones."""
    from qis.portfolio.risk.factor_groups import FactorGroupSpec

    p = market([PortfolioHolding("h", "Future", 0.,
               (InstrumentLeg(InstrumentType.FUTURE, "actual", 1.),))])
    model, date = p.risk_model, p.risk_date
    beta = model.factor_loadings[date].copy()
    beta.loc["stock"] = [1., 0., -0.2, 0.]
    cov = model.factor_covar[date]
    model = replace(model, factor_loadings={date: beta},
                    covar={date: beta @ cov @ beta.T + np.diag(model.residual_vars[date])},
                    factor_groups={"credit": FactorGroupSpec("credit", ("Credit", "Credit EM"))})
    p = replace(p, risk_model=model)
    grid = StressScenarios(pd.DataFrame({"FX": [0.]}, index=[0.]),
                           ScenarioMode.CONDITIONAL, ShockConvention.SIMPLE)
    result = run_portfolio_stress_test(p, grid, factor_grids={"fx": grid})
    factor = result.report_diagnostics["Grid conditional factor Euler"].loc[("fx", 0.)]
    family = result.report_diagnostics["Grid conditional family Euler"].loc[("fx", 0.)]
    assert factor.Credit < 0
    assert family.credit == pytest.approx(factor.Credit+factor["Credit EM"])
    assert factor.drop("Total").sum() == pytest.approx(factor.Total)
    assert factor.drop("Total").abs().sum() > factor.Total


def test_baseline_only_composite_requires_explicit_stressed_derivative(market):
    """Do not silently reuse a custom baseline-only derivative at stressed quotes."""
    class BaselineOnly:
        """A legacy composite that remains supported for direct valuation/current risk."""
        implementation_id = "test.baseline_only.v1"
        coverage = "Funded quote"
        boundary_policy = "Smooth"

        def evaluate(self, context):
            """Value one unit of the actual quote."""
            return context.quotes["actual"]

        def response_jacobian(self, context):
            """Use the original baseline quote, as legacy implementations can do."""
            return context.baseline_quotes["actual"]*context.quote_response_jacobian.loc["actual"]

    p = market([PortfolioHolding("c", "Custom", 100., payoff=BaselineOnly())])
    zero = pd.Series(0., index=p.risk_model.factor_loadings[p.risk_date].columns)
    assert p.response_jacobian().loc["c", "stock"] == 100.
    assert p.get_mtm(zero).iloc[0] == 100.
    with pytest.raises(NotImplementedError, match="scenario_response_jacobian"):
        p.response_jacobian(zero+0.1)

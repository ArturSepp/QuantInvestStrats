"""Independent portfolio risk, historical nonlinear ranking and report-result contracts."""

from dataclasses import replace
import math

import numpy as np
import pandas as pd
import pytest

from qis.portfolio.stress.analytics import StressTestConfig, run_portfolio_stress_test
from qis.portfolio.stress.instruments import InstrumentLeg, InstrumentType
from qis.portfolio.stress.portfolio import PortfolioHolding
from qis.portfolio.stress.scenarios import ScenarioMode, ShockConvention, StressScenarios
from qis.portfolio.stress.tests.scenarios_test import grouped_portfolio


def request():
    """Return simple up/down equity shocks with explicit labels."""
    return StressScenarios(
        pd.DataFrame({"Equity": [-0.2, 0.5]}, index=["down", "up"]),
        convention=ShockConvention.SIMPLE,
    )


def test_current_risk_uses_signed_underlying_units_not_option_marks(market):
    """A zero-mark future retains its full notional factor and residual sensitivities."""
    h = PortfolioHolding(
        "future", "Future", 0.0, (InstrumentLeg(InstrumentType.FUTURE, "actual", -2.0, 5.0),)
    )
    p = market([h])
    result = run_portfolio_stress_test(p, request())
    np.testing.assert_allclose(result.response_exposures, [-1000.0, 0.0, 0.0])
    np.testing.assert_allclose(result.factor_betas, [-1.0, -0.4, -0.2, 0.0])
    # Independent scalar covariance reference for [-1, -.4, -.2, 0].
    systematic_variance = 0.04 + 0.4**2 * 0.01 + 0.2**2 * 0.01
    systematic_variance += 2 * 0.4 * 0.003 + 2 * 0.2 * 0.006
    assert result.risk.annual_systematic_vol == pytest.approx(math.sqrt(systematic_variance))
    assert result.risk.annual_residual_vol == pytest.approx(0.1)
    assert result.risk.annual_total_vol == pytest.approx(math.sqrt(systematic_variance + 0.01))
    assert result.response_risk_contributions.mcte.sum() == pytest.approx(
        result.risk.annual_total_vol
    )
    assert result.holding_risk.loc["future", "annual_total_vol"] == result.risk.annual_total_vol


def test_historical_months_rank_by_full_payoff_not_current_delta(market):
    """A currently OTM short call makes a large up month worse than an equity selloff."""
    stock = PortfolioHolding(
        "stock", "Stock", 100.0, (InstrumentLeg(InstrumentType.DELTA_1, "actual", 1.0),)
    )
    call = PortfolioHolding(
        "call",
        "Short call",
        -3.0,
        (InstrumentLeg(InstrumentType.CALL, "actual", -2.0, strike=110.0),),
    )
    p = market([stock, call])
    history = pd.DataFrame(
        [
            [math.log(0.8), 0.0, 0.0, 0.0],
            [math.log(1.5), 0.0, 0.0, 0.0],
            [np.nan, 0.0, 0.0, 0.0],
            [math.log(0.1), 0.0, 0.0, 0.0],
        ],
        index=pd.to_datetime(["2026-01-31", "2026-02-28", "2026-03-31", "2026-09-30"]),
        columns=p.risk_model.factor_loadings[p.risk_date].columns,
    )
    result = run_portfolio_stress_test(p, request(), history)
    assert result.historical_ranking.index[0] == pd.Timestamp("2026-02-28")
    assert result.historical_ranking.portfolio_pnl.iloc[0] == pytest.approx(-30.0)
    assert result.historical_ranking.portfolio_pnl.iloc[1] == pytest.approx(-20.0)
    assert len(result.historical.pnl) == 2
    assert result.historical_coverage.loc["2026-03-31", "status"] == "incomplete factor vector"
    assert result.historical_coverage.loc["2026-09-30", "status"] == "after valuation date"
    contribution = result.attribution["requested"]
    np.testing.assert_allclose(
        contribution.sum(axis=1), result.valuations["requested"].portfolio_pnl
    )
    assert contribution.loc["up", "Nonlinear payoff adjustment"] < 0.0
    # Independent local factor component on the currently funded stock.
    assert contribution.loc["up", "Equity"] == pytest.approx(50.0)
    assert contribution.loc["up", "Nonlinear payoff adjustment"] == pytest.approx(-80.0)


def test_credit_grid_bands_and_scenario_parity_for_funded_portfolio(market):
    """The same split shock has the same exact P&L in grids and requested scenarios."""
    p = grouped_portfolio(market)
    grid = StressScenarios(
        pd.DataFrame({"credit_family": [-0.1, 0.0, 0.1]}, index=[-0.1, 0.0, 0.1]),
        ScenarioMode.CONDITIONAL,
        ShockConvention.SIMPLE,
    )
    result = run_portfolio_stress_test(p, grid, factor_grids={"Credit": grid})
    pd.testing.assert_frame_equal(result.grids["Credit"].pnl, result.valuations["requested"].pnl)
    assert "lower_bound" in result.grid_summaries["Credit"]
    assert result.grid_summaries["Credit"].loc[0.0, "portfolio_return"] == 0.0
    assert "scenario-local conditional" in result.grid_metadata.loc["Credit", "band_status"]
    np.testing.assert_allclose(
        result.attribution["requested"].iloc[:, :-1].sum(axis=1),
        result.valuations["requested"].portfolio_pnl,
    )
    np.testing.assert_allclose(result.attribution["requested"].iloc[:, -1], 0.0, atol=1e-12)


def test_derivative_grids_use_local_approximation_bands(market):
    """Derivative bounds retain an explicit local-risk approximation status."""
    p = market(
        [
            PortfolioHolding(
                "put", "Put", 2.0, (InstrumentLeg(InstrumentType.PUT, "actual", 1.0, strike=100.0),)
            )
        ]
    )
    grid = StressScenarios(
        pd.DataFrame({"Equity": [-0.2, 0.0, 0.2]}, index=[-0.2, 0.0, 0.2]),
        ScenarioMode.CONDITIONAL,
        ShockConvention.SIMPLE,
    )
    result = run_portfolio_stress_test(p, request(), factor_grids={"Equity": grid})
    assert "lower_1sigma" in result.grid_summaries["Equity"]
    assert "scenario-local" in result.grid_metadata.loc["Equity", "band_status"]
    assert result.grids["Equity"].pnl.loc[0.0, "put"] == 0.0


def test_result_does_not_change_when_live_model_or_inputs_change(market):
    """Report generation receives numerical snapshots, never a live estimator/model."""
    p = grouped_portfolio(market)
    scenarios = request()
    result = run_portfolio_stress_test(p, scenarios)
    before = result.factor_loadings.copy(deep=True)
    p.risk_model.factor_loadings[p.risk_date].iloc[:] = 10.0
    p.risk_model.factor_covar[p.risk_date].iloc[:] = 0.0
    scenarios.anchors.iloc[:] = 0.0
    pd.testing.assert_frame_equal(result.factor_loadings, before)
    assert result.factor_covariance.iloc[0, 0] == 0.04
    assert result.valuations["requested"].portfolio_pnl.loc["down"] < 0.0


def test_denominator_changes_ratios_only(market):
    """Risk and P&L ratios scale without resizing signed positions or contracts."""
    p = grouped_portfolio(market)
    first = run_portfolio_stress_test(p, request())
    second = run_portfolio_stress_test(replace(p, reporting_denominator=2000.0), request())
    pd.testing.assert_series_equal(first.factor_exposures, second.factor_exposures)
    np.testing.assert_allclose(first.factor_betas, 2.0 * second.factor_betas)
    np.testing.assert_allclose(first.risk, 2.0 * second.risk)
    pd.testing.assert_frame_equal(
        first.valuations["requested"].pnl, second.valuations["requested"].pnl
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"historical_count": 0},
        {"historical_count": True},
        {"horizon_years": 0.0},
        {"horizon_years": np.nan},
        {"confidence": 1.0},
    ],
)
def test_invalid_analysis_config_fails(kwargs):
    """Invalid horizon or probability must not reach a risk report."""
    with pytest.raises(ValueError):
        StressTestConfig(**kwargs)

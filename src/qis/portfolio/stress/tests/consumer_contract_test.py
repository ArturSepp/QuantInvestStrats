"""Independent funded and derivatives consumers use the same public interfaces."""

import math

import numpy as np
import pandas as pd

from qis import (
    FactorGroupSpec,
    InstrumentLeg,
    InstrumentPortfolio,
    InstrumentType,
    KinkPolicy,
    PortfolioHolding,
    ResponseBasis,
    RiskModel,
    ScenarioMode,
    ShockConvention,
    StressScenarios,
    Underlying,
    run_portfolio_stress_test,
    project_factor_scenarios,
)


def test_two_public_consumers_share_a_model_without_estimator_types():
    """Ordinary projection parity and signed futures/put economics share one interface."""
    date = pd.Timestamp("2026-08-31")
    beta = pd.DataFrame([[1.0, 0.5], [2.0, 1.0]], index=["a", "proxy"], columns=["F1", "F2"])
    factor_cov = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]], index=beta.columns, columns=beta.columns
    )
    residual = pd.Series([0.01, 0.04], index=beta.index)
    covariance = beta @ factor_cov @ beta.T + pd.DataFrame(
        np.diag(residual), index=beta.index, columns=beta.index
    )
    model = RiskModel(
        {date: covariance},
        {date: beta},
        {date: factor_cov},
        {date: residual},
        {"family": FactorGroupSpec("family", ("F1", "F2"))},
    )
    quotes = {
        "quote_a": Underlying("quote_a", 100.0, "USD", "a", ResponseBasis.REFERENCE),
        "contract": Underlying("contract", 250.0, "USD", "proxy", ResponseBasis.REFERENCE),
    }
    common = dict(
        underlyings=quotes,
        risk_model=model,
        risk_date=date,
        valuation_date=date,
        reference_currency="USD",
        reporting_denominator=1000.0,
        denominator_label="Investment capital",
    )
    funded = InstrumentPortfolio(
        holdings=(
            PortfolioHolding(
                "fund_a", "Fund A", 600.0, (InstrumentLeg(InstrumentType.DELTA_1, "quote_a", 6.0),)
            ),
            PortfolioHolding(
                "fund_b", "Fund B", 400.0, (InstrumentLeg(InstrumentType.DELTA_1, "contract", 1.6),)
            ),
        ),
        **common,
    )
    derivatives = InstrumentPortfolio(
        holdings=(
            PortfolioHolding(
                "hedge",
                "Short future",
                0.0,
                (InstrumentLeg(InstrumentType.FUTURE, "contract", -2.0, 10.0),),
            ),
            PortfolioHolding(
                "put",
                "Protective put",
                7.0,
                (InstrumentLeg(InstrumentType.PUT, "quote_a", 3.0, strike=100.0),),
                kink_policy=KinkPolicy.LEFT,
            ),
        ),
        **common,
    )
    scenarios = StressScenarios(
        pd.DataFrame({"F1": [-0.2], "F2": [0.0]}, index=["stress"]),
        ScenarioMode.INDEPENDENT,
        ShockConvention.SIMPLE,
    )
    first = run_portfolio_stress_test(funded, scenarios)
    second = run_portfolio_stress_test(derivatives, scenarios)
    old = project_factor_scenarios(
        beta,
        pd.Series([600.0, 400.0], index=beta.index),
        pd.DataFrame([[math.log(0.8), 0.0]], index=["stress"], columns=beta.columns),
    )
    np.testing.assert_allclose(first.valuations["requested"].pnl, old.asset_pnl)
    assert second.valuations["requested"].pnl.loc["stress", "hedge"] == 1800.0
    assert second.valuations["requested"].pnl.loc["stress", "put"] == 60.0
    assert second.response_exposures.loc["proxy"] == -5000.0
    assert second.response_exposures.loc["a"] == -300.0

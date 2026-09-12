"""Synthetic funded and derivatives report demonstrations using public QIS imports."""

from enum import Enum
import os
from pathlib import Path

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
    StressReportConfig,
    StressScenarios,
    Underlying,
    generate_portfolio_stress_report,
    run_portfolio_stress_test,
)


class Locals(Enum):
    """Offline report demonstrations."""

    FUNDED_AND_DERIVATIVES = 1


def _portfolio(derivatives: bool) -> InstrumentPortfolio:
    """Construct a synthetic dated book with explicit shared response and currency bases."""
    date = pd.Timestamp("2026-08-31")
    factors = ["Equity", "Rates", "Credit", "Credit EM", "FX"]
    responses = ["stock_a", "stock_b", "bond", "metal", "eur"]
    betas = pd.DataFrame(
        [
            [1.4, 0.0, 0.1, 0.1, 0.2],
            [0.8, 0.1, 0.1, 0.0, 0.0],
            [0.1, 1.0, 1.0, 0.4, 0.0],
            [0.2, -0.1, 0.0, 0.0, 0.1],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        index=responses,
        columns=factors,
    )
    x = np.array(
        [
            [0.18, 0.0, 0.0, 0.0, 0.0],
            [-0.025, 0.07, 0.0, 0.0, 0.0],
            [0.025, 0.01, 0.055, 0.0, 0.0],
            [0.03, 0.0, 0.025, 0.06, 0.0],
            [0.005, 0.0, 0.0, 0.0, 0.09],
        ]
    )
    factor_cov = pd.DataFrame(x @ x.T, index=factors, columns=factors)
    residual = pd.Series([0.12**2, 0.08**2, 0.03**2, 0.16**2, 0.0], index=responses)
    covariance = betas @ factor_cov @ betas.T + pd.DataFrame(
        np.diag(residual), index=responses, columns=responses
    )
    model = RiskModel(
        {date: covariance},
        {date: betas},
        {date: factor_cov},
        {date: residual},
        {
            "credit_family": FactorGroupSpec(
                "credit_family", ("Credit", "Credit EM"), label="Credit + Credit EM"
            )
        },
    )
    underlyings = {
        "share_a": Underlying("share_a", 100.0, "USD", "stock_a", ResponseBasis.REFERENCE),
        "share_b": Underlying("share_b", 75.0, "EUR", "stock_b", ResponseBasis.REFERENCE),
        "bond_fund": Underlying("bond_fund", 100.0, "USD", "bond", ResponseBasis.REFERENCE),
        "metal_contract": Underlying(
            "metal_contract", 2000.0, "USD", "metal", ResponseBasis.REFERENCE
        ),
        "cash": Underlying("cash", 1.0, "USD", None, ResponseBasis.LOCAL),
    }
    holdings = [
        PortfolioHolding(
            "stock_1",
            "Growth equity A",
            3_000_000.0,
            (InstrumentLeg(InstrumentType.DELTA_1, "share_a", 30_000.0),),
        ),
        PortfolioHolding(
            "stock_2",
            "European equity B",
            2_000_000.0,
            (InstrumentLeg(InstrumentType.DELTA_1, "share_b", 20_000.0),),
        ),
        PortfolioHolding(
            "fund_1",
            "Credit and rates fund",
            2_500_000.0,
            (InstrumentLeg(InstrumentType.DELTA_1, "bond_fund", 25_000.0),),
        ),
        PortfolioHolding(
            "metal_1",
            "Funded metal position",
            1_500_000.0,
            (InstrumentLeg(InstrumentType.DELTA_1, "metal_contract", 750.0),),
        ),
        PortfolioHolding(
            "cash_1",
            "Cash reserve",
            1_000_000.0,
            (InstrumentLeg(InstrumentType.DELTA_1, "cash", 1_000_000.0),),
        ),
    ]
    if derivatives:
        holdings.extend(
            [
                PortfolioHolding(
                    "option_1",
                    "Protective equity put",
                    180_000.0,
                    (InstrumentLeg(InstrumentType.PUT, "share_a", 20_000.0, strike=95.0),),
                ),
                PortfolioHolding(
                    "option_2",
                    "Short equity call",
                    -90_000.0,
                    (InstrumentLeg(InstrumentType.CALL, "share_a", -15_000.0, strike=110.0),),
                ),
                PortfolioHolding(
                    "future_1",
                    "Short metal futures",
                    0.0,
                    (InstrumentLeg(InstrumentType.FUTURE, "metal_contract", -20.0, 50.0),),
                ),
                PortfolioHolding(
                    "accumulator_1",
                    "Continuing accumulator proxy",
                    -120_000.0,
                    (
                        InstrumentLeg(InstrumentType.CALL, "share_b", 10_000.0, strike=75.0),
                        InstrumentLeg(InstrumentType.PUT, "share_b", -20_000.0, strike=75.0),
                    ),
                    kink_policy=KinkPolicy.RIGHT,
                    metadata={"contract_policy": "remaining-quantity intrinsic proxy"},
                ),
            ]
        )
    return InstrumentPortfolio(
        holdings=tuple(holdings),
        underlyings=underlyings,
        risk_model=model,
        risk_date=date,
        valuation_date=date,
        reference_currency="USD",
        reporting_denominator=10_000_000.0,
        denominator_label="Investment capital",
        fx_rates={"EUR": Underlying("EURUSD", 1.15, "USD", "eur", ResponseBasis.REFERENCE)},
    )


def _generate(derivatives: bool, output_dir: Path):
    """Run independent/conditional shocks, historical months and split factor grids."""
    portfolio = _portfolio(derivatives)
    requests = pd.DataFrame.from_dict(
        {
            "Equity -20%": {"Equity": -0.2},
            "Equity -10%": {"Equity": -0.1},
            "Equity +10%": {"Equity": 0.1},
            "Equity +30%": {"Equity": 0.3},
            "Rates -4%": {"Rates": -0.04},
            "Rates +4%": {"Rates": 0.04},
            "Credit family -10%": {"credit_family": -0.1},
            "Credit family +10%": {"credit_family": 0.1},
            "FX -10%": {"FX": -0.1},
            "FX +10%": {"FX": 0.1},
        },
        orient="index",
    )
    scenarios = StressScenarios(requests, convention=ShockConvention.SIMPLE)
    grids = {}
    for name, key in [
        ("Equity", "Equity"),
        ("Rates", "Rates"),
        ("Credit + Credit EM (equal split)", "credit_family"),
        ("FX", "FX"),
    ]:
        axis = pd.Index(np.linspace(-0.2, 0.2, 41), name=f"Total {name} simple bump")
        grids[name] = StressScenarios(
            pd.DataFrame({key: axis.to_numpy()}, index=axis),
            ScenarioMode.CONDITIONAL,
            ShockConvention.SIMPLE,
        )
    rng = np.random.default_rng(17)
    covariance = portfolio.risk_model.factor_covar[portfolio.risk_date]
    history = pd.DataFrame(
        rng.multivariate_normal(np.zeros(len(covariance)), covariance.to_numpy() / 12.0, size=104),
        index=pd.date_range("2018-01-31", periods=104, freq="ME"),
        columns=covariance.columns,
    )
    result = run_portfolio_stress_test(portfolio, scenarios, history, grids)
    responses = result.factor_loadings.index
    diagnostics = pd.DataFrame({"r2": [0.68, 0.76, 0.88, 0.4, 1.0]}, index=responses)
    membership = pd.Series([0, 0, 1, 1, 1], index=responses)
    linkage = np.array(
        [[0.0, 1.0, 0.2, 2.0], [2.0, 3.0, 0.4, 2.0], [4.0, 6.0, 0.6, 3.0], [5.0, 7.0, 1.0, 5.0]]
    )
    config = StressReportConfig(
        title="Synthetic derivatives portfolio" if derivatives else "Synthetic funded portfolio",
        model_label="Generic illustrative five-factor model",
        response_diagnostics=diagnostics,
        cluster_memberships={"illustration": membership},
        cluster_linkages={"illustration": linkage},
        cluster_cutoffs={"illustration": 0.7},
        write_previews=True,
        notes=(
            "Synthetic data only. This demonstrates the public interface and is not a client analysis.",
            "R-squared and cluster topology are illustrative caller-supplied diagnostics.",
        ),
    )
    return generate_portfolio_stress_report(result, output_dir, config)


def run_local(local: Locals = Locals.FUNDED_AND_DERIVATIVES):
    """Generate two offline reports in an explicitly C-local output directory."""
    if local is not Locals.FUNDED_AND_DERIVATIVES:
        raise ValueError(f"unsupported local demonstration: {local}")
    supplied = os.environ.get("QIS_STRESS_DEMO_OUTPUT")
    if supplied:
        output_dir = Path(supplied)
    else:
        output_dir = (
            Path(os.environ.get("AGENT_LOCAL_ROOT", Path.home() / ".cache" / "qis"))
            / "analyses"
            / f"portfolio_stress_{pd.Timestamp.now():%Y%m%d_%H%M%S}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    for derivatives in (False, True):
        artifacts = _generate(
            derivatives, output_dir / ("derivatives" if derivatives else "funded")
        )
        print(artifacts.pdf_path)


if __name__ == "__main__":
    run_local(local=Locals.FUNDED_AND_DERIVATIVES)

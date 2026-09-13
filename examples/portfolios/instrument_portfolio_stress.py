"""Offline funded and derivative portfolios using the same public stress/report interface.

Run: python -m examples.portfolios.instrument_portfolio_stress
Optional: --case funded|mixed|all --output-dir <fresh directory>
No files are written by default. Prices come from qis.datasets.synthetic; factor names,
loadings, residual variances and contract terms are teaching inputs, not a fitted MATF model.
"""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import qis
from qis.datasets.synthetic import (
    SYNTHETIC_UNIVERSE, SyntheticInstrument, generate_synthetic_prices,
)


def build_model_and_history():
    """Assign a synthetic seven-factor model with annual covariance and monthly log history."""
    names = {
        "SEQ_US": "Equity", "SBD_TSY": "Rates", "SBD_IG": "Credit", "SBD_HY": "Credit EM",
        "SAL_HF": "Carry G10", "SAL_PE": "Carry EM", "SFX_EURUSD": "FX",
    }
    instruments = tuple(item for item in SYNTHETIC_UNIVERSE if item.ticker in names)
    instruments += (SyntheticInstrument("SFX_EURUSD", "FX", 0.08, 0.0, 0.0, 0.0),)
    prices = generate_synthetic_prices(
        instruments=instruments, start="2014-01-02", end="2025-12-31", apply_quirks=False,
    ).rename(columns=names).reindex(columns=list(names.values()))
    covariances = qis.estimate_rolling_ewma_covar(
        prices=prices, returns_freq="W-WED", rebalancing_freq="QE", span=52,
        demean=True, apply_an_factor=True,
    )
    date = max(covariances)
    factor_covariance = covariances[date]
    betas = pd.DataFrame(
        [[1.2, 0.0, 0.1, 0.1, 0.15, 0.05, 0.2],
         [0.1, 0.8, 1.0, 0.8, 0.10, 0.10, 0.0],
         [0.25, -0.1, 0.0, 0.0, 0.15, 0.10, -0.25],
         [0.0, 0.0, 0.0, 0.0, 0.00, 0.00, 1.0]],
        index=["equity_response", "credit_response", "metal_response", "eurusd_response"],
        columns=factor_covariance.columns,
    )
    residual = pd.Series([0.10, 0.03, 0.15, 0.0], index=betas.index).pow(2)
    # Define the covariance of this prescribed toy model; RiskModel owns portfolio analytics.
    covariance = betas @ factor_covariance @ betas.T + pd.DataFrame(
        np.diag(residual), index=betas.index, columns=betas.index,
    )
    model = qis.RiskModel(
        covar={date: covariance}, factor_loadings={date: betas},
        factor_covar={date: factor_covariance}, residual_vars={date: residual},
        factor_groups={
            "family:Credit": qis.FactorGroupSpec(
                "family:Credit", ("Credit", "Credit EM"), label="Credit",
            ),
            "family:Carry": qis.FactorGroupSpec(
                "family:Carry", ("Carry G10", "Carry EM"), label="Carry",
            ),
        },
    )
    history = qis.to_returns(prices.loc[:date], freq="ME", is_log_returns=True).dropna()
    return model, date, history


def build_portfolio(model, date, derivatives=False):
    """Keep marks, local strike quotes and shared fitted responses as separate inputs."""
    reference = qis.ResponseBasis.REFERENCE
    quotes = {
        "share_usd": qis.Underlying("share_usd", 100.0, "USD", "equity_response", reference),
        "share_eur": qis.Underlying("share_eur", 100.0, "EUR", "equity_response", reference),
        "bond_fund": qis.Underlying("bond_fund", 100.0, "USD", "credit_response", reference),
        "metal_contract": qis.Underlying(
            "metal_contract", 2_000.0, "USD", "metal_response", reference,
        ),
        "cash": qis.Underlying("cash", 1.0, "USD", None, qis.ResponseBasis.LOCAL),
    }
    funded = [
        ("stock_usd", "US equity", "share_usd", 2_000_000.0, 20_000.0),
        ("stock_eur", "European equity", "share_eur", 1_000_000.0, 1_000_000 / 120),
        ("fund", "Credit fund", "bond_fund", 1_500_000.0, 15_000.0),
        ("metal", "Funded metal", "metal_contract", 500_000.0, 250.0),
        ("cash", "Cash reserve", "cash", 1_000_000.0, 1_000_000.0),
    ]
    holdings = [qis.PortfolioHolding(
        key, name, mark, (qis.InstrumentLeg(qis.InstrumentType.DELTA_1, quote, units),),
    ) for key, name, quote, mark, units in funded]
    if derivatives:
        holdings += [
            qis.PortfolioHolding(
                "put", "Protective put", 100_000.0,
                (qis.InstrumentLeg(qis.InstrumentType.PUT, "share_usd", 10_000, strike=100),),
                kink_policy=qis.KinkPolicy.LEFT,
            ),
            qis.PortfolioHolding(
                "call", "Short EUR call", -25_000.0,
                (qis.InstrumentLeg(qis.InstrumentType.CALL, "share_eur", -5_000, strike=110),),
            ),
            qis.PortfolioHolding(
                "future", "Short metal future", 0.0,
                (qis.InstrumentLeg(qis.InstrumentType.FUTURE, "metal_contract", -5, 50),),
            ),
            qis.PortfolioHolding(
                "accumulator", "Accumulator proxy", -40_000.0,
                (qis.InstrumentLeg(qis.InstrumentType.CALL, "share_eur", 2_000, strike=100),
                 qis.InstrumentLeg(qis.InstrumentType.PUT, "share_eur", -4_000, strike=100)),
                kink_policy=qis.KinkPolicy.RIGHT,
                metadata={"quantity_basis": "2000 base units; leverage 2 applied once"},
            ),
            qis.PortfolioHolding(
                "decumulator", "Decumulator proxy", -20_000.0,
                (qis.InstrumentLeg(qis.InstrumentType.PUT, "share_usd", 1_000, strike=100),
                 qis.InstrumentLeg(qis.InstrumentType.CALL, "share_usd", -2_000, strike=100)),
                kink_policy=qis.KinkPolicy.LEFT,
                metadata={"quantity_basis": "1000 base units; leverage 2 applied once"},
            ),
        ]
    holdings = tuple(replace(h, metadata={**h.metadata, "short_name": h.name}) for h in holdings)
    return qis.InstrumentPortfolio(
        holdings=holdings, underlyings=quotes, risk_model=model, risk_date=date,
        valuation_date=date, reference_currency="USD", reporting_denominator=6_000_000.0,
        denominator_label="Investment capital",
        fx_rates={"EUR": qis.Underlying("EURUSD", 1.2, "USD", "eurusd_response", reference)},
    )


def scenario_inputs(model, date):
    """Use sparse anchors for free factors and explicit zeros to pin a factor unchanged."""
    anchors = pd.DataFrame.from_dict({
        "Equity down": {"Equity": -0.2},
        "Equity up": {"Equity": 0.4},
        "Rates down": {"Rates": -0.05},
        "Credit down": {"family:Credit": -0.1},
        "Carry up": {"family:Carry": 0.1},
        "FX up": {"FX": 0.1},
        "Equity; FX fixed": {"Equity": -0.2, "FX": 0.0},
        "No move": {factor: 0.0 for factor in model.factor_covar[date].columns},
    }, orient="index")
    requests = qis.StressScenarios(anchors, convention=qis.ShockConvention.SIMPLE)
    axis = pd.Index(np.arange(-20, 21) / 100, name="Total simple factor/family bump")
    grids = {name: qis.StressScenarios(
        pd.DataFrame({key: axis.to_numpy()}, index=axis),
        mode=qis.ScenarioMode.CONDITIONAL, convention=qis.ShockConvention.SIMPLE,
    ) for name, key in [("Equity", "Equity"), ("Rates", "Rates"),
                       ("Credit", "family:Credit"), ("FX", "FX")]}
    return requests, grids


def report_config(portfolio, title):
    """Supply a consumer appendix without inventing fitted R-squared or dendrograms."""
    appendix = pd.DataFrame({
        "Value": [str(len(portfolio.holdings)), str(portfolio.risk_date.date()),
                  portfolio.denominator_label, "Synthetic inputs only"],
    }, index=["Source holdings", "Assigned risk date", "Denominator", "Data origin"])
    return qis.StressReportConfig(
        title=title, model_label="Illustrative seven-factor model",
        selected_grids=("Equity", "Rates", "Credit", "FX"),
        appendix_table=appendix, appendix_title="Example inputs and assumptions",
        appendix_notes=(
            "All quantities and quotes are synthetic; factor names do not identify a MATF fit.",
            "Accumulator/decumulator legs represent continuing remaining quantities; "
            "fixing paths are omitted.",
            "No fitted R-squared or original clustering topology is supplied; "
            "these remain unavailable.",
        ),
    )


def verify_result(portfolio, result, derivatives):
    """Check economic identities independently of the report's display tables."""
    requested = result.valuations["requested"]
    np.testing.assert_allclose(requested.pnl.loc["No move"], 0.0, atol=1e-10)
    pd.testing.assert_series_equal(
        requested.mtm.loc["No move"], result.positions.observed_mtm, check_names=False,
    )
    np.testing.assert_allclose(
        requested.factor_log_shocks.loc["Credit down", ["Credit", "Credit EM"]], np.log(0.95),
    )
    np.testing.assert_allclose(
        result.attribution["requested"].sum(axis=1), requested.portfolio_pnl, atol=1e-8,
    )
    risk = result.report_diagnostics["Annualised portfolio risk"]
    np.testing.assert_allclose(
        result.report_diagnostics["Factor Euler volatility"].euler_vol.sum()
        + risk.loc["Idiosyncratic", "euler_vol"], risk.loc["Total", "annual_vol"],
    )
    assert ("lower_bound" in result.grid_summaries["Credit"]) is (not derivatives)
    regressions = result.report_diagnostics["Grid polynomial regressions"]
    assert regressions["order"].eq(2).all()
    assert set(regressions.index) == set(result.grid_summaries)
    if derivatives:
        # -5 contracts x multiplier 50 x spot 2000: zero MTM still has -500,000 exposure.
        assert result.response_jacobian.loc["future", "metal_response"] == -500_000.0
        # USD-return beta 0.2 minus EUR FX beta 1: EUR spot falls in an isolated FX-up scenario.
        local_spot = 100 * 1.1 ** -0.8
        expected_accumulator = 4_000 * (local_spot - 100) * (1.2 * 1.1)
        np.testing.assert_allclose(
            requested.pnl.loc["FX up", "accumulator"], expected_accumulator, atol=1e-8,
        )


def run_example(output_dir=None, case="all"):
    """Compare funded and mixed books; optionally write both standard report packs."""
    if output_dir is not None:
        output_dir = Path(output_dir)
        if output_dir.exists():
            raise FileExistsError("Use a fresh example output directory")
    model, date, history = build_model_and_history()
    requests, grids = scenario_inputs(model, date)
    results = {}
    for label in (("funded", "mixed") if case == "all" else (case,)):
        derivatives = label == "mixed"
        portfolio = build_portfolio(model, date, derivatives)
        result = qis.run_portfolio_stress_test(portfolio, requests, history, grids)
        verify_result(portfolio, result, derivatives)
        if output_dir is not None:
            artifact = qis.generate_portfolio_stress_report(
                result, output_dir / label,
                report_config(portfolio, f"Synthetic {label} portfolio"),
            )
            print(artifact.pdf_path)
        print(f"{label}: {len(portfolio.holdings)} holdings; "
              f"{len(result.historical.pnl)} synthetic months; valuation and Euler checks passed")
        results[label] = result
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("funded", "mixed", "all"), default="all")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run_example(args.output_dir, args.case)

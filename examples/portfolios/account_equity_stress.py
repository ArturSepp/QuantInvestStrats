"""Offline account statement, signed borrowing and equity-after-stress example.

Run: python -m examples.portfolios.account_equity_stress
Select Locals.REPORT; run_local accepts output_path and with_beta_intervals overrides.
No files are written by default. Uses the existing seeded synthetic factor example;
prescribed loadings are teaching inputs. No client data, Bloomberg or network access.
"""
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import qis

from examples.portfolios.instrument_portfolio_stress import (
    build_model_and_history, scenario_inputs,
)


def build_account(model, date):
    """Supply 150m invested bonds, 50m borrowing and 100m net account equity."""
    ledger = pd.DataFrame({
        "mtm": [150_000_000., -50_000_000.],
        "role": [qis.PositionRole.ASSET, qis.PositionRole.FINANCING],
        "asset_class": [qis.AccountAssetClass.FIXED_INCOME, qis.AccountAssetClass.BORROWING],
        "currency": ["USD", "USD"],
    }, index=["bond_fund", "loan"])
    accounting = qis.PortfolioAccounting(ledger, net_equity=100_000_000.)
    quotes = {
        "bond_quote": qis.Underlying("bond_quote", 100., "USD", "credit_response",
                                    qis.ResponseBasis.REFERENCE),
        "loan_quote": qis.Underlying("loan_quote", 1., "USD", None, qis.ResponseBasis.LOCAL),
    }
    holdings = (
        qis.PortfolioHolding("bond_fund", "Synthetic bond investment", 150_000_000.,
            (qis.InstrumentLeg(qis.InstrumentType.DELTA_1, "bond_quote", 1_500_000.),)),
        qis.PortfolioHolding("loan", "USD borrowing", -50_000_000.,
            (qis.InstrumentLeg(qis.InstrumentType.DELTA_1, "loan_quote", -50_000_000.),)),
    )
    return qis.InstrumentPortfolio(holdings, quotes, model, date, date, "USD",
        accounting.denominator, accounting.denominator_label, accounting=accounting,
        response_provenance={"credit_response": qis.ResponseProvenance(
            qis.HistorySource.SYNTHETIC, "USD", qis.ReturnBasis.TOTAL,
            "Prescribed synthetic bond response to seven generated factors",
            limitations=("Illustrative loadings; not an estimated client product",))})


def run_example(output_dir=None, with_beta_intervals=False):
    """Check the accounting example and optionally create the standard PDF/workbook."""
    if output_dir is not None:
        output_dir = Path(qis.get_output_path(output_dir, create=False))
    model, date, history = build_model_and_history()
    portfolio = build_account(model, date)
    scenarios, grids = scenario_inputs(model, date)
    # Scenario history is explicit and independent of any product fitting period.
    start = str(history.index.min().to_period("M").start_time.date())
    selection = qis.HistoricalScenarioSelection("Equity", start, 10)
    result = qis.run_portfolio_stress_test(portfolio, scenarios, history, grids,
        qis.StressTestConfig(historical_selection=selection))
    shares = result.report_diagnostics["Asset class allocation with borrowing"]
    np.testing.assert_allclose(shares.loc["Fixed Income", "gross_asset_share"], 1.)
    np.testing.assert_allclose(shares.loc["Borrowing", "gross_asset_share"], -1/3)
    value = result.valuations["requested"]
    np.testing.assert_allclose(value.pnl.loan, 0., atol=1e-8)
    np.testing.assert_allclose(value.pnl.loc["No move"], 0., atol=1e-8)
    # Independent scalar response: family -10% is split into -5% on two factors.
    beta = model.factor_loadings[date].loc["credit_response"]
    expected = 150_000_000. * (.95**(beta.Credit + beta["Credit EM"]) - 1.)
    np.testing.assert_allclose(value.portfolio_pnl.loc["Credit down"], expected, atol=1e-7)
    equity = result.report_diagnostics["requested equity after stress"]
    np.testing.assert_allclose(equity.equity_after_stress, 100_000_000.+value.portfolio_pnl)
    np.testing.assert_allclose(equity.reporting_return, value.portfolio_pnl/100_000_000.)
    print("Gross assets USD 150m; borrowing USD 50m; net equity USD 100m")
    print("Fixed Income 100.00%; Borrowing -33.33%; assets/equity 1.50x")
    print("Scenario checks passed; chart order: % P&L, Total P&L, Equity after")
    if output_dir is not None:
        # These supplied bounds demonstrate presentation only. In production the
        # caller supplies estimates and intervals from its inference implementation.
        intervals = None
        if with_beta_intervals:
            center = result.factor_betas
            intervals = pd.DataFrame({'estimate': center, 'lower': center-.15,
                                      'upper': center+.15, 'status': 'Illustrative'})
        artifacts = qis.generate_portfolio_stress_report(result, Path(output_dir),
            qis.StressReportConfig(title="Synthetic financed account",
                report_name="Account accounting example", model_name="Illustrative factors",
                model_label="Synthetic seven-factor model", write_previews=True,
                factor_beta_intervals=intervals,
                beta_interval_label="Illustrative production ranges (not fitted)",
                beta_interval_target="production", beta_interval_kind="stability",
                beta_interval_notes=("Prescribed 95% bounds demonstrate the interface; "
                                     "they are not estimated statistical intervals.",),
                notes=("All data and loadings are synthetic teaching inputs.",)))
        print(artifacts.pdf_path)
    return result


class Locals(Enum):
    CHECK = 1
    REPORT = 2


def run_local(local: Locals, output_path: str = None, with_beta_intervals: bool = False):
    """Check offline or write reports using YAML defaults/an explicit destination."""
    if local not in (Locals.CHECK, Locals.REPORT):
        raise ValueError(local)
    output = None
    if local == Locals.REPORT or output_path is not None:
        output = Path(qis.get_output_path(output_path, create=False))
        if output_path is None:
            output /= 'account_equity_stress'
    return run_example(output_dir=output, with_beta_intervals=with_beta_intervals)


if __name__ == '__main__':
    run_local(local=Locals.CHECK)

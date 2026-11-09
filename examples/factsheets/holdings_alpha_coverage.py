"""Offline strict/proxy-assisted holdings-alpha reports on unchanged synthetic NAV.

The missing instrument histories are a reporting-only simulation, applied after the
backtest. An identical synthetic bond path is the declared teaching proxy; it is not
an empirically selected real-asset mapping. No downloads or files are required.
"""
from copy import deepcopy
from enum import Enum

import matplotlib.pyplot as plt
import pandas as pd
import qis
from qis.datasets.synthetic import generate_synthetic_prices
from qis.portfolio.reports.config import ReportingFrequency, fetch_default_report_kwargs


def run_example() -> dict:
    """Render quarterly strict and assisted reports and verify capital is unchanged."""
    prices = generate_synthetic_prices(end="2025-12-31", apply_quirks=False)[
        ["SEQ_US", "SBD_TSY", "SCM_GLD"]]
    portfolio = qis.backtest_model_portfolio(
        prices, weights=[.65, .34999995, .00000005], ticker="Synthetic portfolio")
    reporting = deepcopy(portfolio)
    reporting.prices.loc[:"2018-12-31", "SBD_TSY"] = float("nan")
    reporting.prices["SCM_GLD"] = float("nan")
    coverage = qis.HoldingsAlphaCoverage(
        weight_tolerance=1e-6, total_weight_tolerance=1e-5,
        proxy_map={"SBD_TSY": "Declared bond proxy"},
        proxy_prices=prices[["SBD_TSY"]].rename(columns={"SBD_TSY": "Declared bond proxy"}))
    time_period = qis.TimePeriod("2009-01-01", "2025-12-31")
    options = fetch_default_report_kwargs(
        time_period=time_period, reporting_frequency=ReportingFrequency.QUARTERLY,
        add_rates_data=False)
    reports = {}
    for name, policy in (("strict", None), ("assisted", coverage)):
        reports[name] = qis.generate_strategy_factsheet(
            reporting, benchmark_prices=prices[["SEQ_US"]], time_period=time_period,
            alpha_coverage=policy, add_performance_inference_sheet=True,
            add_current_position_var_risk_sheet=False, **options)
        pd.testing.assert_series_equal(reporting.nav, portfolio.nav)
        pd.testing.assert_frame_equal(reporting.weights, portfolio.weights)
        pd.testing.assert_frame_equal(reporting.units, portfolio.units)
    assert any("coverage audit" in fig.get_suptitle() for fig in reports["assisted"])
    return {"reports": reports, "coverage": coverage, "portfolio": portfolio,
            "reporting_portfolio": reporting, "options": options}


class Locals(Enum):
    """Select the offline report workflow."""

    COVERAGE = "coverage"


def run_local(local: Locals):
    """Display the quarterly factsheets, including the explicit coverage audit."""
    if local is not Locals.COVERAGE:
        raise ValueError(local)
    result = run_example()
    plt.show()
    return result


if __name__ == "__main__":
    run_local(local=Locals.COVERAGE)

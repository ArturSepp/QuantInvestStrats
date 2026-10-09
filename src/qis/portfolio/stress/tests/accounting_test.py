"""Independent funding, denominator, FX liability and provenance contracts."""

from dataclasses import replace
import math

import numpy as np
import pandas as pd
import pytest

from qis.portfolio.stress.accounting import (
    HistorySource, PortfolioAccounting, PositionRole, ReportingBasis,
    ResponseProvenance, ReturnBasis,
)
from qis.portfolio.stress.analytics import run_portfolio_stress_test
from qis.portfolio.stress.instruments import (
    InstrumentLeg, InstrumentType, Underlying, ResponseBasis,
)
from qis.portfolio.stress.portfolio import PortfolioHolding
from qis.portfolio.stress.tests.analytics_test import request


def account_portfolio(market, basis=ReportingBasis.NET_EQUITY, fixed=None):
    """Use a complete synthetic account: stock 100 + cash 10 - borrowing 20 = equity 90."""
    rows = pd.DataFrame({"mtm": [100., 10., -20.],
                         "role": [PositionRole.ASSET, PositionRole.CASH, PositionRole.FINANCING]},
                        index=["stock", "cash", "loan"])
    account = PortfolioAccounting(rows, 90., basis, fixed)
    holdings = [PortfolioHolding(key, key, mark,
                (InstrumentLeg(InstrumentType.DELTA_1, quote, 1. if mark >= 0 else -1.),))
                for key, mark, quote in [("stock", 100., "actual"), ("cash", 10., "cash_quote"),
                                         ("loan", -20., "loan_quote")]]
    quotes = {"actual": Underlying("actual", 100., "USD", "stock", ResponseBasis.REFERENCE),
              "cash_quote": Underlying("cash_quote", 1., "USD", None, ResponseBasis.LOCAL),
              "loan_quote": Underlying("loan_quote", 1., "USD", None, ResponseBasis.LOCAL)}
    return market(holdings, denominator=account.denominator, underlyings=quotes, accounting=account)


def test_borrowing_reduces_equity_denominator_and_preserves_asset_pnl(market):
    p = account_portfolio(market)
    result = run_portfolio_stress_test(p, request())
    assert result.valuations["requested"].pnl.loc["down", "stock"] == pytest.approx(-20.)
    np.testing.assert_allclose(result.valuations["requested"].pnl[["cash", "loan"]], 0.)
    assert result.summaries["requested"].loc["down", "portfolio_return"] == pytest.approx(-20/90)
    equity = result.report_diagnostics["requested equity after stress"]
    assert equity.loc["down", "equity_after_stress"] == pytest.approx(70.)
    assert equity.loc["down", "pnl_to_net_equity"] == pytest.approx(-20/90)
    assert p.accounting.summary()["assets_to_equity"] == pytest.approx(110/90)
    assert p.accounting.borrowing == 20.
    assert p.denominator_label == "Net equity / NAV"
    legacy = replace(p, accounting=None)
    old = run_portfolio_stress_test(legacy, request())
    pd.testing.assert_frame_equal(result.valuations["requested"].pnl,
                                  old.valuations["requested"].pnl)
    pd.testing.assert_frame_equal(result.summaries["requested"], old.summaries["requested"])
    pd.testing.assert_series_equal(result.risk, old.risk)


@pytest.mark.parametrize("basis,fixed,denominator", [
    (ReportingBasis.NET_EQUITY, None, 90.),
    (ReportingBasis.GROSS_ASSETS, None, 110.),
    (ReportingBasis.FIXED_NOTIONAL, 200., 200.),
])
def test_denominator_basis_is_explicit_and_equity_table_remains_equity_based(
        market, basis, fixed, denominator):
    p = account_portfolio(market, basis, fixed)
    result = run_portfolio_stress_test(p, request())
    assert result.summaries["requested"].loc[
        "down", "portfolio_return"] == pytest.approx(-20/denominator)
    assert result.report_diagnostics["requested equity after stress"].loc[
        "down", "pnl_to_net_equity"] == pytest.approx(-20/90)
    with pytest.raises(ValueError, match="denominator differs"):
        replace(p, reporting_denominator=denominator+1)


def test_short_and_derivative_liabilities_are_not_misclassified_as_borrowing():
    p = pd.DataFrame({"mtm": [150., -20., -10., -5.],
        "role": [PositionRole.ASSET, PositionRole.FINANCING, PositionRole.SHORT,
                 PositionRole.DERIVATIVE]}, index=["asset", "loan", "short", "option"])
    account = PortfolioAccounting(p, 115.)
    assert account.borrowing == 20.
    assert account.summary().other_liabilities == 15.
    with pytest.raises(ValueError, match="explicit PositionRole"):
        PortfolioAccounting(p.assign(role="loan"), 115.)
    with pytest.raises(ValueError, match="reconcile"):
        PortfolioAccounting(p, 150.)


def test_foreign_currency_borrowing_retains_fx_risk(market):
    rows = pd.DataFrame({"mtm": [100., -24.],
        "role": [PositionRole.ASSET, PositionRole.FINANCING]}, index=["stock", "loan"])
    holdings = [PortfolioHolding("stock", "stock", 100.,
        (InstrumentLeg(InstrumentType.DELTA_1, "actual", 1.),)),
        PortfolioHolding("loan", "EUR loan", -24.,
        (InstrumentLeg(InstrumentType.DELTA_1, "loan_quote", -1.),))]
    p = market(holdings, currency="EUR", denominator=76., accounting=PortfolioAccounting(rows, 76.),
        underlyings={"actual": Underlying("actual", 100., "USD", "stock", ResponseBasis.REFERENCE),
                     "loan_quote": Underlying("loan_quote", 1., "EUR", None, ResponseBasis.LOCAL)})
    shocks = pd.DataFrame(0., index=["fx"], columns=p.risk_model.factor_covar[p.risk_date].columns)
    shocks.loc["fx", "FX"] = math.log(.9)
    value = p.evaluate(shocks)
    assert value.pnl.loc["fx", "loan"] == pytest.approx(2.4)
    assert value.mtm.loc["fx", "loan"] == pytest.approx(-21.6)


def test_accounting_and_provenance_exports_are_detached_from_caller_data(market):
    p = account_portfolio(market)
    provenance = ResponseProvenance(HistorySource.SYNTHETIC, "USD", ReturnBasis.TOTAL,
        "Spot return + coupon accrual + declared credit factor return", ("ACT/365 funding",),
        ("Contractual optionality excluded",), "2022-01-31", "2026-08-31")
    p = replace(p, response_provenance={"stock": provenance})
    result = run_portfolio_stress_test(p, request())
    p.accounting.positions.loc["loan", "mtm"] = -100.
    assert result.report_diagnostics["Account positions"].loc["loan", "mtm"] == -20.
    assert result.report_diagnostics["Response history provenance"].loc[
        "stock", "source"] == "synthetic"
    assert result.metadata["accounting"]["net_equity"] == 90.


def test_uncovered_ledger_position_is_disclosed_without_claiming_zero_risk(market):
    p = account_portfolio(market)
    ledger = p.accounting.positions.copy()
    ledger.loc["uncovered"] = [30., PositionRole.ASSET]
    p = replace(p, accounting=PortfolioAccounting(ledger, 120.), reporting_denominator=120.)
    result = run_portfolio_stress_test(p, request())
    assert result.metadata["accounting_unvalued_position_ids"] == ["uncovered"]
    assert not result.report_diagnostics["Account positions"].loc["uncovered", "valuation_covered"]
    assert result.report_diagnostics["requested equity after stress"].loc[
        "down", "equity_after_stress"] == pytest.approx(100.)

"""Gross-allocation mapping and the three scenario panels have explicit denominators."""

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.text import Text

from qis.portfolio.stress.accounting import (
    AccountAssetClass, PortfolioAccounting, ReportingBasis, PositionRole,
)
from qis.portfolio.stress.analytics import StressTestConfig, run_portfolio_stress_test
from qis.portfolio.stress.historical import HistoricalScenarioSelection
from qis.portfolio.stress.reporting import StressReportConfig
from qis.portfolio.stress._figures import report_pages
from qis.portfolio.stress._statement import displayed_allocation
from qis.portfolio.stress.tests.accounting_test import account_portfolio
from qis.portfolio.stress.tests.analytics_test import request
from qis.portfolio.stress.tests.historical_selection_test import history


def mapped_portfolio(market, basis=ReportingBasis.NET_EQUITY):
    p = account_portfolio(market, basis)
    ledger = p.accounting.positions.copy()
    ledger["asset_class"] = [AccountAssetClass.EQUITY, AccountAssetClass.LIQUIDITY,
                             AccountAssetClass.BORROWING]
    ledger["currency"] = ["EUR", "USD", "USD"]
    return replace(p, accounting=PortfolioAccounting(ledger, 90., basis))


def test_allocation_excludes_borrowing_and_uses_gross_assets_not_net_equity(market):
    p = mapped_portfolio(market)
    result = run_portfolio_stress_test(p, request())
    allocation = result.report_diagnostics["Gross asset allocation"]
    assert allocation.sum().sum() == 110.
    assert allocation.loc["Borrowing"].sum() == 0.
    funded = result.report_diagnostics["Asset class allocation with borrowing"]
    assert funded.loc["Borrowing", "amount_ref"] == -20.
    assert funded.loc["Borrowing", "gross_asset_share"] == pytest.approx(-20/110)
    assert allocation.loc["Liquidity", "USD"] == 10.
    assert allocation.loc["Equity", "EUR"] == 100.
    shares = result.report_diagnostics["Gross currency allocation"].gross_asset_share
    assert shares.loc["EUR"] == pytest.approx(100/110)
    assert shares.sum() == pytest.approx(1.)
    assert result.summaries["requested"].loc["down", "portfolio_return"] == pytest.approx(-20/90)


def test_currency_display_is_capped_at_five_without_losing_small_currency_exposures():
    ledger = pd.DataFrame({"mtm": [70., 60., 50., 40., 30., 20., 10., -40.],
        "role": [PositionRole.ASSET]*7 + [PositionRole.FINANCING],
        "asset_class": [AccountAssetClass.EQUITY]*7 + [AccountAssetClass.BORROWING],
        "currency": ["USD", "EUR", "GBP", "JPY", "CHF", "TRY", "AUD", "USD"]})
    account = PortfolioAccounting(ledger, 240.)
    full = account.gross_asset_allocation()
    shown = displayed_allocation(full)
    assert len(full.columns) == 7 and len(shown.columns) == 5
    assert shown.columns.tolist() == ["USD", "EUR", "GBP", "JPY", "Other"]
    assert shown.loc["Equity", "Other"] == 60.
    assert shown.sum().sum() == full.sum().sum() == 280.


@pytest.mark.parametrize('missing', ['currency', 'asset_class'])
def test_partial_allocation_mapping_is_rejected(market, missing):
    p = mapped_portfolio(market)
    with pytest.raises(ValueError, match='both asset_class and currency'):
        PortfolioAccounting(p.accounting.positions.drop(columns=missing), 90.)


@pytest.mark.parametrize('basis', [ReportingBasis.NET_EQUITY, ReportingBasis.GROSS_ASSETS])
def test_statement_and_scenario_panel_order_values_and_page_numbers(market, basis):
    p = mapped_portfolio(market, basis)
    result = run_portfolio_stress_test(p, request(), history(p),
        config=StressTestConfig(historical_selection=HistoricalScenarioSelection(count=2)))
    valuations = [result.valuations['requested'], result.valuations['conditional'],
                  result.historical]
    for page, (title, fig) in enumerate(report_pages(result, StressReportConfig()), 1):
        try:
            texts = '\n'.join(t.get_text() for t in fig.findobj(Text))
            if page == 1:
                assert 'Statement of assets' in title
                assert 'Reconciled balance sheet' in texts
                assert 'Asset allocation and borrowing' in texts
                assert 'Asset-class allocation' in texts and 'Currency allocation' in texts
                assert 'Requested stress (' not in texts
                assert 'maturities' not in texts
            elif 2 <= page <= 4:
                assert 'portfolio_pnl' not in texts
                axes = fig.axes[:3]
                assert axes[0].get_title().startswith('Portfolio P&L')
                assert axes[1].get_title().startswith('Total P&L')
                assert axes[2].get_title().startswith('Equity after')
                value = valuations[page-2]
                rows = (result.historical_ranking.index if page == 4 else value.pnl.index)[:12]
                pnl = value.portfolio_pnl.loc[rows].to_numpy()
                expected = [pnl/p.reporting_denominator, pnl, 90. + pnl]
                for ax, reference in zip(axes, expected):
                    widths = [bar.get_width() for container in ax.containers for bar in container]
                    np.testing.assert_allclose(sorted(widths), sorted(reference), atol=1e-12)
            assert str(page) == fig.texts[-1].get_text() or str(page) in [
                t.get_text() for t in fig.texts if t.get_position() == (.96, .025)]
        finally:
            plt.close(fig)


def test_fully_invested_levered_fixed_income_account_shows_100_and_minus_one_third():
    ledger = pd.DataFrame({"mtm": [150e6, -50e6],
        "role": [PositionRole.ASSET, PositionRole.FINANCING],
        "asset_class": [AccountAssetClass.FIXED_INCOME, AccountAssetClass.BORROWING],
        "currency": ["USD", "USD"]}, index=["bonds", "loan"])
    account = PortfolioAccounting(ledger, 100e6)
    allocation = account.allocation_with_borrowing()
    shares = allocation.sum(axis=1)/account.gross_assets
    assert shares.loc["Fixed Income"] == 1.
    assert shares.loc["Borrowing"] == pytest.approx(-1/3)
    assert shares.drop(index="Borrowing").sum() == 1.
    assert account.gross_assets == 150e6 and account.net_equity == 100e6


def test_credit_alias_is_accepted_for_financing_and_displayed_as_borrowing():
    assert AccountAssetClass.CREDIT is AccountAssetClass.BORROWING
    ledger = pd.DataFrame({"mtm": [150., -50.],
        "role": [PositionRole.ASSET, PositionRole.FINANCING],
        "asset_class": ["Fixed Income", "Credit"], "currency": ["USD", "USD"]})
    account = PortfolioAccounting(ledger, 100.)
    assert account.positions.loc[1, "asset_class"] == "Borrowing"
    assert account.allocation_with_borrowing().loc["Borrowing", "USD"] == -50.

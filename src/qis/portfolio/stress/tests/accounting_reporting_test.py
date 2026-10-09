"""Accounting and factor-selection report exports and explicit ratio captions."""

from dataclasses import replace
import hashlib
import json

import pandas as pd

from qis.portfolio.stress.accounting import HistorySource, ResponseProvenance, ReturnBasis
from qis.portfolio.stress.analytics import StressTestConfig, run_portfolio_stress_test
from qis.portfolio.stress.historical import HistoricalScenarioSelection
from qis.portfolio.stress.reporting import StressReportConfig, generate_portfolio_stress_report
from qis.portfolio.stress.tests.accounting_test import account_portfolio
from qis.portfolio.stress.tests.analytics_test import request
from qis.portfolio.stress.tests.historical_selection_test import history


def test_account_equity_page_and_exported_provenance_require_no_repricing(market, tmp_path):
    p = account_portfolio(market)
    p = replace(p, response_provenance={"stock": ResponseProvenance(
        HistorySource.SYNTHETIC, "USD", ReturnBasis.TOTAL,
        "Complete synthetic product NAV", ("Declared coupon accrual",),
        ("Contractual optionality excluded",))})
    result = run_portfolio_stress_test(p, request(), history(p),
        config=StressTestConfig(historical_selection=HistoricalScenarioSelection(count=2)))
    artifacts = generate_portfolio_stress_report(result, tmp_path/"account",
        StressReportConfig(write_previews=True))
    manifest = json.loads(artifacts.manifest_path.read_text())
    assert manifest["page_count"] == 14
    assert manifest["page_titles"][0] == "Statement of assets as of 31.08.2026"
    assert manifest["page_titles"][3] == "2 worst Equity months since 2006"
    assert "Account equity, borrowing and stress impact" not in manifest["page_titles"]
    assert "Net equity / NAV USD 90" in manifest["report_heading"]
    assert "Notional" not in manifest["report_heading"]
    assert len(artifacts.preview_paths) == 14
    equity = pd.read_csv(artifacts.table_paths["requested equity after stress"], index_col=0)
    assert equity.loc["down", "equity_after_stress"] == 70.
    provenance = pd.read_csv(artifacts.table_paths["Response history provenance"], index_col=0)
    assert provenance.loc["stock", "source"] == "synthetic"
    assert provenance.loc["stock", "return_basis"] == "total"
    for path, digest in manifest["hashes"].items():
        assert hashlib.sha256((artifacts.pdf_path.parent/path).read_bytes()).hexdigest() == digest

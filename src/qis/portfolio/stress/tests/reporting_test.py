"""Artifact and result-only rendering contracts on synthetic portfolios."""

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest

from qis.portfolio.stress.analytics import run_portfolio_stress_test
from qis.portfolio.stress.reporting import StressReportConfig, generate_portfolio_stress_report
from qis.portfolio.stress.scenarios import ScenarioMode, ShockConvention, StressScenarios
from qis.portfolio.stress.tests.scenarios_test import grouped_portfolio


def test_nine_core_pages_export_all_values_without_repricing(market, tmp_path, monkeypatch):
    """The renderer consumes results; original payoff/model access is unnecessary."""
    p = grouped_portfolio(market)
    request = StressScenarios(
        pd.DataFrame({"Equity": [-0.2, 0.2]}, index=["down", "up"]),
        convention=ShockConvention.SIMPLE,
    )
    grid = StressScenarios(
        pd.DataFrame(
            {"credit_family": [-0.1, 0.0, 0.1]},
            index=pd.Index([-0.1, 0.0, 0.1], name="Total Credit bump"),
        ),
        ScenarioMode.CONDITIONAL,
        ShockConvention.SIMPLE,
    )
    history = pd.DataFrame(
        [[np.log(0.8), 0.0, 0.0, 0.0], [np.log(0.7), 0.0, 0.0, 0.0]],
        index=pd.to_datetime(["2026-01-31", "2026-02-28"]),
        columns=p.risk_model.factor_loadings[p.risk_date].columns,
    )
    result = run_portfolio_stress_test(p, request, history, {"Credit": grid})

    def forbidden(*args, **kwargs):
        """Reject any attempt to repeat portfolio valuation during rendering."""
        raise AssertionError("report attempted to reprice")

    monkeypatch.setattr(type(p), "evaluate", forbidden)
    config = StressReportConfig(
        title="Synthetic funded consumer",
        model_label="Generic four-factor model",
        response_diagnostics=pd.DataFrame({"r2": [0.8]}, index=["proxy"]),
        cluster_memberships={"monthly": pd.Series([0, 0], index=["stock", "proxy"])},
        cluster_linkages={"monthly": np.array([[0.0, 1.0, 0.6, 2.0]])},
        cluster_cutoffs={"monthly": 0.7},
    )
    artifact = generate_portfolio_stress_report(result, tmp_path / "report", config)
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    assert manifest["page_count"] == 9
    assert len(manifest["page_titles"]) == 9
    assert artifact.pdf_path.read_bytes().startswith(b"%PDF-")
    for path, checksum in manifest["hashes"].items():
        assert (
            hashlib.sha256((artifact.pdf_path.parent / path).read_bytes()).hexdigest() == checksum
        )
    values = pd.read_csv(artifact.table_paths["requested holding pnl"], index_col=0)
    pd.testing.assert_frame_equal(
        values, result.valuations["requested"].pnl, check_names=False, rtol=1e-12
    )
    assert artifact.workbook_path.exists()
    from openpyxl import load_workbook
    workbook = load_workbook(artifact.workbook_path)
    total_share = workbook["01 Annualised portfolio risk"]["D2"]
    assert total_share.value == 1
    assert "%" in total_share.number_format
    ci_sheet_name = next(table["workbook_sheet"] for table in manifest["tables"]
                         if table["name"] == "Grid regression confidence bands")
    ci_sheet = workbook[ci_sheet_name]
    assert ci_sheet.freeze_panes == "C2"
    assert "%" in ci_sheet["C2"].number_format
    assert "%" in ci_sheet["E2"].number_format
    workbook.close()
    with zipfile.ZipFile(artifact.workbook_path) as archive:
        sheets = [name for name in archive.namelist() if name.startswith("xl/worksheets/sheet")]
        assert len(sheets) == len(artifact.table_paths) + 1
    with pytest.raises(FileExistsError):
        generate_portfolio_stress_report(result, tmp_path / "report", config)


def test_report_rejects_unknown_diagnostics_before_writing(market, tmp_path):
    """Presentation aliases must not silently relabel the model or add fitted rows."""
    p = grouped_portfolio(market)
    request = StressScenarios(pd.DataFrame({"Equity": [-0.2]}))
    result = run_portfolio_stress_test(p, request)
    for config in [
        StressReportConfig(selected_grids=("Unknown",)),
        StressReportConfig(factor_labels={"Unknown": "Equity"}),
        StressReportConfig(response_diagnostics=pd.DataFrame({"r2": [0.9]}, index=["Unknown"])),
    ]:
        with pytest.raises(ValueError):
            generate_portfolio_stress_report(result, tmp_path / "bad", config)
        assert not Path(tmp_path / "bad").exists()


def test_scenario_descriptions_are_visible_without_changing_ids(market):
    """Consumer scenario labels must reach both charts and contribution tables."""
    import matplotlib.pyplot as plt
    from qis.portfolio.stress._figures import _scenario_page
    p = grouped_portfolio(market)
    request = StressScenarios(
        pd.DataFrame({"Equity": [-0.2]}, index=["S01"]),
        descriptions={"S01": "Global equities -20%"},
    )
    result = run_portfolio_stress_test(p, request)
    fig = _scenario_page(result, StressReportConfig(), 1, "Requested",
                         result.valuations["requested"], "Synthetic scenario")
    try:
        fig.canvas.draw()
        strings = [str(item.get_text()) for item in fig.findobj()
                   if hasattr(item, "get_text")]
        assert any("Global equities" in text for text in strings)
        assert result.valuations["requested"].pnl.index.tolist() == ["S01"]
    finally:
        plt.close(fig)

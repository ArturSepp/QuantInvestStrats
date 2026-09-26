"""The manifest's grid-panel display limit matches the sensitivity page.

``StressReportConfig.selected_grids`` accepts six names and the sensitivity page lays out a
two-by-three grid of panels, but the manifest recorded ``"grid_panels": 4``.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from qis.portfolio.stress._figures import _grid_page
from qis.portfolio.stress.analytics import run_portfolio_stress_test
from qis.portfolio.stress.reporting import StressReportConfig, generate_portfolio_stress_report
from qis.portfolio.stress.scenarios import ScenarioMode, ShockConvention, StressScenarios
from qis.portfolio.stress.tests.scenarios_test import grouped_portfolio


def _result(market):
    """A small completed result with one conditional credit grid."""
    portfolio = grouped_portfolio(market)
    request = StressScenarios(pd.DataFrame({"Equity": [-0.2, 0.2]}, index=["down", "up"]),
                              convention=ShockConvention.SIMPLE)
    grid = StressScenarios(
        pd.DataFrame({"credit_family": [-0.1, 0.0, 0.1]},
                     index=pd.Index([-0.1, 0.0, 0.1], name="Total Credit bump")),
        ScenarioMode.CONDITIONAL,
        ShockConvention.SIMPLE,
    )
    history = pd.DataFrame(
        [[np.log(0.8), 0.0, 0.0, 0.0], [np.log(0.7), 0.0, 0.0, 0.0]],
        index=pd.to_datetime(["2026-01-31", "2026-02-28"]),
        columns=portfolio.risk_model.factor_loadings[portfolio.risk_date].columns,
    )
    return run_portfolio_stress_test(portfolio, request, history, {"Credit": grid})


def test_selected_grids_accept_six_names_and_no_more():
    StressReportConfig(selected_grids=tuple("abcdef"))
    with pytest.raises(ValueError, match="at most six"):
        StressReportConfig(selected_grids=tuple("abcdefg"))


def test_manifest_grid_panel_limit_is_the_number_of_panels_drawn(market, tmp_path):
    result = _result(market)
    config = StressReportConfig(title="Grid panel limit", write_workbook=False)
    figure = _grid_page(result, config)
    try:
        panels = [ax for ax in figure.axes if ax.get_subplotspec() is not None]
    finally:
        plt.close(figure)
    artifact = generate_portfolio_stress_report(result, tmp_path / "report", config)
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    assert len(panels) == 6
    assert manifest["display_limits"]["grid_panels"] == len(panels)

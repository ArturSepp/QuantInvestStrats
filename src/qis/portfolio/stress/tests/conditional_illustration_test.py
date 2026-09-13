"""Independent references for single-factor conditional-shock illustrations."""

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("bump", [-0.10, 0.10])
def test_each_column_is_one_simple_anchor_with_conditional_comoves(bump):
    """Distinguish row/column orientation, covariance scaling and simple/log units."""
    from qis.portfolio.stress._diagnostics import _conditional_shock_tables

    factors = ["Equity", "Oil", "Rates"]
    cov = pd.DataFrame([[.04, -.012, .002], [-.012, .09, .003], [.002, .003, .01]],
                       index=factors, columns=factors)
    before = cov.copy(deep=True)
    tables = _conditional_shock_tables(cov)
    table = tables[f"Conditional factor shocks {bump:+.0%}"]
    expected = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            expected[i, j] = (1.0 + bump)**(cov.iloc[i, j] / cov.iloc[j, j]) - 1.0
    assert list(table.index) == factors == list(table.columns)
    assert table.index.name == "Affected factor"
    assert table.columns.name == "Anchored factor"
    np.testing.assert_allclose(table, expected, atol=1e-14)
    np.testing.assert_allclose(np.diag(table), bump, atol=1e-14)
    assert np.sign(table.loc["Equity", "Oil"]) == -np.sign(bump)
    assert table.loc["Equity", "Oil"] != pytest.approx(table.loc["Oil", "Equity"])
    other = tables[f"Conditional factor shocks {-bump:+.0%}"]
    assert table.loc["Equity", "Oil"] != pytest.approx(-other.loc["Equity", "Oil"])
    pd.testing.assert_frame_equal(cov, before)


def test_zero_variance_anchor_is_unavailable_not_zero_comoves():
    """A nonzero shock cannot condition a deterministic factor under this model."""
    from qis.portfolio.stress._diagnostics import _conditional_shock_tables

    cov = pd.DataFrame([[.04, 0.0], [0.0, 0.0]], index=["Equity", "Fixed"],
                       columns=["Equity", "Fixed"])
    tables = _conditional_shock_tables(cov)
    for bump in [-.1, .1]:
        table = tables[f"Conditional factor shocks {bump:+.0%}"]
        assert table.Fixed.isna().all()
        assert table.loc["Fixed", "Equity"] == 0.0
        assert table.loc["Equity", "Equity"] == pytest.approx(bump)


def test_report_uses_detached_illustrations_and_moves_formulas(market):
    """The new appendix has two percentage tables, with the conditional maths moved to it."""
    import matplotlib.pyplot as plt
    from qis.portfolio.stress.analytics import run_portfolio_stress_test
    from qis.portfolio.stress.reporting import StressReportConfig, _report_tables
    from qis.portfolio.stress.scenarios import StressScenarios
    from qis.portfolio.stress.tests.scenarios_test import grouped_portfolio
    from qis.portfolio.stress._figures import _conditional_page, _methodology_page

    result = run_portfolio_stress_test(
        grouped_portfolio(market), StressScenarios(pd.DataFrame({"Equity": [-.1]})))
    config = StressReportConfig(model_name="Test model", factor_labels={"FX": "Currency"})
    exported = _report_tables(result, config)
    for bump in [-.1, .1]:
        key = f"Conditional factor shocks {bump:+.0%}"
        pd.testing.assert_frame_equal(exported[key], result.report_diagnostics[key])
    new, old = _conditional_page(result, config), _methodology_page(result, config)
    try:
        assert len(new.axes) == 2
        assert all(len(ax.collections) for ax in new.axes)
        assert all("Currency" in [label.get_text() for label in ax.get_yticklabels()]
                   for ax in new.axes)
        text_new = " ".join(item.get_text() for item in new.texts)
        text_old = " ".join(item.get_text() for item in old.texts)
        assert "Test model conditional shocks and covariance" in text_new
        assert r"\Sigma_{F|A}" in text_new and r"\Sigma_{F|A}" not in text_old
        assert "conditional mean" in text_new.lower()
        assert min(item.get_fontsize() for item in new.texts) >= 9
    finally:
        plt.close(new)
        plt.close(old)

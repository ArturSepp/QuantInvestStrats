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


@pytest.mark.parametrize("direction", [-1, 1])
def test_sigma_columns_use_annual_vol_magnitudes_as_simple_anchors(direction):
    """Unequal annual vols must remain exact diagonals, not exponentiated or monthly vols."""
    from qis.portfolio.stress._diagnostics import _conditional_shock_tables

    names = ["Equity", "Oil", "Rates"]
    vols = np.array([.133, .51, .04])
    correlation = np.array([[1., -.25, .1], [-.25, 1., -.58], [.1, -.58, 1.]])
    covariance = pd.DataFrame(correlation * np.outer(vols, vols), index=names, columns=names)
    tables = _conditional_shock_tables(covariance)
    table = tables[f"Conditional factor shocks {direction:+d}sigma"]
    anchors = direction * vols
    expected = np.array([
        [(1 + anchors[j]) ** (correlation[i, j] * vols[i] / vols[j]) - 1
         for j in range(3)] for i in range(3)])
    np.testing.assert_allclose(table, expected, atol=1e-14)
    np.testing.assert_allclose(np.diag(table), anchors, atol=1e-14)
    assert table.index.name == "Affected factor"
    assert table.columns.name == "Anchored factor"
    scaled = _conditional_shock_tables(4 * covariance)
    np.testing.assert_allclose(
        np.diag(scaled["Conditional factor shocks +1sigma"]), 2 * vols, atol=1e-14)
    pd.testing.assert_frame_equal(
        tables["Conditional factor shocks -10%"], scaled["Conditional factor shocks -10%"])


def test_sigma_unavailable_anchors_are_explicit_and_not_clipped():
    """Zero variance and a simple downside beyond -100% cannot be conditional anchors."""
    from qis.portfolio.stress._diagnostics import _conditional_shock_tables

    names = ["Equity", "Extreme", "Boundary", "Fixed"]
    cov = pd.DataFrame(np.diag([.04, 1.44, 1., 0.]), index=names, columns=names)
    tables = _conditional_shock_tables(cov)
    downside = tables["Conditional factor shocks -1sigma"]
    upside = tables["Conditional factor shocks +1sigma"]
    assert downside[["Extreme", "Boundary", "Fixed"]].isna().all().all()
    assert upside.Fixed.isna().all()
    assert upside.loc["Extreme", "Extreme"] == pytest.approx(1.2)
    assert upside.loc["Boundary", "Boundary"] == pytest.approx(1.)
    assert downside.loc["Equity", "Equity"] == pytest.approx(-.2)
    assert downside.loc["Fixed", "Equity"] == 0.


def test_report_uses_detached_illustrations_and_moves_formulas(market):
    """The new appendix has two percentage tables, with the conditional maths moved to it."""
    import matplotlib.pyplot as plt
    from qis.portfolio.stress.analytics import run_portfolio_stress_test
    from qis.portfolio.stress.reporting import StressReportConfig, _report_tables
    from qis.portfolio.stress.scenarios import StressScenarios
    from qis.portfolio.stress.tests.scenarios_test import grouped_portfolio
    from qis.portfolio.stress._figures import (
        _conditional_page, _conditional_sigma_page, _methodology_page)

    result = run_portfolio_stress_test(
        grouped_portfolio(market), StressScenarios(pd.DataFrame({"Equity": [-.1]})))
    config = StressReportConfig(model_name="Test model", factor_labels={"FX": "Currency"})
    exported = _report_tables(result, config)
    for bump in [-.1, .1]:
        key = f"Conditional factor shocks {bump:+.0%}"
        pd.testing.assert_frame_equal(exported[key], result.report_diagnostics[key])
    new, old = _conditional_page(result, config), _methodology_page(result, config)
    sigma = _conditional_sigma_page(result, config)
    try:
        assert len(new.axes) == 2
        assert all(len(ax.collections) for ax in new.axes)
        assert all("Currency" in [label.get_text() for label in ax.get_yticklabels()]
                   for ax in new.axes)
        text_new = " ".join(item.get_text() for item in new.texts)
        text_old = " ".join(item.get_text() for item in old.texts)
        assert "Test model conditional shocks and covariance at 10% shocks" in text_new
        sigma_text = " ".join(item.get_text() for item in sigma.texts)
        assert "Test model conditional shocks and covariance at 1-sigma shocks" in sigma_text
        assert "annual" in sigma_text and "simple-return" in sigma_text
        for page in (new, sigma):
            text = " ".join(item.get_text() for item in page.texts)
            assert "Each column is a separate scenario" in text
            assert "diagonal is the factor shock" in text
            assert "off-diagonal cells are induced conditional shocks" in text
            assert all(ax.get_xlabel() == "Anchored factor (column)" for ax in page.axes)
            assert all(ax.get_ylabel() == "Responding factor (row)" for ax in page.axes)
        for direction in (-1, 1):
            key = f"Conditional factor shocks {direction:+d}sigma"
            pd.testing.assert_frame_equal(exported[key], result.report_diagnostics[key])
        assert r"\Sigma_{F|A}" in text_new and r"\Sigma_{F|A}" not in text_old
        assert "conditional mean" in text_new.lower()
        assert min(item.get_fontsize() for item in new.texts) >= 9
    finally:
        plt.close(new)
        plt.close(old)
        plt.close(sigma)

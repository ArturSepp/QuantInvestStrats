"""Tracking error distinguishes floating-point cancellation from negative risk."""

from decimal import Decimal, localcontext

import numpy as np
import pandas as pd
import pytest

from qis.portfolio.risk.risk_model import RiskModel


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
@pytest.mark.parametrize("dimension", [2, 7, 25])
def test_analytical_null_direction_has_zero_tracking_error(scale, dimension):
    """A rank-one PSD model has zero risk when its single exposure cancels exactly."""
    loading = np.zeros(dimension)
    loading[:2] = [.3, .7]
    weights = np.zeros(dimension)
    weights[:2] = [.7, -.3]
    with localcontext() as context:
        context.prec = 80
        exposure = sum(Decimal(str(w)) * Decimal(str(b))
                       for w, b in zip(weights, loading))
        assert exposure * exposure == 0
    labels = pd.Index([f"asset_{i}" for i in range(dimension)])
    covariance = pd.DataFrame(scale * np.outer(loading, loading), index=labels, columns=labels)
    date = pd.Timestamp("2026-11-17")
    model = RiskModel(covar={date: covariance})
    actual = model.compute_tre_at_date(
        pd.Series(0., index=labels), pd.Series(weights, index=labels), date)
    # Positive roundoff is deliberately not clipped; its risk remains bounded by arithmetic error.
    assert np.isfinite(actual)
    assert actual <= np.sqrt(10 * dimension * np.finfo(float).eps * scale)


def test_grouped_and_ungrouped_null_total_agree():
    """Standalone sleeves keep positive risk even when their combined exposure cancels."""
    labels = pd.Index(["a", "b"])
    date = pd.Timestamp("2026-11-17")
    covariance = pd.DataFrame(np.outer([.3, .7], [.3, .7]), index=labels, columns=labels)
    model = RiskModel(covar={date: covariance})
    benchmark = pd.Series(0., index=labels)
    weights = pd.Series([.7, -.3], index=labels)
    grouped = model.compute_tre_at_date(
        benchmark, weights, date, group_data=pd.Series(["A", "B"], index=labels))
    assert grouped["Total"] == model.compute_tre_at_date(benchmark, weights, date) == 0.
    np.testing.assert_allclose(grouped[["A", "B"]], [.21, .21], rtol=1e-15)


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
def test_material_negative_form_stays_unavailable(scale):
    """A genuinely negative direction is not made valid by a general zero floor."""
    covariance = pd.DataFrame(scale * np.diag([1., -1e-6]))
    with pytest.warns(RuntimeWarning, match="invalid value"):
        actual = RiskModel._compute_tracking_error(pd.Series([0., 1.]), covariance)
    assert np.isnan(actual)


@pytest.mark.parametrize("empty_risk", ["weights", "covariance"])
def test_zero_risk_is_exact(empty_risk):
    """Zero holdings and zero covariance retain an exact zero."""
    weights = pd.Series([.7, -.3])
    covariance = pd.DataFrame([[.09, .21], [.21, .49]])
    if empty_risk == "weights":
        weights *= 0
    else:
        covariance *= 0
    assert RiskModel._compute_tracking_error(weights, covariance) == 0.


def test_positive_quadratic_forms_keep_the_existing_operation_order():
    """No arithmetic path changes for ordinary positive forms."""
    rng = np.random.default_rng(17112026)
    for size in (2, 7, 25):
        matrix = rng.normal(size=(size, size))
        covariance = pd.DataFrame(matrix @ matrix.T + np.eye(size))
        weights = pd.Series(rng.normal(size=size))
        expected = float(np.sqrt(float(weights @ covariance @ weights)))
        assert RiskModel._compute_tracking_error(weights, covariance) == expected

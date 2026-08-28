"""Tests for generic OLS regression inference utilities."""

import numpy as np
from scipy.stats import norm

from qis.utils.regression import estimate_ols_alpha_beta_hac


def _manual_hac_alpha_beta(
        x: np.ndarray,
        y: np.ndarray,
        lags: int,
) -> tuple[np.ndarray, float, float, float, tuple[float, float]]:
    """Compute OLS and Bartlett HAC inference independently of the production helper."""
    design = np.column_stack((np.ones_like(x), x))
    xtx_inv = np.linalg.inv(design.T @ design)
    params = xtx_inv @ design.T @ y
    residuals = y - design @ params
    scores = design * residuals[:, None]
    meat = scores.T @ scores
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        lagged_cross_product = scores[lag:].T @ scores[:-lag]
        meat += weight * (lagged_cross_product + lagged_cross_product.T)
    n_obs, n_params = design.shape
    covariance = (n_obs / (n_obs - n_params)) * xtx_inv @ meat @ xtx_inv
    alpha_se = float(np.sqrt(covariance[0, 0]))
    alpha_pvalue = float(2.0 * norm.sf(abs(params[0] / alpha_se)))
    critical_value = norm.ppf(0.975)
    alpha_interval = (
        float(params[0] - critical_value * alpha_se),
        float(params[0] + critical_value * alpha_se),
    )
    fitted = design @ params
    r_squared = 1.0 - np.square(y - fitted).sum() / np.square(y - y.mean()).sum()
    return params, float(r_squared), alpha_pvalue, alpha_se, alpha_interval


def test_estimate_ols_alpha_beta_hac_matches_matrix_reference() -> None:
    """Generic HAC estimates match an independent matrix-form Bartlett calculation."""
    x = np.array([
        -0.030, 0.012, 0.021, -0.008, 0.025, -0.017,
        0.009, 0.018, -0.011, 0.026, 0.004, -0.019,
        0.014, 0.007, -0.006, 0.023, -0.013, 0.016,
        0.005, -0.010, 0.020, -0.004, 0.011, 0.015,
    ])
    noise = np.array([
        0.0020, -0.0010, 0.0015, -0.0020, 0.0005, 0.0010,
        -0.0015, 0.0025, -0.0005, 0.0010, -0.0010, 0.0015,
        -0.0025, 0.0005, 0.0020, -0.0010, 0.0010, -0.0005,
        0.0015, -0.0020, 0.0005, 0.0010, -0.0015, 0.0020,
    ])
    y = 0.002 + 0.82 * x + noise
    expected_params, expected_r2, expected_pvalue, expected_se, expected_interval = (
        _manual_hac_alpha_beta(x=x, y=y, lags=3)
    )

    result = estimate_ols_alpha_beta_hac(
        x=x,
        y=y,
        hac_lags=3,
        confidence_level=0.95,
    )

    np.testing.assert_allclose(result.alpha, expected_params[0], atol=1.0e-12)
    np.testing.assert_allclose(result.beta, expected_params[1], atol=1.0e-12)
    np.testing.assert_allclose(result.r_squared, expected_r2, atol=1.0e-12)
    np.testing.assert_allclose(result.alpha_pvalue, expected_pvalue, atol=1.0e-12)
    np.testing.assert_allclose(result.alpha_hac_se, expected_se, atol=1.0e-12)
    np.testing.assert_allclose(
        result.alpha_confidence_interval,
        expected_interval,
        atol=1.0e-12,
    )

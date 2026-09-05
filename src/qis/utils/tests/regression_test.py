"""Tests for generic OLS regression inference utilities."""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels import api as sm

from qis.utils.regression import (
    estimate_ewma_alpha_beta_hac,
    estimate_hac_mean,
    estimate_ols_alpha_beta_hac,
    newey_west_lag_rule,
)


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


def _manual_joint_ewma_hac(
        x: np.ndarray,
        y: np.ndarray,
        span: float,
        lags: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute joint EWMA-WLS coefficients and Bartlett HAC covariance from matrices."""
    design = np.column_stack((np.ones_like(x), x))
    decay = 1.0 - 2.0 / (span + 1.0)
    weights = np.power(decay, np.arange(x.shape[0] - 1, -1, -1, dtype=float))
    bread = np.linalg.inv(design.T @ (weights[:, None] * design))
    params = bread @ design.T @ (weights[:, None] * y)
    residuals = y - design @ params
    scores = np.einsum(
        'ti,tj->tji',
        weights[:, None] * design,
        residuals,
    ).reshape(x.shape[0], -1)
    meat = scores.T @ scores
    for lag in range(1, lags + 1):
        kernel_weight = 1.0 - lag / (lags + 1.0)
        lagged_cross_product = scores[lag:].T @ scores[:-lag]
        meat += kernel_weight * (lagged_cross_product + lagged_cross_product.T)
    joint_bread = np.kron(np.eye(y.shape[1]), bread)
    covariance = (
        x.shape[0]
        / (x.shape[0] - design.shape[1])
        * joint_bread
        @ meat
        @ joint_bread
    )
    weighted_mean = np.sum(weights[:, None] * y, axis=0) / weights.sum()
    r_squared = 1.0 - np.sum(weights[:, None] * np.square(residuals), axis=0) / np.sum(
        weights[:, None] * np.square(y - weighted_mean),
        axis=0,
    )
    return params, covariance, weights, r_squared


def test_estimate_ewma_alpha_beta_hac_matches_joint_matrix_reference() -> None:
    """EWMA-WLS estimates and joint HAC covariance match an independent matrix calculation."""
    index = pd.date_range('2020-01-31', periods=30, freq='ME')
    x = pd.Series(
        0.01 * np.sin(np.arange(index.size)) + 0.002 * np.arange(index.size),
        index=index,
        name='Benchmark',
    )
    shocks = np.column_stack((
        0.002 * np.cos(np.arange(index.size)),
        -0.0015 * np.sin(0.7 * np.arange(index.size)),
        0.001 * np.cos(0.4 * np.arange(index.size)),
    ))
    y = pd.DataFrame(
        np.array([0.001, -0.0005, 0.002])
        + x.to_numpy()[:, None] * np.array([0.8, 1.1, 0.95])
        + shocks,
        index=index,
        columns=['Risk', 'Signal', 'Full'],
    )
    expected_params, expected_covariance, expected_weights, expected_r2 = (
        _manual_joint_ewma_hac(
            x=x.to_numpy(),
            y=y.to_numpy(),
            span=18.0,
            lags=3,
        )
    )

    result = estimate_ewma_alpha_beta_hac(
        x=x,
        y=y,
        span=18.0,
        hac_lags=3,
        confidence_level=0.95,
    )

    np.testing.assert_allclose(result.alpha, expected_params[0], atol=1.0e-12)
    np.testing.assert_allclose(result.beta, expected_params[1], atol=1.0e-12)
    np.testing.assert_allclose(result.r_squared, expected_r2, atol=1.0e-12)
    np.testing.assert_allclose(result.parameter_covariance, expected_covariance, atol=1.0e-12)
    np.testing.assert_allclose(result.weights, expected_weights, atol=1.0e-15)
    assert result.nobs == index.size
    assert result.hac_lags == 3
    assert result.confidence_level == 0.95
    np.testing.assert_allclose(result.ewm_lambda, 1.0 - 2.0 / 19.0, atol=1.0e-15)
    np.testing.assert_allclose(
        result.effective_nobs,
        np.square(expected_weights.sum()) / np.square(expected_weights).sum(),
        atol=1.0e-12,
    )
    assert result.alpha.index.tolist() == y.columns.tolist()
    assert result.alpha_confidence_interval.columns.tolist() == ['Lower', 'Upper']
    expected_parameter_index = pd.MultiIndex.from_product(
        [y.columns, ['Intercept', 'Beta']],
        names=['equation', 'parameter'],
    )
    assert result.parameter_covariance.index.equals(expected_parameter_index)
    assert result.parameter_covariance.columns.equals(expected_parameter_index)

    expected_alpha_se = np.sqrt(np.diag(expected_covariance)[::2])
    expected_alpha_pvalue = 2.0 * norm.sf(np.abs(expected_params[0] / expected_alpha_se))
    critical_value = norm.ppf(0.975)
    expected_alpha_interval = np.column_stack((
        expected_params[0] - critical_value * expected_alpha_se,
        expected_params[0] + critical_value * expected_alpha_se,
    ))
    np.testing.assert_allclose(result.alpha_hac_se, expected_alpha_se, atol=1.0e-12)
    np.testing.assert_allclose(result.alpha_pvalue, expected_alpha_pvalue, atol=1.0e-12)
    np.testing.assert_allclose(
        result.alpha_confidence_interval,
        expected_alpha_interval,
        atol=1.0e-12,
    )

    design = sm.add_constant(x.to_numpy())
    for equation_number, equation in enumerate(y.columns):
        robust_model = sm.WLS(
            y[equation].to_numpy(),
            design,
            weights=expected_weights,
        ).fit().get_robustcov_results(
            cov_type='HAC',
            maxlags=3,
            use_correction=True,
            use_t=False,
        )
        block_start = 2 * equation_number
        block = expected_covariance[
            block_start:block_start + 2,
            block_start:block_start + 2,
        ]
        np.testing.assert_allclose(result.alpha[equation], robust_model.params[0], atol=1.0e-12)
        np.testing.assert_allclose(result.beta[equation], robust_model.params[1], atol=1.0e-12)
        np.testing.assert_allclose(block, robust_model.cov_params(), atol=1.0e-12)


def test_estimate_ewma_alpha_beta_hac_uses_one_common_finite_sample() -> None:
    """Every equation uses one finite sample and returned weights retain its row labels."""
    index = pd.Index(['z', 'a', 'q', 'b', 'x', 'c', 'm', 'd'], name='row')
    x = pd.Series([0.01, 0.02, np.nan, -0.01, 0.03, 0.00, 0.04, -0.02], index=index)
    y = pd.DataFrame(
        {
            'One': [0.01, 0.03, 0.02, -0.01, 0.04, 0.01, 0.05, -0.02],
            'Two': [0.02, 0.01, 0.03, -0.02, np.inf, 0.00, 0.06, -0.01],
        },
        index=index,
    )

    result = estimate_ewma_alpha_beta_hac(x=x, y=y, span=12.0, hac_lags=20)

    assert result.weights.index.tolist() == ['z', 'a', 'b', 'c', 'm', 'd']
    assert result.nobs == 6
    assert result.hac_lags == 5
    assert result.weights.iloc[-1] == 1.0
    np.testing.assert_allclose(
        result.weights.iloc[:-1] / result.weights.iloc[1:].to_numpy(),
        result.ewm_lambda,
        atol=1.0e-15,
    )


def test_estimate_ewma_alpha_beta_hac_rejects_invalid_inputs() -> None:
    """The joint estimator rejects ambiguous alignment and degenerate weighted regressions."""
    index = pd.RangeIndex(6)
    x = pd.Series(np.arange(6, dtype=float), index=index)
    y = pd.DataFrame({'Layer': np.square(np.arange(6, dtype=float))}, index=index)

    with np.testing.assert_raises(ValueError):
        estimate_ewma_alpha_beta_hac(x=x.set_axis(index[::-1]), y=y)
    with np.testing.assert_raises(ValueError):
        estimate_ewma_alpha_beta_hac(x=x, y=y, span=2.0)
    with np.testing.assert_raises(ValueError):
        estimate_ewma_alpha_beta_hac(x=np.ones(6), y=y)
    with np.testing.assert_raises(TypeError):
        estimate_ewma_alpha_beta_hac(x=x, y=y, hac_lags=1.5)
    with np.testing.assert_raises(ValueError):
        estimate_ewma_alpha_beta_hac(x=x, y=y, confidence_level=1.0)


def test_newey_west_lag_rule() -> None:
    """Newey-West's rule returns the documented floor and rejects empty samples."""
    assert newey_west_lag_rule(nobs=260) == 4
    assert newey_west_lag_rule(nobs=100) == 4
    assert newey_west_lag_rule(nobs=86) == 3
    with np.testing.assert_raises(ValueError):
        newey_west_lag_rule(nobs=0)


def _manual_hac_mean(y: np.ndarray, lags: int) -> tuple[float, float, float, tuple[float, float]]:
    """Compute a constant-only Bartlett HAC mean independently of the production helper."""
    n_obs = y.shape[0]
    mean = float(y.mean())
    scores = y - mean
    meat = float(scores @ scores)
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        meat += 2.0 * weight * float(scores[lag:] @ scores[:-lag])
    variance = (n_obs / (n_obs - 1)) * meat / n_obs ** 2
    se = float(np.sqrt(variance))
    pvalue = float(2.0 * norm.sf(abs(mean / se)))
    critical_value = norm.ppf(0.975)
    return mean, se, pvalue, (mean - critical_value * se, mean + critical_value * se)


def test_estimate_hac_mean_matches_matrix_reference_without_warnings() -> None:
    """The constant-only HAC mean matches the hand-rolled estimator and emits no warning."""
    y = np.array([
        0.0120, -0.0080, 0.0150, -0.0020, 0.0050, 0.0100, -0.0150, 0.0250,
        -0.0050, 0.0100, -0.0100, 0.0150, -0.0250, 0.0050, 0.0200, -0.0100,
        0.0100, -0.0050, 0.0150, -0.0200, 0.0050, 0.0100, -0.0150, 0.0200,
    ])
    expected_mean, expected_se, expected_pvalue, expected_interval = _manual_hac_mean(y=y, lags=3)

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result = estimate_hac_mean(y=y, hac_lags=3, confidence_level=0.95)

    assert result.nobs == y.shape[0]
    np.testing.assert_allclose(result.mean, expected_mean, atol=1.0e-12)
    np.testing.assert_allclose(result.hac_se, expected_se, atol=1.0e-12)
    np.testing.assert_allclose(result.pvalue, expected_pvalue, atol=1.0e-12)
    np.testing.assert_allclose(result.confidence_interval, expected_interval, atol=1.0e-12)


def test_estimate_hac_mean_drops_non_finite_and_rejects_short_samples() -> None:
    """Non-finite observations are removed before estimation and short samples raise."""
    y = np.array([0.01, np.nan, -0.02, np.inf, 0.03, 0.005])
    result = estimate_hac_mean(y=y, hac_lags=1)
    assert result.nobs == 4
    expected_mean = np.array([0.01, -0.02, 0.03, 0.005]).mean()
    np.testing.assert_allclose(result.mean, expected_mean, atol=1.0e-15)
    with np.testing.assert_raises(ValueError):
        estimate_hac_mean(y=np.array([0.01]), hac_lags=1)
    with np.testing.assert_raises(ValueError):
        estimate_hac_mean(y=y, hac_lags=-1)

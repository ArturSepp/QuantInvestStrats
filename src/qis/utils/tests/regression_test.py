"""Tests for generic OLS regression inference utilities."""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from statsmodels import api as sm

from qis.utils.regression import (
    estimate_ewma_alpha_beta_hac,
    estimate_hac_mean,
    estimate_hac_regression,
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


def test_estimate_ols_alpha_beta_hac_exactly_zero_response() -> None:
    """A deterministic zero effect has a zero interval and no evidence against zero alpha."""
    x = np.array([-0.03, 0.01, 0.02, -0.01, 0.04, -0.02])
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        result = estimate_ols_alpha_beta_hac(x=x, y=np.zeros_like(x))
    assert result.alpha == result.beta == result.alpha_hac_se == 0.0
    assert result.alpha_confidence_interval == (0.0, 0.0)
    assert result.alpha_pvalue == 1.0
    assert np.isnan(result.r_squared)


def _joint_hac_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Small deterministic, correlated equations with three distinct factor columns."""
    index = pd.date_range('2000-01-31', periods=47, freq='ME')
    t = np.arange(len(index), dtype=float)
    x = pd.DataFrame({
        'Market': np.sin(t * 0.73) * 0.02,
        'Value': np.cos(t * 0.37) * 0.015,
        'Momentum': (t % 7 - 3) * 0.004,
    }, index=index)
    noise = 0.003 * np.sin(t * 0.31) + 0.001 * np.cos(t * 1.17)
    y = pd.DataFrame({
        'One': 0.002 + x.to_numpy() @ np.array([0.8, 0.4, -0.3]) + noise,
        'Two': -0.001 + x.to_numpy() @ np.array([1.1, -0.2, 0.5])
        + 0.6 * noise + 0.002 * np.cos(t * 0.91),
    }, index=index)
    return x, y


def _qr_dense_hac_reference(y: np.ndarray, x: np.ndarray, weights: np.ndarray,
                            lags: int) -> tuple[np.ndarray, np.ndarray]:
    """QR fit and dense time-kernel covariance without production's score/lag loop."""
    nobs = len(y)
    design = np.column_stack((np.ones(nobs), x))
    root_weights = np.sqrt(weights)
    q, r = np.linalg.qr(root_weights[:, None] * design, mode='reduced')
    parameters = np.linalg.solve(r, q.T @ (root_weights[:, None] * y))
    residuals = y - design @ parameters
    coefficient_map = np.linalg.solve(r, q.T) * root_weights[None, :]
    distances = np.abs(np.subtract.outer(np.arange(nobs), np.arange(nobs)))
    kernel = np.maximum(1.0 - distances / (lags + 1.0), 0.0)
    influence = np.vstack([
        coefficient_map * residuals[:, equation][None, :]
        for equation in range(y.shape[1])
    ])
    covariance = nobs / (nobs - design.shape[1]) * influence @ kernel @ influence.T
    return parameters, covariance


@pytest.mark.parametrize('factor_count', [0, 1, 3])
@pytest.mark.parametrize('span', [None, 12.0])
def test_shared_hac_matches_qr_dense_kernel(factor_count: int, span: float | None) -> None:
    """All designs/weight modes agree with QR and a dense Bartlett time kernel."""
    x, y = _joint_hac_fixture()
    factors = x.iloc[:, :factor_count]
    result = estimate_hac_regression(y, x=factors, span=span, hac_lags=4,
                                     confidence_level=0.9)
    nobs = len(y)
    weights = np.ones(nobs) if span is None else (
        (span - 1.0) / (span + 1.0)
    ) ** np.arange(nobs - 1, -1, -1)
    expected, covariance = _qr_dense_hac_reference(
        y.to_numpy(), factors.to_numpy(), weights, lags=4,
    )
    np.testing.assert_allclose(result.coefficients, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(result.parameter_covariance, covariance, rtol=1e-10, atol=1e-12)
    se = np.sqrt(np.diag(covariance)).reshape(y.shape[1], factor_count + 1).T
    np.testing.assert_allclose(result.hac_se, se, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(result.pvalues, 2 * norm.sf(np.abs(expected / se)), atol=1e-12)
    np.testing.assert_allclose(result.confidence_lower, expected - norm.ppf(0.95) * se,
                               rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(result.confidence_upper, expected + norm.ppf(0.95) * se,
                               rtol=1e-10, atol=1e-12)
    expected_parameters = ['Intercept'] + factors.columns.tolist()
    assert result.coefficients.index.tolist() == expected_parameters
    assert result.coefficients.index.name == 'parameter'
    assert result.coefficients.columns.name == 'equation'
    assert result.coefficients.columns.equals(y.columns)
    assert result.parameter_covariance.index.tolist() == [
        (equation, parameter) for equation in y.columns for parameter in expected_parameters
    ]
    assert result.parameter_covariance.columns.equals(result.parameter_covariance.index)
    assert result.weights.index.equals(y.index)
    if span is None:
        np.testing.assert_array_equal(result.weights, np.ones(nobs))
        assert result.effective_nobs == nobs
        assert result.span is result.ewm_lambda is None
        assert result.weighting == 'equal'
    else:
        np.testing.assert_allclose(result.weights, weights, rtol=1e-14, atol=0)
        assert result.span == span
        assert result.weighting == 'ewma'
        assert result.effective_nobs == pytest.approx(weights.sum() ** 2 / (weights @ weights))
    if factor_count == 0:
        np.testing.assert_allclose(result.coefficients.loc['Intercept'],
                                   np.average(y, weights=weights, axis=0), atol=1e-14)


@pytest.mark.parametrize('lags', [0, 6])
def test_shared_hac_six_factor_intercepts_and_full_covariance(lags: int) -> None:
    """Significance-table inputs retain every slope and joint covariance in the core."""
    x, y = _joint_hac_fixture()
    t = np.arange(len(x))
    x = x.assign(Size=0.01 * np.sin(0.43 * t), Quality=0.02 * np.cos(0.89 * t),
                 Investment=0.015 * np.sin(1.31 * t))
    result = estimate_hac_regression(y=y, x=x, span=None, hac_lags=lags)
    coefficients, covariance = _qr_dense_hac_reference(
        y.to_numpy(), x.to_numpy(), np.ones(len(x)), lags,
    )
    np.testing.assert_allclose(result.coefficients, coefficients, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(result.parameter_covariance, covariance, rtol=1e-9, atol=1e-12)
    intercept_se = np.sqrt(np.diag(covariance)[::7])
    np.testing.assert_allclose(result.hac_se.loc['Intercept'], intercept_se, rtol=1e-9)
    np.testing.assert_allclose(result.pvalues.loc['Intercept'],
                               2.0 * norm.sf(np.abs(coefficients[0] / intercept_se)),
                               rtol=1e-9, atol=1e-12)
    assert result.nobs == len(y) and result.coefficients.shape == (7, 2)
    assert result.parameter_covariance.shape == (14, 14)
    for name in y:
        legacy = estimate_ols_alpha_beta_hac(x=x, y=y[name], hac_lags=lags)
        np.testing.assert_allclose(
            [result.coefficients.loc['Intercept', name], result.hac_se.loc['Intercept', name]],
            [legacy.alpha, legacy.alpha_hac_se], rtol=1e-9, atol=1e-12,
        )


def test_shared_hac_equal_weights_matches_legacy_ols_and_mean() -> None:
    """The generic equal-weight path retains each existing full-sample HAC convention."""
    x, y = _joint_hac_fixture()
    mean_fit = estimate_hac_regression(y)
    regression_fit = estimate_hac_regression(y, x=x)
    for equation in y:
        mean = estimate_hac_mean(y[equation])
        legacy = estimate_ols_alpha_beta_hac(x=x, y=y[equation])
        np.testing.assert_allclose(
            [mean_fit.coefficients.loc['Intercept', equation],
             mean_fit.hac_se.loc['Intercept', equation],
             mean_fit.pvalues.loc['Intercept', equation]],
            [mean.mean, mean.hac_se, mean.pvalue], atol=1e-12,
        )
        np.testing.assert_allclose(
            [regression_fit.coefficients.loc['Intercept', equation],
             regression_fit.coefficients.loc['Market', equation],
             regression_fit.hac_se.loc['Intercept', equation],
             regression_fit.pvalues.loc['Intercept', equation]],
            [legacy.alpha, legacy.beta, legacy.alpha_hac_se, legacy.alpha_pvalue], atol=1e-12,
        )


def test_shared_hac_additive_contrast_uses_cross_equation_covariance() -> None:
    """A fit of the response contrast equals the contrast of joint coefficients/covariance."""
    x, y = _joint_hac_fixture()
    result = estimate_hac_regression(y, x=x, span=18)
    contrast = estimate_hac_regression(y['One'] - 2.0 * y['Two'], x=x, span=18)
    transform = np.hstack((np.eye(4), -2.0 * np.eye(4)))
    np.testing.assert_allclose(contrast.coefficients.iloc[:, 0],
                               result.coefficients['One'] - 2 * result.coefficients['Two'],
                               atol=1e-12)
    np.testing.assert_allclose(contrast.parameter_covariance,
                               transform @ result.parameter_covariance @ transform.T,
                               atol=1e-12)


def test_shared_hac_common_finite_sample_preserves_order_and_nullable_values() -> None:
    """Filtering occurs jointly before assigning retained-row EWMA ages and lags."""
    x, y = _joint_hac_fixture()
    x = x.iloc[:10].astype('Float64')
    y = y.iloc[:10].astype('Float64')
    x.iloc[2, 2] = pd.NA
    y.iloc[5, 1] = np.inf
    y = y.iloc[::-1]
    x = x.iloc[::-1]
    finite = np.isfinite(x.to_numpy(dtype=float, na_value=np.nan)).all(axis=1)
    finite &= np.isfinite(y.to_numpy(dtype=float, na_value=np.nan)).all(axis=1)
    result = estimate_hac_regression(y, x=x, span=18, hac_lags=99)
    expected = estimate_hac_regression(y.loc[finite], x=x.loc[finite], span=18, hac_lags=99)
    pd.testing.assert_frame_equal(result.parameter_covariance, expected.parameter_covariance)
    pd.testing.assert_frame_equal(result.coefficients, expected.coefficients)
    assert result.weights.index.equals(y.index[finite])
    assert result.hac_lags == result.nobs - 1 == 7
    assert result.weights.iloc[-1] == 1


@pytest.mark.parametrize('span', [True, 1, 0, np.nan, np.inf, '12'])
def test_shared_hac_invalid_spans(span: object) -> None:
    """Finite spans must be numerical and strictly greater than one."""
    _, y = _joint_hac_fixture()
    with pytest.raises(ValueError, match='span'):
        estimate_hac_regression(y, span=span)


def test_shared_hac_validation_and_deterministic_zero() -> None:
    """Generic labels, identification and degenerate-response semantics are explicit."""
    x, y = _joint_hac_fixture()
    for factors in [x.set_axis(x.index[::-1]), x.rename(columns={'Market': 'Intercept'}),
                    pd.concat([x, x], axis=1)]:
        with pytest.raises(ValueError):
            estimate_hac_regression(y, x=factors)
    with pytest.raises(ValueError, match='unique'):
        estimate_hac_regression(pd.concat([y, y], axis=1))
    with pytest.raises(TypeError, match='pandas'):
        estimate_hac_regression(y.to_numpy())
    with pytest.raises(TypeError, match='pandas'):
        estimate_hac_regression(y, x=x.to_numpy())
    for lags in [True, 1.5]:
        with pytest.raises(TypeError, match='integer'):
            estimate_hac_regression(y, hac_lags=lags)
    for settings in [{'hac_lags': -1}, {'confidence_level': 0}, {'confidence_level': np.nan}]:
        with pytest.raises(ValueError):
            estimate_hac_regression(y, **settings)
    with pytest.raises(ValueError, match='finite rows'):
        estimate_hac_regression(y.iloc[:4], x=x.iloc[:4])
    with pytest.raises(ValueError, match='effective sample size'):
        estimate_hac_regression(y, x=x, span=3)
    with pytest.raises(ValueError, match='singular'):
        estimate_hac_regression(y, x=pd.Series(1.0, index=y.index))
    with pytest.raises(ValueError, match='numeric'):
        estimate_hac_regression(pd.Series(['a', 'b']))
    zero = estimate_hac_regression(pd.Series(np.zeros(8)))
    assert zero.coefficients.index.tolist() == ['Intercept']
    assert zero.coefficients.columns.tolist() == ['y']
    assert zero.coefficients.iloc[0, 0] == zero.hac_se.iloc[0, 0] == 0
    assert zero.pvalues.iloc[0, 0] == 1
    assert np.isnan(zero.r_squared.iloc[0])


def test_legacy_hac_wrappers_keep_input_and_result_contracts() -> None:
    """Legacy array, default, undefined-sample and label conventions remain available."""
    x, y = _joint_hac_fixture()
    array = np.array([[0.1, np.nan], [-0.2, 0.4]])
    expected = estimate_hac_mean(np.array([0.1, -0.2, 0.4]), hac_lags=1)
    assert estimate_hac_mean(array, hac_lags=True) == expected
    assert estimate_hac_mean(array, hac_lags=99.5) == estimate_hac_mean(array, hac_lags=2)
    with pytest.raises(ValueError):
        estimate_hac_mean(array, hac_lags=1.5)
    with pytest.raises(ValueError, match='non-finite outputs'):
        estimate_hac_mean(np.zeros(8))
    with pytest.raises(ValueError):
        estimate_ewma_alpha_beta_hac(x.iloc[:, 0], y, span=None)
    with pytest.raises(TypeError):
        estimate_ewma_alpha_beta_hac(x.iloc[:, 0], y['One'])
    with pytest.raises(ValueError):
        estimate_ewma_alpha_beta_hac(x.iloc[:, 0], y * 0)
    actual = estimate_ewma_alpha_beta_hac(x.iloc[:, [0]].to_numpy(), y)
    generic = estimate_hac_regression(y, x=x.iloc[:, 0].rename('Beta'), span=36)
    pd.testing.assert_frame_equal(actual.parameter_covariance, generic.parameter_covariance)
    pd.testing.assert_series_equal(
        actual.alpha, generic.coefficients.loc['Intercept'].rename('Alpha'),
    )
    assert actual.alpha.index.name == 'equation'
    assert actual.alpha_hac_se.name == 'Alpha HAC SE'
    assert actual.alpha_pvalue.name == 'Alpha p-value'
    assert actual.beta.name == 'Beta'
    assert actual.weights.name == 'EWMA weight'
    assert actual.alpha_confidence_interval.columns.tolist() == ['Lower', 'Upper']


def test_legacy_hac_wrappers_delegate_to_common_estimator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wrapper inference is taken from the common estimator rather than recomputed."""
    import qis.utils.regression as regression

    calls = []
    original = regression.estimate_hac_regression

    def record_call(*args: object, **kwargs: object) -> object:
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(regression, 'estimate_hac_regression', record_call)
    x, y = _joint_hac_fixture()
    estimate_hac_mean(y['One'])
    estimate_ewma_alpha_beta_hac(x.iloc[:, 0], y)
    assert len(calls) == 2
    assert calls[0].get('x') is calls[0].get('span') is None
    assert calls[1]['span'] == 36.0

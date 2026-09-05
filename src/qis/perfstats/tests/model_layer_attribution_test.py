"""Tests for full-sample and rolling model-layer log-return attribution."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import qis.perfstats.model_layer_attribution as model_layer
from qis.models.linear.ewm import MeanAdjType
from qis.perfstats.model_layer_attribution import (
    ModelLayerAlphaBetaAttribution,
    ModelLayerCumulativeAlphaAttribution,
    ModelLayerEwmaAlphaAttribution,
    ModelLayerEwmaRegressionAttribution,
    compute_model_layer_alpha_beta_attribution,
    compute_model_layer_cumulative_alpha_after_warmup,
    compute_model_layer_ewma_alpha_attribution,
    compute_model_layer_ewma_regression_attribution,
    compute_model_layer_ewma_stage_sharpes,
)


def _nav_from_log_returns(log_returns: np.ndarray, index: pd.DatetimeIndex) -> pd.Series:
    """Convert a deterministic log-return path to a NAV series."""
    values = np.exp(np.concatenate(([0.0], np.cumsum(log_returns))))
    return pd.Series(values, index=index)


def _manual_hac_alpha_interval(
        benchmark_returns: np.ndarray,
        layer_returns: np.ndarray,
        lags: int = 3,
) -> tuple[float, float, float, float]:
    """Compute a Bartlett-kernel HAC alpha interval independently of the production helper."""
    x = np.column_stack((np.ones_like(benchmark_returns), benchmark_returns))
    xtx_inv = np.linalg.inv(x.T @ x)
    params = xtx_inv @ x.T @ layer_returns
    residuals = layer_returns - x @ params
    scores = x * residuals[:, None]
    meat = scores.T @ scores
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        lagged_cross_product = scores[lag:].T @ scores[:-lag]
        meat += weight * (lagged_cross_product + lagged_cross_product.T)
    n_obs, n_params = x.shape
    covariance = (n_obs / (n_obs - n_params)) * xtx_inv @ meat @ xtx_inv
    alpha_se = np.sqrt(covariance[0, 0])
    critical_value = norm.ppf(0.975)
    return (
        params[0],
        alpha_se,
        params[0] - critical_value * alpha_se,
        params[0] + critical_value * alpha_se,
    )


def test_model_layer_attribution_recovers_log_alpha_beta_and_reconstructs() -> None:
    """Known layer alphas and betas are recovered on one common quarterly sample."""
    benchmark_returns = np.array([
        -0.12, 0.08, 0.03, -0.04, 0.11, -0.02, 0.05, 0.01,
    ])
    risk_returns = 0.002 + 0.80 * benchmark_returns
    signal_returns = 0.003 + 0.20 * benchmark_returns
    integration_returns = 0.001 - 0.10 * benchmark_returns
    full_returns = risk_returns + signal_returns + integration_returns
    index = pd.date_range('2019-12-31', periods=9, freq='QE')

    result = compute_model_layer_alpha_beta_attribution(
        benchmark_nav=_nav_from_log_returns(benchmark_returns, index),
        risk_layer_nav=_nav_from_log_returns(risk_returns, index),
        signal_layer_nav=_nav_from_log_returns(signal_returns, index),
        full_model_nav=_nav_from_log_returns(full_returns, index),
    )

    assert isinstance(result, ModelLayerAlphaBetaAttribution)
    assert (
        'Alpha Layer' not in result.regression_table.index
        and 'Alpha Layer Alpha' not in result.component_returns.columns
    )
    np.testing.assert_allclose(
        result.periodic_returns['Full Model'].to_numpy(),
        full_returns,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        result.regression_table['Beta'].to_numpy(),
        np.array([1.0, 0.80, 0.20, -0.10, 0.90]),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        result.regression_table['An Alpha'].to_numpy(),
        4.0 * np.array([0.0, 0.002, 0.003, 0.001, 0.006]),
        atol=1.0e-12,
    )

    bridge_columns = [
        'Systematic Return',
        'Risk Layer Alpha',
        'Signal Layer Alpha',
        'Integration Alpha',
    ]
    pd.testing.assert_series_equal(
        result.component_returns[bridge_columns].sum(axis=1),
        result.component_returns['Full Model Return'],
        check_names=False,
        atol=1.0e-14,
        rtol=0.0,
    )


def test_model_layer_component_means_equal_ols_intercepts() -> None:
    """Residual component means match OLS intercepts without a production assertion."""
    benchmark_returns = np.array([-0.05, 0.02, 0.06, -0.01, 0.04, 0.03])
    risk_returns = 0.004 + 0.70 * benchmark_returns
    signal_returns = 0.002 + 0.30 * benchmark_returns
    integration_returns = -0.001 + 0.05 * benchmark_returns
    full_returns = risk_returns + signal_returns + integration_returns
    index = pd.date_range('2020-12-31', periods=7, freq='QE')

    result = compute_model_layer_alpha_beta_attribution(
        benchmark_nav=_nav_from_log_returns(benchmark_returns, index),
        risk_layer_nav=_nav_from_log_returns(risk_returns, index),
        signal_layer_nav=_nav_from_log_returns(signal_returns, index),
        full_model_nav=_nav_from_log_returns(full_returns, index),
    )

    mapping = {
        'Risk Layer Alpha': 'Risk Layer',
        'Signal Layer Alpha': 'Signal Layer',
        'Integration Alpha': 'Integration',
    }
    for component, regression_layer in mapping.items():
        np.testing.assert_allclose(
            result.annualised_components[component],
            result.regression_table.loc[regression_layer, 'An Alpha'],
            atol=1.0e-12,
        )


def test_optional_net_nav_adds_exact_cost_drag_and_net_reconstruction() -> None:
    """An optional net NAV extends the gross bridge by its realised trading-cost drag."""
    benchmark_returns = np.array([-0.05, 0.03, 0.06, -0.02, 0.04, 0.01])
    risk_returns = 0.003 + 0.75 * benchmark_returns
    signal_returns = 0.004 + 0.20 * benchmark_returns
    integration_returns = 0.001 - 0.05 * benchmark_returns
    full_returns = risk_returns + signal_returns + integration_returns
    cost_drag = np.array([0.0002, 0.0004, 0.0001, 0.0003, 0.0005, 0.0002])
    net_returns = full_returns - cost_drag
    index = pd.date_range('2021-12-31', periods=7, freq='QE')

    result = compute_model_layer_alpha_beta_attribution(
        benchmark_nav=_nav_from_log_returns(benchmark_returns, index),
        risk_layer_nav=_nav_from_log_returns(risk_returns, index),
        signal_layer_nav=_nav_from_log_returns(signal_returns, index),
        full_model_nav=_nav_from_log_returns(full_returns, index),
        full_model_net_nav=_nav_from_log_returns(net_returns, index),
    )

    np.testing.assert_allclose(
        result.component_returns['Trading Cost Drag'].to_numpy(),
        -cost_drag,
        atol=1.0e-14,
    )
    bridge_columns = [
        'Systematic Return',
        'Risk Layer Alpha',
        'Signal Layer Alpha',
        'Integration Alpha',
        'Trading Cost Drag',
    ]
    pd.testing.assert_series_equal(
        result.component_returns[bridge_columns].sum(axis=1),
        result.component_returns['Full Model Net Return'],
        check_names=False,
        atol=1.0e-14,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        result.annualised_components['Trading Cost Drag'],
        -4.0 * cost_drag.mean(),
        atol=1.0e-14,
    )
    assert 'Full Model Net' in result.regression_table.index


def test_monthly_alpha_intervals_use_hac3_and_annualise_the_bounds() -> None:
    """Monthly alpha bounds match an independent Bartlett HAC(3) calculation."""
    benchmark_returns = np.array([
        -0.030, 0.012, 0.021, -0.008, 0.025, -0.017,
        0.009, 0.018, -0.011, 0.026, 0.004, -0.019,
        0.014, 0.007, -0.006, 0.023, -0.013, 0.016,
        0.005, -0.010, 0.020, -0.004, 0.011, 0.015,
    ])
    risk_noise = np.array([
        0.0020, -0.0010, 0.0015, -0.0020, 0.0005, 0.0010,
        -0.0015, 0.0025, -0.0005, 0.0010, -0.0010, 0.0015,
        -0.0025, 0.0005, 0.0020, -0.0010, 0.0010, -0.0005,
        0.0015, -0.0020, 0.0005, 0.0010, -0.0015, 0.0020,
    ])
    risk_returns = 0.002 + 0.82 * benchmark_returns + risk_noise
    signal_returns = 0.003 + 0.18 * benchmark_returns - 0.5 * risk_noise
    integration_returns = 0.001 - 0.07 * benchmark_returns + 0.25 * risk_noise
    full_returns = risk_returns + signal_returns + integration_returns
    index = pd.date_range('2020-12-31', periods=25, freq='ME')

    result = compute_model_layer_alpha_beta_attribution(
        benchmark_nav=_nav_from_log_returns(benchmark_returns, index),
        risk_layer_nav=_nav_from_log_returns(risk_returns, index),
        signal_layer_nav=_nav_from_log_returns(signal_returns, index),
        full_model_nav=_nav_from_log_returns(full_returns, index),
        freq='ME',
    )
    alpha, alpha_se, ci_low, ci_high = _manual_hac_alpha_interval(
        benchmark_returns=benchmark_returns,
        layer_returns=risk_returns,
    )

    risk_row = result.regression_table.loc['Risk Layer']
    np.testing.assert_allclose(risk_row['An Alpha'], 12.0 * alpha, atol=1.0e-12)
    np.testing.assert_allclose(risk_row['Alpha HAC SE'], alpha_se, atol=1.0e-12)
    np.testing.assert_allclose(risk_row['An Alpha CI Low'], 12.0 * ci_low, atol=1.0e-12)
    np.testing.assert_allclose(risk_row['An Alpha CI High'], 12.0 * ci_high, atol=1.0e-12)
    assert (result.regression_table['An Alpha CI Low']
            <= result.regression_table['An Alpha']).all()
    assert (result.regression_table['An Alpha']
            <= result.regression_table['An Alpha CI High']).all()


def test_common_sample_trims_to_latest_start_and_earliest_end() -> None:
    """Ragged daily NAVs are trimmed before resampling to the common valid range."""
    index = pd.bdate_range('2005-01-03', '2007-12-31')
    generator = np.random.default_rng(seed=1)
    log_returns = 0.0003 + 0.01 * generator.standard_normal((len(index) - 1, 4))
    navs = pd.DataFrame(
        np.vstack((np.ones(4), np.exp(np.cumsum(log_returns, axis=0)))),
        index=index,
        columns=['Benchmark', 'Risk Layer', 'Signal Layer', 'Full Model'],
    )
    risk_nav = navs['Risk Layer'].where(navs.index >= pd.Timestamp('2005-03-15'))
    signal_nav = navs['Signal Layer'].where(navs.index <= pd.Timestamp('2007-06-15'))

    result = compute_model_layer_alpha_beta_attribution(
        benchmark_nav=navs['Benchmark'],
        risk_layer_nav=risk_nav,
        signal_layer_nav=signal_nav,
        full_model_nav=navs['Full Model'],
        freq='ME',
    )

    assert result.periodic_returns.index[0] == pd.Timestamp('2005-04-30')
    assert result.periodic_returns.index[-1] == pd.Timestamp('2007-05-31')
    assert len(result.periodic_returns) == 26
    assert (result.periodic_returns == 0.0).sum().sum() == 0

    with np.testing.assert_raises(ValueError):
        compute_model_layer_alpha_beta_attribution(
            benchmark_nav=navs['Benchmark'],
            risk_layer_nav=risk_nav,
            signal_layer_nav=pd.Series(np.nan, index=index),
            full_model_nav=navs['Full Model'],
            freq='ME',
        )


def test_hac_lags_and_confidence_level_are_recorded_and_applied() -> None:
    """Custom HAC settings affect inference only and are recorded on the result."""
    benchmark_returns = np.array([
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
    risk_returns = 0.002 + 0.82 * benchmark_returns + noise
    signal_returns = 0.003 + 0.18 * benchmark_returns - 0.5 * noise
    integration_returns = 0.001 - 0.07 * benchmark_returns + 0.25 * noise
    full_returns = risk_returns + signal_returns + integration_returns
    index = pd.date_range('2020-12-31', periods=25, freq='ME')
    kwargs = {
        'benchmark_nav': _nav_from_log_returns(benchmark_returns, index),
        'risk_layer_nav': _nav_from_log_returns(risk_returns, index),
        'signal_layer_nav': _nav_from_log_returns(signal_returns, index),
        'full_model_nav': _nav_from_log_returns(full_returns, index),
        'freq': 'ME',
    }

    default_result = compute_model_layer_alpha_beta_attribution(**kwargs)
    custom_result = compute_model_layer_alpha_beta_attribution(
        **kwargs,
        hac_lags=5,
        confidence_level=0.90,
    )

    columns = ['Alpha', 'An Alpha', 'Beta']
    pd.testing.assert_frame_equal(
        custom_result.regression_table[columns],
        default_result.regression_table[columns],
    )
    assert not np.allclose(
        custom_result.regression_table['Alpha HAC SE'],
        default_result.regression_table['Alpha HAC SE'],
    )
    risk_row = custom_result.regression_table.loc['Risk Layer']
    expected_half_width = 12.0 * norm.ppf(0.95) * risk_row['Alpha HAC SE']
    np.testing.assert_allclose(
        0.5 * (risk_row['An Alpha CI High'] - risk_row['An Alpha CI Low']),
        expected_half_width,
        atol=1.0e-12,
    )
    assert custom_result.freq == 'ME'
    assert custom_result.hac_lags == 5
    assert custom_result.confidence_level == 0.90


def _rolling_layer_returns() -> pd.DataFrame:
    """Return a non-degenerate synthetic monthly model-layer panel."""
    index = pd.date_range('2024-01-31', periods=16, freq='ME')
    benchmark = pd.Series(
        [
            0.020, -0.010, 0.030, -0.020, 0.010, 0.025, -0.015, 0.018,
            0.006, -0.012, 0.021, 0.014, -0.009, 0.017, 0.004, -0.006,
        ],
        index=index,
    )
    return pd.DataFrame({
        'Benchmark': benchmark,
        'Risk Layer': 0.90 * benchmark + pd.Series(
            [
                0.002, 0.001, -0.001, 0.002, 0.001, 0.003, -0.001, 0.002,
                0.000, 0.002, 0.001, -0.001, 0.003, 0.001, 0.002, -0.001,
            ],
            index=index,
        ),
        'Signal Layer': 1.10 * benchmark + pd.Series(
            [
                0.004, -0.002, 0.003, 0.001, -0.001, 0.004, 0.002, 0.003,
                -0.002, 0.001, 0.003, 0.002, -0.001, 0.004, 0.001, 0.002,
            ],
            index=index,
        ),
        'Full Model': 0.95 * benchmark + pd.Series(
            [
                0.006, -0.001, 0.005, 0.002, -0.002, 0.007, 0.001, 0.005,
                -0.001, 0.003, 0.004, 0.001, -0.002, 0.006, 0.002, 0.003,
            ],
            index=index,
        ),
    })


def _rolling_navs(returns: pd.DataFrame) -> dict[str, pd.Series]:
    """Convert the synthetic rolling return panel to named NAV inputs."""
    nav_index = pd.date_range(
        returns.index[0] - pd.offsets.MonthEnd(1),
        periods=len(returns.index) + 1,
        freq='ME',
    )
    return {
        'benchmark_nav': _nav_from_log_returns(returns['Benchmark'].to_numpy(), nav_index),
        'risk_layer_nav': _nav_from_log_returns(returns['Risk Layer'].to_numpy(), nav_index),
        'signal_layer_nav': _nav_from_log_returns(
            returns['Signal Layer'].to_numpy(), nav_index
        ),
        'full_model_nav': _nav_from_log_returns(returns['Full Model'].to_numpy(), nav_index),
    }


def _manual_ewma_beta(
        benchmark: np.ndarray,
        layer: np.ndarray,
        span: int,
        beta_init_value: float,
) -> np.ndarray:
    """Compute the EWMA-mean-adjusted seeded beta by scalar recursion."""
    decay = 1.0 - 2.0 / (span + 1.0)
    benchmark_mean = np.empty_like(benchmark, dtype=float)
    layer_mean = np.empty_like(layer, dtype=float)
    benchmark_mean[0] = benchmark[0]
    layer_mean[0] = layer[0]
    for index in range(1, len(benchmark)):
        benchmark_mean[index] = (
            decay * benchmark_mean[index - 1] + (1.0 - decay) * benchmark[index]
        )
        layer_mean[index] = decay * layer_mean[index - 1] + (1.0 - decay) * layer[index]
    benchmark_centered = benchmark - benchmark_mean
    layer_centered = layer - layer_mean
    first = int(np.flatnonzero(~np.isclose(benchmark_centered, 0.0))[0])
    beta = np.full_like(benchmark, np.nan, dtype=float)
    variance = benchmark_centered[first] ** 2
    covariance = beta_init_value * variance
    beta[first] = beta_init_value
    for index in range(first + 1, len(benchmark)):
        variance = (
            decay * variance + (1.0 - decay) * benchmark_centered[index] ** 2
        )
        covariance = decay * covariance + (
            (1.0 - decay) * benchmark_centered[index] * layer_centered[index]
        )
        beta[index] = covariance / variance
    return beta


def test_rolling_ewma_betas_match_independent_recursion_and_are_lagged() -> None:
    """EWMA-mean-adjusted betas use X0 seeding, a prior, and point-in-time application."""
    returns = _rolling_layer_returns()
    span = 4
    prior = 1.0
    result = compute_model_layer_ewma_alpha_attribution(
        **_rolling_navs(returns),
        freq='ME',
        beta_span=span,
        beta_lag=1,
        beta_init_value=prior,
    )

    expected_estimated = pd.DataFrame(
        {
            layer: _manual_ewma_beta(
                benchmark=returns['Benchmark'].to_numpy(),
                layer=returns[layer].to_numpy(),
                span=span,
                beta_init_value=prior,
            )
            for layer in ('Risk Layer', 'Signal Layer', 'Full Model')
        },
        index=returns.index,
    )
    expected_applied = expected_estimated.shift(1).fillna(prior)

    assert isinstance(result, ModelLayerEwmaAlphaAttribution)
    assert result.mean_adj_type is MeanAdjType.EWMA
    assert result.beta_span == span
    assert result.beta_lag == 1
    assert result.beta_init_value == prior
    pd.testing.assert_frame_equal(result.estimated_betas, expected_estimated, check_freq=False)
    pd.testing.assert_frame_equal(result.applied_betas, expected_applied, check_freq=False)
    assert result.estimated_betas.iloc[0].isna().all()
    assert np.isfinite(result.applied_betas.to_numpy()).all()


def test_rolling_alpha_is_step_ahead_and_reconstructs_every_return() -> None:
    """A return updates only the next applied beta and all alpha identities hold exactly."""
    returns = _rolling_layer_returns()
    shocked = returns.copy()
    shocked.iloc[-1, shocked.columns.get_loc('Full Model')] += 0.20
    baseline = compute_model_layer_ewma_alpha_attribution(
        **_rolling_navs(returns), freq='ME', beta_span=4
    )
    changed = compute_model_layer_ewma_alpha_attribution(
        **_rolling_navs(shocked), freq='ME', beta_span=4
    )

    pd.testing.assert_series_equal(
        baseline.applied_betas.iloc[-1], changed.applied_betas.iloc[-1]
    )
    assert not np.isclose(
        baseline.estimated_betas.iloc[-1]['Full Model'],
        changed.estimated_betas.iloc[-1]['Full Model'],
    )
    components = baseline.component_returns
    np.testing.assert_allclose(
        components['Full Model Return'],
        components['Systematic Return']
        + components['Risk Layer Alpha']
        + components['Signal Layer Alpha']
        + components['Integration Alpha'],
        atol=1.0e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        components['Total Model Alpha'],
        components['Risk Layer Alpha']
        + components['Signal Layer Alpha']
        + components['Integration Alpha'],
        atol=1.0e-12,
        rtol=0.0,
    )
    expected_expanding = (
        components[[
            'Total Model Alpha',
            'Risk Layer Alpha',
            'Signal Layer Alpha',
            'Integration Alpha',
        ]]
        .expanding(min_periods=1)
        .mean()
        .multiply(12.0)
    )
    pd.testing.assert_frame_equal(baseline.expanding_annualised_alpha, expected_expanding)


def test_rolling_ewma_alpha_summary_matches_independent_pandas_recursion() -> None:
    """Current EWMA estimates smooth the exact lagged-beta components with the beta span."""
    span = 4
    result = compute_model_layer_ewma_alpha_attribution(
        **_rolling_navs(_rolling_layer_returns()),
        freq='ME',
        beta_span=span,
    )
    alpha_columns = [
        'Total Model Alpha',
        'Risk Layer Alpha',
        'Signal Layer Alpha',
        'Integration Alpha',
    ]
    expected_components = (
        result.component_returns.ewm(span=span, adjust=False).mean().multiply(12.0)
    )
    expected_alpha = expected_components[alpha_columns]

    pd.testing.assert_frame_equal(
        result.ewma_annualised_components,
        expected_components,
    )
    pd.testing.assert_frame_equal(result.ewma_annualised_alpha, expected_alpha)
    pd.testing.assert_series_equal(
        result.current_ewma_annualised_components,
        expected_components.iloc[-1],
    )
    pd.testing.assert_series_equal(
        result.current_ewma_annualised_alpha,
        expected_alpha.iloc[-1],
    )
    np.testing.assert_allclose(
        result.ewma_annualised_alpha['Total Model Alpha'],
        result.ewma_annualised_alpha[[
            'Risk Layer Alpha',
            'Signal Layer Alpha',
            'Integration Alpha',
        ]].sum(axis=1),
        atol=1.0e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        result.ewma_annualised_components['Full Model Return'],
        result.ewma_annualised_components[[
            'Systematic Return',
            'Risk Layer Alpha',
            'Signal Layer Alpha',
            'Integration Alpha',
        ]].sum(axis=1),
        atol=1.0e-12,
        rtol=0.0,
    )


def test_current_ewma_regression_attribution_is_additive_and_ci_centred() -> None:
    """EWMA-WLS bars reconstruct the model and match their joint HAC intervals."""
    result = compute_model_layer_ewma_regression_attribution(
        **_rolling_navs(_rolling_layer_returns()),
        freq='ME',
        span=8,
        hac_lags=3,
        confidence_level=0.95,
    )

    assert isinstance(result, ModelLayerEwmaRegressionAttribution)
    assert result.span == 8
    assert result.hac_lags == 3
    assert 0.0 < result.effective_nobs <= result.nobs
    bridge_columns = [
        'Systematic Return',
        'Risk Layer Alpha',
        'Signal Layer Alpha',
        'Integration Alpha',
    ]
    np.testing.assert_allclose(
        result.component_returns[bridge_columns].sum(axis=1),
        result.component_returns['Full Model Return'],
        atol=1.0e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        result.annualised_components[bridge_columns].sum(),
        result.annualised_components['Full Model Return'],
        atol=1.0e-12,
        rtol=0.0,
    )
    for layer, component in (
        ('Risk Layer', 'Risk Layer Alpha'),
        ('Signal Layer', 'Signal Layer Alpha'),
        ('Integration', 'Integration Alpha'),
    ):
        row = result.regression_table.loc[layer]
        midpoint = 0.5 * (row['An Alpha CI Low'] + row['An Alpha CI High'])
        np.testing.assert_allclose(
            result.annualised_components[component],
            row['An Alpha'],
            atol=1.0e-12,
            rtol=0.0,
        )
        np.testing.assert_allclose(midpoint, row['An Alpha'], atol=1.0e-12, rtol=0.0)


def test_current_ewma_regression_integration_covariance_matches_full_alpha() -> None:
    """Joint risk, signal and integration covariance reconstructs full-model uncertainty."""
    result = compute_model_layer_ewma_regression_attribution(
        **_rolling_navs(_rolling_layer_returns()),
        freq='ME',
        span=8,
    )
    ones = np.ones(3)
    reconstructed_variance = float(
        ones @ result.annualised_alpha_covariance.to_numpy() @ ones
    )
    full_se = float(result.regression_table.loc['Full Model', 'Alpha HAC SE']) * 12.0
    np.testing.assert_allclose(
        reconstructed_variance,
        full_se ** 2,
        atol=1.0e-14,
        rtol=1.0e-10,
    )


def test_current_ewma_integration_matches_a_direct_weighted_hac_regression() -> None:
    """The joint covariance contrast agrees with fitting the integration return directly."""
    result = compute_model_layer_ewma_regression_attribution(
        **_rolling_navs(_rolling_layer_returns()),
        freq='ME',
        span=8,
        hac_lags=3,
    )
    integration_returns = (
        result.periodic_returns['Full Model']
        - result.periodic_returns['Risk Layer']
        - result.periodic_returns['Signal Layer']
    ).to_frame('Integration')
    direct = model_layer.ols.estimate_ewma_alpha_beta_hac(
        x=result.periodic_returns['Benchmark'],
        y=integration_returns,
        span=result.span,
        hac_lags=result.hac_lags,
        confidence_level=result.confidence_level,
    )
    row = result.regression_table.loc['Integration']
    np.testing.assert_allclose(row['Alpha'], direct.alpha['Integration'], atol=1.0e-12)
    np.testing.assert_allclose(row['Beta'], direct.beta['Integration'], atol=1.0e-12)
    np.testing.assert_allclose(
        row['Alpha HAC SE'],
        direct.alpha_hac_se['Integration'],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        row[['An Alpha CI Low', 'An Alpha CI High']].to_numpy(dtype=float),
        12.0 * direct.alpha_confidence_interval.loc['Integration'].to_numpy(dtype=float),
        atol=1.0e-12,
    )


def _near_additive_ewma_navs(integration_scale: float) -> dict[str, pd.Series]:
    """Return layers whose integration return is tiny relative to the observed layers."""
    nobs = 96
    time = np.arange(nobs, dtype=float)
    benchmark_returns = 0.01 * np.sin(time / 4.0) + 0.005 * np.cos(time / 9.0)
    risk_returns = 0.001 + 0.8 * benchmark_returns + 0.003 * np.sin(time / 3.0)
    signal_returns = 0.002 + 1.1 * benchmark_returns + 0.004 * np.cos(time / 5.0)
    integration_returns = integration_scale * (0.005 + 0.0001 * np.sin(time))
    full_returns = risk_returns + signal_returns + integration_returns
    index = pd.date_range('2017-12-31', periods=nobs + 1, freq='ME')
    return {
        'benchmark_nav': _nav_from_log_returns(benchmark_returns, index),
        'risk_layer_nav': _nav_from_log_returns(risk_returns, index),
        'signal_layer_nav': _nav_from_log_returns(signal_returns, index),
        'full_model_nav': _nav_from_log_returns(full_returns, index),
    }


@pytest.mark.parametrize(('statistic', 'absolute_tolerance'), [
    ('R2', 1.0e-12),
    ('p-Alpha', 1.0e-15),
])
def test_current_ewma_integration_statistics_are_scale_invariant(
        statistic: str,
        absolute_tolerance: float,
) -> None:
    """Small return units do not trigger zero shortcuts in an integration fit statistic."""
    ordinary = compute_model_layer_ewma_regression_attribution(
        **_near_additive_ewma_navs(integration_scale=1.0),
        freq='ME',
        span=36,
        hac_lags=3,
    )
    tiny = compute_model_layer_ewma_regression_attribution(
        **_near_additive_ewma_navs(integration_scale=1.0e-6),
        freq='ME',
        span=36,
        hac_lags=3,
    )
    ordinary_row = ordinary.regression_table.loc['Integration']
    tiny_row = tiny.regression_table.loc['Integration']

    np.testing.assert_allclose(
        tiny_row[statistic],
        ordinary_row[statistic],
        rtol=1.0e-6,
        atol=absolute_tolerance,
    )


def test_current_ewma_tiny_integration_covariance_matches_direct_fit() -> None:
    """Joint integration inference avoids cancellation between large layer covariances."""
    result = compute_model_layer_ewma_regression_attribution(
        **_near_additive_ewma_navs(integration_scale=1.0e-6),
        freq='ME',
        span=36,
        hac_lags=3,
    )
    integration_returns = (
        result.periodic_returns['Full Model']
        - result.periodic_returns['Risk Layer']
        - result.periodic_returns['Signal Layer']
    ).to_frame('Integration')
    direct = model_layer.ols.estimate_ewma_alpha_beta_hac(
        x=result.periodic_returns['Benchmark'],
        y=integration_returns,
        span=result.span,
        hac_lags=result.hac_lags,
        confidence_level=result.confidence_level,
    )
    row = result.regression_table.loc['Integration']

    np.testing.assert_allclose(
        row['Alpha HAC SE'],
        direct.alpha_hac_se['Integration'],
        rtol=1.0e-10,
        atol=1.0e-20,
    )
    np.testing.assert_allclose(
        result.annualised_alpha_covariance.loc['Integration', 'Integration'],
        (12.0 * direct.alpha_hac_se['Integration']) ** 2,
        rtol=1.0e-10,
        atol=1.0e-30,
    )


def test_current_ewma_stage_sharpes_match_existing_qis_ewma_engine() -> None:
    """Sequential stage Sharpes reuse the QIS norm-type-two EWMA estimator."""
    result = compute_model_layer_ewma_regression_attribution(
        **_rolling_navs(_rolling_layer_returns()),
        freq='ME',
        span=8,
    )
    actual = compute_model_layer_ewma_stage_sharpes(result, norm_type=2)
    components = result.component_returns
    expected_returns = pd.DataFrame({
        'Benchmark': components['Benchmark Return'],
        'Systematic': components['Systematic Return'],
        'Risk Layer': components['Systematic Return'] + components['Risk Layer Alpha'],
        'Signal Layer': (
            components['Systematic Return']
            + components['Risk Layer Alpha']
            + components['Signal Layer Alpha']
        ),
        'Full Model Gross': components['Full Model Return'],
    })
    expected = model_layer.compute_ewm_sharpe(
        returns=expected_returns,
        span=result.span,
        norm_type=2,
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_current_ewma_regression_optional_net_extends_both_bridges() -> None:
    """An optional net NAV contributes an exact cost step to return and Sharpe stages."""
    returns = _rolling_layer_returns()
    net_returns = returns['Full Model'] - np.linspace(0.0001, 0.0004, len(returns.index))
    navs = _rolling_navs(returns)
    navs['full_model_net_nav'] = _nav_from_log_returns(
        net_returns.to_numpy(),
        navs['full_model_nav'].index,
    )
    result = compute_model_layer_ewma_regression_attribution(
        **navs,
        freq='ME',
        span=8,
    )

    np.testing.assert_allclose(
        result.component_returns['Full Model Return']
        + result.component_returns['Trading Cost Drag'],
        result.component_returns['Full Model Net Return'],
        atol=1.0e-12,
        rtol=0.0,
    )
    sharpes = compute_model_layer_ewma_stage_sharpes(result)
    assert sharpes.columns[-1] == 'Full Model Net'


def test_rolling_attribution_honours_a_two_period_beta_lag() -> None:
    """A longer positive lag delays every estimated beta by the requested periods."""
    result = compute_model_layer_ewma_alpha_attribution(
        **_rolling_navs(_rolling_layer_returns()),
        freq='ME',
        beta_span=4,
        beta_lag=2,
        beta_init_value=0.8,
    )

    pd.testing.assert_frame_equal(
        result.applied_betas,
        result.estimated_betas.shift(2).fillna(0.8),
    )
    assert result.beta_lag == 2


def test_cumulative_rolling_alpha_starts_after_warmup_with_base_date_beta() -> None:
    """Post-warm-up cumulative alpha starts at zero and first uses the base-date estimate."""
    attribution = compute_model_layer_ewma_alpha_attribution(
        **_rolling_navs(_rolling_layer_returns()), freq='ME', beta_span=4
    )
    base_date = attribution.periodic_returns.index[11]
    result = compute_model_layer_cumulative_alpha_after_warmup(
        attribution=attribution,
        base_date=base_date,
        warmup_periods=12,
    )
    first_alpha_date = attribution.periodic_returns.index[12]
    alpha_columns = [
        'Total Model Alpha',
        'Risk Layer Alpha',
        'Signal Layer Alpha',
        'Integration Alpha',
    ]
    expected_returns = attribution.component_returns.loc[first_alpha_date:, alpha_columns]
    expected_cumulative = pd.concat([
        pd.DataFrame(0.0, index=pd.DatetimeIndex([base_date]), columns=alpha_columns),
        expected_returns.cumsum(),
    ])

    assert isinstance(result, ModelLayerCumulativeAlphaAttribution)
    assert result.base_date == base_date
    assert result.first_alpha_date == first_alpha_date
    assert result.warmup_periods == 12
    assert result.mean_adj_type is MeanAdjType.EWMA
    pd.testing.assert_series_equal(
        attribution.applied_betas.loc[first_alpha_date],
        attribution.estimated_betas.loc[base_date],
        check_names=False,
    )
    pd.testing.assert_frame_equal(result.alpha_returns, expected_returns)
    pd.testing.assert_frame_equal(result.cumulative_alpha, expected_cumulative)


def test_rolling_attribution_rejects_forward_looking_mean_adjustment() -> None:
    """The point-in-time API rejects the full-sample mean convention."""
    with np.testing.assert_raises_regex(ValueError, 'INSAMPLE'):
        compute_model_layer_ewma_alpha_attribution(
            **_rolling_navs(_rolling_layer_returns()),
            freq='ME',
            mean_adj_type=MeanAdjType.INSAMPLE,
        )


@pytest.mark.parametrize(
    ('overrides', 'error_type', 'message'),
    [
        ({'beta_span': 0}, ValueError, 'beta_span'),
        ({'beta_lag': True}, ValueError, 'beta_lag'),
        ({'beta_init_value': np.inf}, ValueError, 'beta_init_value'),
        ({'mean_adj_type': 'EWMA'}, TypeError, 'MeanAdjType'),
    ],
)
def test_rolling_attribution_validates_estimator_settings(
        overrides: dict[str, object],
        error_type: type[Exception],
        message: str,
) -> None:
    """Invalid spans, lags, priors and mean conventions fail before estimation."""
    kwargs = {**_rolling_navs(_rolling_layer_returns()), 'freq': 'ME', **overrides}
    with pytest.raises(error_type, match=message):
        compute_model_layer_ewma_alpha_attribution(**kwargs)


def test_rolling_attribution_validates_nav_samples() -> None:
    """The rolling API rejects invalid indices, coverage, sample length and benchmark variance."""
    returns = _rolling_layer_returns()
    navs = _rolling_navs(returns)
    invalid_index_navs = {
        name: series.set_axis(pd.RangeIndex(len(series.index)))
        for name, series in navs.items()
    }
    with pytest.raises(TypeError, match='DatetimeIndex'):
        compute_model_layer_ewma_alpha_attribution(**invalid_index_navs)

    all_missing_navs = navs.copy()
    all_missing_navs['signal_layer_nav'] = pd.Series(np.nan, index=navs['signal_layer_nav'].index)
    with pytest.raises(ValueError, match='all-missing'):
        compute_model_layer_ewma_alpha_attribution(**all_missing_navs)

    no_overlap_navs = navs.copy()
    no_overlap_navs['full_model_nav'] = navs['full_model_nav'].set_axis(
        navs['full_model_nav'].index + pd.DateOffset(years=10)
    )
    with pytest.raises(ValueError, match='no common valid sample'):
        compute_model_layer_ewma_alpha_attribution(**no_overlap_navs)

    short_navs = {name: series.iloc[:3] for name, series in navs.items()}
    with pytest.raises(ValueError, match='at least three returns'):
        compute_model_layer_ewma_alpha_attribution(**short_navs)

    constant_returns = returns.copy()
    constant_returns['Benchmark'] = 0.01
    with pytest.raises(ValueError, match='varying benchmark'):
        compute_model_layer_ewma_alpha_attribution(**_rolling_navs(constant_returns))


def test_estimated_beta_validation_rejects_missing_and_broken_paths() -> None:
    """Only leading NaNs before a finite beta path are accepted."""
    with pytest.raises(RuntimeError, match='no finite value'):
        model_layer._validate_estimated_betas(
            pd.DataFrame({'Risk Layer': [np.nan, np.nan]})
        )
    with pytest.raises(RuntimeError, match='non-leading non-finite'):
        model_layer._validate_estimated_betas(
            pd.DataFrame({'Risk Layer': [np.inf, 1.0, np.nan]})
        )


def test_rolling_attribution_rejects_non_finite_components() -> None:
    """Finite beta estimates that overflow the return bridge are rejected."""
    returns = _rolling_layer_returns()
    returns.iloc[2, returns.columns.get_loc('Benchmark')] = 2.0
    index = returns.index
    huge_betas = pd.DataFrame(
        {
            'Risk Layer': [np.nan] + [np.finfo(float).max] * (len(index) - 1),
            'Signal Layer': [np.nan] + [np.finfo(float).max] * (len(index) - 1),
            'Full Model': [np.nan] + [np.finfo(float).max] * (len(index) - 1),
        },
        index=index,
    )
    forecast = (huge_betas, None, None, None, None, None)
    with (
        patch.object(model_layer, 'compute_ewm_beta_alpha_forecast', return_value=forecast),
        pytest.raises(RuntimeError, match='components contain non-finite'),
        np.errstate(over='ignore', invalid='ignore'),
    ):
        compute_model_layer_ewma_alpha_attribution(
            **_rolling_navs(returns), freq='ME', beta_span=4
        )


def test_cumulative_rolling_alpha_validates_dates_and_warmup() -> None:
    """Cumulative alpha rejects duplicate, absent, premature and terminal base dates."""
    attribution = compute_model_layer_ewma_alpha_attribution(
        **_rolling_navs(_rolling_layer_returns()), freq='ME', beta_span=4
    )
    index = attribution.component_returns.index
    duplicate_index = index.to_list()
    duplicate_index[-1] = duplicate_index[-2]
    duplicate_attribution = replace(
        attribution,
        component_returns=attribution.component_returns.set_axis(duplicate_index),
    )
    with pytest.raises(ValueError, match='unique'):
        compute_model_layer_cumulative_alpha_after_warmup(
            duplicate_attribution, base_date=index[11], warmup_periods=12
        )
    with pytest.raises(ValueError, match='not in the periodic return index'):
        compute_model_layer_cumulative_alpha_after_warmup(
            attribution, base_date='1999-12-31', warmup_periods=12
        )
    with pytest.raises(ValueError, match='found 2'):
        compute_model_layer_cumulative_alpha_after_warmup(
            attribution, base_date=index[1], warmup_periods=12
        )
    with pytest.raises(ValueError, match='no subsequent attribution return'):
        compute_model_layer_cumulative_alpha_after_warmup(
            attribution, base_date=index[-1], warmup_periods=12
        )
    with pytest.raises(ValueError, match='warmup_periods'):
        compute_model_layer_cumulative_alpha_after_warmup(
            attribution, base_date=index[11], warmup_periods=0
        )


def test_cumulative_rolling_alpha_checks_the_first_applied_beta() -> None:
    """The one-period-lag audit rejects a changed first post-warm-up applied beta."""
    attribution = compute_model_layer_ewma_alpha_attribution(
        **_rolling_navs(_rolling_layer_returns()), freq='ME', beta_span=4
    )
    base_date = attribution.periodic_returns.index[11]
    first_alpha_date = attribution.periodic_returns.index[12]
    changed_applied_betas = attribution.applied_betas.copy()
    changed_applied_betas.loc[first_alpha_date, 'Full Model'] += 0.1
    changed = replace(attribution, applied_betas=changed_applied_betas)

    with pytest.raises(RuntimeError, match='base-date beta'):
        compute_model_layer_cumulative_alpha_after_warmup(
            changed, base_date=base_date, warmup_periods=12
        )

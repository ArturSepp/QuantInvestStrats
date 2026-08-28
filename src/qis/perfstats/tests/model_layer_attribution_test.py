"""Tests for full-sample model-layer log-return attribution."""

import numpy as np
import pandas as pd

from qis.perfstats.model_layer_attribution import (
    ModelLayerAlphaBetaAttribution,
    compute_model_layer_alpha_beta_attribution,
)


def _nav_from_log_returns(log_returns: np.ndarray, index: pd.DatetimeIndex) -> pd.Series:
    """Convert a deterministic log-return path to a NAV series."""
    values = np.exp(np.concatenate(([0.0], np.cumsum(log_returns))))
    return pd.Series(values, index=index)


def test_model_layer_attribution_recovers_log_alpha_beta_and_reconstructs() -> None:
    """Known layer alphas and betas are recovered on one common quarterly sample."""
    benchmark_returns = np.array([
        -0.12, 0.08, 0.03, -0.04, 0.11, -0.02, 0.05, 0.01,
    ])
    risk_returns = 0.002 + 0.80 * benchmark_returns
    alpha_returns = 0.003 + 0.20 * benchmark_returns
    integration_returns = 0.001 - 0.10 * benchmark_returns
    full_returns = risk_returns + alpha_returns + integration_returns
    index = pd.date_range('2019-12-31', periods=9, freq='QE')

    result = compute_model_layer_alpha_beta_attribution(
        benchmark_nav=_nav_from_log_returns(benchmark_returns, index),
        risk_layer_nav=_nav_from_log_returns(risk_returns, index),
        alpha_layer_nav=_nav_from_log_returns(alpha_returns, index),
        full_model_nav=_nav_from_log_returns(full_returns, index),
    )

    assert isinstance(result, ModelLayerAlphaBetaAttribution)
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
        'Alpha Layer Alpha',
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
    alpha_returns = 0.002 + 0.30 * benchmark_returns
    integration_returns = -0.001 + 0.05 * benchmark_returns
    full_returns = risk_returns + alpha_returns + integration_returns
    index = pd.date_range('2020-12-31', periods=7, freq='QE')

    result = compute_model_layer_alpha_beta_attribution(
        benchmark_nav=_nav_from_log_returns(benchmark_returns, index),
        risk_layer_nav=_nav_from_log_returns(risk_returns, index),
        alpha_layer_nav=_nav_from_log_returns(alpha_returns, index),
        full_model_nav=_nav_from_log_returns(full_returns, index),
    )

    mapping = {
        'Risk Layer Alpha': 'Risk Layer',
        'Alpha Layer Alpha': 'Alpha Layer',
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
    alpha_returns = 0.004 + 0.20 * benchmark_returns
    integration_returns = 0.001 - 0.05 * benchmark_returns
    full_returns = risk_returns + alpha_returns + integration_returns
    cost_drag = np.array([0.0002, 0.0004, 0.0001, 0.0003, 0.0005, 0.0002])
    net_returns = full_returns - cost_drag
    index = pd.date_range('2021-12-31', periods=7, freq='QE')

    result = compute_model_layer_alpha_beta_attribution(
        benchmark_nav=_nav_from_log_returns(benchmark_returns, index),
        risk_layer_nav=_nav_from_log_returns(risk_returns, index),
        alpha_layer_nav=_nav_from_log_returns(alpha_returns, index),
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
        'Alpha Layer Alpha',
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

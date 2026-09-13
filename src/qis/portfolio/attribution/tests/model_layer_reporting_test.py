"""Reporting-date and net-cost tests for lagged model-layer attribution."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from qis.portfolio.attribution.model_layer import (
    compute_model_layer_cumulative_alpha_after_warmup,
    compute_model_layer_ewma_alpha_attribution,
)


def _inputs(cost=0.0002):
    """Return deterministic gross and net monthly layer NAVs."""
    dates = pd.date_range('2020-12-31', periods=49, freq='ME')
    phase = np.arange(48)
    benchmark = 0.003 + 0.02 * np.sin(phase / 2.1)
    full = 0.001 + 0.9 * benchmark + 0.002 * np.cos(phase)
    def nav(returns):
        """Return a NAV including the real initial observation."""
        return pd.Series(np.exp(np.r_[0.0, np.cumsum(returns)]), index=dates)
    return dict(
        benchmark_nav=nav(benchmark),
        risk_layer_nav=nav(0.0005 + 0.7 * benchmark),
        signal_layer_nav=nav(0.001 + 1.1 * benchmark),
        full_model_nav=nav(full),
        full_model_net_nav=nav(full - cost),
    )


def test_net_alpha_from_initial_nav_uses_prior_and_cost_once():
    """A zero base at inception retains lagged betas and reconciles net alpha."""
    inputs = _inputs()
    attribution = compute_model_layer_ewma_alpha_attribution(**inputs)
    result = compute_model_layer_cumulative_alpha_after_warmup(
        attribution, base_date='2020-12-31', warmup_periods=0)
    assert result.first_alpha_date == pd.Timestamp('2021-01-31')
    assert (result.cumulative_alpha.iloc[0] == 0.0).all()
    assert (attribution.applied_betas.iloc[0] == 1.0).all()
    net_returns = np.log(inputs['full_model_net_nav']).diff().dropna()
    gross_returns = np.log(inputs['full_model_nav']).diff().dropna()
    components = attribution.component_returns
    expected = net_returns - components['Systematic Return']
    np.testing.assert_allclose(components['Net Total Model Alpha'], expected, atol=1e-14)
    np.testing.assert_allclose(
        components['Trading Cost Drag'], net_returns - gross_returns, atol=1e-14)
    np.testing.assert_allclose(
        result.cumulative_alpha['Net Total Model Alpha'].iloc[1:],
        expected.cumsum(), atol=1e-14)
    parts = ['Risk Layer Alpha', 'Signal Layer Alpha', 'Integration Alpha', 'Trading Cost Drag']
    np.testing.assert_allclose(
        result.cumulative_alpha[parts].sum(axis=1),
        result.cumulative_alpha['Net Total Model Alpha'], atol=1e-14)


def test_report_date_rebase_keeps_history_and_legacy_warmup():
    """Later report dates rebase the same estimator without allowing pre-inception dates."""
    inputs = _inputs()
    full = compute_model_layer_ewma_alpha_attribution(**inputs)
    result = compute_model_layer_cumulative_alpha_after_warmup(
        full, base_date='2021-12-31', warmup_periods=0)
    expected = full.component_returns.loc['2022-01-31':, 'Net Total Model Alpha'].cumsum()
    np.testing.assert_allclose(
        result.cumulative_alpha['Net Total Model Alpha'].iloc[1:], expected, atol=1e-14)
    with pytest.raises(ValueError, match='warm-up'):
        compute_model_layer_cumulative_alpha_after_warmup(full, '2020-12-31')
    with pytest.raises(ValueError, match='base date'):
        compute_model_layer_cumulative_alpha_after_warmup(
            full, '2020-11-30', warmup_periods=0)
    for invalid in (-1, True, 1.5):
        with pytest.raises(ValueError):
            compute_model_layer_cumulative_alpha_after_warmup(
                full, '2021-12-31', warmup_periods=invalid)
    legacy = compute_model_layer_ewma_alpha_attribution(
        **{key: value for key, value in inputs.items() if key != 'full_model_net_nav'})
    pd.testing.assert_frame_equal(full.estimated_betas, legacy.estimated_betas)
    assert list(legacy.cumulative_alpha.columns) == [
        'Total Model Alpha', 'Risk Layer Alpha', 'Signal Layer Alpha', 'Integration Alpha']
    assert (compute_model_layer_cumulative_alpha_after_warmup(
        legacy, '2021-12-31').cumulative_alpha.iloc[0] == 0.0).all()
    future = {key: value.copy() for key, value in inputs.items()}
    for value in future.values():
        value.loc['2023-01-31':] *= 1.02
    changed = compute_model_layer_ewma_alpha_attribution(**future)
    pd.testing.assert_frame_equal(
        full.component_returns.loc[:'2022-12-31'],
        changed.component_returns.loc[:'2022-12-31'])
    with pytest.raises(ValueError, match='monotonic'):
        compute_model_layer_cumulative_alpha_after_warmup(
            replace(full, component_returns=full.component_returns.iloc[::-1]),
            '2021-12-31', warmup_periods=0)


def test_zero_cost_net_path_equals_gross_and_invalid_baselines_fail():
    """Zero actual drag leaves all net numerical values unchanged."""
    result = compute_model_layer_ewma_alpha_attribution(**_inputs(cost=0.0))
    assert (result.component_returns['Trading Cost Drag'] == 0.0).all()
    pd.testing.assert_series_equal(
        result.component_returns['Net Total Model Alpha'],
        result.component_returns['Total Model Alpha'], check_names=False)
    with pytest.raises(ValueError, match='base date'):
        compute_model_layer_cumulative_alpha_after_warmup(
            replace(result, nav_start_date=None), '2020-12-31', warmup_periods=0)



@pytest.mark.parametrize('cost', [0.0, 1e-10, 0.0002])
def test_cost_display_keeps_small_nonzero_costs_and_gross_view(cost):
    """The display drops only identically zero costs and keeps gross audit data."""
    rolling = compute_model_layer_ewma_alpha_attribution(**_inputs(cost=cost))
    result = compute_model_layer_cumulative_alpha_after_warmup(
        rolling, '2020-12-31', warmup_periods=0)
    net = result.get_cumulative_alpha()
    assert ('Trading Cost Drag' in net) == (cost != 0.0)
    assert 'Net Total Model Alpha' in net
    gross = result.get_cumulative_alpha(is_net=False)
    assert list(gross.columns) == [
        'Total Model Alpha', 'Risk Layer Alpha', 'Signal Layer Alpha', 'Integration Alpha']
    assert 'Trading Cost Drag' in result.cumulative_alpha
    np.testing.assert_allclose(net.iloc[:, 0], net.iloc[:, 1:].sum(axis=1), atol=1e-12)


def test_aggregate_zero_cost_is_visible_and_missing_cost_keeps_legacy_view():
    """A cancelling observed cost series remains visible; absent net NAV remains supported."""
    rolling = compute_model_layer_ewma_alpha_attribution(**_inputs())
    result = compute_model_layer_cumulative_alpha_after_warmup(
        rolling, '2020-12-31', warmup_periods=0)
    alpha = result.alpha_returns.copy()
    alpha['Trading Cost Drag'] = np.tile([.001, -.001], len(alpha) // 2)
    changed = replace(result, alpha_returns=alpha)
    assert alpha['Trading Cost Drag'].sum() == 0.0
    assert 'Trading Cost Drag' in changed.get_cumulative_alpha()
    alpha.iloc[0, alpha.columns.get_loc('Trading Cost Drag')] = np.nan
    with pytest.raises(ValueError, match='finite'):
        changed.get_cumulative_alpha()
    inputs = _inputs()
    inputs.pop('full_model_net_nav')
    legacy = compute_model_layer_cumulative_alpha_after_warmup(
        compute_model_layer_ewma_alpha_attribution(**inputs), '2020-12-31', warmup_periods=0)
    assert 'Trading Cost Drag' not in legacy.get_cumulative_alpha()

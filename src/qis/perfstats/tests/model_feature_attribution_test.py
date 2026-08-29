"""Tests for factorial and Shapley attribution of layered-model features."""

import warnings
from itertools import combinations

import numpy as np
import pandas as pd

from qis.perfstats.model_feature_attribution import (
    ModelFeatureAlphaBetaAttribution,
    ModelLayerNavs,
    _harsanyi_powers,
    compute_model_feature_alpha_beta_attribution,
)


def _to_nav(log_returns: np.ndarray, index: pd.DatetimeIndex, name: str) -> pd.Series:
    """Convert deterministic log returns to a unit-initialised NAV."""
    values = np.exp(np.concatenate(([0.0], np.cumsum(log_returns))))
    return pd.Series(values, index=index, name=name)


def _two_feature_scenarios() -> dict[frozenset[str], ModelLayerNavs]:
    """Build a complete two-feature experiment with layer-specific interaction effects."""
    index = pd.date_range('2010-12-31', periods=121, freq='ME')
    x = np.arange(120, dtype=float)
    benchmark_returns = 0.004 + 0.012 * np.sin(x / 8.0)
    benchmark = _to_nav(benchmark_returns, index, 'Benchmark')
    base_returns = {
        'risk': 0.90 * benchmark_returns + 0.0005,
        'alpha': 1.02 * benchmark_returns + 0.0010,
        'full': 0.84 * benchmark_returns + 0.0018,
    }
    beta_effects = {
        'risk': 0.0002 + 0.0004 * np.cos(x / 9.0),
        'alpha': 0.0001 + 0.0002 * np.sin(x / 7.0),
        'full': 0.0003 + 0.0003 * np.cos(x / 10.0),
    }
    signal_effects = {
        'risk': np.zeros_like(x),
        'alpha': 0.0004 + 0.0005 * np.cos(x / 6.0),
        'full': 0.0005 + 0.0004 * np.sin(x / 11.0),
    }
    interaction_effects = {
        'risk': np.zeros_like(x),
        'alpha': 0.0001 * np.sin(x / 5.0),
        'full': 0.0002 * np.cos(x / 12.0),
    }

    scenarios: dict[frozenset[str], ModelLayerNavs] = {}
    for coalition in (
            frozenset(),
            frozenset({'beta_span'}),
            frozenset({'signal_span'}),
            frozenset({'beta_span', 'signal_span'}),
    ):
        layer_returns = {}
        for layer in ('risk', 'alpha', 'full'):
            values = base_returns[layer].copy()
            if 'beta_span' in coalition:
                values = values + beta_effects[layer]
            if 'signal_span' in coalition:
                values = values + signal_effects[layer]
            if len(coalition) == 2:
                values = values + interaction_effects[layer]
            layer_returns[layer] = values
        label = '+'.join(sorted(coalition)) or 'production'
        scenarios[coalition] = ModelLayerNavs(
            benchmark_nav=benchmark,
            risk_layer_nav=_to_nav(layer_returns['risk'], index, f'{label} Risk'),
            alpha_layer_nav=_to_nav(layer_returns['alpha'], index, f'{label} Alpha'),
            full_model_nav=_to_nav(layer_returns['full'], index, f'{label} Full'),
            full_model_net_nav=_to_nav(
                layer_returns['full'] - 0.00005,
                index,
                f'{label} Full Net',
            ),
        )
    return scenarios


def _log_returns(nav: pd.Series) -> pd.Series:
    """Return consecutive log changes without resampling."""
    return np.log(nav).diff().dropna()


def test_harsanyi_products_use_stable_established_multiplication_order() -> None:
    """Multiply the joint and baseline paths before the proper-subset denominators."""
    beta = frozenset({'beta_span'})
    signal = frozenset({'signal_span'})
    interaction = beta.union(signal)

    assert list(_harsanyi_powers(interaction)) == [
        interaction,
        frozenset(),
        beta,
        signal,
    ]


def test_two_feature_shapley_paths_and_factorial_effects_reconstruct_joint() -> None:
    """Both exact decompositions reconstruct every gross and net layer return path."""
    scenarios = _two_feature_scenarios()

    result = compute_model_feature_alpha_beta_attribution(scenario_layer_navs=scenarios)

    assert isinstance(result, ModelFeatureAlphaBetaAttribution)
    assert tuple(result.feature_attributions) == ('beta_span', 'signal_span')
    assert set(result.factorial_effect_navs) == {
        frozenset({'beta_span'}),
        frozenset({'signal_span'}),
        frozenset({'beta_span', 'signal_span'}),
    }
    for layer_field in (
            'risk_layer_nav',
            'alpha_layer_nav',
            'full_model_nav',
            'full_model_net_nav',
    ):
        joint = getattr(result.joint_effect_navs, layer_field)
        assert joint is not None
        factorial_sum = sum(
            _log_returns(getattr(layer_navs, layer_field))
            for layer_navs in result.factorial_effect_navs.values()
        )
        shapley_sum = sum(
            _log_returns(getattr(layer_navs, layer_field))
            for layer_navs in result.shapley_feature_navs.values()
        )
        pd.testing.assert_series_equal(
            _log_returns(joint),
            factorial_sum,
            check_names=False,
            atol=1.0e-12,
            rtol=0.0,
        )
        pd.testing.assert_series_equal(
            _log_returns(joint),
            shapley_sum,
            check_names=False,
            atol=1.0e-12,
            rtol=0.0,
        )
    assert float(result.identity_errors.max()) < 1.0e-12


def test_two_feature_shapley_path_matches_average_marginal_return() -> None:
    """A feature path equals the average of its two possible marginal contributions."""
    scenarios = _two_feature_scenarios()

    result = compute_model_feature_alpha_beta_attribution(scenario_layer_navs=scenarios)

    empty = scenarios[frozenset()].full_model_nav
    beta = scenarios[frozenset({'beta_span'})].full_model_nav
    signal = scenarios[frozenset({'signal_span'})].full_model_nav
    both = scenarios[frozenset({'beta_span', 'signal_span'})].full_model_nav
    expected = 0.5 * (
        _log_returns(beta) - _log_returns(empty)
        + _log_returns(both) - _log_returns(signal)
    )
    actual = _log_returns(result.shapley_feature_navs['beta_span'].full_model_nav)
    pd.testing.assert_series_equal(actual, expected, check_names=False, atol=1.0e-12)


def test_feature_summary_exposes_midpoint_hac_intervals_and_layer_bridge() -> None:
    """The summary carries symmetric intervals and the exact QIS full-alpha bridge."""
    result = compute_model_feature_alpha_beta_attribution(
        scenario_layer_navs=_two_feature_scenarios(),
    )

    summary = result.summary
    shapley_summary = summary.loc['Shapley']
    assert shapley_summary.index.tolist() == ['beta_span', 'signal_span']
    for prefix in (
            'Annualised Full Model Return',
            'Annualised Full Model Net Return',
            'Risk Layer Alpha',
            'Alpha Layer Alpha',
            'Integration Alpha',
            'Full Model Alpha',
            'Full Model Net Alpha',
    ):
        np.testing.assert_allclose(
            shapley_summary[prefix],
            0.5 * (
                shapley_summary[f'{prefix} CI Low']
                + shapley_summary[f'{prefix} CI High']
            ),
            atol=1.0e-12,
            rtol=0.0,
        )
    np.testing.assert_allclose(
        shapley_summary['Full Model Alpha'],
        shapley_summary[
            ['Risk Layer Alpha', 'Alpha Layer Alpha', 'Integration Alpha']
        ].sum(axis=1),
        atol=1.0e-12,
        rtol=0.0,
    )


def test_three_feature_shapley_engine_accepts_complete_power_set() -> None:
    """The generic engine reconstructs a joint path with three interacting features."""
    features = ('a', 'b', 'c')
    index = pd.date_range('2015-12-31', periods=73, freq='ME')
    x = np.arange(72, dtype=float)
    benchmark_returns = 0.003 + 0.01 * np.sin(x / 7.0)
    benchmark = _to_nav(benchmark_returns, index, 'Benchmark')
    scenarios: dict[frozenset[str], ModelLayerNavs] = {}
    for size in range(len(features) + 1):
        for enabled in combinations(features, size):
            coalition = frozenset(enabled)
            singleton_effect = sum(
                (position + 1) * 0.0001 for position, feature in enumerate(features)
                if feature in coalition
            )
            pair_effect = 0.00007 * max(0, len(coalition) - 1)
            triple_effect = 0.00005 if len(coalition) == 3 else 0.0
            effect = singleton_effect + pair_effect + triple_effect * np.cos(x / 5.0)
            risk_returns = 0.80 * benchmark_returns + 0.0004 + 0.3 * effect
            alpha_returns = 0.25 * benchmark_returns + 0.0006 + 0.4 * effect
            full_returns = 0.92 * benchmark_returns + 0.0012 + effect
            label = '+'.join(enabled) or 'production'
            scenarios[coalition] = ModelLayerNavs(
                benchmark_nav=benchmark,
                risk_layer_nav=_to_nav(risk_returns, index, f'{label} Risk'),
                alpha_layer_nav=_to_nav(alpha_returns, index, f'{label} Alpha'),
                full_model_nav=_to_nav(full_returns, index, f'{label} Full'),
            )

    result = compute_model_feature_alpha_beta_attribution(scenario_layer_navs=scenarios)

    assert tuple(result.shapley_feature_navs) == features
    assert len(result.factorial_effect_navs) == 7
    assert float(result.identity_errors.max()) < 1.0e-12


def test_feature_experiment_validation_rejects_incomplete_and_changed_benchmark() -> None:
    """Incomplete power sets and feature-dependent benchmarks fail before estimation."""
    scenarios = _two_feature_scenarios()
    incomplete = dict(scenarios)
    incomplete.pop(frozenset({'beta_span', 'signal_span'}))
    with np.testing.assert_raises_regex(ValueError, 'complete factorial experiment'):
        compute_model_feature_alpha_beta_attribution(scenario_layer_navs=incomplete)

    changed_benchmark = dict(scenarios)
    coalition = frozenset({'beta_span'})
    layer_navs = changed_benchmark[coalition]
    changed_benchmark[coalition] = ModelLayerNavs(
        benchmark_nav=(
            layer_navs.benchmark_nav
            * np.exp(np.linspace(0.0, 0.01, len(layer_navs.benchmark_nav.index)))
        ),
        risk_layer_nav=layer_navs.risk_layer_nav,
        alpha_layer_nav=layer_navs.alpha_layer_nav,
        full_model_nav=layer_navs.full_model_nav,
        full_model_net_nav=layer_navs.full_model_net_nav,
    )
    with np.testing.assert_raises_regex(ValueError, 'benchmark changed'):
        compute_model_feature_alpha_beta_attribution(scenario_layer_navs=changed_benchmark)


def test_feature_attribution_emits_no_warnings() -> None:
    """The total-return intervals come from the constant-only HAC mean, not a rank-deficient fit."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result = compute_model_feature_alpha_beta_attribution(
            scenario_layer_navs=_two_feature_scenarios(),
        )
    assert isinstance(result, ModelFeatureAlphaBetaAttribution)

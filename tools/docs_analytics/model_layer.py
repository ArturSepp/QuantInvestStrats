"""Adapters around the existing seeded model-layer example and its independent checks."""

from dataclasses import fields

import numpy as np
import pandas as pd

import qis
from examples.portfolios import model_layer_attribution_simulated as example
from tools.docs_analytics.style import model_exhibit


def produce(spec: dict) -> dict:
    """Return figures and full-precision tables from the same attribution objects."""
    params = spec['parameters']
    if params['hac_lags'] != 3 or params['confidence_level'] != 0.95:
        raise ValueError('The existing exhibit labels require 95% Bartlett HAC(3) intervals')
    if (params['start'], params['end'], params['freq']) != (
            example.START, example.END, example.FREQ):
        raise ValueError('Manifest sample must match the dedicated model-layer fixture')
    navs = example.simulate_layer_navs(seed=params['seed'])
    attribution = qis.compute_model_layer_alpha_beta_attribution(
        freq=params['freq'], hac_lags=params['hac_lags'],
        confidence_level=params['confidence_level'], **navs,
    )
    scenarios = example.simulate_feature_scenarios(navs, seed=params['feature_seed'])
    features = qis.compute_model_feature_alpha_beta_attribution(
        scenario_layer_navs=scenarios, freq=params['freq'],
        hac_lags=params['hac_lags'], confidence_level=params['confidence_level'],
    )
    example.check_identities(attribution, navs)
    example.check_feature_design(features)
    identity_error = float(features.identity_errors.abs().max())
    if not np.isfinite(identity_error) or identity_error >= 1e-12:
        raise ValueError('Shapley identity failed')

    bridge = example.plot_return_bridge(attribution)
    cumulative = example.plot_cumulative_alpha(attribution)
    feature_figure = example.plot_feature_decomposition(features)
    model_exhibit(
        bridge, title='Model-layer return attribution',
        subtitle='Synthetic monthly log returns | 2006–2025 | full-sample OLS',
        footer='Annualised mean log returns. Whiskers: 95% Bartlett HAC(3) alpha intervals.\n'
               'Benchmark return is shown separately; the bridge starts at systematic return.',
    )
    model_exhibit(
        cumulative, title='Cumulative model-layer alpha',
        subtitle='Synthetic monthly log returns | 2006–2025 | full-sample OLS',
        footer='Cumulative monthly log-return contributions, with a 31 Dec 2005 zero baseline.\n'
               'Total alpha = risk + signal + integration at every date; these paths are not NAVs.',
    )
    model_exhibit(
        feature_figure, title='Two-feature model sensitivity',
        subtitle='Synthetic monthly log returns | 2006–2025 | Shapley allocation',
        footer='Annualised effects in percentage points; features split the interaction equally.\n'
               'Whiskers: 95% Bartlett HAC(3) intervals, with benchmark OLS for alpha effects.',
    )
    bar_keys = [
        'Benchmark Return', 'Systematic Return', 'Risk Layer Alpha', 'Signal Layer Alpha',
        'Integration Alpha', 'Trading Cost Drag', 'Full Model Net Return',
    ]
    for label, key in zip(bridge.axes[0].texts, bar_keys):
        value = attribution.annualised_components[key]
        label.set_text(f'{value:+.2%}' if 'Alpha' in key or 'Drag' in key else f'{value:.2%}')
    for label in feature_figure.axes[0].texts:
        x_position = label.xy[0]
        label.set_fontsize(10)
        _, vertical_offset = label.get_position()
        label.set_position((-4 if x_position < round(x_position) else 4, vertical_offset))
    axis = bridge.axes[0]
    axis.set_xticks(axis.get_xticks(),
                   [label.get_text().removeprefix('+ ') for label in axis.get_xticklabels()])
    np.testing.assert_allclose(
        [patch.get_height() for patch in bridge.axes[0].patches],
        attribution.annualised_components.loc[bar_keys], atol=1e-12, rtol=0,
    )
    # Export the plotted lines directly, avoiding a second implementation of the plot transform.
    lines = cumulative.axes[0].lines[:4]
    labels = [text.get_text() for text in cumulative.axes[0].get_legend().get_texts()]
    if len(labels) != 4 or len(lines) != 4:
        raise ValueError('Expected four labelled cumulative alpha paths')
    cumulative_table = pd.DataFrame(
        {label: line.get_ydata() for label, line in zip(labels, lines)},
        index=navs['benchmark_nav'].index,
    )
    scenario_tables = {}
    for key, layers in sorted(scenarios.items(), key=lambda item: sorted(item[0])):
        label = '+'.join(sorted(key)) or 'baseline'
        scenario_tables[label] = pd.DataFrame({
            field.name: getattr(layers, field.name)
            for field in fields(layers) if getattr(layers, field.name) is not None
        })
    return {
        'figures': {
            'model_layer_attribution_simulated.png': bridge,
            'model_layer_attribution_cumulative_alpha_simulated.png': cumulative,
            'model_feature_attribution_simulated.png': feature_figure,
        },
        'tables': {
            'navs': pd.DataFrame(navs),
            'regressions': attribution.regression_table,
            'periodic_returns': attribution.periodic_returns,
            'components': attribution.component_returns,
            'annualised_components': attribution.annualised_components,
            'feature_summary': features.summary,
            'feature_identity_errors': features.identity_errors,
            'feature_scenarios': pd.concat(scenario_tables, axis=1),
            'cumulative_alpha_lines': cumulative_table,
        },
        'checks': {
            'ols_linearity_and_excess_basis': True, 'feature_design': True,
            'shapley_identity': True, 'displayed_bridge_matches_components': True,
        },
        'parameters': params,
        'summary': {
            'monthly_log_return_observations': len(attribution.periodic_returns),
            'annualised_components': {
                str(k): float(v) for k, v in attribution.annualised_components.items()
            },
            'shapley_max_identity_error': identity_error,
            'cumulative_alpha_terminal': {
                str(k): float(v) for k, v in cumulative_table.iloc[-1].items()
            },
        },
    }

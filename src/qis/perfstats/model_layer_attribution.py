"""Full-sample alpha/beta attribution for layered portfolio models.

The decomposition is defined in log-return space so its systematic, standalone-layer and
integration contributions add exactly through time. It is descriptive and full-sample by design;
it is not a point-in-time estimator for use inside a backtest.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

import qis.perfstats.returns as ret
import qis.utils.regression as ols
from qis.perfstats.config import PerfStat
from qis.utils.annualisation import get_annualization_factor


@dataclass(frozen=True)
class ModelLayerAlphaBetaAttribution:
    """Full-sample alpha/beta decomposition of model-layer NAVs.

    Attributes:
        periodic_returns: Common-sample log returns for the supplied gross and optional net NAVs.
        regression_table: Full-sample OLS statistics for each layer, integration, and optional net
            model.
        component_returns: Exact periodic log-return decomposition of the gross and optional net
            full model.
        annualised_components: Annualised mean log-return contributions.
    """

    periodic_returns: pd.DataFrame
    regression_table: pd.DataFrame
    component_returns: pd.DataFrame
    annualised_components: pd.Series


def compute_model_layer_alpha_beta_attribution(
        benchmark_nav: pd.Series,
        risk_layer_nav: pd.Series,
        alpha_layer_nav: pd.Series,
        full_model_nav: pd.Series,
        freq: str = 'QE',
        full_model_net_nav: Optional[pd.Series] = None,
) -> ModelLayerAlphaBetaAttribution:
    """Compute a full-sample OLS decomposition of model-layer log returns.

    Each observed layer is regressed on the benchmark. The full-model log return is then
    decomposed into its systematic return, risk-layer alpha, alpha-layer alpha and integration
    alpha. The integration term captures the non-additivity of the two standalone layers relative
    to the fully constrained model. The four gross contributions reconstruct the full-model log
    return in every observation. When ``full_model_net_nav`` is supplied, trading-cost drag is the
    exact log-return difference between the net and gross full models and extends the bridge to the
    net return.

    Args:
        benchmark_nav: Benchmark NAV or price index.
        risk_layer_nav: NAV produced using the risk layer without alpha signals.
        alpha_layer_nav: NAV of the standalone alpha or signal portfolio.
        full_model_nav: NAV of the fully integrated model.
        freq: Regression and return frequency. Defaults to quarter-end.
        full_model_net_nav: Optional NAV of the same fully integrated model after trading costs.

    Returns:
        Alpha/beta regression statistics and exact log-return components.

    Raises:
        TypeError: If the inputs do not have a DatetimeIndex.
        ValueError: If the common sample is too short or the benchmark has no variation.
        RuntimeError: If the resulting decomposition does not reconcile.
    """
    nav_series = {
        'Benchmark': benchmark_nav,
        'Risk Layer': risk_layer_nav,
        'Alpha Layer': alpha_layer_nav,
        'Full Model': full_model_nav,
    }
    if full_model_net_nav is not None:
        nav_series['Full Model Net'] = full_model_net_nav
    navs = pd.concat(
        nav_series,
        axis=1,
        sort=True,
    ).sort_index()
    if not isinstance(navs.index, pd.DatetimeIndex):
        raise TypeError('model-layer NAVs must have a DatetimeIndex')

    periodic_returns = ret.to_returns(
        prices=navs,
        freq=freq,
        is_log_returns=True,
    )
    periodic_returns = periodic_returns.replace([np.inf, -np.inf], np.nan).dropna(how='any')
    if len(periodic_returns.index) < 3:
        raise ValueError('model-layer attribution requires at least three common returns')
    if np.isclose(periodic_returns['Benchmark'].var(ddof=0), 0.0):
        raise ValueError('model-layer attribution requires varying benchmark returns')

    layer_returns = periodic_returns.copy()
    layer_returns.insert(
        3,
        'Integration',
        (
            periodic_returns['Full Model']
            - periodic_returns['Risk Layer']
            - periodic_returns['Alpha Layer']
        ),
    )

    alpha_column = PerfStat.ALPHA.to_str()
    annualised_alpha_column = PerfStat.ALPHA_AN.to_str()
    beta_column = PerfStat.BETA.to_str()
    r2_column = PerfStat.R2.to_str()
    pvalue_column = PerfStat.ALPHA_PVALUE.to_str()
    annualisation = get_annualization_factor(freq=freq)
    regression_rows = {
        'Benchmark': {
            alpha_column: 0.0,
            annualised_alpha_column: 0.0,
            beta_column: 1.0,
            r2_column: 1.0,
            pvalue_column: 1.0,
        }
    }
    benchmark_returns = layer_returns['Benchmark']
    regression_layers = ['Risk Layer', 'Alpha Layer', 'Integration', 'Full Model']
    if full_model_net_nav is not None:
        regression_layers.append('Full Model Net')
    for layer in regression_layers:
        alpha, beta, r_squared, alpha_pvalue = ols.estimate_ols_alpha_beta(
            x=benchmark_returns,
            y=layer_returns[layer],
        )
        if not np.isfinite(alpha) or not np.isfinite(beta):
            raise ValueError(f'OLS failed for {layer}')
        regression_rows[layer] = {
            alpha_column: alpha,
            annualised_alpha_column: annualisation * alpha,
            beta_column: beta,
            r2_column: r_squared,
            pvalue_column: alpha_pvalue,
        }
    regression_table = pd.DataFrame.from_dict(regression_rows, orient='index')

    betas = regression_table[beta_column]
    component_returns = pd.DataFrame({
        'Benchmark Return': benchmark_returns,
        'Alpha Layer Return': layer_returns['Alpha Layer'],
        'Systematic Return': betas['Full Model'] * benchmark_returns,
        'Risk Layer Alpha': (
            layer_returns['Risk Layer'] - betas['Risk Layer'] * benchmark_returns
        ),
        'Alpha Layer Alpha': (
            layer_returns['Alpha Layer'] - betas['Alpha Layer'] * benchmark_returns
        ),
    })
    initial_bridge_columns = [
        'Systematic Return',
        'Risk Layer Alpha',
        'Alpha Layer Alpha',
    ]
    component_returns['Integration Alpha'] = (
        layer_returns['Full Model']
        - component_returns[initial_bridge_columns].sum(axis=1)
    )
    component_returns['Full Model Return'] = layer_returns['Full Model']

    bridge_columns = initial_bridge_columns + ['Integration Alpha']
    reconstructed = component_returns[bridge_columns].sum(axis=1)
    if not np.allclose(
            reconstructed,
            component_returns['Full Model Return'],
            atol=1.0e-12,
            rtol=0.0,
    ):
        raise RuntimeError(
            'model-layer components do not reconstruct the full-model log return'
        )
    if full_model_net_nav is not None:
        component_returns['Trading Cost Drag'] = (
            layer_returns['Full Model Net'] - layer_returns['Full Model']
        )
        component_returns['Full Model Net Return'] = layer_returns['Full Model Net']
        net_bridge_columns = bridge_columns + ['Trading Cost Drag']
        net_reconstructed = component_returns[net_bridge_columns].sum(axis=1)
        if not np.allclose(
                net_reconstructed,
                component_returns['Full Model Net Return'],
                atol=1.0e-12,
                rtol=0.0,
        ):
            raise RuntimeError(
                'model-layer components do not reconstruct the net full-model log return'
            )
    annualised_components = annualisation * component_returns.mean(axis=0)
    return ModelLayerAlphaBetaAttribution(
        periodic_returns=periodic_returns,
        regression_table=regression_table,
        component_returns=component_returns,
        annualised_components=annualised_components,
    )

"""Full-sample alpha/beta attribution for layered portfolio models.

The methodology starts from benchmark, risk-layer, signal-layer and fully integrated model NAVs.
An optional net full-model NAV may be supplied to measure realised trading
cost drag. All NAVs are trimmed to the range between the latest first valid observation and the
earliest last valid observation, forward-filled inside that range, and converted at ``freq`` to
log returns, ``r[t] = log(NAV[t] / NAV[t-1])``. Log returns are required because their additive
identity lets the component bridge reconstruct the full model in every observation and over time.

For each observed or derived layer L, the module estimates the descriptive full-sample regression
``r_L[t] = alpha_L + beta_L * r_B[t] + epsilon_L[t]``, where B is the supplied benchmark. The point
estimates are OLS. Alpha inference uses a Bartlett-kernel HAC covariance with ``hac_lags`` Bartlett
lags (default three), the statsmodels small-sample correction, a normal reference distribution and
a two-sided 95% interval.
Alpha and its confidence bounds are annualised linearly by the factor implied by ``freq``; beta,
R² and the periodic HAC standard error are not annualised. The generic regression and HAC
calculation lives in ``qis.utils.regression``; this module only assigns ``PerfStat`` labels.
The result records the return frequency, HAC lag count and confidence level used in estimation.

The gross full-model return is separated into four exact periodic components: systematic return is
``beta_F * r_B``; risk-layer alpha is ``r_R - beta_R * r_B``; signal-layer alpha is
``r_S - beta_S * r_B``; and integration alpha is the residual required to reconcile these three
terms to ``r_F``. Thus integration measures what the constrained full model adds beyond simply
combining the risk- and signal-layer effects. If a net NAV is supplied, trading-cost drag is the
exact log-return difference ``r_F_net - r_F`` and extends the same identity to net performance.
All generated component returns are checked for finiteness before the result is returned.

This is an ex-post explanatory analysis, not a point-in-time estimator and not an input to the
portfolio backtest. Its full-sample coefficients must not be interpreted as implementable forecasts.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

import qis.perfstats.returns as ret
import qis.utils.regression as ols
from qis.perfstats.config import PerfStat
from qis.utils.annualisation import get_annualization_factor


ALPHA_HAC_LAGS: int = 3
ALPHA_CONFIDENCE_LEVEL: float = 0.95
ALPHA_HAC_SE_COLUMN: str = 'Alpha HAC SE'
ALPHA_AN_CI_LOW_COLUMN: str = 'An Alpha CI Low'
ALPHA_AN_CI_HIGH_COLUMN: str = 'An Alpha CI High'


@dataclass(frozen=True)
class ModelLayerAlphaBetaAttribution:
    """Full-sample alpha/beta decomposition of model-layer NAVs.

    Attributes:
        periodic_returns: Common-sample log returns for the supplied gross and optional net NAVs.
        regression_table: Full-sample OLS statistics for each layer, integration, and optional net
            model, including annualised Bartlett HAC alpha confidence bounds.
        component_returns: Exact periodic log-return decomposition in this order: Benchmark Return,
            Risk Layer Return, Signal Layer Return, Systematic Return, Risk Layer Alpha, Signal
            Layer Alpha, Integration Alpha, Full Model Return, then Trading Cost Drag and Full
            Model Net Return when a net NAV is supplied.
        annualised_components: Annualised mean log-return contributions.
        freq: Regression and return frequency.
        hac_lags: Bartlett-kernel lag count used for alpha inference.
        confidence_level: Two-sided confidence level used for alpha intervals.
    """

    periodic_returns: pd.DataFrame
    regression_table: pd.DataFrame
    component_returns: pd.DataFrame
    annualised_components: pd.Series
    freq: str
    hac_lags: int
    confidence_level: float


def compute_model_layer_alpha_beta_attribution(
        benchmark_nav: pd.Series,
        risk_layer_nav: pd.Series,
        signal_layer_nav: pd.Series,
        full_model_nav: pd.Series,
        freq: str = 'QE',
        full_model_net_nav: Optional[pd.Series] = None,
        hac_lags: int = ALPHA_HAC_LAGS,
        confidence_level: float = ALPHA_CONFIDENCE_LEVEL,
) -> ModelLayerAlphaBetaAttribution:
    """Compute a full-sample OLS decomposition of model-layer log returns.

    Each observed layer is regressed on the benchmark. The full-model log return is then
    decomposed into its systematic return, risk-layer alpha, signal-layer alpha and integration
    alpha. The integration term captures the non-additivity of the risk and signal layers relative
    to the fully constrained model. The four gross contributions reconstruct the full-model log
    return in every observation. When ``full_model_net_nav`` is supplied, trading-cost drag is the
    exact log-return difference between the net and gross full models and extends the bridge to the
    net return.

    Alpha p-values and confidence intervals use Bartlett-kernel heteroskedasticity and
    autocorrelation-consistent standard errors with ``hac_lags`` Bartlett lags (default three).
    Confidence bounds use ``confidence_level`` and are annualised linearly using the factor implied
    by ``freq``.

    Args:
        benchmark_nav: Benchmark NAV or price index.
        risk_layer_nav: NAV produced using the risk layer without alpha signals.
        signal_layer_nav: NAV of the portfolio built from the signals alone.
        full_model_nav: NAV of the fully integrated model.
        freq: Regression and return frequency. Defaults to quarter-end.
        full_model_net_nav: Optional NAV of the same fully integrated model after trading costs.
        hac_lags: Bartlett-kernel lag count for alpha HAC covariance. Defaults to three periods.
        confidence_level: Two-sided alpha confidence-interval level. Defaults to 0.95.

    Returns:
        Alpha/beta regression statistics and exact log-return components.

    Raises:
        TypeError: If the inputs do not have a DatetimeIndex.
        ValueError: If HAC settings are invalid, any NAV is entirely missing, the NAVs have no
            common valid range, the common sample is too short, or the benchmark has no variation.
        RuntimeError: If any component return is non-finite.
    """
    if hac_lags < 0:
        raise ValueError(f'hac_lags must be non-negative, got {hac_lags}')
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            f'confidence_level must be between zero and one, got {confidence_level}'
        )
    nav_series = {
        'Benchmark': benchmark_nav,
        'Risk Layer': risk_layer_nav,
        'Signal Layer': signal_layer_nav,
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
    first_valid_dates = [nav.first_valid_index() for _, nav in navs.items()]
    last_valid_dates = [nav.last_valid_index() for _, nav in navs.items()]
    if any(date is None for date in first_valid_dates + last_valid_dates):
        raise ValueError('model-layer NAV inputs contain an all-missing series')
    first_valid, last_valid = max(first_valid_dates), min(last_valid_dates)
    if first_valid >= last_valid:
        raise ValueError(
            f'model-layer NAVs have no common valid sample: {first_valid=}, {last_valid=}'
        )
    navs = navs.loc[first_valid:last_valid]

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
            - periodic_returns['Signal Layer']
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
            ALPHA_HAC_SE_COLUMN: 0.0,
            ALPHA_AN_CI_LOW_COLUMN: 0.0,
            ALPHA_AN_CI_HIGH_COLUMN: 0.0,
        }
    }
    benchmark_returns = layer_returns['Benchmark']
    regression_layers = ['Risk Layer', 'Signal Layer', 'Integration', 'Full Model']
    if full_model_net_nav is not None:
        regression_layers.append('Full Model Net')
    for layer in regression_layers:
        try:
            regression_result = ols.estimate_ols_alpha_beta_hac(
                x=benchmark_returns,
                y=layer_returns[layer],
                hac_lags=hac_lags,
                confidence_level=confidence_level,
            )
        except Exception as exception:
            raise ValueError(f'OLS failed for {layer}') from exception
        regression_rows[layer] = {
            alpha_column: regression_result.alpha,
            annualised_alpha_column: annualisation * regression_result.alpha,
            beta_column: regression_result.beta,
            r2_column: regression_result.r_squared,
            pvalue_column: regression_result.alpha_pvalue,
            ALPHA_HAC_SE_COLUMN: regression_result.alpha_hac_se,
            ALPHA_AN_CI_LOW_COLUMN: (
                annualisation * regression_result.alpha_confidence_interval[0]
            ),
            ALPHA_AN_CI_HIGH_COLUMN: (
                annualisation * regression_result.alpha_confidence_interval[1]
            ),
        }
    regression_table = pd.DataFrame.from_dict(regression_rows, orient='index')

    betas = regression_table[beta_column]
    component_returns = pd.DataFrame({
        'Benchmark Return': benchmark_returns,
        'Risk Layer Return': layer_returns['Risk Layer'],
        'Signal Layer Return': layer_returns['Signal Layer'],
        'Systematic Return': betas['Full Model'] * benchmark_returns,
        'Risk Layer Alpha': (
            layer_returns['Risk Layer'] - betas['Risk Layer'] * benchmark_returns
        ),
        'Signal Layer Alpha': (
            layer_returns['Signal Layer'] - betas['Signal Layer'] * benchmark_returns
        ),
    })
    initial_bridge_columns = [
        'Systematic Return',
        'Risk Layer Alpha',
        'Signal Layer Alpha',
    ]
    component_returns['Integration Alpha'] = (
        layer_returns['Full Model']
        - component_returns[initial_bridge_columns].sum(axis=1)
    )
    component_returns['Full Model Return'] = layer_returns['Full Model']

    if full_model_net_nav is not None:
        component_returns['Trading Cost Drag'] = (
            layer_returns['Full Model Net'] - layer_returns['Full Model']
        )
        component_returns['Full Model Net Return'] = layer_returns['Full Model Net']
    if not np.isfinite(component_returns.to_numpy(dtype=float)).all():
        raise RuntimeError('model-layer components contain non-finite values')
    annualised_components = annualisation * component_returns.mean(axis=0)
    return ModelLayerAlphaBetaAttribution(
        periodic_returns=periodic_returns,
        regression_table=regression_table,
        component_returns=component_returns,
        annualised_components=annualised_components,
        freq=freq,
        hac_lags=hac_lags,
        confidence_level=confidence_level,
    )

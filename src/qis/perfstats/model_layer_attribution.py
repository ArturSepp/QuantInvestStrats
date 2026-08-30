"""Full-sample and point-in-time attribution for layered portfolio models.

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

The full-sample estimator is an ex-post explanatory analysis, not an input to a portfolio
backtest. Its coefficients must not be interpreted as implementable forecasts. The rolling
estimator instead uses QIS EWMA betas, applies each estimate only after ``beta_lag`` periods, and
supports expanding annualised alpha and an unannualised cumulative-alpha view after a warm-up.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

import qis.perfstats.returns as ret
import qis.utils.regression as ols
from qis.models.linear.ewm import (
    InitType,
    MeanAdjType,
    compute_ewm_beta_alpha_forecast,
)
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


@dataclass(frozen=True)
class ModelLayerEwmaAlphaAttribution:
    """Point-in-time model-layer attribution under lagged EWMA betas.

    Attributes:
        periodic_returns: Common-sample model-layer log returns.
        estimated_betas: EWMA betas after observing each return, preserving leading NaNs.
        applied_betas: Betas available before each return, using the prior when unavailable.
        component_returns: Exact realised systematic and alpha return components.
        cumulative_alpha: Cumulative realised log-return alpha components from inception.
        expanding_annualised_alpha: Annualised expanding means of realised alpha components.
        freq: Return frequency used by the estimator.
        beta_span: EWMA beta span in return periods.
        beta_lag: Number of periods between beta estimation and application.
        beta_init_value: Point-in-time beta prior used before an estimate is available.
        mean_adj_type: Point-in-time mean adjustment used in beta estimation.
    """

    periodic_returns: pd.DataFrame
    estimated_betas: pd.DataFrame
    applied_betas: pd.DataFrame
    component_returns: pd.DataFrame
    cumulative_alpha: pd.DataFrame
    expanding_annualised_alpha: pd.DataFrame
    freq: str
    beta_span: int
    beta_lag: int
    beta_init_value: float
    mean_adj_type: MeanAdjType


@dataclass(frozen=True)
class ModelLayerCumulativeAlphaAttribution:
    """Post-warm-up cumulative realised alpha under point-in-time EWMA betas.

    Attributes:
        alpha_returns: Lagged-beta alpha returns after the warm-up base date.
        cumulative_alpha: Unannualised cumulative alpha with a zero row at the base date.
        base_date: Date on which the cumulative paths are rebased to zero.
        first_alpha_date: First return date accrued after the warm-up.
        warmup_periods: Minimum number of estimator returns through the base date.
        freq: Return frequency inherited from the underlying attribution.
        beta_span: EWMA beta span in return periods.
        beta_lag: Number of periods between beta estimation and application.
        beta_init_value: Point-in-time beta prior used by the underlying estimator.
        mean_adj_type: Point-in-time mean adjustment used by the underlying estimator.
    """

    alpha_returns: pd.DataFrame
    cumulative_alpha: pd.DataFrame
    base_date: pd.Timestamp
    first_alpha_date: pd.Timestamp
    warmup_periods: int
    freq: str
    beta_span: int
    beta_lag: int
    beta_init_value: float
    mean_adj_type: MeanAdjType


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


def _validate_positive_integer(value: int, name: str) -> None:
    """Validate an integer estimator setting that must be strictly positive."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f'{name} must be a positive integer, got {value!r}')


def _to_common_model_layer_returns(
        benchmark_nav: pd.Series,
        risk_layer_nav: pd.Series,
        signal_layer_nav: pd.Series,
        full_model_nav: pd.Series,
        freq: str,
) -> pd.DataFrame:
    """Convert four model-layer NAVs to finite common-sample log returns."""
    navs = pd.concat(
        {
            'Benchmark': benchmark_nav,
            'Risk Layer': risk_layer_nav,
            'Signal Layer': signal_layer_nav,
            'Full Model': full_model_nav,
        },
        axis=1,
        sort=True,
    ).sort_index()
    if not isinstance(navs.index, pd.DatetimeIndex):
        raise TypeError('model-layer NAVs must have a DatetimeIndex')
    first_valid_dates = [series.first_valid_index() for _, series in navs.items()]
    last_valid_dates = [series.last_valid_index() for _, series in navs.items()]
    if any(date is None for date in first_valid_dates + last_valid_dates):
        raise ValueError('model-layer NAV inputs contain an all-missing series')
    first_valid = max(first_valid_dates)
    last_valid = min(last_valid_dates)
    if first_valid >= last_valid:
        raise ValueError(
            'model-layer NAVs have no common valid sample: '
            f'{first_valid=}, {last_valid=}'
        )
    periodic_returns = ret.to_returns(
        prices=navs.loc[first_valid:last_valid],
        freq=freq,
        is_log_returns=True,
    )
    periodic_returns = periodic_returns.replace([np.inf, -np.inf], np.nan).dropna(how='any')
    if len(periodic_returns.index) < 3:
        raise ValueError('rolling model-layer attribution requires at least three returns')
    if np.isclose(periodic_returns['Benchmark'].var(ddof=0), 0.0):
        raise ValueError('rolling model-layer attribution requires varying benchmark returns')
    return periodic_returns


def _validate_estimated_betas(estimated_betas: pd.DataFrame) -> None:
    """Allow only leading NaNs before each beta series becomes finite."""
    for column in estimated_betas.columns:
        values = estimated_betas[column].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if not finite.any():
            raise RuntimeError(f'EWMA beta estimates contain no finite value for {column}')
        first_finite = int(np.flatnonzero(finite)[0])
        if not np.isnan(values[:first_finite]).all() or not finite[first_finite:].all():
            raise RuntimeError(
                f'EWMA beta estimates contain non-leading non-finite values for {column}'
            )


def compute_model_layer_ewma_alpha_attribution(
        benchmark_nav: pd.Series,
        risk_layer_nav: pd.Series,
        signal_layer_nav: pd.Series,
        full_model_nav: pd.Series,
        freq: str = 'ME',
        beta_span: int = 36,
        beta_lag: int = 1,
        beta_init_value: float = 1.0,
        mean_adj_type: MeanAdjType = MeanAdjType.EWMA,
) -> ModelLayerEwmaAlphaAttribution:
    """Estimate model-layer alpha using point-in-time, lagged EWMA betas.

    QIS estimates each layer beta after observing return ``t`` and applies it only from
    ``t + beta_lag`` onward. An explicit beta prior is applied until a lagged finite estimate is
    available. With the default EWMA mean adjustment and ``InitType.X0``, the first centered
    observation is zero, so the first estimated beta remains NaN for audit while applied betas
    remain finite. Alpha is the realised step-ahead beta-adjusted log return; the EWMA alpha
    forecast returned by the lower-level estimator is deliberately not used.

    Args:
        benchmark_nav: Benchmark NAV or price index.
        risk_layer_nav: NAV produced using the risk layer without alpha signals.
        signal_layer_nav: NAV of the portfolio built from the signals alone.
        full_model_nav: NAV of the fully integrated model.
        freq: Return frequency. Defaults to month-end.
        beta_span: EWMA beta span in return periods. Defaults to 36.
        beta_lag: Periods between beta estimation and application. Defaults to one.
        beta_init_value: Finite beta prior used before an estimate is available. Defaults to one.
        mean_adj_type: Point-in-time beta mean adjustment. Defaults to EWMA. ``INSAMPLE`` is
            rejected because it is forward-looking.

    Returns:
        Rolling betas, exact realised alpha components and expanding alpha estimates.

    Raises:
        TypeError: If NAV inputs do not have a DatetimeIndex or ``mean_adj_type`` is not an enum.
        ValueError: If estimator settings or the common return sample are invalid.
        RuntimeError: If beta estimates contain non-leading missing values or outputs are not
            finite.
    """
    _validate_positive_integer(value=beta_span, name='beta_span')
    _validate_positive_integer(value=beta_lag, name='beta_lag')
    if not np.isfinite(beta_init_value):
        raise ValueError(f'beta_init_value must be finite, got {beta_init_value!r}')
    if not isinstance(mean_adj_type, MeanAdjType):
        raise TypeError(f'mean_adj_type must be MeanAdjType, got {mean_adj_type!r}')
    if mean_adj_type is MeanAdjType.INSAMPLE:
        raise ValueError('MeanAdjType.INSAMPLE is forward-looking and not allowed here')
    periodic_returns = _to_common_model_layer_returns(
        benchmark_nav=benchmark_nav,
        risk_layer_nav=risk_layer_nav,
        signal_layer_nav=signal_layer_nav,
        full_model_nav=full_model_nav,
        freq=freq,
    )
    layer_columns = ['Risk Layer', 'Signal Layer', 'Full Model']
    estimated_betas, *_ = compute_ewm_beta_alpha_forecast(
        x_data=periodic_returns['Benchmark'],
        y_data=periodic_returns[layer_columns],
        span=beta_span,
        mean_adj_type=mean_adj_type,
        init_type=InitType.X0,
        beta_init_value=beta_init_value,
    )
    _validate_estimated_betas(estimated_betas=estimated_betas)
    applied_betas = estimated_betas.shift(beta_lag).fillna(float(beta_init_value))

    benchmark_returns = periodic_returns['Benchmark']
    systematic_returns = applied_betas['Full Model'] * benchmark_returns
    risk_alpha = (
        periodic_returns['Risk Layer']
        - applied_betas['Risk Layer'] * benchmark_returns
    )
    signal_alpha = (
        periodic_returns['Signal Layer']
        - applied_betas['Signal Layer'] * benchmark_returns
    )
    total_alpha = periodic_returns['Full Model'] - systematic_returns
    integration_alpha = total_alpha - risk_alpha - signal_alpha
    component_returns = pd.DataFrame({
        'Benchmark Return': benchmark_returns,
        'Risk Layer Return': periodic_returns['Risk Layer'],
        'Signal Layer Return': periodic_returns['Signal Layer'],
        'Systematic Return': systematic_returns,
        'Total Model Alpha': total_alpha,
        'Risk Layer Alpha': risk_alpha,
        'Signal Layer Alpha': signal_alpha,
        'Integration Alpha': integration_alpha,
        'Full Model Return': periodic_returns['Full Model'],
    })
    if not np.isfinite(component_returns.to_numpy(dtype=float)).all():
        raise RuntimeError('rolling model-layer components contain non-finite values')
    alpha_columns = [
        'Total Model Alpha',
        'Risk Layer Alpha',
        'Signal Layer Alpha',
        'Integration Alpha',
    ]
    cumulative_alpha = component_returns[alpha_columns].cumsum()
    expanding_annualised_alpha = (
        component_returns[alpha_columns]
        .expanding(min_periods=1)
        .mean()
        .multiply(get_annualization_factor(freq=freq))
    )
    return ModelLayerEwmaAlphaAttribution(
        periodic_returns=periodic_returns,
        estimated_betas=estimated_betas,
        applied_betas=applied_betas,
        component_returns=component_returns,
        cumulative_alpha=cumulative_alpha,
        expanding_annualised_alpha=expanding_annualised_alpha,
        freq=freq,
        beta_span=int(beta_span),
        beta_lag=int(beta_lag),
        beta_init_value=float(beta_init_value),
        mean_adj_type=mean_adj_type,
    )


def compute_model_layer_cumulative_alpha_after_warmup(
        attribution: ModelLayerEwmaAlphaAttribution,
        base_date: pd.Timestamp | str,
        warmup_periods: int = 12,
) -> ModelLayerCumulativeAlphaAttribution:
    """Rebase cumulative lagged-beta alpha after a point-in-time warm-up.

    The base date is the final estimator-only observation. Cumulative alpha is zero on that date,
    and the first accrued residual uses the beta available on the base date. With a one-period
    beta lag this is the base-date estimate, or the explicit prior if that estimate is unavailable.

    Args:
        attribution: Point-in-time EWMA-beta attribution containing realised alpha residuals.
        base_date: Date on which cumulative alpha is rebased to zero.
        warmup_periods: Minimum estimator observations through ``base_date``. Defaults to 12.

    Returns:
        Post-warm-up alpha returns and their unannualised cumulative sums.

    Raises:
        ValueError: If settings, dates or available warm-up observations are invalid.
        RuntimeError: If the first post-warm-up return does not use the base-date beta.
    """
    _validate_positive_integer(value=warmup_periods, name='warmup_periods')
    requested_base_date = pd.Timestamp(base_date)
    index = attribution.component_returns.index
    if not index.is_unique:
        raise ValueError('rolling attribution index must be unique')
    if requested_base_date not in index:
        raise ValueError(
            f'cumulative alpha base date {requested_base_date:%Y-%m-%d} is not in the '
            'periodic return index'
        )
    base_position = int(index.get_loc(requested_base_date))
    observed_warmup_periods = base_position + 1
    if observed_warmup_periods < warmup_periods:
        raise ValueError(
            f'cumulative alpha requires at least {warmup_periods} warm-up returns through '
            f'{requested_base_date:%Y-%m-%d}, found {observed_warmup_periods}'
        )
    first_alpha_position = base_position + 1
    if first_alpha_position >= len(index):
        raise ValueError('cumulative alpha base date leaves no subsequent attribution return')
    first_alpha_date = index[first_alpha_position]
    alpha_columns = [
        'Total Model Alpha',
        'Risk Layer Alpha',
        'Signal Layer Alpha',
        'Integration Alpha',
    ]
    alpha_returns = attribution.component_returns.loc[first_alpha_date:, alpha_columns].copy()
    baseline = pd.DataFrame(
        0.0,
        index=pd.DatetimeIndex([requested_base_date]),
        columns=alpha_columns,
    )
    cumulative_alpha = pd.concat([baseline, alpha_returns.cumsum()])
    if attribution.beta_lag == 1:
        expected_first_beta = attribution.estimated_betas.loc[
            requested_base_date
        ].fillna(attribution.beta_init_value)
        if not np.allclose(
                attribution.applied_betas.loc[first_alpha_date],
                expected_first_beta,
                atol=1.0e-12,
                rtol=0.0,
        ):
            raise RuntimeError('first post-warm-up return does not use the base-date beta')
    return ModelLayerCumulativeAlphaAttribution(
        alpha_returns=alpha_returns,
        cumulative_alpha=cumulative_alpha,
        base_date=requested_base_date,
        first_alpha_date=first_alpha_date,
        warmup_periods=int(warmup_periods),
        freq=attribution.freq,
        beta_span=attribution.beta_span,
        beta_lag=attribution.beta_lag,
        beta_init_value=attribution.beta_init_value,
        mean_adj_type=attribution.mean_adj_type,
    )

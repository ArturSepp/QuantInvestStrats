"""AR(1)-prewhitened Bartlett HAC inference for OLS intercepts, one response at a time.

Mean and alpha tests need the long-run variance of the intercept score. With a fixed
Bartlett lag count L the plain kernel estimator is biased towards zero when the score is
persistent, and the bias does not vanish as T grows. Andrews and Monahan (1992, Econometrica
60(4), 953-966) prewhiten with a low-order autoregression, apply the kernel to the innovations
and recolour at frequency zero. Nothing here is new: this module is their estimator for the
linear combination of the score that defines each intercept, applied to one response at a
time rather than as a VAR(1) of the joint score. The qis choices are a fixed Bartlett lag on
the innovations instead of their automatic bandwidth, and an abort at |rho| >= 1 instead of
their truncation of rho at 0.97.

For T observations, k design columns (intercept included) and the intercept row a of
(X'X)^{-1}X', the influence is psi_t = T a_t u_t with OLS residuals u_t, so that
alpha_hat - alpha = (1/T) sum_t psi_t and sum_t psi_t = 0 exactly. A zero-intercept AR(1)
fitted to psi, rho = sum_t psi_t psi_{t-1} / sum_t psi_{t-1}^2, leaves the T-1 innovations
v_t = psi_t - rho psi_{t-1}. With Bartlett weights w_l = 1 - l/(L+1),

    S_v = sum_t v_t^2 + 2 sum_{l=1}^{L} w_l sum_t v_t v_{t-l},
    Var(alpha_hat) = T/(T-k) * S_v / ((1-rho)^2 T^2).

The normaliser keeps the original T, as in Andrews and Monahan, so with rho = 0 the estimate
equals the unprewhitened sandwich minus the first observation's contribution. The fit never
falls back to the plain kernel.

Inputs are complete per-observation values in chronological order; the estimates, standard
errors and interval bounds are in input units per observation. There is no calendar
inference, resampling, funding subtraction or annualisation here, and no joint parameter
covariance. Family tests, multiple-testing corrections and the bootstrap live in qis.perfstats.
"""

# packages
import numpy as np
import pandas as pd
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Optional, Tuple, Union
from scipy.stats import norm
# qis
from qis.utils import regression


# Residual RMS or standard error at or below this fraction of the response's maximum absolute
# value is a degenerate fit, not a hypothesis. qis.perfstats.significance applies the same rule.
DEGENERACY_RTOL = 1.0e-12


@dataclass(frozen=True)
class PrewhitenedInterceptResult:
    """Marginal prewhitened intercept inference for every response.

    Attributes:
        table: Response-indexed estimate, standard_error, statistic, pvalue_normal,
            ci_lower_pointwise, ci_upper_pointwise and rho, in input units per observation.
        nobs: Observations T; the AR(1) filter leaves T-1 innovations.
        n_parameters: Design columns k, including the intercept.
        hac_lags: Bartlett lag count L applied to the innovations.
        confidence_level: Two-sided level of the pointwise normal intervals.
        influence: Response-indexed psi_t = T a_t u_t in input units, on all T rows.
        innovations: Filtered influence v_t on the final T-1 rows, in input units.
    """

    table: pd.DataFrame
    nobs: int
    n_parameters: int
    hac_lags: int
    confidence_level: float
    influence: pd.DataFrame
    innovations: pd.DataFrame


def _reject(mask: np.ndarray, columns: pd.Index, reason: str) -> None:
    """Raise one ValueError naming every response that fails a feasibility check."""
    if np.any(mask):
        raise ValueError(f'{reason}; responses={columns[np.asarray(mask)].tolist()!r}')


def _validate_settings(nobs: int, n_parameters: int, hac_lags: int) -> int:
    """Check the sample, design and lag counts and return the lag count as an int."""
    if nobs < 3:
        raise ValueError(f'prewhitening requires at least three observations, got T={nobs}')
    if not 1 <= n_parameters < nobs:
        raise ValueError(f'require 1 <= k < T, got k={n_parameters} and T={nobs}')
    if (isinstance(hac_lags, (bool, np.bool_)) or not isinstance(hac_lags, Integral)
            or not 0 <= hac_lags <= nobs - 2):
        raise ValueError(f'hac_lags must be an integer in [0, {nobs - 2}], got {hac_lags!r}')
    return int(hac_lags)


def _ar1_bartlett(influence: np.ndarray, n_parameters: int, hac_lags: int,
                  columns: pd.Index) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the scalar AR(1) filters and evaluate the recoloured Bartlett variances.

    Args:
        influence: Finite (T, M) influence values psi in chronological row order.
        n_parameters: Design columns k, including the intercept, for the T/(T-k) correction.
        hac_lags: Bartlett lag count on the T-1 innovations; no selection or capping.
        columns: Response labels for error messages.

    Returns:
        Variances of the intercept estimates, the AR(1) coefficients and the (T-1, M)
        innovations, all in influence units.

    Raises:
        ValueError: If an influence column is degenerate, its AR(1) fit is unstable
            (|rho| >= 1) or its variance is not finite and positive. There is no fallback.
    """
    nobs = influence.shape[0]
    lags = _validate_settings(nobs, n_parameters, hac_lags)
    _reject(~np.isfinite(influence).all(axis=0), columns, 'nonfinite influence')
    scales = np.max(np.abs(influence), axis=0)
    _reject(scales == 0.0, columns, 'degenerate zero influence')
    # A per-column rescaling prevents underflow in the AR dot products; rho is unchanged.
    unit = influence / scales
    denominator = np.sum(np.square(unit[:-1]), axis=0)
    _reject((denominator <= 0.0) | ~np.isfinite(denominator), columns,
            'zero/nonfinite AR denominator')
    rho = np.sum(unit[1:] * unit[:-1], axis=0) / denominator
    _reject(~np.isfinite(rho) | (np.abs(rho) >= 1.0), columns, 'unstable/nonfinite AR(1) filter')
    innovations = unit[1:] - rho * unit[:-1]
    meat = np.sum(np.square(innovations), axis=0)
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        meat += 2.0 * weight * np.sum(innovations[lag:] * innovations[:-lag], axis=0)
    correction = nobs / (nobs - n_parameters)
    variances = correction * np.square(scales / (1.0 - rho)) * meat / float(nobs ** 2)
    _reject(~np.isfinite(variances) | (variances <= 0.0), columns,
            'invalid/nonpositive prewhitened variance')
    return variances, rho, innovations * scales


def estimate_prewhitened_intercepts(
        y: Union[pd.Series, pd.DataFrame],
        x: Optional[Union[pd.Series, pd.DataFrame]] = None,
        hac_lags: int = 3,
        confidence_level: float = 0.95,
) -> PrewhitenedInterceptResult:
    """Estimate OLS intercepts with AR(1)-prewhitened Bartlett HAC standard errors.

    Point estimates come from the canonical equal-weight fit in qis.utils.regression.
    Only the intercept's scalar influence is filtered, separately for each response.

    Args:
        y: Finite real response Series or DataFrame on a unique, increasing index; an
            unnamed Series is labelled 'y'.
        x: Optional finite factor Series or DataFrame on the same index. An intercept is
            always fitted; 'Intercept' is a reserved label.
        hac_lags: Bartlett lag count in [0, T-2] on the T-1 filtered innovations.
        confidence_level: Two-sided pointwise normal level strictly between zero and one.

    Returns:
        Per-response estimates, prewhitened standard errors, normal statistics, p-values,
        pointwise intervals and fitted rho, with the influence and innovation diagnostics.

    Raises:
        ValueError: If inputs or settings are invalid, the design is rank deficient, a
            response or its influence is degenerate, an AR(1) fit has |rho| >= 1, or a
            standard error is not finite and positive. No response is dropped.
    """
    if (isinstance(confidence_level, (bool, np.bool_)) or not isinstance(confidence_level, Real)
            or not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0):
        raise ValueError(f'confidence_level must be strictly between zero and one, '
                         f'got {confidence_level!r}')
    y_frame, x_frame = regression._prepare_hac_inputs(y, x)
    if not y_frame.index.is_unique or not y_frame.index.is_monotonic_increasing:
        raise ValueError('observation index must be unique and increasing')
    nobs = len(y_frame)
    n_parameters = 1 if x_frame is None else 1 + x_frame.shape[1]
    lags = _validate_settings(nobs, n_parameters, hac_lags)
    y_values = y_frame.to_numpy()
    _reject(~np.isfinite(y_values).all(axis=0), y_frame.columns, 'nonfinite responses')
    scales = np.max(np.abs(y_values), axis=0)
    _reject(scales == 0.0, y_frame.columns, 'zero response scale')
    # Max-absolute scaling keeps the fit well conditioned; it cancels from statistics.
    normalized_y = y_frame / scales
    normalized_x = None
    if x_frame is not None:
        x_values = x_frame.to_numpy()
        if not np.isfinite(x_values).all():
            raise ValueError('factors must be complete and finite; no row deletion')
        factor_scales = np.max(np.abs(x_values), axis=0)
        if np.any(factor_scales == 0.0):
            raise ValueError(f'constant zero factors: '
                             f'{x_frame.columns[factor_scales == 0.0].tolist()!r}')
        normalized_x = x_frame / factor_scales
    fitted = regression.estimate_hac_regression(
        y=normalized_y, x=normalized_x, span=None, hac_lags=lags,
        confidence_level=float(confidence_level),
    )
    if fitted.nobs != nobs or fitted.hac_lags != lags:
        raise ValueError('canonical OLS fit changed the sample or lag count')
    design = (np.ones((nobs, 1)) if normalized_x is None else
              np.column_stack((np.ones(nobs), normalized_x.to_numpy())))
    residuals = normalized_y.to_numpy() - design @ fitted.coefficients.to_numpy()
    residual_rms = np.sqrt(np.mean(np.square(residuals), axis=0))
    _reject(~np.isfinite(residual_rms) | (residual_rms <= DEGENERACY_RTOL), y_frame.columns,
            'degenerate residuals relative to response scale')
    q, r = np.linalg.qr(design, mode='reduced')
    intercept_row = np.linalg.solve(r, q.T)[0]
    influence = nobs * intercept_row[:, None] * residuals
    variances, rho, innovations = _ar1_bartlett(influence, n_parameters, lags, y_frame.columns)
    standard_errors = np.sqrt(variances)
    _reject(standard_errors <= DEGENERACY_RTOL, y_frame.columns,
            'degenerate standard errors relative to response scale')
    intercepts = fitted.coefficients.loc['Intercept'].to_numpy()
    statistics = intercepts / standard_errors
    critical = norm.ppf(0.5 + 0.5 * confidence_level)
    with np.errstate(over='ignore', invalid='ignore'):
        table = pd.DataFrame(dict(
            estimate=intercepts * scales,
            standard_error=standard_errors * scales,
            statistic=statistics,
            pvalue_normal=2.0 * norm.sf(np.abs(statistics)),
            ci_lower_pointwise=(intercepts - critical * standard_errors) * scales,
            ci_upper_pointwise=(intercepts + critical * standard_errors) * scales,
            rho=rho,
        ), index=pd.Index(y_frame.columns, name='equation'))
    _reject(~np.isfinite(table.to_numpy()).all(axis=1) | (table['standard_error'] <= 0.0),
            y_frame.columns, 'nonfinite/unrepresentable prewhitened output')
    return PrewhitenedInterceptResult(
        table=table, nobs=nobs, n_parameters=n_parameters, hac_lags=lags,
        confidence_level=float(confidence_level),
        influence=pd.DataFrame(influence * scales, index=y_frame.index, columns=y_frame.columns),
        innovations=pd.DataFrame(innovations * scales, index=y_frame.index[1:],
                                 columns=y_frame.columns),
    )

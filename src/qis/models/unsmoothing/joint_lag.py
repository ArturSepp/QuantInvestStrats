"""
Joint own-lag + factor-lag unsmoothing for illiquid / appraisal-based returns.

Single-regression alternative to ``adjust_returns_with_ar`` (in ``ar_lag``)
followed by ``adjust_returns_with_factor_lag`` (in ``factor_lag``). The smoothing
process is fitted as ONE regression,

    r_t = phi_1 r_{t-1} + beta_1 F_{t-1} + e_t,

estimating the own-lag coefficient phi_1 and the lagged-factor beta beta_1
JOINTLY by minimizing the rolling EWMA squared error subject to the supplied bounds,
using the cross-moments of ``compute_ewm_xy_beta_tensor``. The regressors r_{t-1} and
F_{t-1} are correlated in
proportion to the contemporaneous beta. The two-stage pipeline fits phi_1 from
r_t on r_{t-1} alone, so the omitted F_{t-1} biases it
(phi_1^seq - phi_1^joint = beta_1 Cov(r_{t-1}, F_{t-1}) / Var(r_{t-1})), and then
fits beta_1 on the AR-transformed series rather than the raw one. Fitting them
together removes that bias and the order dependence of the two stages.

Inversion. The innovation is e_t = r_t - phi_1 r_{t-1} - beta_1 F_{t-1}. The
unsmoothed return scales the innovation up by 1 / (1 - phi_1) to undo the own-lag
damping and re-times the deferred market response from t-1 to t by adding
beta_1 F_t:

    r_t^u = (e_t + beta_1 F_t) / (1 - phi_1)
          = (r_t - phi_1 r_{t-1}) / (1 - phi_1)
            + beta_1 (F_t - F_{t-1}) / (1 - phi_1).

The first term is Geltner own-lag unsmoothing, the second the Dimson re-timing of
the lagged factor. For fixed coefficients in a stationary process, the correction preserves the
population mean. Time-varying estimated coefficients do not preserve cumulative return exactly. The
implied unsmoothed contemporaneous loading is (beta_0 + beta_1) / (1 - phi_1), an
OUTPUT of the fit.

Limits. ``beta_1 = 0`` recovers pure Geltner, ``phi_1 = 0`` recovers
``adjust_returns_with_factor_lag`` at lag 1, so this engine nests both. phi_1 is
capped below 1 to keep the inflation factor 1 / (1 - phi_1) finite.

Reference:
    Geltner, D. (1991), "Smoothing in Appraisal-Based Returns," Journal of Real
    Estate Finance and Economics 4(3), 327-345.
    Dimson, E. (1979), "Risk Measurement When Shares Are Subject to Infrequent
    Trading," Journal of Financial Economics 7(2), 197-226.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union

from qis.models.linear.ewm import (compute_ewm, compute_ewm_xy_beta_tensor, MeanAdjType,
                                    compute_rolling_mean_adj, InitType, NanBackfill)


UNSMOOTHING_INITIALIZATION_VERSION = 2
UNSMOOTHING_JOINT_BOUNDS_VERSION = 1


def _fit_bounded_joint_betas(x: np.ndarray, y: np.ndarray, span: int,
                             lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Minimize each two-variable EWMA objective over its coefficient box.

    Reuse the linear engine's unconstrained fit and rank safeguard. If a bound
    binds, enumerate the boundary optima, including corners via conditional
    clipping. Scalar moment filters use the same zero seed and gap decay as the
    joint tensor; unit-diagonal scaling keeps the objective independent of units.
    """
    betas = compute_ewm_xy_beta_tensor(
        x=x, y=y, span=span, warmup_period=-1, is_x_correlated=True,
        nan_backfill=NanBackfill.DEFLATED_FFILL)[:, :, 0]
    bounded = np.clip(betas, lower, upper)
    active = np.isfinite(betas).all(axis=1) & np.any(
        (betas < lower) | (betas > upper), axis=1)
    if not active.any():
        return bounded

    products = np.column_stack((x[:, 0] ** 2, x[:, 0] * x[:, 1], x[:, 1] ** 2,
                                x[:, 0] * y, x[:, 1] * y))
    moments = compute_ewm(data=products, span=span, init_type=InitType.ZERO,
                          nan_backfill=NanBackfill.DEFLATED_FFILL)
    for pos in np.flatnonzero(active):
        a, b, c, d, e = moments[pos]
        scale = np.sqrt(np.array([a, c]))
        hessian = np.array([[1.0, b / (scale[0] * scale[1])],
                            [b / (scale[0] * scale[1]), 1.0]])
        cross = np.array([d, e]) / scale
        lo, hi = lower * scale, upper * scale
        candidates = []
        for fixed in (0, 1):
            free = 1 - fixed
            for boundary in (lo[fixed], hi[fixed]):
                if not np.isfinite(boundary):
                    continue
                candidate = np.empty(2)
                candidate[fixed] = boundary
                candidate[free] = np.clip(
                    cross[free] - hessian[free, fixed] * boundary, lo[free], hi[free])
                candidates.append(candidate)
        losses = [z @ hessian @ z - 2.0 * cross @ z for z in candidates]
        best = candidates[int(np.argmin(losses))]
        bounded[pos] = np.clip(best / scale, lower, upper)
    return bounded


def adjust_returns_with_joint_unsmoothing(returns: pd.DataFrame,
                                          factor_returns: pd.Series,
                                          span: int = 40,
                                          mean_adj_type: MeanAdjType = MeanAdjType.NONE,
                                          warmup_period: Optional[int] = 4,
                                          max_ar_coeff: float = 0.75,
                                          min_ar_coeff: Optional[float] = 0.0,
                                          max_factor_beta: Optional[float] = 1.0,
                                          min_factor_beta: Optional[float] = 0.0,
                                          apply_ewma_mean_smoother: bool = False,
                                          return_diagnostics: bool = False
                                          ) -> Union[pd.DataFrame,
                                                     Tuple[pd.DataFrame,
                                                           pd.DataFrame,
                                                           pd.DataFrame]]:
    """Jointly EWMA-fit phi_1 (own-lag) and beta_1 (lagged factor) and unsmooth.

    The fitted model is ``r[t] = phi1 * r[t-1] + beta1 * F[t-1] + e[t]``. The inversion is
    ``r_u[t] = (r[t] - phi1 * r[t-1] + beta1 * (F[t] - F[t-1])) / (1 - phi1)``.

    Fits phi_1 and beta_1 in one regression of r_t on [r_{t-1}, F_{t-1}] with a
    time-varying bounded EWMA beta pair. When a bound binds, the other coefficient
    is re-estimated conditional on it, rather than clipped independently. Lags the pair
    one period to avoid look-ahead, and applies the mean-preserving inversion on
    the raw levels. Single-stage counterpart to running ``adjust_returns_with_ar``
    then ``adjust_returns_with_factor_lag``, without the omitted-variable bias in
    the AR step or the order dependence.

    Args:
        returns: Observed period returns (rows = dates, columns = assets), in the
            reference currency. Every column passed is unsmoothed; exclude liquid
            columns upstream.
        factor_returns: The liquid factor return (e.g. the MATF Equity factor),
            same index and frequency as ``returns``.
        span: EWMA span in observations, default 40 (ten years at QE).
            Monthly callers must explicitly use span=120 and warmup_period=12.
        mean_adj_type: Default NONE fits raw r_t, r_{t-1}, F_{t-1}. Explicit
            EWMA uses a point-in-time ``InitType.ZERO`` seed. Inversion always uses
            raw returns; time-varying coefficients need not preserve the sample mean.
        warmup_period: Number of jointly observable coefficient dates masked once
            per asset. The application lag then makes the next return the first
            corrected one. Default 4; None disables the warmup mask.
        max_ar_coeff: Upper cap on phi_1, strictly < 1, bounding the inflation
            factor 1 / (1 - phi_1). Default 0.75 caps this multiplier at 4x.
        min_ar_coeff: Lower cap on phi_1, default 0.0. None disables.
        max_factor_beta: Upper cap on beta_1, default 1.0. None disables.
        min_factor_beta: Lower cap on beta_1. Default 0.0 floors the lagged-factor
            beta at non-negative (the forward-only re-timing the factor-lag engine
            enforces via its sign tie). Bounds are imposed jointly in the weighted
            fit; None disables this bound.
        apply_ewma_mean_smoother: If True, apply an extra EWMA smoother to each
            coefficient series after the bounded fit. Default False.
        return_diagnostics: If True, return ``(corrected, phi_1, beta_1)``, the
            unsmoothed returns and the two rolling coefficient panels.

    Returns:
        Unsmoothed returns (DataFrame, input shape). If ``return_diagnostics``,
        the tuple ``(corrected, phi_1, beta_1)``.

    Raises:
        ValueError: if ``span <= 0``, ``warmup_period < 0``, ``factor_returns``
            is not an aligned Series, ``max_ar_coeff >= 1``, or bounds are invalid.
    """
    if span <= 0:
        raise ValueError(f"span must be positive, got {span!r}")
    if warmup_period is not None and warmup_period < 0:
        raise ValueError(f"warmup_period must be >= 0 or None, got {warmup_period!r}")
    if not isinstance(factor_returns, pd.Series):
        raise ValueError(f"factor_returns must be a pd.Series, got {type(factor_returns)}")
    if max_ar_coeff >= 1.0:
        raise ValueError(f"max_ar_coeff must be < 1 to keep 1 - phi_1 > 0, got {max_ar_coeff!r}")

    lo_phi = -np.inf if min_ar_coeff is None else min_ar_coeff
    lo_beta = -np.inf if min_factor_beta is None else min_factor_beta
    hi_beta = np.inf if max_factor_beta is None else max_factor_beta
    lower = np.array([lo_phi, lo_beta], dtype=float)
    upper = np.array([max_ar_coeff, hi_beta], dtype=float)
    if (np.isnan(lower).any() or np.isnan(upper).any()
            or np.isposinf(lower).any() or np.isneginf(upper).any()
            or np.any(lower > upper)):
        raise ValueError("coefficient bounds must be ordered and cannot contain NaN")

    cols = returns.columns
    factor = factor_returns.reindex(returns.index)

    # Raw lags drive the mean-preserving inversion.
    r_lag_raw = returns.shift(1)                          # r_{t-1}, per asset
    f_lag_raw = factor.shift(1)                           # F_{t-1}, common

    # Demean the target and the two regressors for the beta fit only.
    if mean_adj_type != MeanAdjType.NONE:
        y_adj = compute_rolling_mean_adj(data=returns, mean_adj_type=mean_adj_type,
                                         span=span, init_type=InitType.ZERO)
        r_lag_adj = compute_rolling_mean_adj(data=r_lag_raw, mean_adj_type=mean_adj_type,
                                             span=span, init_type=InitType.ZERO)
        f_lag_adj = compute_rolling_mean_adj(data=f_lag_raw, mean_adj_type=mean_adj_type,
                                             span=span, init_type=InitType.ZERO)
    else:
        y_adj, r_lag_adj, f_lag_adj = returns, r_lag_raw, f_lag_raw

    f_lag_np = f_lag_adj.to_numpy(float)
    phi1 = pd.DataFrame(index=returns.index, columns=cols, dtype=float)
    beta1 = pd.DataFrame(index=returns.index, columns=cols, dtype=float)
    jointly_observable = pd.DataFrame(False, index=returns.index, columns=cols)
    for col in cols:
        # Asset-specific design [r_{t-1}, F_{t-1}]; the own lag differs per asset,
        # so unlike the all-factor engine the design cannot be shared.
        x = np.column_stack([r_lag_adj[col].to_numpy(float), f_lag_np])
        y = y_adj[col].to_numpy(float)
        joint_valid = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
        jointly_observable[col] = joint_valid

        # Update every regression moment from the same rows. In particular, do not
        # accumulate factor variance before a ragged asset itself is observable.
        x_for_fit = np.where(joint_valid[:, None], x, np.nan)
        y_for_fit = np.where(joint_valid, y, np.nan)
        bt = _fit_bounded_joint_betas(
            x=x_for_fit, y=y_for_fit, span=span, lower=lower, upper=upper)
        has_joint_observation = np.logical_or.accumulate(joint_valid)
        publish = returns[col].notna().to_numpy() & has_joint_observation
        phi1[col] = np.where(publish, bt[:, 0], np.nan)
        beta1[col] = np.where(publish, bt[:, 1], np.nan)

    if warmup_period is not None:
        # Mask exactly the requested number of usable regression dates per asset.
        # The tensor itself is deliberately unmasked, so this is not applied twice.
        coefficient_ready = jointly_observable.cumsum().gt(warmup_period)
        phi1 = phi1.where(coefficient_ready)
        beta1 = beta1.where(coefficient_ready)

    # Excluded estimates must never seed or update the coefficient smoother.
    if apply_ewma_mean_smoother:
        phi1 = compute_ewm(data=phi1, span=span, init_type=InitType.ZERO)
        beta1 = compute_ewm(data=beta1, span=span, init_type=InitType.ZERO)
        phi1 = phi1.clip(lower=lo_phi, upper=max_ar_coeff)
        beta1 = beta1.clip(lower=lo_beta, upper=hi_beta)

    # Lag the coefficients one period (no look-ahead), invert with raw levels.
    phi1_l = phi1.shift(1)
    beta1_l = beta1.shift(1)
    one_minus_phi = (1.0 - phi1_l).clip(lower=1.0 - max_ar_coeff)   # > 0 by the phi_1 cap
    f_diff = factor - f_lag_raw                          # F_t - F_{t-1}, a Series
    numerator = (returns
                 - phi1_l.multiply(r_lag_raw)
                 + beta1_l.multiply(f_diff, axis=0))
    corrected = numerator.divide(one_minus_phi).where(returns.notna())

    if return_diagnostics:
        return corrected, phi1, beta1
    return corrected


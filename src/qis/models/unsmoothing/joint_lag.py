"""
Joint own-lag + factor-lag unsmoothing for illiquid / appraisal-based returns.

Single-regression alternative to ``adjust_returns_with_ar`` (in ``ar_lag``)
followed by ``adjust_returns_with_factor_lag`` (in ``factor_lag``). The smoothing
process is fitted as ONE regression,

    r_t = phi_1 r_{t-1} + beta_1 F_{t-1} + e_t,

estimating the own-lag coefficient phi_1 and the lagged-factor beta beta_1
JOINTLY with the rolling EWMA cross-moment estimator (``compute_ewm_xy_beta_tensor``,
``is_x_correlated=True``), because r_{t-1} and F_{t-1} are correlated in
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
the lagged factor. The correction is mean-preserving (E[F_t - F_{t-1}] = 0 and the
denominator restores the level), so the cumulative return is unchanged. The
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

from qis.utils.np_ops import set_nans_for_warmup_period
from qis.models.linear.ewm import (compute_ewm, compute_ewm_xy_beta_tensor, MeanAdjType,
                                    compute_rolling_mean_adj, InitType, NanBackfill)


def adjust_returns_with_joint_unsmoothing(returns: pd.DataFrame,
                                          factor_returns: pd.Series,
                                          span: int = 40,
                                          mean_adj_type: MeanAdjType = MeanAdjType.EWMA,
                                          warmup_period: Optional[int] = 16,
                                          max_ar_coeff: float = 0.9,
                                          min_ar_coeff: Optional[float] = None,
                                          max_factor_beta: Optional[float] = None,
                                          min_factor_beta: Optional[float] = None,
                                          apply_ewma_mean_smoother: bool = True,
                                          return_diagnostics: bool = False
                                          ) -> Union[pd.DataFrame,
                                                     Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Jointly EWMA-fit phi_1 (own-lag) and beta_1 (lagged factor) and unsmooth.

    Model:      r_t = phi_1 r_{t-1} + beta_1 F_{t-1} + e_t      (fitted jointly)
    Inversion:  r_t^u = (r_t - phi_1 r_{t-1}) / (1 - phi_1)
                        + beta_1 (F_t - F_{t-1}) / (1 - phi_1)

    Fits phi_1 and beta_1 in one regression of r_t on [r_{t-1}, F_{t-1}] with a
    time-varying EWMA beta pair (``compute_ewm_xy_beta_tensor``,
    ``is_x_correlated=True``, the two regressors are correlated), lags the pair
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
        span: EWMA span for the rolling coefficients, in periods of ``returns``.
        mean_adj_type: How to demean r_t, r_{t-1}, F_{t-1} for the beta fit. The
            inversion uses raw levels and stays mean-preserving regardless.
        warmup_period: Initial periods masked before the first valid coefficient,
            then back-filled (mirrors ``adjust_returns_with_ar`` /
            ``adjust_returns_with_factor_lag``). None disables.
        max_ar_coeff: Upper cap on phi_1, strictly < 1, bounding the inflation
            factor 1 / (1 - phi_1). Default 0.9 caps inflation at 10x.
        min_ar_coeff: Lower cap on phi_1. None disables.
        max_factor_beta: Upper cap on beta_1. None disables.
        min_factor_beta: Lower cap on beta_1. Pass 0.0 to floor the lagged-factor
            beta at non-negative (the forward-only re-timing the factor-lag engine
            enforces via its sign tie), at the cost of boundary bias. None
            disables.
        apply_ewma_mean_smoother: If True, apply an extra EWMA smoother to each
            coefficient series after the caps.
        return_diagnostics: If True, return ``(corrected, phi_1, beta_1)``, the
            unsmoothed returns and the two rolling coefficient panels.

    Returns:
        Unsmoothed returns (DataFrame, input shape). If ``return_diagnostics``,
        the tuple ``(corrected, phi_1, beta_1)``.

    Raises:
        ValueError: if ``span <= 0``, ``factor_returns`` is not an aligned Series,
            or ``max_ar_coeff >= 1``.
    """
    if span <= 0:
        raise ValueError(f"span must be positive, got {span!r}")
    if not isinstance(factor_returns, pd.Series):
        raise ValueError(f"factor_returns must be a pd.Series, got {type(factor_returns)}")
    if max_ar_coeff >= 1.0:
        raise ValueError(f"max_ar_coeff must be < 1 to keep 1 - phi_1 > 0, got {max_ar_coeff!r}")

    cols = returns.columns
    factor = factor_returns.reindex(returns.index)

    # Raw lags drive the mean-preserving inversion.
    r_lag_raw = returns.shift(1)                          # r_{t-1}, per asset
    f_lag_raw = factor.shift(1)                           # F_{t-1}, common

    # Demean the target and the two regressors for the beta fit only.
    if mean_adj_type != MeanAdjType.NONE:
        y_adj = compute_rolling_mean_adj(data=returns, mean_adj_type=mean_adj_type,
                                         span=span, init_type=InitType.MEAN)
        r_lag_adj = compute_rolling_mean_adj(data=r_lag_raw, mean_adj_type=mean_adj_type,
                                             span=span, init_type=InitType.MEAN)
        f_lag_adj = compute_rolling_mean_adj(data=f_lag_raw, mean_adj_type=mean_adj_type,
                                             span=span, init_type=InitType.MEAN)
    else:
        y_adj, r_lag_adj, f_lag_adj = returns, r_lag_raw, f_lag_raw

    f_lag_np = f_lag_adj.to_numpy(float)
    phi1 = pd.DataFrame(index=returns.index, columns=cols, dtype=float)
    beta1 = pd.DataFrame(index=returns.index, columns=cols, dtype=float)
    for col in cols:
        # Asset-specific design [r_{t-1}, F_{t-1}]; the own lag differs per asset,
        # so unlike the all-factor engine the design cannot be shared.
        x = np.column_stack([r_lag_adj[col].to_numpy(float), f_lag_np])
        y = y_adj[col].to_numpy(float)
        bt = compute_ewm_xy_beta_tensor(
            x=x, y=y, span=span,
            warmup_period=warmup_period if warmup_period is not None else 20,
            is_x_correlated=True,                        # joint 2x2 cross-moment inverse
            nan_backfill=NanBackfill.FFILL,
        )                                                # shape [t, 2, 1]
        valid = returns[col].notna().to_numpy()
        phi1[col] = np.where(valid, bt[:, 0, 0], np.nan)
        beta1[col] = np.where(valid, bt[:, 1, 0], np.nan)

    # Caps. phi_1 capped below 1 for inversion stability; beta_1 optional.
    lo_phi = -np.inf if min_ar_coeff is None else min_ar_coeff
    phi1 = phi1.clip(lower=lo_phi, upper=max_ar_coeff)
    if max_factor_beta is not None or min_factor_beta is not None:
        lo_b = -np.inf if min_factor_beta is None else min_factor_beta
        hi_b = np.inf if max_factor_beta is None else max_factor_beta
        beta1 = beta1.clip(lower=lo_b, upper=hi_b)

    if apply_ewma_mean_smoother:
        phi1 = compute_ewm(data=phi1, span=span)
        beta1 = compute_ewm(data=beta1, span=span)
        phi1 = phi1.clip(lower=lo_phi, upper=max_ar_coeff)   # smoother can overshoot the cap

    if warmup_period is not None:
        phi1 = set_nans_for_warmup_period(a=phi1, warmup_period=warmup_period).reindex(index=returns.index).bfill()
        beta1 = set_nans_for_warmup_period(a=beta1, warmup_period=warmup_period).reindex(index=returns.index).bfill()

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
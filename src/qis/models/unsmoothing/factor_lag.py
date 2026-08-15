"""
Factor-lag (Dimson) unsmoothing for illiquid / appraisal-based return series.

Companion to ``adjust_returns_with_ar`` (the own-lag AR(q) engine in
``qis.models.unsmoothing``). The AR engine removes smoothing that manifests
as OWN autocorrelation; this engine removes smoothing that manifests as a
LAGGED response to a liquid factor (typically equity), which the own-lag AR
cannot see (a fund-of-funds with NAV timing has near-zero own autocorrelation
but a real lagged-equity beta).

Model. With the observed return following a one-factor representation whose
true loading is split by smoothing across the contemporaneous and lagged
factor,

    r_obs_t = a + b_0 F_t + sum_{l=1..p} b_l F_{t-l} + e_t,

the true (summed) loading is beta_D = b_0 + sum_{l>=1} b_l. Moving the lagged
response to contemporaneous gives the unsmoothed return

    r_corrected_t = r_obs_t + sum_{l=1..p} b_l (F_t - F_{t-l}).

The correction is mean-preserving (E[F_t - F_{t-l}] = 0) and lifts the
contemporaneous loading to beta_D. The key property for the ROSAA pipeline:
a plain contemporaneous regression of r_corrected on F returns beta_D, so the
existing HCGL / factor-covariance estimator recovers the true loading with no
change once it consumes the corrected series.

Betas are rolling EWMA (regime-adaptive, same primitives as the AR engine),
fitted jointly over [F_t, F_{t-1}, ..., F_{t-p}] (the factor and its lags are
mutually correlated), lagged one period before the shift to avoid look-ahead,
sign-tied to the contemporaneous loading (a lagged response carries the
contemporaneous sign), and the lagged-beta sum is optionally capped.

Reference:
    Dimson, E. (1979), "Risk Measurement When Shares Are Subject to Infrequent
    Trading," Journal of Financial Economics 7(2), 197-226.
    Couts, S.J., Goncalves, A.S., Rossi, A. (2024), "Unsmoothing Returns of
    Illiquid Funds," Review of Financial Studies 37(7), 2110-2155.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union

from qis.utils.np_ops import set_nans_for_warmup_period
from qis.models.linear.ewm import (compute_ewm, compute_ewm_xy_beta_tensor, MeanAdjType,
                                    compute_rolling_mean_adj, InitType, NanBackfill)


def adjust_returns_with_factor_lag(returns: pd.DataFrame,
                                   factor_returns: pd.Series,
                                   factor_lag_order: int = 1,
                                   span: int = 40,
                                   mean_adj_type: MeanAdjType = MeanAdjType.EWMA,
                                   warmup_period: Optional[int] = 16,
                                   max_value_for_beta: Optional[float] = None,
                                   min_value_for_beta: Optional[float] = None,
                                   sign_tie_to_contemporaneous: bool = True,
                                   sign_tie_tol: float = 0.10,
                                   apply_ewma_mean_smoother: bool = True,
                                   return_diagnostics: bool = False
                                   ) -> Union[pd.DataFrame,
                                              Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Unsmooth observed returns by shifting the lagged-factor response to contemporaneous.

    Regresses each asset return on the factor and its first ``factor_lag_order``
    lags with a time-varying EWMA beta vector, then shifts:

        r_corrected_t = r_obs_t + sum_{l=1..p} b_{l,t-1} (F_t - F_{t-l})

    where ``b_{l,t-1}`` are the lagged (shift-1) rolling betas, aligning with the
    lagged factor in the correction and avoiding look-ahead. The lags are fitted
    JOINTLY via the rolling cross-moment inverse (``compute_ewm_xy_beta_tensor``,
    ``is_x_correlated=True``) because the factor and its lags are mutually
    correlated; independent univariate fits would double-count the shared factor
    variance.

    Args:
        returns: Observed period returns (rows = dates, columns = assets), in the
            reference currency. Columns not requiring unsmoothing should be
            excluded by the caller; every column passed is corrected.
        factor_returns: The liquid factor return (e.g. the MATF Equity factor),
            same index and frequency as ``returns``. A single Series in Phase 1;
            a per-asset factor mapping is a later extension.
        factor_lag_order: Number of factor lags p (1 = lag-1; p > 1 ok).
        span: EWMA span for the rolling beta, in periods of ``returns``. For
            quarterly private-market data use ~40 (10y); a p >= 2 fit needs more
            effective observations.
        mean_adj_type: How to demean returns / factor lags before beta
            estimation. The shift itself uses the raw factor differences, so the
            correction stays mean-preserving regardless.
        warmup_period: Initial periods masked before the first valid beta, then
            back-filled (mirrors ``adjust_returns_with_ar``). None disables.
        max_value_for_beta: Upper bound on the lagged-beta SUM
            ``L = sum_{l>=1} b_l`` (the shifted amount). The lagged betas are
            rescaled by ``bound / L`` to preserve direction. Bounds the
            vol-inflation contributed by the shift. Pass None to disable.
        min_value_for_beta: Lower bound on ``L``. Pass None to disable.
        sign_tie_to_contemporaneous: If True, force each lagged beta to carry the
            sign of the contemporaneous beta ``b_0`` via a deadzone: a lagged beta
            is zeroed only when it is opposite-signed beyond ``sign_tie_tol``
            (small opposite-sign noise is left untouched). For positive equity
            exposure (``b_0 > 0``) this is non-negativity on the lagged betas, so
            a noisy negative lag cannot spuriously DEFLATE the recovered vol.
        sign_tie_tol: Deadzone width for the sign tie. Ignored when
            ``sign_tie_to_contemporaneous`` is False.
        apply_ewma_mean_smoother: If True, apply an extra EWMA smoother to each
            beta series after the sign tie and cap.
        return_diagnostics: If True, return ``(corrected, beta_d, r2)`` where
            ``beta_d`` is the per-period summed loading ``b_0 + L`` and ``r2`` is
            the EWMA R^2 of the contemporaneous + lagged factor fit.

    Returns:
        Corrected returns (DataFrame, input shape). If ``return_diagnostics``,
        the tuple ``(corrected, beta_d, r2)``.

    Raises:
        ValueError: if ``factor_lag_order < 1`` or ``factor_returns`` is not a
            Series aligned to ``returns``.
    """
    if factor_lag_order < 1:
        raise ValueError(f"factor_lag_order must be >= 1, got {factor_lag_order!r}")
    if not isinstance(factor_returns, pd.Series):
        raise ValueError(f"factor_returns must be a pd.Series, got {type(factor_returns)}")

    cols = returns.columns
    p = factor_lag_order
    factor = factor_returns.reindex(returns.index)

    # Raw factor lags (used by the mean-preserving shift).
    f_raw = [factor.shift(l) for l in range(p + 1)]      # l = 0..p

    # Demean y and the factor lags consistently for beta estimation only.
    if mean_adj_type != MeanAdjType.NONE:
        f_adj = [compute_rolling_mean_adj(data=f_raw[l], mean_adj_type=mean_adj_type,
                                          span=span, init_type=InitType.MEAN)
                 for l in range(p + 1)]
        y_adj = compute_rolling_mean_adj(data=returns, mean_adj_type=mean_adj_type,
                                         span=span, init_type=InitType.MEAN)
    else:
        f_adj, y_adj = f_raw, returns

    # Design matrix [F_t, F_{t-1}, ..., F_{t-p}] is shared across assets.
    x = np.column_stack([f_adj[l].to_numpy(float) for l in range(p + 1)])

    betas = [pd.DataFrame(index=returns.index, columns=cols, dtype=float)
             for _ in range(p + 1)]
    for col in cols:
        y = y_adj[col].to_numpy(float)
        bt = compute_ewm_xy_beta_tensor(
            x=x, y=y, span=span,
            warmup_period=warmup_period if warmup_period is not None else 20,
            is_x_correlated=True,                        # joint (p+1)x(p+1) inverse
            nan_backfill=NanBackfill.FFILL,
        )                                                # shape [t, p+1, 1]
        valid = returns[col].notna().to_numpy()
        for l in range(p + 1):
            betas[l][col] = np.where(valid, bt[:, l, 0], np.nan)

    # Sign-tie lagged betas to the contemporaneous loading (deadzone).
    if sign_tie_to_contemporaneous:
        s0 = np.sign(betas[0])
        for l in range(1, p + 1):
            keep = (s0 * betas[l]) > -sign_tie_tol
            betas[l] = betas[l].where(keep, other=0.0)

    # Cap the lagged-beta sum L, rescaling lagged betas to preserve direction.
    if (max_value_for_beta is not None or min_value_for_beta is not None) and p >= 1:
        lag_sum = sum(betas[1:])
        lo = -np.inf if min_value_for_beta is None else min_value_for_beta
        hi = np.inf if max_value_for_beta is None else max_value_for_beta
        lag_sum_clipped = lag_sum.clip(lower=lo, upper=hi)
        with np.errstate(divide='ignore', invalid='ignore'):
            scale = (lag_sum_clipped / lag_sum).where(lag_sum.abs() > 1e-12, other=1.0)
        betas = [betas[0]] + [b.multiply(scale) for b in betas[1:]]

    if apply_ewma_mean_smoother:
        betas = [compute_ewm(data=b, span=span) for b in betas]

    if warmup_period is not None:
        betas = [set_nans_for_warmup_period(a=b, warmup_period=warmup_period)
                 .reindex(index=returns.index).bfill() for b in betas]

    # Shift: lagged betas (no look-ahead) times raw factor differences.
    betas_l = [b.shift(1) for b in betas]
    correction = pd.DataFrame(0.0, index=returns.index, columns=cols)
    for l in range(1, p + 1):
        diff = f_raw[0] - f_raw[l]                       # (F_t - F_{t-l}), a Series
        correction = correction.add(betas_l[l].multiply(diff, axis=0).fillna(0.0))
    corrected = (returns + correction).where(returns.notna())

    if return_diagnostics:
        beta_d = betas[0] + sum(betas[1:])               # summed loading b_0 + L
        fitted = sum(betas[l].multiply(f_adj[l], axis=0) for l in range(p + 1))
        resid = y_adj - fitted
        resid_var = compute_ewm(data=resid.pow(2.0), span=span)
        y_var = compute_ewm(data=y_adj.pow(2.0), span=span)
        with np.errstate(divide='ignore', invalid='ignore'):
            r2 = (1.0 - resid_var.divide(y_var)).where(y_var > 0.0)
        r2 = r2.clip(lower=0.0, upper=1.0)
        return corrected, beta_d, r2

    return corrected

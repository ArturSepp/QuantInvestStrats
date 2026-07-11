"""
Return unsmoothing for illiquid / appraisal-based NAV series.

Two methodologies for recovering "true" returns from observed (smoothed)
NAV returns:

1. ``adjust_returns_with_ar``: rolling EWMA AR(q) beta vector with theta-sum
   clipping. The single, canonical AR-unsmoothing engine for any lag order.
   ``ar_order=1`` is the rolling AR(1) (adapts to regime changes, integrates
   with qis EWM primitives); ``ar_order>=2`` adds the higher-order appraisal
   lags that AR(1) misses on heavily smoothed private-market sleeves.
   ``unsmooth_returns_ar1_ewma`` / ``compute_ar1_unsmoothed_prices`` are kept
   as backward-compatible shims over this engine.

2. ``unsmooth_returns_glm``: static Getmansky-Lo-Makarov AR(q) fit.
   Classical methodology from the 2004 JFE paper. Useful for academic
   reproducibility or when a single smoothing-parameter summary is needed.

Both methods assume the observed return is a weighted combination of current
and lagged true returns with weights summing to one:

    r_obs_t = theta_0 * r_true_t + theta_1 * r_true_{t-1} + ...

The rolling AR engine lets theta vary over time; the GLM method allows q > 1
but assumes theta is constant over the sample.

Note on the single-engine consolidation: for ``ar_order=1`` the rolling
engine reproduces the previous panel-vectorised AR(1) path to ~1e-4 at any
given estimation date (the residual difference is an early-warmup transient
that decays within ~span periods as the EWMA forgets its initial condition).
The production-relevant parity gate is therefore the current-date max|delta-beta|,
not the full-history unsmoothed vol.

Guards (opt-in; defaults reproduce the previous behaviour exactly):

    ``insufficient_data``  Controls what happens when a column comes out entirely
        NaN. ``set_nans_for_warmup_period`` discards the first ``warmup_period``
        FINITE beta values and blanks the whole column when fewer remain, and the
        rolling tensor already masks its own first ``warmup_period`` rows, so the
        returns frame needs ``2 * warmup_period + 2`` ROWS before any beta
        survives (see ``min_obs_for_ar_unsmoothing``). Individual columns can also
        degenerate when they are entirely NaN. The default ``NAN`` keeps the
        historical silent all-NaN column. ``RAISE`` reports the offending columns.
        ``PASSTHROUGH`` returns those columns unsmoothed.

    ``check_denominator``  The inversion ``r_true = numerator / (1 - theta_sum)``
        requires ``theta_sum < 1``. Above that the denominator is non-positive
        and the unsmoothed series flips sign rather than inflating. The default
        ``max_value_for_beta=0.75`` holds the denominator at or above 0.25, so
        this check is inert under the default arguments. It fires only when the
        cap is disabled, where the previous behaviour was to return sign-flipped
        values with no error.

    ``validate_inputs``  Type, shape, index and parameter-range checks that
        raise ``ValueError`` carrying the offending value.

Reference:
    Getmansky, M., Lo, A.W., and Makarov, I. (2004),
    "An Econometric Model of Serial Correlation and Illiquidity in Hedge Fund Returns,"
    Journal of Financial Economics, 74(3), 529-609.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union

from qis.utils.np_ops import set_nans_for_warmup_period
from qis.perfstats.returns import to_returns, returns_to_nav
from qis.models.linear.ewm import (compute_ewm, compute_ewm_xy_beta_tensor, MeanAdjType,
                                    compute_rolling_mean_adj, InitType, NanBackfill)


# compute_ewm_xy_beta_tensor's own default when warmup_period is None
TENSOR_DEFAULT_WARMUP: int = 20
# with warmup_period=None the outer warmup masking is skipped, so only the tensor warmup binds,
# and the inversion consumes one further shift. verified for ar_order = 1, 2.
WARMUP_NONE_MIN_OBS_OFFSET: int = 3
# theta_sum >= 1 makes the inversion denominator non-positive: the unsmoothed series flips sign
# rather than inflating. observed at theta_sum = 1.03 for AR(1) rho >= 0.9 with the beta smoother
# off and max_value_for_beta=None. the default cap of 0.75 keeps the denominator at or above 0.25.
DENOM_FLOOR: float = 1e-8


class InsufficientData(str, Enum):
    """Policy for columns with too few observations to identify the AR(q) beta.

    NAN: return an all-NaN column with no error. This is the historical behaviour
        and remains the default so that existing callers are unaffected.
    RAISE: raise ValueError naming the columns and their observation counts.
    PASSTHROUGH: return the observed returns unchanged for those columns.
    """
    NAN = 'nan'
    RAISE = 'raise'
    PASSTHROUGH = 'passthrough'


def min_obs_for_ar_unsmoothing(ar_order: int,
                               warmup_period: Optional[int],
                               ) -> int:
    """Minimum number of ROWS in the returns frame for which AR(q) unsmoothing yields output.

    The rolling beta tensor masks its first ``warmup_period`` rows, and
    ``set_nans_for_warmup_period`` then discards the first ``warmup_period`` finite betas,
    blanking the column when fewer remain. The inversion consumes one further shift.
    The bound does not depend on ``ar_order``.

    This is a bound on the row count, not on a column's finite observation count. A column
    with few finite returns inside a long frame still receives betas, because the tensor
    backfills across the full index.

    Args:
        ar_order: Lag order q, must be >= 1.
        warmup_period: Periods masked before the first valid beta. None defers to the
            tensor default of ``TENSOR_DEFAULT_WARMUP``.

    Returns:
        Minimum number of rows in the returns frame.

    Raises:
        ValueError: If ``ar_order`` < 1 or ``warmup_period`` < 0.
    """
    if ar_order < 1:
        raise ValueError(f"ar_order must be >= 1, got {ar_order!r}")
    if warmup_period is None:
        return TENSOR_DEFAULT_WARMUP + WARMUP_NONE_MIN_OBS_OFFSET
    if warmup_period < 0:
        raise ValueError(f"warmup_period must be >= 0 or None, got {warmup_period!r}")
    return 2 * warmup_period + 2


def _validate_ar_inputs(returns: pd.DataFrame,
                        ar_order: int,
                        span: int,
                        warmup_period: Optional[int],
                        max_value_for_beta: Optional[float],
                        min_value_for_beta: Optional[float],
                        non_negative_tol: float,
                        ) -> None:
    """Validate the arguments of ``adjust_returns_with_ar``, raising on the offending value."""
    if not isinstance(returns, pd.DataFrame):
        raise ValueError(f"returns must be a pd.DataFrame, got {type(returns).__name__}. "
                         f"pass a Series as returns.to_frame()")
    if returns.empty:
        raise ValueError("returns is empty")
    if returns.columns.has_duplicates:
        dupes = returns.columns[returns.columns.duplicated()].tolist()
        raise ValueError(f"returns has duplicate columns {dupes}")
    if not returns.index.is_unique:
        raise ValueError("returns index is not unique")
    if not returns.index.is_monotonic_increasing:
        raise ValueError("returns index must be sorted ascending")
    if ar_order < 1:
        raise ValueError(f"ar_order must be >= 1, got {ar_order!r}")
    if span < 1:
        raise ValueError(f"span must be >= 1, got {span!r}")
    if warmup_period is not None and warmup_period < 0:
        raise ValueError(f"warmup_period must be >= 0 or None, got {warmup_period!r}")
    if (min_value_for_beta is not None and max_value_for_beta is not None
            and min_value_for_beta > max_value_for_beta):
        raise ValueError(f"min_value_for_beta {min_value_for_beta!r} exceeds "
                         f"max_value_for_beta {max_value_for_beta!r}")
    if non_negative_tol < 0.0:
        raise ValueError(f"non_negative_tol must be >= 0, got {non_negative_tol!r}")


def _degenerate_columns(unsmoothed: pd.DataFrame) -> List[str]:
    """Return the columns of the unsmoothed frame that carry no finite observation."""
    return [c for c in unsmoothed.columns if int(unsmoothed[c].notna().sum()) == 0]


def _check_denominator(denom: pd.DataFrame) -> None:
    """Raise when the inversion denominator 1 - theta_sum is non-positive.

    The Getmansky-Lo-Makarov inversion r_true = numerator / (1 - theta_sum) requires
    theta_sum < 1. At theta_sum >= 1 the denominator is non-positive and the unsmoothed
    series flips sign instead of inflating, which is silent nonsense rather than a large number.
    """
    unsafe = denom <= DENOM_FLOOR
    if not bool(unsafe.to_numpy().any()):
        return
    offenders = []
    for col in denom.columns:
        mask = unsafe[col].fillna(False)
        if bool(mask.any()):
            first = denom.index[mask][0]
            theta_sum = 1.0 - float(denom.loc[first, col])
            offenders.append(f"{col!r} at {first} (theta_sum={theta_sum:.4f})")
    raise ValueError(f"inversion denominator 1 - theta_sum <= {DENOM_FLOOR} for "
                     f"{'; '.join(offenders)}. the unsmoothed series would flip sign. "
                     f"set max_value_for_beta below 1.0 to bound theta_sum, or pass "
                     f"check_denominator=False to restore the previous behaviour")


# =============================================================================
# METHOD 1: ROLLING EWMA AR(q) UNSMOOTHING (single engine, preferred default)
# =============================================================================

def _active_set_nonneg(sxx: np.ndarray, sxy: np.ndarray,
                       tol: float = 0.0, ridge: float = 1e-10) -> np.ndarray:
    """Deadzone non-negative solve on EWMA normal equations.

    Minimises 0.5 b'Sxx b - Sxy'b subject to a *deadzone* sign rule: a lag is
    forced to zero only if its unconstrained coefficient is materially negative
    (below -tol); a lag in [-tol, 0) keeps its (small, noise-level) negative
    value. With tol=0 this is ordinary non-negative LS (Lawson-Hanson active
    set), so a strongly negative higher lag collapses to the lower-order
    univariate solution; with tol>0 the small negative excursions of a
    near-zero lag are left untouched, which avoids the upward truncation bias on
    funds whose true higher-lag coefficient is ~0.

    Solved by active-set descent: solve on the free lags, drop the most-negative
    lag below -tol, re-solve, repeat. A tiny relative ridge keeps the
    sub-systems solvable.
    """
    q = sxx.shape[0]
    sxx = sxx + ridge * (np.trace(sxx) / q + ridge) * np.eye(q)
    free = list(range(q))
    for _ in range(q + 1):
        if not free:
            return np.zeros(q)
        idx = np.array(free)
        sub = sxx[np.ix_(idx, idx)]
        try:
            bf = np.linalg.solve(sub, sxy[idx])
        except Exception:
            bf = np.linalg.lstsq(sub, sxy[idx], rcond=None)[0]
        cand = np.zeros(q)
        cand[idx] = bf
        below = [(cand[i], i) for i in free if cand[i] < -tol]
        if not below:
            return cand                     # dropped lags are zero in cand
        free.remove(min(below)[1])          # drop most-negative violator, re-solve
    return np.zeros(q)


def _rolling_nonneg_betas(x_lags, y_adj, cols, ar_order, span, warmup_period,
                          tol: float = 0.0):
    """Per-period deadzone-non-negative AR(q) betas from EWMA cross-moments.

    Builds the EWMA second-moment matrix Sxx_t (q x q) and cross-moment Sxy_t
    (q,) from the demeaned lags / target via ``compute_ewm`` on the element-wise
    products (same span -> same lambda as the unconstrained tensor), then solves
    a per-period deadzone non-negative LS (``_active_set_nonneg``). When the
    unconstrained higher-order lag is materially negative (< -tol) it is dropped
    and the lower lags re-solved — for q = 2 this is exactly the collapse to the
    univariate AR(1) coefficient — while small (noise-level) negative excursions
    are kept, so near-zero higher lags are not biased upward.
    """
    q = ar_order
    betas = [pd.DataFrame(index=y_adj.index, columns=cols, dtype=float)
             for _ in range(q)]
    nt = len(y_adj.index)
    wp = -1 if warmup_period is None else warmup_period
    for col in cols:
        xc = [x_lags[i][col] for i in range(q)]
        yc = y_adj[col]
        sxx = [[compute_ewm(data=(xc[i] * xc[j]), span=span).to_numpy(float)
                for j in range(q)] for i in range(q)]
        sxy = [compute_ewm(data=(xc[i] * yc), span=span).to_numpy(float)
               for i in range(q)]
        bmat = np.full((nt, q), np.nan)
        for t in range(nt):
            if t <= wp:
                continue
            A = np.empty((q, q))
            ok = True
            for i in range(q):
                for j in range(q):
                    v = sxx[i][j][t]
                    if not np.isfinite(v):
                        ok = False; break
                    A[i, j] = v
                if not ok:
                    break
            if not ok:
                continue
            bvec = np.array([sxy[i][t] for i in range(q)])
            if not np.all(np.isfinite(bvec)) or np.min(np.diag(A)) <= 1e-12:
                continue
            bmat[t] = _active_set_nonneg(A, bvec, tol=tol)
        for i in range(q):
            betas[i][col] = bmat[:, i]
    return betas


def adjust_returns_with_ar(returns: pd.DataFrame,
                           ar_order: int = 2,
                           span: int = 20,
                           mean_adj_type: MeanAdjType = MeanAdjType.EWMA,
                           warmup_period: Optional[int] = 10,
                           max_value_for_beta: Optional[float] = 0.75,
                           min_value_for_beta: Optional[float] = -0.25,
                           apply_ewma_mean_smoother: bool = True,
                           non_negative: bool = False,
                           non_negative_tol: float = 0.0,
                           return_diagnostics: bool = False,
                           insufficient_data: InsufficientData = InsufficientData.NAN,
                           check_denominator: bool = True,
                           validate_inputs: bool = True,
                           ) -> Union[pd.DataFrame,
                                      Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Unsmooth observed returns with a rolling EWMA AR(q) beta vector.

    Single implementation for any lag order. The observed return is regressed
    on its first ``ar_order`` lags with a time-varying EWMA beta vector, then
    inverted period-by-period:

        r_obs_t  = sum_i b_i,t * r_obs_{t-i} + eps_t,            i = 1..q
        r_true_t = (r_obs_t - sum_i b_i,{t-1} * r_obs_{t-i}) / (1 - sum_i b_i,{t-1})

    Betas are the lagged (shift-1) estimates, aligning with the lagged returns
    in the numerator and avoiding look-ahead. The lags are fitted JOINTLY via
    the rolling cross-moment inverse b_t = Sxx_t^{-1} Sxy_t
    (``compute_ewm_xy_beta_tensor``); for q >= 2 this is required because the
    lags of an autocorrelated series are mutually correlated, so independent
    univariate fits would double-count the shared autocorrelation. For q = 1
    it reduces to the scalar rolling EWMA beta.

    Denominator stability is governed by the smoothing total
    ``theta_sum = sum_i b_i``, so clipping is applied to the SUM (not to each
    beta): if theta_sum leaves [min_value_for_beta, max_value_for_beta] the
    beta vector is rescaled by boundary / theta_sum, preserving direction and
    bounding the vol-inflation factor 1 / (1 - theta_sum). For q = 1 this is
    identical to clipping the single beta, so the bounds carry the same meaning
    as in the legacy AR(1) path.

    With the default arguments the output is identical to the pre-guard
    implementation, including the all-NaN column returned for a series shorter
    than ``min_obs_for_ar_unsmoothing(ar_order, warmup_period)``.

    Args:
        returns: Observed period returns (rows = dates, columns = assets).
        ar_order: Lag order q (1 = rolling AR(1); 2 = rolling AR(2); q > 2 ok).
        span: EWMA span for the rolling beta. Default 20. A q >= 2 fit needs
            more effective observations than AR(1) — for quarterly data use
            ~40 (10y); for monthly ~24-36.
        mean_adj_type: How to demean returns/lags before beta estimation.
        warmup_period: Initial periods masked before the first valid beta.
            Default 10; raise to ~16 for q >= 2 (the q x q estimate is noisier
            early). The underlying tensor also self-guards via its own warmup.
        max_value_for_beta: Upper bound on theta_sum (sum of betas). Bounds the
            inversion denominator away from zero. Pass None to disable. For
            q = 1 this is the upper beta cap (legacy semantics).
        min_value_for_beta: Lower bound on theta_sum. Permits genuine
            mean-reversion / symmetric treatment of estimation noise. Pass None
            to disable; pass 0.0 for one-sided clipping (legacy behaviour).
        apply_ewma_mean_smoother: If True, apply an extra EWMA smoother to each
            beta series after the sum-clip.
        non_negative: If True, constrain every lag beta to be non-negative via
            per-period NNLS on the EWMA normal equations (mirrors the GLM
            theta >= 0 smoothing weights). A spurious negative higher-order lag
            self-reduces: for q = 2 the fit collapses to the univariate AR(1)
            coefficient rather than an oscillating b1 > 1, b2 < 0 solution. With
            non_negative=True the theta_sum lower bound (min_value_for_beta) is
            moot since theta_sum >= 0 by construction; the upper cap still
            applies. Applied only to flagged (smoothed) assets whose lag-1 is
            positive, so the b1 >= 0 constraint does not bind in practice.
        non_negative_tol: Deadzone width for the non-negativity. A lag is
            forced to zero only when its unconstrained coefficient is below
            -non_negative_tol; lags in [-tol, 0) keep their small negative
            value. tol=0 is strict non-negativity. A small tol (~0.1) catches a
            genuinely destabilising negative higher lag (e.g. an oscillating
            b2 ~ -0.45) while leaving the negative noise of a near-zero lag
            untouched, removing the upward bias that strict non-negativity puts
            on funds whose true higher-lag coefficient is ~0. Ignored when
            non_negative is False.
        return_diagnostics: If True, return (unsmoothed, betas, r2) where
            ``betas`` is the applied per-period theta_sum (sum of betas; equals
            the single beta for q = 1) and ``r2`` is the EWMA R^2 of the fit.
        insufficient_data: Policy for columns that come out entirely NaN, which
            happens when the frame has fewer than ``min_obs_for_ar_unsmoothing``
            rows or when a column is itself all NaN. Default NAN preserves the
            historical silent all-NaN column. RAISE reports the columns, their
            finite input counts and the row count. PASSTHROUGH returns their
            observed returns unchanged, with a theta_sum of 0.0 and an r2 of NaN
            in the diagnostics.
        check_denominator: If True, raise when the inversion denominator
            1 - theta_sum is non-positive. Inert under the default
            ``max_value_for_beta=0.75``, which keeps the denominator at or above
            0.25. Pass False to restore the previous unchecked division.
        validate_inputs: If True, validate types, index and parameter ranges.

    Returns:
        Unsmoothed returns (DataFrame, input shape). If ``return_diagnostics``,
        the tuple (unsmoothed, betas, r2).

    Raises:
        ValueError: If ``validate_inputs`` and an argument is invalid.
        ValueError: If ``insufficient_data`` is RAISE and any column is too short.
        ValueError: If ``check_denominator`` and 1 - theta_sum is non-positive.
    """
    if validate_inputs:
        _validate_ar_inputs(returns=returns, ar_order=ar_order, span=span,
                            warmup_period=warmup_period, max_value_for_beta=max_value_for_beta,
                            min_value_for_beta=min_value_for_beta,
                            non_negative_tol=non_negative_tol)
    elif ar_order < 1:
        raise ValueError(f"ar_order must be >= 1, got {ar_order}")

    min_obs = min_obs_for_ar_unsmoothing(ar_order=ar_order, warmup_period=warmup_period)
    if insufficient_data == InsufficientData.RAISE and len(returns) < min_obs:
        raise ValueError(f"AR({ar_order}) unsmoothing needs {min_obs} rows with "
                         f"warmup_period={warmup_period!r}, got {len(returns)}. every column "
                         f"would come out all-NaN. pass insufficient_data="
                         f"InsufficientData.PASSTHROUGH to leave them unsmoothed")

    core = returns
    cols = core.columns
    lags_raw = [core.shift(i + 1) for i in range(ar_order)]   # raw lags for the inversion

    # Demean y and the lags consistently (single demean; the tensor does not demean).
    if mean_adj_type != MeanAdjType.NONE:
        y_adj = compute_rolling_mean_adj(data=core, mean_adj_type=mean_adj_type,
                                         span=span, init_type=InitType.MEAN)
        x_lags = [compute_rolling_mean_adj(data=lag, mean_adj_type=mean_adj_type,
                                           span=span, init_type=InitType.MEAN)
                  for lag in lags_raw]
    else:
        y_adj, x_lags = core, lags_raw

    # Per-asset rolling q x q EWMA regression: x = [r_{t-1}, ..., r_{t-q}], y = r_t.
    if non_negative:
        # Non-negative per-period fit (NNLS on EWMA normal equations); a spurious
        # negative higher lag collapses the fit to the lower-order solution.
        betas = _rolling_nonneg_betas(x_lags=x_lags, y_adj=y_adj, cols=cols,
                                      ar_order=ar_order, span=span,
                                      warmup_period=warmup_period,
                                      tol=non_negative_tol)
    else:
        betas = [pd.DataFrame(index=core.index, columns=cols, dtype=float)
                 for _ in range(ar_order)]
        for col in cols:
            x = np.column_stack([x_lags[i][col].to_numpy(float) for i in range(ar_order)])
            y = y_adj[col].to_numpy(float)
            betas_ts = compute_ewm_xy_beta_tensor(
                x=x, y=y, span=span,
                warmup_period=warmup_period if warmup_period is not None else TENSOR_DEFAULT_WARMUP,
                is_x_correlated=(ar_order > 1),     # q x q inverse for q >= 2; scalar for q = 1
                nan_backfill=NanBackfill.FFILL,
            )                                       # shape [t, q, 1]
            for i in range(ar_order):
                betas[i][col] = betas_ts[:, i, 0]

    # Clip on the SUM, rescaling the vector to preserve direction.
    if max_value_for_beta is not None or min_value_for_beta is not None:
        ts = sum(betas)
        lo = -np.inf if min_value_for_beta is None else min_value_for_beta
        hi = np.inf if max_value_for_beta is None else max_value_for_beta
        ts_clipped = ts.clip(lower=lo, upper=hi)
        with np.errstate(divide='ignore', invalid='ignore'):
            scale = (ts_clipped / ts).where(ts.abs() > 1e-12, other=1.0)
        betas = [b.multiply(scale) for b in betas]

    if apply_ewma_mean_smoother:
        betas = [compute_ewm(data=b, span=span) for b in betas]

    if warmup_period is not None:
        betas = [set_nans_for_warmup_period(a=b, warmup_period=warmup_period)
                 .reindex(index=core.index).bfill() for b in betas]

    # Invert the AR(q) smoothing using lagged betas (align with lagged returns).
    betas_l = [b.shift(1) for b in betas]
    numerator = core.copy()
    for i in range(ar_order):
        numerator = numerator - lags_raw[i].multiply(betas_l[i])
    denom = 1.0 - sum(betas_l)
    if check_denominator:
        _check_denominator(denom)
    unsmoothed = numerator.divide(denom)

    skipped: List[str] = []
    if insufficient_data != InsufficientData.NAN:
        skipped = _degenerate_columns(unsmoothed)
        if skipped and insufficient_data == InsufficientData.RAISE:
            counts = {c: int(returns[c].notna().sum()) for c in skipped}
            raise ValueError(f"AR({ar_order}) unsmoothing produced no finite observation for "
                             f"columns {skipped} (finite input returns {counts}, frame rows "
                             f"{len(returns)}, floor {min_obs}). pass insufficient_data="
                             f"InsufficientData.PASSTHROUGH to leave them unsmoothed")
        for col in skipped:
            unsmoothed[col] = returns[col]

    if return_diagnostics:
        betas_sum = sum(betas)
        # EWMA R^2 of the in-sample AR(q) fit (contemporaneous betas on demeaned lags).
        fitted = sum(betas[i].multiply(x_lags[i]) for i in range(ar_order))
        resid = y_adj - fitted
        resid_var = compute_ewm(data=resid.pow(2.0), span=span)
        y_var = compute_ewm(data=y_adj.pow(2.0), span=span)
        with np.errstate(divide='ignore', invalid='ignore'):
            r2 = (1.0 - resid_var.divide(y_var)).where(y_var > 0.0)
        r2 = r2.clip(lower=0.0, upper=1.0)
        for col in skipped:
            betas_sum[col] = 0.0
            r2[col] = np.nan
        return unsmoothed, betas_sum, r2

    return unsmoothed


def unsmooth_returns_ar1_ewma(returns: pd.DataFrame,
                              span: int = 20,
                              mean_adj_type: MeanAdjType = MeanAdjType.EWMA,
                              warmup_period: Optional[int] = 10,
                              max_value_for_beta: Optional[float] = 0.75,
                              min_value_for_beta: Optional[float] = -0.25,
                              apply_ewma_mean_smoother: bool = True,
                              non_negative: bool = False,
                              non_negative_tol: float = 0.0,
                              insufficient_data: InsufficientData = InsufficientData.NAN,
                              ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Backward-compatible rolling AR(1) unsmoother (shim over ``adjust_returns_with_ar``).

    Equivalent to ``adjust_returns_with_ar(returns, ar_order=1, ...)``. Retained
    for existing callers and for the AR(1) defaults (span=20, warmup=10). See
    the module docstring for the ~1e-4 current-date parity note vs the previous
    panel-vectorised implementation.

    Returns:
        Tuple (unsmoothed_returns, betas, r2), each matching the input shape.
    """
    return adjust_returns_with_ar(
        returns=returns, ar_order=1, span=span, mean_adj_type=mean_adj_type,
        warmup_period=warmup_period, max_value_for_beta=max_value_for_beta,
        min_value_for_beta=min_value_for_beta,
        apply_ewma_mean_smoother=apply_ewma_mean_smoother, non_negative=non_negative,
        non_negative_tol=non_negative_tol, return_diagnostics=True,
        insufficient_data=insufficient_data,
    )


def compute_ar_unsmoothed_prices(prices: pd.DataFrame,
                                 ar_order: int = 2,
                                 freq: Union[str, pd.Series] = 'QE',
                                 span: int = 40,
                                 mean_adj_type: MeanAdjType = MeanAdjType.EWMA,
                                 warmup_period: Optional[int] = 8,
                                 max_value_for_beta: Optional[float] = 0.75,
                                 min_value_for_beta: Optional[float] = -0.25,
                                 apply_ewma_mean_smoother: bool = True,
                                 non_negative: bool = False,
                                 non_negative_tol: float = 0.0,
                                 is_log_returns: bool = True,
                                 insufficient_data: InsufficientData = InsufficientData.NAN,
                                 ) -> Tuple[pd.DataFrame, pd.DataFrame,
                                            pd.DataFrame, pd.DataFrame]:
    """Apply rolling EWMA AR(q) unsmoothing to a price panel, optionally mixed-frequency.

    Converts prices to returns at the specified frequency (or per-asset
    frequencies), applies ``adjust_returns_with_ar``, converts back to NAVs.

    Args:
        prices: Price level DataFrame.
        ar_order: Lag order q (1 = AR(1); 2 = AR(2); q > 2 ok).
        freq: Either a single pandas resample frequency string applied to all
            assets, or a Series mapping asset name to frequency for
            mixed-frequency panels (e.g. monthly HFs alongside quarterly PE).
        span: EWMA span for the rolling beta.
        mean_adj_type: Mean adjustment type for beta regression.
        warmup_period: Initial warmup periods to mask.
        max_value_for_beta: Upper bound on theta_sum (default 0.75).
        min_value_for_beta: Lower bound on theta_sum (default -0.25). See
            ``adjust_returns_with_ar`` for the rationale behind clipping the sum.
        apply_ewma_mean_smoother: If True, second EWMA pass on the beta series
            (forwarded to ``adjust_returns_with_ar``).
        is_log_returns: If True, use log returns for the regression and convert
            back with expm1 at the end. Generally recommended for unsmoothing.
        insufficient_data: Forwarded to ``adjust_returns_with_ar``. Default NAN.

    Returns:
        Tuple (navs, unsmoothed_returns, betas, r2).
    """
    if isinstance(freq, str):
        y = to_returns(prices, freq=freq, drop_first=False, is_log_returns=is_log_returns)
        unsmoothed, betas, r2 = adjust_returns_with_ar(
            returns=y, ar_order=ar_order, span=span, mean_adj_type=mean_adj_type,
            warmup_period=warmup_period, max_value_for_beta=max_value_for_beta,
            min_value_for_beta=min_value_for_beta,
            apply_ewma_mean_smoother=apply_ewma_mean_smoother, non_negative=non_negative,
            non_negative_tol=non_negative_tol, return_diagnostics=True,
            insufficient_data=insufficient_data,
        )
    else:
        unsmoothed_dict, betas_dict, r2_dict = {}, {}, {}
        for frequency, assets in freq.groupby(freq):
            asset_list = assets.index.tolist()
            if len(asset_list) == 0:
                continue
            y = to_returns(prices[asset_list], freq=str(frequency),
                           drop_first=False, is_log_returns=is_log_returns)
            u, b, r = adjust_returns_with_ar(
                returns=y, ar_order=ar_order, span=span, mean_adj_type=mean_adj_type,
                warmup_period=warmup_period, max_value_for_beta=max_value_for_beta,
                min_value_for_beta=min_value_for_beta,
                apply_ewma_mean_smoother=apply_ewma_mean_smoother, non_negative=non_negative,
                non_negative_tol=non_negative_tol, return_diagnostics=True,
                insufficient_data=insufficient_data,
            )
            unsmoothed_dict[frequency] = u
            betas_dict[frequency] = b
            r2_dict[frequency] = r
        unsmoothed = pd.concat(unsmoothed_dict.values(), axis=1).reindex(columns=prices.columns)
        betas = pd.concat(betas_dict.values(), axis=1).reindex(columns=prices.columns)
        r2 = pd.concat(r2_dict.values(), axis=1).reindex(columns=prices.columns)

    if is_log_returns:
        unsmoothed = np.expm1(unsmoothed)
    navs = returns_to_nav(returns=unsmoothed)
    return navs, unsmoothed, betas, r2


def compute_ar1_unsmoothed_prices(prices: pd.DataFrame,
                                  freq: Union[str, pd.Series] = 'QE',
                                  span: int = 40,
                                  mean_adj_type: MeanAdjType = MeanAdjType.EWMA,
                                  warmup_period: Optional[int] = 8,
                                  max_value_for_beta: Optional[float] = 0.75,
                                  min_value_for_beta: Optional[float] = -0.25,
                                  non_negative: bool = False,
                                  non_negative_tol: float = 0.0,
                                  is_log_returns: bool = True,
                                  insufficient_data: InsufficientData = InsufficientData.NAN,
                                  ) -> Tuple[pd.DataFrame, pd.DataFrame,
                                             pd.DataFrame, pd.DataFrame]:
    """Backward-compatible AR(1) price-level unsmoother (shim over ``compute_ar_unsmoothed_prices``).

    Equivalent to ``compute_ar_unsmoothed_prices(prices, ar_order=1, ...)``.
    Returns (navs, unsmoothed_returns, betas, r2).
    """
    return compute_ar_unsmoothed_prices(
        prices=prices, ar_order=1, freq=freq, span=span, mean_adj_type=mean_adj_type,
        warmup_period=warmup_period, max_value_for_beta=max_value_for_beta,
        min_value_for_beta=min_value_for_beta, non_negative=non_negative,
        non_negative_tol=non_negative_tol, is_log_returns=is_log_returns,
        insufficient_data=insufficient_data,
    )


# =============================================================================
# METHOD 2: STATIC GETMANSKY-LO-MAKAROV UNSMOOTHING
# =============================================================================

@dataclass
class GLMUnsmoothingDiagnostics:
    """Diagnostics from a static Getmansky-Lo-Makarov unsmoothing fit.

    Attributes:
        theta: AR coefficients (length q) interpreted as smoothing weights.
        theta_sum: Sum of theta coefficients. Smoothing parameter = 1 - theta_sum.
        vol_inflation_factor: 1 / (1 - theta_sum).
        ar_order: Lag order q used in the fit.
        is_severe: True if |theta_sum| > 0.95 (near-singular inversion).
    """
    theta: np.ndarray
    theta_sum: float
    vol_inflation_factor: float
    ar_order: int
    is_severe: bool


def unsmooth_returns_glm(returns: Union[pd.Series, pd.DataFrame],
                         ar_order: int = 3,
                         return_diagnostics: bool = False
                         ) -> Union[pd.Series, pd.DataFrame,
                                    Tuple[pd.Series, GLMUnsmoothingDiagnostics],
                                    Tuple[pd.DataFrame, dict]]:
    """Apply static AR(q) unsmoothing following Getmansky-Lo-Makarov (2004).

    Fits a single AR(q) model over the entire sample to extract constant
    smoothing weights, then inverts period-by-period. Less adaptive than the
    rolling EWMA method but useful for academic reproducibility.

    Args:
        returns: Observed return Series or DataFrame.
        ar_order: Lag order q. Standard values: 2 for monthly, 3 for higher frequency.
        return_diagnostics: If True, return tuple of (unsmoothed, diagnostics).

    Returns:
        Unsmoothed returns in the same shape as input. If ``return_diagnostics``,
        a tuple with ``GLMUnsmoothingDiagnostics`` (single) or dict (per column).

    Raises:
        ValueError: If returns has fewer than 4*ar_order observations.

    Note:
        Prefer ``adjust_returns_with_ar`` for most practical applications.
        Use this method for academic reproducibility or single-parameter summaries.

        When ``theta_sum`` exceeds 1 the denominator turns negative and the
        inversion flips the sign of the unsmoothed series. ``is_severe`` flags
        ``|theta_sum| > 0.95``; check it before using the result.
    """
    if isinstance(returns, pd.Series):
        unsmoothed, diag = _unsmooth_glm_single(returns, ar_order=ar_order)
        if return_diagnostics:
            return unsmoothed, diag
        return unsmoothed

    if isinstance(returns, pd.DataFrame):
        unsmoothed_cols = {}
        diagnostics_dict = {}
        for col in returns.columns:
            us, diag = _unsmooth_glm_single(returns[col], ar_order=ar_order)
            unsmoothed_cols[col] = us
            diagnostics_dict[col] = diag
        unsmoothed_df = pd.DataFrame(unsmoothed_cols)
        if return_diagnostics:
            return unsmoothed_df, diagnostics_dict
        return unsmoothed_df

    raise ValueError(f"returns must be Series or DataFrame, got {type(returns)}")


def _unsmooth_glm_single(returns: pd.Series,
                         ar_order: int
                         ) -> Tuple[pd.Series, GLMUnsmoothingDiagnostics]:
    """Internal single-series worker for ``unsmooth_returns_glm``."""
    from statsmodels.tsa.ar_model import AutoReg

    clean = returns.dropna()
    if len(clean) < 4 * ar_order:
        raise ValueError(
            f"insufficient observations: {len(clean)} returns for AR({ar_order}) "
            f"(need at least {4 * ar_order})"
        )

    model = AutoReg(clean.values, lags=ar_order, old_names=False).fit()
    theta = np.asarray(model.params[1:])
    theta_sum = float(theta.sum())
    vol_inflation = 1.0 / (1.0 - theta_sum) if theta_sum < 1.0 else np.inf
    is_severe = abs(theta_sum) > 0.95

    denom = 1.0 - theta_sum
    if abs(denom) < 1e-10:
        diag = GLMUnsmoothingDiagnostics(theta=theta, theta_sum=theta_sum,
                                         vol_inflation_factor=np.inf,
                                         ar_order=ar_order, is_severe=True)
        return returns.copy(), diag

    vals = returns.values.copy().astype(float)
    out = vals.copy()
    for t in range(ar_order, len(vals)):
        if np.isnan(vals[t]):
            continue
        correction = 0.0
        for i in range(ar_order):
            lag_val = vals[t - i - 1]
            if np.isnan(lag_val):
                correction = np.nan
                break
            correction += theta[i] * lag_val
        if np.isnan(correction):
            out[t] = np.nan
        else:
            out[t] = (vals[t] - correction) / denom

    unsmoothed = pd.Series(out, index=returns.index, name=returns.name)
    diag = GLMUnsmoothingDiagnostics(theta=theta, theta_sum=theta_sum,
                                     vol_inflation_factor=vol_inflation,
                                     ar_order=ar_order, is_severe=is_severe)
    return unsmoothed, diag
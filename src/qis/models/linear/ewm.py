"""
the exponentially weighted moving-average engine: one recursion, and everything built on it.

Every estimator here is the same recursion,

    s_t = (1 - λ) x_t + λ s_{t-1}

applied to the observations for a mean, to their squares for a variance, and to outer products
for a covariance or a beta. Decay is given as ``ewm_lambda`` or as ``span``, which overrides it
through λ = 1 - 2/(span + 1). That is the pandas span-to-decay mapping, but the recursion is
run unadjusted, so it matches ``.ewm(span=..., adjust=False)`` and not the ``.ewm()`` default.
The column-wise path takes a vector decay, one per column; the covariance kernels take a scalar.

The seed is the state before a column's first finite observation, s_{t0-1}, and every finite
observation, the first included, updates it. Rows before the first observation are NaN.

Four enums carry the conventions, and they are the arguments worth getting right:

    ``NanBackfill`` what the recursion carries across a missing observation: ``FFILL`` holds the
        last state, ``DEFLATED_FFILL`` decays it by λ (a missing value is a zero observation),
        ``ZERO_FILL`` resets it to zero, and ``NAN_FILL`` resets it to zero and reports NaN at
        the gap. The covariance functions default to ``DEFLATED_FFILL``, which keeps every
        matrix positive semidefinite; the column-wise functions default to ``FFILL``
    ``MeanAdjType`` which mean is removed before a second moment is taken. ``NONE`` is the
        second moment about zero, the usual choice for returns; ``INSAMPLE`` subtracts the
        full-sample mean and is forward-looking, so descriptive exhibits only, never inside a
        backtest; ``EXPANDING`` and ``EWMA`` are point-in-time given a point-in-time seed
    ``InitType`` how the recursion is seeded - ``ZERO`` and ``X0`` are point in time, while
        ``MEAN`` and ``VAR`` seed it with full-sample statistics, so they carry the same
        look-ahead near the start of the sample as ``INSAMPLE`` does throughout.
        ``CrossXyType`` selects covariance, beta or correlation

Two layers, and the difference matters at the call site. ``compute_ewm`` and ``compute_ewm_vol``
are wrappers: they take pandas or ndarray, preserve the container, and handle the NaN policy,
the warmup mask and annualisation. ``compute_ewm_covar`` for the covariance at the last date,
``compute_ewm_covar_tensor`` for the (t, n, n) tensor at every date and
``compute_ewm_xy_beta_tensor`` for rolling multivariate betas are ndarray-only - pass
``.to_numpy()`` and rebuild the frame yourself. Second moments are per period unless
annualisation is asked for; the factor is inferred from the index frequency of pandas input,
and a bare ndarray, having no frequency, falls back to 1 with a warning.

Factor-structured covariance belongs in ``factorlasso``, not here.
"""
# packages
import warnings
import numpy as np
import pandas as pd
from numba import njit
from typing import Union, Tuple, Optional
from enum import Enum

# qis
from qis.utils.annualisation import infer_annualisation_factor_from_df
import qis.utils.np_ops as npo


class NanBackfill(Enum):
    """
    how the EWM recursion treats a missing observation.

    The recursion ``s_t = (1-λ) x_t + λ s_{t-1}`` has no value to carry forward at a NaN, so a
    policy is required. Which one is right depends on whether the gap means "no observation
    arrived" or "the series is genuinely absent here". Every policy applies only after a
    column's first finite observation: before it the output is NaN under all four. In a
    covariance or beta recursion the policy applies entry by entry, to the entries whose update
    involves a missing value.

    Attributes:
        FFILL: hold the last state, ``s_t = s_{t-1}``: time stops for the series. The default of
            the column-wise functions. In a covariance matrix with gaps that differ across
            assets it can break positive semidefiniteness
        DEFLATED_FFILL: decay the last state, ``s_t = λ s_{t-1}``, which is exactly the update
            with the missing observation replaced by zero. The usual reading for a return series,
            where a missing return is economically zero; it keeps covariance matrices positive
            semidefinite and is the default of the covariance functions
        ZERO_FILL: reset the state to zero, ``s_t = 0``: the history is erased and the next
            observation restarts the recursion at ``(1-λ) x_t``. For a series that is genuinely
            absent at the gap
        NAN_FILL: carry the state as ZERO_FILL does, and report NaN at the gap instead of the
            zero state, so the output flags where the input was missing. A genuine zero estimate
            is reported as zero
    """
    FFILL = 1  # hold the last state
    DEFLATED_FFILL = 2  # decay the last state by lambda: a missing value is a zero observation
    ZERO_FILL = 3  # reset the state to zero
    NAN_FILL = 4  # reset the state to zero and report nan at the gap


class InitType(Enum):
    """
    how an EWM recursion is seeded when no explicit ``init_value`` is given.

    The seed is the state before a column's first finite observation; that observation then
    updates it, ``s_{t0} = λ seed + (1-λ) x_{t0}``. The seed keeps weight ``λ^(t-t0+1)`` at row
    ``t``, below 5% after about ``1.5 N`` rows for span ``N``, so a full-sample seed leaks later
    information into the early estimates.

    Attributes:
        ZERO: seed zero. Point in time; the early estimates are shrunk towards zero by the
            factor ``1 - λ^(t-t0+1)``
        X0: seed with the column's first finite observation (its square, or cross product, in a
            second-moment recursion), so ``s_{t0} = x_{t0}``. Point in time and the qis default;
            it is pandas ``adjust=False``
        MEAN: seed with the full-sample mean of the series the recursion runs on (for a
            variance, the mean of ``x^2``). Uses the whole sample: look-ahead
        VAR: seed a second-moment recursion with the full-sample variance of the observations
            (a covariance for a cross moment). Look-ahead. A mean recursion cannot take it:
            ``compute_ewm`` and ``compute_rolling_mean_adj`` raise ``ValueError``, and where the
            same ``init_type`` also seeds a mean adjustment that mean is seeded with ``MEAN``
    """
    ZERO = 1
    X0 = 2
    MEAN = 3
    VAR = 4


class MeanAdjType(Enum):
    """
    which mean is subtracted before an EWM second moment is computed.

    The choice sets whether the estimate is a variance or a second moment about zero, and
    whether it uses information the estimation date did not have. ``INSAMPLE`` subtracts the
    full-sample mean and is therefore forward-looking: correct for a descriptive exhibit,
    wrong inside a backtest.

    Attributes:
        NONE: subtract nothing. The second moment about zero, which is the convention for
            return volatility where the mean is small relative to the standard deviation
        INSAMPLE: subtract the full-sample mean. Uses the whole sample at every date, so it is
            forward-looking and belongs only in descriptive output
        EXPANDING: subtract the expanding mean up to each date. Point-in-time
        EWMA: subtract the EWM mean at the same span. Point-in-time, and tracks a drifting mean
    """
    NONE = 1
    INSAMPLE = 2
    EXPANDING = 3
    EWMA = 4


class CrossXyType(Enum):
    """
    which cross statistic ``compute_ewm_cross_xy`` returns.

    Attributes:
        COVAR: the EWM cross moment ``M^{xy}`` (a covariance only if the inputs are centred)
        BETA: ``M^{xy} / M^{xx}``, the EWM regression slope of y on x through the origin
        CORR: ``M^{xy} / sqrt(M^{xx} M^{yy})``, an uncentred correlation unless mean-adjusted
    """
    COVAR = 1
    BETA = 2
    CORR = 3


def _njit_cached(func):
    """``njit`` with an on-disk cache, compiled in memory when no cache location is writable.

    Only for kernels that call no jitted function outside this module: numba does not
    invalidate a cached function when a callee in another file changes.
    """
    try:
        return njit(cache=True)(func)
    except RuntimeError:  # numba raises at decoration when it finds no writable cache location
        return njit(func)


def _first_finite(x: np.ndarray) -> Union[float, np.ndarray]:
    """First finite value of a 1-d array, or of each column of a 2-d array; 0 where none."""
    finite = np.isfinite(x)
    if x.ndim == 1:
        idx = np.flatnonzero(finite)
        return float(x[idx[0]]) if idx.size > 0 else 0.0
    first = np.argmax(finite, axis=0)
    values = x[first, np.arange(x.shape[1])]
    return np.where(finite.any(axis=0), values, 0.0).astype(float)


def set_init_dim1(data: Union[pd.DataFrame, pd.Series, np.ndarray],
                  init_type: InitType = InitType.X0
                  ) -> Union[float, np.ndarray]:
    """Seed of an EWM recursion run on ``data``, one value per column.

    Args:
        data: the series the recursion runs on: observations for a mean, squares or cross
            products for a second moment
        init_type: ``ZERO`` gives 0; ``X0`` the first finite value of each column; ``MEAN`` the
            full-sample ``nanmean``; ``VAR`` the full-sample ``nanvar`` (``ddof=0``) of ``data``
            itself. A column with no finite value is seeded with 0

    Returns:
        a float for 1-d data, an array with one seed per column otherwise

    Raises:
        TypeError: if ``init_type`` is not an ``InitType``
    """
    x = npo.to_finite_np(data=data, fill_value=np.nan)
    has_data = np.isfinite(x).any(axis=0)

    if init_type == InitType.ZERO:
        init_value = np.zeros_like(x[0], dtype=float)
    elif init_type == InitType.X0:
        init_value = _first_finite(x)
    elif init_type in (InitType.MEAN, InitType.VAR):
        func = np.nanmean if init_type == InitType.MEAN else np.nanvar
        with warnings.catch_warnings():  # an empty column is seeded with zero below
            warnings.simplefilter('ignore', RuntimeWarning)
            init_value = func(x, axis=0)
        init_value = np.where(has_data, init_value, 0.0)
    else:
        raise TypeError(f"in set_init_dim1: unsupported init_type={init_type}")

    if x.ndim == 1:
        return float(init_value)
    return np.asarray(init_value, dtype=float)


def set_init_dim2(data: Union[pd.DataFrame, pd.Series, np.ndarray],
                  init_type: InitType = InitType.X0
                  ) -> np.ndarray:
    """Seed matrix of an EWM covariance recursion: zeros for ``ZERO`` and ``X0``.

    Args:
        data: observations, shape (t, n)
        init_type: ``ZERO`` or ``X0``; both give the (n, n) zero matrix, because the matrix
            recursions update at the first row from their seed

    Returns:
        the (n, n) seed matrix

    Raises:
        TypeError: for ``MEAN`` or ``VAR``
    """
    x = npo.to_finite_np(data=data, fill_value=np.nan)
    n = 1 if x.ndim == 1 else x.shape[1]
    if init_type in (InitType.ZERO, InitType.X0):
        init_value = np.zeros((n, n))
    else:
        raise TypeError("in set_initial_condition_dim2: unsupported init_type")

    return init_value


def _second_moment_init(x: np.ndarray,
                        y: np.ndarray,
                        init_type: InitType
                        ) -> Union[float, np.ndarray]:
    """Seed of the EWM second moment of ``x * y`` on the scale of that product.

    ``VAR`` gives the full-sample (co)variance of the observations about their means over the
    rows where both are finite; the other types apply :func:`set_init_dim1` to the product.
    """
    if init_type != InitType.VAR:
        return set_init_dim1(data=np.multiply(x, y), init_type=init_type)
    both = np.isfinite(x) & np.isfinite(y)
    xm = np.where(both, x, np.nan)
    ym = np.where(both, y, np.nan)
    with warnings.catch_warnings():  # an empty column is seeded with zero below
        warnings.simplefilter('ignore', RuntimeWarning)
        covar = np.nanmean((xm - np.nanmean(xm, axis=0)) * (ym - np.nanmean(ym, axis=0)), axis=0)
    covar = np.where(both.any(axis=0), covar, 0.0)
    return float(covar) if np.ndim(x) == 1 else np.asarray(covar, dtype=float)


def _mean_init_type(init_type: InitType) -> InitType:
    """Seed of a mean recursion when ``init_type`` also seeds a variance: VAR maps to MEAN."""
    return InitType.MEAN if init_type == InitType.VAR else init_type


def _check_mean_init_type(init_type: InitType, name: str) -> None:
    """A mean recursion cannot be seeded with a variance."""
    if init_type == InitType.VAR:
        raise ValueError(f"{name}: InitType.VAR seeds a second-moment recursion; a mean "
                         f"recursion takes ZERO, X0 or MEAN")


def _to_decay(span: Optional[Union[float, np.ndarray]],
              ewm_lambda: Union[float, np.ndarray]
              ) -> Union[float, np.ndarray]:
    """``lambda = 1 - 2 / (span + 1)`` when ``span`` is given, else ``ewm_lambda``."""
    if span is None:
        return ewm_lambda
    if isinstance(span, np.ndarray):
        return 1.0 - 2.0 / (span.astype(float) + 1.0)
    return 1.0 - 2.0 / (span + 1.0)


def _kernel_args(a: np.ndarray,
                 init_value: Union[float, np.ndarray],
                 ewm_lambda: Union[float, np.ndarray]
                 ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Seed and decay in the types the numba kernel needs: floats for a 1-d array, and a seed
    vector with one entry per column for a 2-d array."""
    if a.ndim == 1:
        seed = float(np.asarray(init_value, dtype=float).reshape(-1)[0])
        decay = float(np.asarray(ewm_lambda, dtype=float).reshape(-1)[0])
        return seed, decay
    seed = np.broadcast_to(np.asarray(init_value, dtype=float), (a.shape[1],)).copy()
    if isinstance(ewm_lambda, np.ndarray):
        return seed, ewm_lambda.astype(float)
    return seed, float(ewm_lambda)


def _run_ewm(a: np.ndarray,
             init_value: Union[float, np.ndarray],
             ewm_lambda: Union[float, np.ndarray],
             nan_backfill: NanBackfill = NanBackfill.FFILL,
             is_unit_vol_scaling: bool = False
             ) -> np.ndarray:
    """:func:`ewm_recursion` on a float array with the seed and decay typed for numba."""
    a = np.asarray(a, dtype=float)
    seed, decay = _kernel_args(a=a, init_value=init_value, ewm_lambda=ewm_lambda)
    return ewm_recursion(a=a, init_value=seed, ewm_lambda=decay, nan_backfill=nan_backfill,
                         is_unit_vol_scaling=is_unit_vol_scaling)


@_njit_cached
def ewm_recursion(a: np.ndarray,
                  init_value: Union[float, np.ndarray],
                  span: Union[float, np.ndarray] = None,
                  ewm_lambda: Union[float, np.ndarray] = 0.94,
                  is_start_from_first_nonan: bool = True,
                  is_unit_vol_scaling: bool = False,
                  nan_backfill: NanBackfill = NanBackfill.FFILL
                  ) -> np.ndarray:

    """
    exponentially weighted moving average by the recursion ``s_t = λ s_{t-1} + (1-λ) x_t``.

    ``init_value`` is the state before a column's first finite observation, and every finite
    observation, the first included, updates it: ``s_{t0} = λ init_value + (1-λ) x_{t0}``. So
    ``init_value = x_{t0}`` gives ``s_{t0} = x_{t0}`` (pandas ``adjust=False``) and a zero seed
    gives ``(1-λ) x_{t0}``, whether the column starts on row 0 or later. After the start a
    missing observation (or a non-finite update) follows ``nan_backfill``, see
    :class:`NanBackfill`.

    Args:
        a: observations, shape (t,) or (t, n)
        init_value: the seed, a float for 1-d ``a`` and a float or an (n,) array for 2-d ``a``
        span: if given, overrides ``ewm_lambda`` via ``λ = 1 - 2 / (span + 1)``
        ewm_lambda: decay in [0, 1), a float or one per column
        is_start_from_first_nonan: True (the default) keeps each column NaN until its first
            finite observation and applies the seed there. False applies the seed before row 0
            and the missing-value policy from row 0 on
        is_unit_vol_scaling: multiply the output by ``sqrt((1+λ)/(1-λ))``, which gives unit
            variance for IID unit-variance input in the stationary limit
        nan_backfill: the missing-observation policy after the start

    Returns:
        the EWM path, same shape as ``a``
    """
    if span is not None:
        ewm_lambda = 1.0 - 2.0 / (span + 1.0)

    ewm_lambda_1 = 1.0 - ewm_lambda
    is_nan_fill = nan_backfill == NanBackfill.NAN_FILL
    ewm = np.full_like(a, fill_value=np.nan, dtype=np.double)

    if a.ndim == 1:
        state = init_value
        started = not is_start_from_first_nonan
        for t in range(a.shape[0]):
            a_t = a[t]
            if not started:
                if not np.isfinite(a_t):
                    continue  # before the first observation the output stays nan
                started = True
            current_ewm = ewm_lambda * state + ewm_lambda_1 * a_t
            if np.isfinite(current_ewm):
                state = current_ewm
                ewm[t] = state
            else:
                if nan_backfill == NanBackfill.FFILL:
                    pass
                elif nan_backfill == NanBackfill.DEFLATED_FFILL:
                    state = ewm_lambda * state
                else:  # ZERO_FILL and NAN_FILL reset the state
                    state = 0.0
                if not is_nan_fill:
                    ewm[t] = state
    else:
        n = a.shape[1]
        state = np.empty(n)
        state[:] = init_value
        started = np.zeros(n, dtype=np.bool_)
        if not is_start_from_first_nonan:
            started[:] = True
        zeros = np.zeros(n)
        nans = np.full(n, np.nan)
        for t in range(a.shape[0]):
            a_t = a[t]
            started = np.logical_or(started, np.isfinite(a_t))
            current_ewm = ewm_lambda * state + ewm_lambda_1 * a_t
            is_updated = np.isfinite(current_ewm)
            if nan_backfill == NanBackfill.FFILL:
                fill_value = state
            elif nan_backfill == NanBackfill.DEFLATED_FFILL:
                fill_value = ewm_lambda * state
            else:  # ZERO_FILL and NAN_FILL reset the state
                fill_value = zeros
            state = np.where(started, np.where(is_updated, current_ewm, fill_value), state)
            if is_nan_fill:
                ewm[t] = np.where(np.logical_and(started, is_updated), state, nans)
            else:
                ewm[t] = np.where(started, state, nans)

    if is_unit_vol_scaling:
        if np.any(np.asarray(ewm_lambda) >= 1.0):
            raise ValueError("ewm_lambda must be < 1 for unit-variance scaling")
        vol_ratio = np.sqrt((1 + ewm_lambda) / (1 - ewm_lambda))
        ewm = vol_ratio * ewm

    return ewm


def _validate_long_short_spans(long_span, short_span):
    """Validate spans for the long/short EWM filter (compute_ewm_long_short).

    The EWM decay is ``lambda = 1 - 2/(span + 1)``, so ``span = 1`` gives
    ``lambda = 0`` (a VALID degenerate case: that leg passes the input through
    unsmoothed) and ``span -> inf`` gives ``lambda -> 1``. Two hard limits follow:

      1. every span must be ``>= 1``. Below 1 ``lambda < 0`` (a sign-alternating
         recursion, not a smoother), and for ``span <= 0`` the unit-variance load
         ``sqrt((1 + lambda)/(1 - lambda))`` takes the root of a non-positive number.
      2. with two legs, ``short_span`` must be STRICTLY LESS than ``long_span``.
         Equal spans make the legs identical, collapsing the unit-variance
         normaliser ``covar = sqrt(1/(1-lL^2) + 1/(1-lS^2) - 2/(1-lL*lS))`` to
         ``sqrt(0) = 0`` so the leg weights divide by zero; ``short_span`` above
         ``long_span`` inverts the intended fast-minus-slow band-pass.

    Scalars and per-asset ``np.ndarray`` spans are both accepted.
    """
    long_arr = np.asarray(long_span, dtype=float)
    if np.any(long_arr < 1.0):
        raise ValueError(f"compute_ewm_long_short: long_span must be >= 1 "
                         f"(lambda = 1 - 2/(span+1) is negative below span 1); "
                         f"got long_span={long_span}")
    if short_span is not None:
        short_arr = np.asarray(short_span, dtype=float)
        if np.any(short_arr < 1.0):
            raise ValueError(f"compute_ewm_long_short: short_span must be >= 1 "
                             f"(lambda = 1 - 2/(span+1) is negative below span 1); "
                             f"got short_span={short_span}")
        if np.any(short_arr >= long_arr):
            raise ValueError(f"compute_ewm_long_short: short_span must be strictly less "
                             f"than long_span. Equal spans collapse the unit-variance "
                             f"normaliser to 0 (division by zero); short_span > long_span "
                             f"inverts the fast/slow bands. "
                             f"got short_span={short_span}, long_span={long_span}")


@_njit_cached
def compute_ewm_long_short(a: np.ndarray,
                           init_value: Union[float, np.ndarray],
                           long_span: Union[float, np.ndarray] = 63,
                           short_span: Optional[Union[float, np.ndarray]] = 5
                           ) -> np.ndarray:
    """
    Long/short EWM band-pass filter, unit-variance normalised.

    Forms ``weight_long*load_long*EWM(long_lambda) - weight_short*load_short*
    EWM(short_lambda)`` (or the long leg alone when ``short_span is None``), where
    ``lambda = 1 - 2/(span + 1)`` and the weights/loads renormalise the output to
    unit variance for unit-variance white-noise input.

    Span limits (enforced by compute_ewm_long_short_filter via
    _validate_long_short_spans):
      * every span must be ``>= 1``. ``span = 1`` -> ``lambda = 0`` -> the EWM is a
        pass-through (no smoothing): well defined. ``span < 1`` -> ``lambda < 0``:
        rejected.
      * two legs require ``short_span < long_span``: equal spans give ``covar = 0``
        and the leg weights divide by zero.
      * the unstable end is LARGE spans: ``span -> inf`` drives ``lambda -> 1`` and
        the ``(1 - lambda)`` terms in the loads/weights toward zero.

    Both legs start from ``init_value`` as the state before the first observation, so the
    first finite row enters the filter; the weight at lag 0 is zero with two legs.

    @njit kernel: assumes pre-validated spans (no f-string raises inside njit), so
    direct callers should validate first or use compute_ewm_long_short_filter.
    """
    long_lambda = 1.0 - 2.0 / (long_span+1.0)
    if short_span is not None:  # use short + long filter
        short_lambda = 1.0 - 2.0 / (short_span + 1.0)
        short_lambda2 = np.square(short_lambda)
        long_lambda2 = np.square(long_lambda)
        covar = np.sqrt(1.0 / (1.0 - long_lambda2) + 1.0 / (1.0 - short_lambda2) - 2.0 / (1.0 - long_lambda * short_lambda))
        weight_long = 1.0 / (np.sqrt(1.0 - long_lambda2) * covar)
        weight_short = 1.0 / (np.sqrt(1.0 - short_lambda2) * covar)
        load_long = np.sqrt((1.0 + long_lambda) / (1.0 - long_lambda))
        load_short = np.sqrt((1.0 + short_lambda) / (1.0 - short_lambda))
        long_signal = weight_long * load_long * ewm_recursion(a=a, ewm_lambda=long_lambda, init_value=init_value)
        short_signal = weight_short * load_short * ewm_recursion(a=a, ewm_lambda=short_lambda, init_value=init_value)
        ls_filter = long_signal - short_signal

    else:
        weight_long = np.sqrt((1.0 + long_lambda) / (1.0 - long_lambda))
        ls_filter = weight_long * ewm_recursion(a=a, ewm_lambda=long_lambda, init_value=init_value)
    return ls_filter


def compute_ewm_long_short_filter(data: Union[pd.DataFrame, pd.Series, np.ndarray],
                                  long_span: Union[float, np.ndarray] = 63,
                                  short_span: Optional[Union[float, np.ndarray]] = 5,
                                  warmup_period: Optional[Union[int, np.ndarray]] = 21
                                  ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Signal smoother (long/short EWM band-pass) for DataFrame / Series / ndarray.

    Validates the spans, applies compute_ewm_long_short, then masks the first
    ``warmup_period`` observations.

    Span limits (validated here, raising ValueError):
      * ``long_span >= 1`` and, if given, ``short_span >= 1``. ``span = 1`` means
        ``lambda = 1 - 2/(span+1) = 0`` -> that leg is an unsmoothed pass-through.
      * if ``short_span`` is given, ``short_span < long_span``: equal spans collapse
        the unit-variance normaliser to 0 (division by zero) and a larger
        ``short_span`` inverts the band-pass.

    Args:
        data: observations, time along the first axis; a 1-d ndarray is accepted
        long_span: span of the slow leg
        short_span: span of the fast leg; None uses the long leg alone
        warmup_period: number of leading finite values of each column set to NaN

    Returns:
        the unit-variance filter, same container and shape as ``data``

    Raises:
        ValueError: if the spans violate the limits above
    """

    _validate_long_short_spans(long_span=long_span, short_span=short_span)

    data_np = np.asarray(npo.to_finite_np(data=data, fill_value=np.nan), dtype=float)
    if data_np.ndim == 1:  # numba needs a float seed for a single series
        init_value = 0.0
    else:
        init_value = np.zeros(data_np.shape[1])

    ls_filter = compute_ewm_long_short(a=data_np,
                                       init_value=init_value,
                                       long_span=long_span,
                                       short_span=short_span)

    if warmup_period is not None:   # set to nan first nonnan in warmup_period
        ls_filter = npo.set_nans_for_warmup_period(a=ls_filter, warmup_period=warmup_period)

    if isinstance(data, pd.DataFrame):
        ls_filter = pd.DataFrame(data=ls_filter, index=data.index, columns=data.columns)
    elif isinstance(data, pd.Series):
        ls_filter = pd.Series(data=ls_filter, index=data.index, name=data.name)

    return ls_filter


@_njit_cached
def _covar_update(last_covar: np.ndarray,
                  product: np.ndarray,
                  ewm_lambda: Union[float, np.ndarray],
                  nan_backfill: NanBackfill
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """One step of the matrix recursion with the missing-value policy applied entry by entry.

    Returns the new state and the mask of entries whose update was finite.
    """
    covar = (1.0 - ewm_lambda) * product + ewm_lambda * last_covar
    is_updated = np.isfinite(covar)
    if nan_backfill == NanBackfill.FFILL:
        fill_value = last_covar
    elif nan_backfill == NanBackfill.DEFLATED_FFILL:
        fill_value = ewm_lambda * last_covar
    else:  # ZERO_FILL and NAN_FILL reset the entry
        fill_value = np.zeros_like(last_covar)
    return np.where(is_updated, covar, fill_value), is_updated


@njit
def compute_ewm_covar(a: np.ndarray,
                      b: np.ndarray = None,
                      span: Union[int, np.ndarray] = None,
                      ewm_lambda: float = 0.94,
                      covar0: np.ndarray = None,
                      is_corr: bool = False,
                      nan_backfill: NanBackfill = NanBackfill.DEFLATED_FFILL
                      ) -> np.ndarray:
    """
    exponentially weighted covariance matrix at the final observation.

    Runs ``S_t = (1 - lambda) x_t x_t' + lambda S_{t-1}`` over the sample and returns the last
    state, not the path; use :func:`compute_ewm_covar_tensor` for the time series of matrices.
    The matrix recursion updates at the first row from ``covar0``.

    Args:
        a: observations, shape (t, n) or (n,) for a single cross-section
        b: second panel of the same shape. When given, the result is the cross-covariance of
            ``a`` with ``b`` and is not symmetric
        span: if given, overrides ``ewm_lambda`` via ``lambda = 1 - 2 / (span + 1)``
        ewm_lambda: decay in [0, 1); ignored when ``span`` is given
        covar0: seed matrix, shape (n, n); non-finite entries are treated as zero. Zeros when
            None
        is_corr: normalise the result to a correlation matrix, for 1-d input as well
        nan_backfill: how a missing observation is carried, entry by entry; see
            :class:`NanBackfill`. The default ``DEFLATED_FFILL`` treats a missing value as a
            zero observation and keeps the matrix positive semidefinite; ``FFILL`` does not
            when the gaps differ across assets. ``NAN_FILL`` reports NaN at the entries of the
            assets missing on the last row

    Returns:
        covariance matrix, shape (n, n)

    Raises:
        ValueError: if ``b`` is given with a shape different from ``a``
    """
    if b is None:
        b = a
    elif a.shape != b.shape:
        raise ValueError("a and b must have the same shape")

    if span is not None:
        ewm_lambda = 1.0 - 2.0 / (span + 1.0)

    if a.ndim == 1:  # ndarry
        n = a.shape[0]
    else:
        n = a.shape[1]  # array of ndarray

    if covar0 is None:
        last_covar = np.zeros((n, n))
    else:
        last_covar = np.where(np.isfinite(covar0), covar0, 0.0)
    is_updated = np.ones((n, n), dtype=np.bool_)

    if a.ndim == 1:  # a single cross-section: one update
        last_covar, is_updated = _covar_update(last_covar, np.outer(a, b), ewm_lambda,
                                               nan_backfill)
    else:  # loop over rows
        for idx in range(0, a.shape[0]):
            last_covar, is_updated = _covar_update(last_covar, np.outer(a[idx], b[idx]),
                                                   ewm_lambda, nan_backfill)

    if is_corr:
        covar, _, _ = npo._covar_to_corr_array(last_covar)
    else:
        covar = last_covar.copy()
    if nan_backfill == NanBackfill.NAN_FILL:
        covar = np.where(is_updated, covar, np.nan)
    return covar


@njit
def compute_ewm_covar_newey_west(a: np.ndarray,
                                 num_lags: int = 2,
                                 span: Union[int, np.ndarray] = None,
                                 ewm_lambda: float = 0.94,
                                 covar0: np.ndarray = None,
                                 is_corr: bool = False,
                                 nan_backfill: NanBackfill = NanBackfill.DEFLATED_FFILL
                                 ) -> np.ndarray:
    """
    exponentially weighted Newey-West covariance matrix at the final observation.

    ``S_T + sum_{k=1}^{L} (1 - k / (L + 1)) λ^(k/2) (C_k + C_k')``, where ``S_T`` is
    :func:`compute_ewm_covar` and ``C_k`` the EWM of ``x_t x_{t-k}'`` from a zero seed, all with
    the same decay. The factor ``λ^(k/2)`` is the geometric mean of the EWM weights of the two
    dates a lag-k product pairs; with it the estimator is a quadratic form with the Bartlett
    kernel and is positive semidefinite for complete data and for ``DEFLATED_FFILL`` gaps.

    Args:
        a: observations, shape (t, n)
        num_lags: Bartlett lag count ``L``; 0 returns :func:`compute_ewm_covar`
        span: if given, overrides ``ewm_lambda`` for every term
        ewm_lambda: decay in [0, 1) of every term when ``span`` is None
        covar0: seed of ``S``; zeros when None
        is_corr: normalise the result to a correlation matrix
        nan_backfill: missing-observation policy of every term; see :class:`NanBackfill`

    Returns:
        the Newey-West covariance matrix, shape (n, n)
    """
    if span is not None:
        ewm_lambda = 1.0 - 2.0 / (span + 1.0)
    ewm0 = compute_ewm_covar(a=a, ewm_lambda=ewm_lambda, covar0=covar0, is_corr=False,
                             nan_backfill=nan_backfill)
    if num_lags > 0:
        nw_adjustment = np.zeros_like(ewm0)
        for m in range(1, num_lags + 1):
            # lagged value
            a_m = np.empty_like(a)
            a_m[m:] = a[:-m]
            a_m[:m] = np.nan
            ewm_m1 = compute_ewm_covar(a=a, b=a_m, ewm_lambda=ewm_lambda,
                                       nan_backfill=nan_backfill)
            weight = (1.0 - m / (num_lags + 1)) * ewm_lambda ** (0.5 * m)
            nw_adjustment += weight * (ewm_m1 + np.transpose(ewm_m1))
        ewm_nw = ewm0 + nw_adjustment
    else:
        ewm_nw = ewm0

    if is_corr:
        ewm_nw, _, _ = npo._covar_to_corr_array(ewm_nw)

    return ewm_nw


@njit
def compute_ewm_covar_tensor(a: np.ndarray,
                             span: Union[int, np.ndarray] = None,
                             ewm_lambda: float = 0.94,
                             covar0: np.ndarray = None,
                             is_corr: bool = False,
                             nan_backfill: NanBackfill = NanBackfill.DEFLATED_FFILL
                             ) -> np.ndarray:
    """
    exponentially weighted covariance matrix at every date, as a 3-d tensor.

    The path version of :func:`compute_ewm_covar`: same recursion, but every intermediate
    state is kept rather than only the last. Memory is t * n * n floats, so this is for
    rolling risk attribution over a modest universe, not for a wide panel.

    Args:
        a: observations, shape (t, n); 1-d input is rejected
        span: if given, overrides ``ewm_lambda`` via ``lambda = 1 - 2 / (span + 1)``
        ewm_lambda: decay in [0, 1); ignored when ``span`` is given
        covar0: seed matrix, shape (n, n); zeros when None
        is_corr: normalise each matrix to a correlation matrix
        nan_backfill: how a missing observation is carried, entry by entry. The default
            ``DEFLATED_FFILL`` treats it as a zero observation and keeps every matrix positive
            semidefinite; ``FFILL`` can break that when gaps differ across assets.
            ``NAN_FILL`` reports NaN at the entries of the assets missing on that row

    Returns:
        covariance tensor, shape (t, n, n), one matrix per observation date

    Raises:
        ValueError: if ``a`` is not 2-d
    """
    if span is not None:
        ewm_lambda = 1.0 - 2.0 / (span + 1.0)

    if not a.ndim == 2:
        raise ValueError("only 2-d arrays are supported")

    t = a.shape[0]
    n = a.shape[1]  # array of ndarray

    if covar0 is None:
        last_covar = np.zeros((n, n))
    else:
        last_covar = np.where(np.isfinite(covar0), covar0, 0.0)

    output_covar = np.empty((t, n, n))
    # loop over rows
    for idx in range(0, t):  # row in x:
        row = a[idx]
        last_covar, is_updated = _covar_update(last_covar, np.outer(row, row), ewm_lambda,
                                               nan_backfill)
        if is_corr:
            last_covar_, _, _ = npo._covar_to_corr_array(last_covar)
        else:
            last_covar_ = last_covar

        if nan_backfill == NanBackfill.NAN_FILL:  # report the gaps, keep genuine zeros
            last_covar_ = np.where(is_updated, last_covar_, np.nan)

        output_covar[idx] = last_covar_

    return output_covar


@njit
def compute_ewm_covar_tensor_vol_norm_returns(a: np.ndarray,
                                              span: Union[int, np.ndarray] = None,
                                              ewm_lambda: float = 0.94,
                                              covar0: np.ndarray = None,
                                              is_corr: bool = False,
                                              nan_backfill: NanBackfill = NanBackfill.DEFLATED_FFILL
                                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    EWM covariance tensor computed on vol-normalised returns, with the vols returned alongside.

    Two-step estimator: each series is divided by its own EWM volatility, the correlation is
    estimated on the normalised series, and the covariance is rebuilt as ``D C D`` with ``D`` the
    diagonal of vols. Normalising first stops a single volatile asset dominating the correlation
    estimate, which the direct covariance recursion does not avoid. The volatility recursion is
    seeded, point in time, with each column's first finite squared return (``InitType.X0``), so
    the first vol of a column is the absolute value of its first return.

    Args:
        a: returns, rows are dates and columns are assets. 2-d only
        span: EWM span. Takes precedence over ``ewm_lambda``
        ewm_lambda: decay used when ``span`` is None
        covar0: initial covariance of the normalised returns. None starts from zero
        is_corr: return the correlation tensor of the normalised returns as the second output
            instead of their covariance. The first output is the covariance either way
        nan_backfill: how a missing observation is handled in both recursions; see
            :class:`NanBackfill`

    Returns:
        the covariance tensor of shape ``(t, n, n)``, the normalised-return covariance (or, with
        ``is_corr``, correlation) tensor, and the EWM vols of shape ``(t, n)``

    Raises:
        ValueError: if ``a`` is not 2-d
    """
    if span is not None:
        ewm_lambda = 1.0 - 2.0 / (span + 1.0)

    if not a.ndim == 2:
        raise ValueError("only 2-d arrays are supported")

    t = a.shape[0]
    n = a.shape[1]  # array of ndarray

    if covar0 is None:
        last_covar = np.zeros((n, n))
    else:
        last_covar = np.where(np.isfinite(covar0), covar0, 0.0)

    output_covar_norm = np.empty((t, n, n))
    output_covar = np.empty((t, n, n))

    # point-in-time vols: the variance recursion is seeded with the first finite squared return
    a_var = np.square(a)
    init_var = np.zeros(n)
    for column in range(n):
        for row in range(t):
            if np.isfinite(a_var[row, column]):
                init_var[column] = a_var[row, column]
                break
    ewm_vol = np.sqrt(ewm_recursion(a=a_var, init_value=init_var, ewm_lambda=ewm_lambda,
                                    nan_backfill=nan_backfill))
    safe_ewm_vol = np.where(np.greater(ewm_vol, 0.0), ewm_vol, np.nan)
    a_norm = a / safe_ewm_vol

    # loop over rows
    for idx in range(0, t):  # row in x:
        row = a_norm[idx]
        last_covar, is_updated = _covar_update(last_covar, np.outer(row, row), ewm_lambda,
                                               nan_backfill)
        if is_corr:
            last_covar_, _, _ = npo._covar_to_corr_array(last_covar)
        else:
            last_covar_ = last_covar

        if nan_backfill == NanBackfill.NAN_FILL:  # report the gaps, keep genuine zeros
            last_covar_ = np.where(is_updated, last_covar_, np.nan)

        # normalise to preserve vols for output_covar
        _, normalized_vols, _ = npo._covar_to_corr_array(last_covar_)
        norm_to_ewm_vols = ewm_vol[idx] / normalized_vols
        output_covar[idx] = last_covar_ * np.outer(norm_to_ewm_vols, norm_to_ewm_vols)
        output_covar_norm[idx] = last_covar_

    return output_covar, output_covar_norm, ewm_vol


# relative eigenvalue threshold below which the unit-diagonal factor moment matrix is singular
_BETA_SINGULAR_RCOND = 1e-12


def _solve_betas(covar_xx: np.ndarray,
                 cross_xy: np.ndarray,
                 is_x_correlated: bool
                 ) -> np.ndarray:
    """Betas ``covar_xx^{-1} cross_xy`` with a scale-free singularity test.

    A factor whose second moment is not strictly positive carries no information: its betas are
    NaN and the others are solved from the reduced system. The reduced matrix is rescaled to
    unit diagonal, so the singularity test does not depend on the units of the factors; when its
    smallest eigenvalue is below ``_BETA_SINGULAR_RCOND`` times the largest, the system is
    singular and every beta is NaN.
    """
    betas = np.full(cross_xy.shape, np.nan)
    diag = np.diag(covar_xx)
    valid = np.flatnonzero(np.isfinite(diag) & (diag > 0.0))
    if valid.size == 0:
        return betas
    scale = 1.0 / np.sqrt(diag[valid])
    cross_v = cross_xy[valid]
    if is_x_correlated and valid.size > 1:
        unit_diag = covar_xx[np.ix_(valid, valid)] * np.outer(scale, scale)
        if not np.all(np.isfinite(unit_diag)):
            return betas
        eigenvalues = np.linalg.eigvalsh(unit_diag)
        if eigenvalues[0] <= _BETA_SINGULAR_RCOND * eigenvalues[-1]:
            return betas
        betas[valid] = scale[:, None] * np.linalg.solve(unit_diag, scale[:, None] * cross_v)
    else:
        betas[valid] = cross_v * np.square(scale)[:, None]
    return betas


def compute_ewm_xy_beta_tensor(x: np.ndarray,  # factor returns
                               y: np.ndarray,  # asset returns
                               span: Union[int, np.ndarray] = None,
                               ewm_lambda: float = 0.94,
                               warmup_period: int = 20,  # to avoid excessive betas at start,
                               is_x_correlated: bool = True,  # computation of [x,x]
                               nan_backfill: NanBackfill = NanBackfill.FFILL
                               ) -> np.ndarray:
    """Compute an EWM beta tensor from factor and asset returns.

    The cross moment ``E[x y']`` and the factor second moment ``E[x x']`` run the zero-seeded
    matrix recursion, which updates at the first row, and ``beta_t = E[x x']_t^{-1} E[x y']_t``.
    The singularity test is scale free: a factor with a non-positive second moment gets NaN
    betas while the other factors are solved from the reduced system, and a reduced system whose
    unit-diagonal rescaling is numerically singular gives NaN betas. A beta is never replaced by
    the raw cross moment.

    Args:
        x: Factor returns with shape ``(time,)`` or ``(time, factors)``.
        y: Asset returns with shape ``(time,)`` or ``(time, assets)``.
        span: Optional EWM span overriding ``ewm_lambda``.
        ewm_lambda: EWM decay when ``span`` is not supplied.
        warmup_period: Last time position masked during estimator warm-up.
        is_x_correlated: Whether to invert the full factor cross-moment matrix; False uses its
            diagonal, one-factor betas for each factor.
        nan_backfill: Missing-observation policy applied to both EWM moments in the beta ratio.
            ``NAN_FILL`` reports NaN where the factor or the asset is missing on that row.

    Returns:
        EWM betas with shape ``(time, factors, assets)``.

    Raises:
        TypeError: If an input is not one- or two-dimensional, or time dimensions differ.
    """
    if x.ndim not in [1, 2] or y.ndim not in [1, 2]:
        raise TypeError("Expected 1- or 2-dimensional NumPy array for x and y")
    if x.shape[0] != y.shape[0]:
        raise TypeError("first time series dimension of x and y must be equal")

    if x.ndim == 1:  # numba is sensetive to how dimensions and initial value
        nx = 1
        is_x_correlated = False  # for 1-d factor no need to compute outer product
    else:
        nx = x.shape[1]
    last_covar_xx = np.zeros((nx, nx))
    if y.ndim == 1:
        ny = 1
    else:
        ny = y.shape[1]
    last_cross_xy = np.zeros((nx, ny))
    beta_nan = np.full((nx, ny), np.nan)

    nt = x.shape[0]
    betas_ts = np.full((nt, nx, ny), np.nan)

    if span is not None:
        ewm_lambda = 1.0 - 2.0 / (span + 1.0)
    for t in range(nt):  # over time index
        row_x = x[t]  # time series row
        row_y = y[t]
        # A missing factor row must age both moments identically so their ratio stays coherent.
        last_cross_xy, is_cross_updated = _covar_update(last_cross_xy, np.outer(row_x, row_y),
                                                        ewm_lambda, nan_backfill)
        last_covar_xx, _ = _covar_update(last_covar_xx, np.outer(row_x, row_x), ewm_lambda,
                                         nan_backfill)

        if t > warmup_period:
            betas_t = _solve_betas(covar_xx=last_covar_xx, cross_xy=last_cross_xy,
                                   is_x_correlated=is_x_correlated)
            if nan_backfill == NanBackfill.NAN_FILL:
                betas_t = np.where(is_cross_updated, betas_t, np.nan)
        else:
            betas_t = beta_nan

        betas_ts[t] = betas_t

    return betas_ts


def compute_one_factor_ewm_betas(x: pd.Series,
                                 y: pd.DataFrame,
                                 span: Union[int, np.ndarray] = None,
                                 ewm_lambda: float = 0.94,
                                 nan_backfill: NanBackfill = NanBackfill.FFILL,
                                 warmup_period: int = 20
                                 ) -> pd.DataFrame:
    """EWM betas of every asset column on one factor.

    ``beta_t = E[x y]_t / E[x^2]_t`` from zero-seeded recursions that update at the first row,
    with no mean adjustment; see :func:`compute_ewm_xy_beta_tensor`.

    Args:
        x: factor returns
        y: asset returns, one column per asset, on the same index as ``x``
        span: if given, overrides ``ewm_lambda`` via ``lambda = 1 - 2 / (span + 1)``
        ewm_lambda: decay in [0, 1); ignored when ``span`` is given
        nan_backfill: missing-observation policy of both moments; see :class:`NanBackfill`
        warmup_period: rows ``t <= warmup_period`` are NaN, to suppress the unstable start.
            The default 20 masks the first 21 rows

    Returns:
        betas with the index and columns of ``y``; NaN where the factor has no variance

    Raises:
        ValueError: if the indices of ``x`` and ``y`` differ
    """
    if not x.index.equals(y.index):
        raise ValueError(f"x.index={x.index} is not equal to y.index={y.index}")

    x_np = npo.to_finite_np(data=x, fill_value=np.nan)
    y_np = npo.to_finite_np(data=y, fill_value=np.nan)

    betas_ts = compute_ewm_xy_beta_tensor(x=x_np, y=y_np, span=span,
                                          ewm_lambda=ewm_lambda,
                                          warmup_period=warmup_period,
                                          nan_backfill=nan_backfill)
    # the x factor dimension is 1, we get slice [t, y] using [:, 0, :]
    one_factor_ewm_betas = pd.DataFrame(data=betas_ts[:, 0, :], index=y.index, columns=y.columns)
    return one_factor_ewm_betas


def compute_ewm(data: Union[pd.DataFrame, pd.Series, np.ndarray],
                span: Union[float, np.ndarray] = None,
                ewm_lambda: Union[float, np.ndarray] = 0.94,
                init_value: Union[float, np.ndarray, None] = None,
                init_type: InitType = InitType.X0,
                is_unit_vol_scaling: bool = False,
                nan_backfill: NanBackfill = NanBackfill.FFILL
                ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
    """
    exponentially weighted moving average of a t-by-n panel.

    Implements the recursion ``m_t = (1 - lambda) x_t + lambda m_{t-1}``, evaluated column-wise
    by the numba kernel :func:`ewm_recursion`. The seed is the state before each column's first
    finite observation, and that observation updates it, so with the default ``X0`` seed the
    result is pandas ``ewm(adjust=False)`` column by column, however late a column starts. The
    container type of ``data`` is preserved.

    Args:
        data: observations, time along the first axis
        span: if given, overrides ``ewm_lambda`` via ``lambda = 1 - 2 / (span + 1)``, the same
            correspondence pandas uses
        ewm_lambda: decay in [0, 1); higher is smoother. Ignored when ``span`` is given
        init_value: explicit seed, the state before the first observation; overrides
            ``init_type``
        init_type: how the seed is set when ``init_value`` is None — the first observation
            (``X0``), zero, or the full-sample mean (look-ahead); see :class:`InitType`
        is_unit_vol_scaling: rescale the output to unit unconditional variance
        nan_backfill: how the recursion carries over missing observations

    Returns:
        smoothed data, same container and shape as ``data``

    Raises:
        ValueError: if ``init_type`` is ``InitType.VAR``, a variance seed for a mean
    """
    a = npo.to_finite_np(data=data, fill_value=np.nan)

    if init_value is None:
        _check_mean_init_type(init_type=init_type, name='compute_ewm')
        init_value = set_init_dim1(data=a, init_type=init_type)

    ewm = _run_ewm(a=a,
                   init_value=init_value,
                   ewm_lambda=_to_decay(span=span, ewm_lambda=ewm_lambda),
                   nan_backfill=nan_backfill,
                   is_unit_vol_scaling=is_unit_vol_scaling)

    if isinstance(data, pd.DataFrame):  # return of data type
        ewm = pd.DataFrame(data=ewm, index=data.index, columns=data.columns)

    elif isinstance(data, pd.Series):  # return of data type
        ewm = pd.Series(data=ewm, index=data.index, name=data.name)

    return ewm


def _annualise(ewm: np.ndarray,
               data: Union[pd.DataFrame, pd.Series, np.ndarray],
               annualize: bool,
               annualization_factor: Optional[float]
               ) -> np.ndarray:
    """Multiply a variance path by the annualisation factor when asked."""
    if annualize or annualization_factor is not None:
        if annualization_factor is None:
            if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
                annualization_factor = infer_annualisation_factor_from_df(data=data)
            else:
                warnings.warn("in compute_ewm  annualization_factor for np array default is 1")
                annualization_factor = 1.0
        ewm = annualization_factor * ewm
    return ewm


def _wrap_like(values: np.ndarray,
               data: Union[pd.DataFrame, pd.Series, np.ndarray]
               ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
    """Return ``values`` in the container of ``data``."""
    if isinstance(data, pd.DataFrame):
        return pd.DataFrame(data=values, index=data.index, columns=data.columns)
    if isinstance(data, pd.Series):
        return pd.Series(data=values, index=data.index, name=data.name)
    return values


def compute_ewm_vol(data: Union[pd.DataFrame, pd.Series, np.ndarray],
                    span: Optional[Union[float, np.ndarray]] = None,
                    ewm_lambda: Union[float, np.ndarray] = 0.94,
                    mean_adj_type: MeanAdjType = MeanAdjType.NONE,
                    init_type: InitType = InitType.X0,
                    init_value: Optional[Union[float, np.ndarray]] = None,
                    apply_sqrt: bool = True,
                    annualize: bool = False,
                    annualization_factor: Optional[float] = None,
                    vol_floor_quantile: Optional[float] = None,  # to floor the volatility = 0.16
                    vol_floor_quantile_roll_period: int = 5*260,  # 5y for daily returns
                    warmup_period: Optional[int] = None,
                    nan_backfill: NanBackfill = NanBackfill.FFILL
                    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
    """
    exponentially weighted volatility, or variance when ``apply_sqrt`` is False.

    Runs the EWM recursion on squared observations: ``v_t = (1 - lambda) x_t^2 + lambda
    v_{t-1}``, seeded before each column's first finite observation, which then updates the
    seed; with the default ``X0`` seed the first variance is the first squared observation.
    Whether the result is a volatility per period or per annum depends on ``annualize``, and the
    annualisation factor is inferred from the index frequency for pandas input.

    Args:
        data: observations, time along the first axis
        span: if given, overrides ``ewm_lambda`` via ``lambda = 1 - 2 / (span + 1)``
        ewm_lambda: decay in [0, 1); ignored when ``span`` is given
        mean_adj_type: how the mean is removed before squaring. NONE treats the data as
            already centred, which is the usual choice for returns
        init_type: how the recursion is seeded when ``init_value`` is None, on the scale of the
            squares: ``X0`` the first squared observation, ``ZERO`` zero, ``MEAN`` the
            full-sample mean square and ``VAR`` the full-sample variance of the observations
            (both look-ahead). It also seeds an EWMA mean adjustment, with ``VAR`` read as
            ``MEAN`` there
        init_value: explicit seed for the variance recursion, the state before the first
            observation
        apply_sqrt: return volatility rather than variance
        annualize: scale to annual terms
        annualization_factor: periods per year; inferred from the index for pandas input,
            and defaults to 1 with a warning for a bare ndarray
        vol_floor_quantile: floor the estimate at this rolling quantile of itself, so a quiet
            sample does not produce a vol that collapses toward zero. 0.16 is a usual choice.
            Works for a Series, a DataFrame and 1-d or 2-d arrays
        vol_floor_quantile_roll_period: lookback for that quantile, in periods
        warmup_period: number of leading observations set to nan, so an estimate is not
            reported before the recursion has data
        nan_backfill: how the recursion carries over missing observations

    Returns:
        volatility or variance, same container and shape as ``data``
    """
    a = npo.to_finite_np(data=data, fill_value=np.nan)
    ewm_lambda = _to_decay(span=span, ewm_lambda=ewm_lambda)

    if mean_adj_type != MeanAdjType.NONE:
        a = compute_rolling_mean_adj(data=a,
                                     mean_adj_type=mean_adj_type,
                                     ewm_lambda=ewm_lambda,
                                     init_type=_mean_init_type(init_type),
                                     nan_backfill=nan_backfill)

    # the variance recursion runs on squared observations and is seeded on that scale
    if init_value is None:
        init_value = _second_moment_init(x=a, y=a, init_type=init_type)
    ewm = _run_ewm(a=np.square(a), init_value=init_value, ewm_lambda=ewm_lambda,
                   nan_backfill=nan_backfill)

    # apply quantile
    if vol_floor_quantile is not None:
        ewm_2d = ewm.reshape(-1, 1) if ewm.ndim == 1 else ewm
        ewm_quantiles = pd.DataFrame(ewm_2d).rolling(
            vol_floor_quantile_roll_period,
            min_periods=int(0.2*vol_floor_quantile_roll_period)
        ).quantile(vol_floor_quantile, interpolation="lower")
        vol_floor = ewm_quantiles.to_numpy()
        ewm_2d = np.where(np.less(ewm_2d, vol_floor), vol_floor, ewm_2d)
        ewm = ewm_2d[:, 0] if ewm.ndim == 1 else ewm_2d

    if warmup_period is not None:   # set to nan first nonnan in warmup_period
        ewm = npo.set_nans_for_warmup_period(a=ewm, warmup_period=warmup_period)

    ewm = _annualise(ewm=ewm, data=data, annualize=annualize,
                     annualization_factor=annualization_factor)

    if apply_sqrt:
        ewm = np.sqrt(ewm)

    return _wrap_like(values=ewm, data=data)


@_njit_cached
def _newey_west_variance(a: np.ndarray,
                         init_value: np.ndarray,
                         ewm_lambda: np.ndarray,
                         num_lags: int,
                         nan_backfill: NanBackfill
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """EWM variance and EWM Newey-West variance of every column of ``a`` (t, n).

    ``v^NW_t = v_t + sum_k (1 - k/(L+1)) 2 λ^(k/2) c_{k,t}``, with ``c_{k,t}`` the zero-seeded
    EWM of ``x_t x_{t-k}``. The lag partner of an observation is the k-th previous observation
    of the current run: ``FFILL`` skips a gap (time stops), ``DEFLATED_FFILL`` makes it a zero
    observation, and ``ZERO_FILL`` and ``NAN_FILL`` reset every state and the lag history. Under
    each policy the estimator is a quadratic form with the Bartlett kernel plus the non-negative
    seed term, so it is never negative for a non-negative seed.
    """
    t, n = a.shape
    variance = np.full((t, n), np.nan)
    nw_variance = np.full((t, n), np.nan)
    cross = np.zeros(num_lags)
    history = np.full(num_lags, np.nan)
    for column in range(n):
        lam = ewm_lambda[column]
        state = init_value[column]
        cross[:] = 0.0
        history[:] = np.nan
        started = False
        for row in range(t):
            x = a[row, column]
            if not started:
                if not np.isfinite(x):
                    continue
                started = True
            if not np.isfinite(x):
                if nan_backfill == NanBackfill.FFILL:  # time stops for the series
                    variance[row, column] = variance[row - 1, column]
                    nw_variance[row, column] = nw_variance[row - 1, column]
                    continue
                elif nan_backfill == NanBackfill.DEFLATED_FFILL:  # a zero observation
                    x = 0.0
                else:  # ZERO_FILL and NAN_FILL erase the history
                    state = 0.0
                    cross[:] = 0.0
                    history[:] = np.nan
                    if nan_backfill == NanBackfill.ZERO_FILL:
                        variance[row, column] = 0.0
                        nw_variance[row, column] = 0.0
                    continue
            state = lam * state + (1.0 - lam) * x * x
            nw = state
            for k in range(num_lags):
                partner = history[k]
                product = x * partner if np.isfinite(partner) else 0.0
                cross[k] = lam * cross[k] + (1.0 - lam) * product
                weight = (1.0 - (k + 1.0) / (num_lags + 1.0)) * 2.0 * lam ** (0.5 * (k + 1.0))
                nw += weight * cross[k]
            for k in range(num_lags - 1, 0, -1):
                history[k] = history[k - 1]
            if num_lags > 0:
                history[0] = x
            variance[row, column] = state
            nw_variance[row, column] = nw
    return variance, nw_variance


def compute_ewm_newey_west_vol(data: Union[pd.DataFrame, pd.Series, np.ndarray],
                               num_lags: int = 2,
                               span: Optional[Union[float, np.ndarray]] = None,
                               ewm_lambda: Union[float, np.ndarray] = 0.94,
                               mean_adj_type: MeanAdjType = MeanAdjType.NONE,
                               init_type: InitType = InitType.X0,
                               init_value: Optional[Union[float, np.ndarray]] = None,
                               apply_sqrt: bool = True,
                               annualize: bool = False,
                               annualization_factor: Optional[float] = None,
                               warmup_period: Optional[int] = None,
                               nan_backfill: NanBackfill = NanBackfill.FFILL
                               ) -> Tuple[Union[pd.DataFrame, pd.Series, np.ndarray],
                                          Union[pd.DataFrame, pd.Series, np.ndarray]]:
    """
    exponentially weighted Newey-West variance or volatility.

    The EWM variance ``v_t`` of ``compute_ewm_vol`` is corrected for serial correlation with
    Bartlett-weighted EWM autocovariances,
    ``v_t + sum_{m=1}^{L} (1 - m / (L + 1)) * 2 * λ^(m/2) * EWM(x_t x_{t-m})``, all with the same
    decay. The factor ``λ^(m/2)`` is the geometric mean of the EWM weights of the two dates a
    lag-m product pairs: with it the estimator is a quadratic form in the EWM-weighted
    observations with the Bartlett kernel, which is positive semidefinite, so the corrected
    variance is never negative.

    Args:
        data: observations in rows
        num_lags: Bartlett lag count ``L``; 0 returns the EWM variance itself
        span: EWM span; overrides ``ewm_lambda`` via ``lambda = 1 - 2 / (span + 1)``
        ewm_lambda: EWM decay used when ``span`` is None
        mean_adj_type: mean subtracted before the second moments are formed
        init_type: seed of the variance recursion, applied to the squared observations; see
            :func:`compute_ewm_vol`
        init_value: explicit seed of the variance recursion
        apply_sqrt: return a volatility rather than a variance
        annualize: multiply the variance by the annualisation factor
        annualization_factor: explicit annualisation factor; inferred from the index if None
        warmup_period: number of initial observations set to NaN
        nan_backfill: treatment of missing observations in the variance and in every lag term:
            ``FFILL`` holds the estimate and pairs each observation with the previous
            observed ones, ``DEFLATED_FFILL`` treats a gap as a zero observation, and
            ``ZERO_FILL`` and ``NAN_FILL`` restart the estimator after the gap

    Returns:
        the corrected estimate and its ratio to the uncorrected EWM variance; the ratio is NaN
        where the EWM variance is not positive
    """
    a = npo.to_finite_np(data=data, fill_value=np.nan)
    ewm_lambda = _to_decay(span=span, ewm_lambda=ewm_lambda)

    if mean_adj_type != MeanAdjType.NONE:
        a = compute_rolling_mean_adj(data=a,
                                     mean_adj_type=mean_adj_type,
                                     ewm_lambda=ewm_lambda,
                                     init_type=_mean_init_type(init_type),
                                     nan_backfill=nan_backfill)

    # the variance recursion runs on squared observations, so it is seeded on that scale
    if init_value is None:
        init_value = _second_moment_init(x=a, y=a, init_type=init_type)

    # the kernel works on columns, so a single series is treated as one column
    is_1d = a.ndim == 1
    a_2d = np.asarray(a.reshape(-1, 1) if is_1d else a, dtype=float)
    ncols = a_2d.shape[1]
    seed = np.broadcast_to(np.asarray(init_value, dtype=float), (ncols,)).copy()
    decay = np.broadcast_to(np.asarray(ewm_lambda, dtype=float), (ncols,)).copy()
    ewm0, ewm_nw = _newey_west_variance(a_2d, seed, decay, int(num_lags), nan_backfill)
    if is_1d:
        ewm0, ewm_nw = ewm0[:, 0], ewm_nw[:, 0]

    # NumPy 2.x: explicit out= so masked positions (ewm0<=0 or nan) are deterministic nan.
    nw_ratio = np.divide(ewm_nw, ewm0, out=np.full_like(ewm_nw, np.nan, dtype=float),
                         where=np.greater(np.nan_to_num(ewm0, nan=0.0), 0.0))

    if warmup_period is not None:   # set to nan first nonnan in warmup_period
        ewm_nw = npo.set_nans_for_warmup_period(a=ewm_nw, warmup_period=warmup_period)
        nw_ratio = npo.set_nans_for_warmup_period(a=nw_ratio, warmup_period=warmup_period)

    ewm_nw = _annualise(ewm=ewm_nw, data=data, annualize=annualize,
                        annualization_factor=annualization_factor)

    if apply_sqrt:
        ewm_nw = np.sqrt(ewm_nw)

    return _wrap_like(values=ewm_nw, data=data), _wrap_like(values=nw_ratio, data=data)


def compute_roll_mean(data: Union[pd.DataFrame, pd.Series, np.ndarray],
                      mean_adj_type: MeanAdjType = MeanAdjType.EWMA,
                      span: Union[float, np.ndarray] = None,
                      ewm_lambda: Union[float, np.ndarray] = 0.94,
                      init_value: Union[float, np.ndarray] = None,
                      nan_backfill: NanBackfill = NanBackfill.FFILL
                      ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
    """
    mean path of each column under a mean-adjustment convention, same shape as the input.

    Args:
        data: observations, time along the first axis
        mean_adj_type: ``NONE`` gives zeros; ``INSAMPLE`` the full-sample mean of each column,
            ignoring missing values and repeated on every row, which is forward-looking and
            belongs in descriptive exhibits, never inside a backtest; ``EXPANDING`` the
            expanding mean up to each row; ``EWMA`` the EWM mean, point in time
        span: EWMA span; overrides ``ewm_lambda``
        ewm_lambda: EWMA decay
        init_value: EWMA seed; the first observation when None
        nan_backfill: EWMA missing-observation policy

    Returns:
        the mean path, same container and shape as ``data``

    Raises:
        TypeError: for an unsupported container or ``mean_adj_type``
    """
    if not isinstance(data, (pd.DataFrame, pd.Series, np.ndarray)):
        raise TypeError(f"unsupported type {type(data)}")

    if mean_adj_type == MeanAdjType.NONE:
        values = np.asarray(data, dtype=float)
        mean = _wrap_like(values=np.zeros_like(values), data=data)

    elif mean_adj_type == MeanAdjType.INSAMPLE:  # forward-looking: the full-sample mean
        values = npo.to_finite_np(data=data, fill_value=np.nan).astype(float)
        with warnings.catch_warnings():  # an all-nan column has a nan mean
            warnings.simplefilter('ignore', RuntimeWarning)
            column_mean = np.nanmean(values, axis=0)
        mean = _wrap_like(values=np.broadcast_to(column_mean, values.shape).copy(), data=data)

    elif mean_adj_type == MeanAdjType.EXPANDING:  # use pandas core
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            x = data
        else:  # convert to pandas
            x = pd.DataFrame(data=data)
        mean = x.expanding(min_periods=1).mean()  # apply pandas expanding
        if isinstance(data, np.ndarray):  # return of np.ndarray data type
            mean = mean.to_numpy()
            if data.ndim == 1:
                mean = mean[:, 0]

    elif mean_adj_type == MeanAdjType.EWMA:
        mean = compute_ewm(data=data,
                           span=span,
                           ewm_lambda=ewm_lambda,
                           init_value=init_value,
                           nan_backfill=nan_backfill)
    else:
        raise TypeError(f"mean_adj_type={mean_adj_type} is not implemented")

    return mean


def compute_rolling_mean_adj(data: Union[pd.DataFrame, pd.Series, np.ndarray],
                             mean_adj_type: MeanAdjType = MeanAdjType.EWMA,
                             span: Union[float, np.ndarray] = None,
                             ewm_lambda: Union[float, np.ndarray] = 0.94,
                             init_type: InitType = InitType.X0,
                             init_value: Union[float, np.ndarray, None] = None,
                             nan_backfill: NanBackfill = NanBackfill.FFILL
                             ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
    """
    observations minus their mean path under ``mean_adj_type``; see :func:`compute_roll_mean`.

    Args:
        data: observations, time along the first axis
        mean_adj_type: which mean is removed; ``INSAMPLE`` is forward-looking
        span: EWMA span; overrides ``ewm_lambda``
        ewm_lambda: EWMA decay
        init_type: EWMA seed when ``init_value`` is None; ``X0`` makes the first centred value 0
        init_value: explicit EWMA seed
        nan_backfill: EWMA missing-observation policy

    Returns:
        the centred data, same container and shape as ``data``

    Raises:
        ValueError: if ``init_type`` is ``InitType.VAR``, a variance seed for a mean
    """
    if mean_adj_type == MeanAdjType.NONE:
        x_mean = data
    else:
        if init_value is None and mean_adj_type == MeanAdjType.EWMA:
            _check_mean_init_type(init_type=init_type, name='compute_rolling_mean_adj')
            init_value = set_init_dim1(data=data, init_type=init_type)

        mean = compute_roll_mean(data=data,
                                 mean_adj_type=mean_adj_type,
                                 span=span,
                                 ewm_lambda=ewm_lambda,
                                 init_value=init_value,
                                 nan_backfill=nan_backfill)
        x_mean = data - mean

    return x_mean


def compute_ewm_cross_xy(x_data: Union[pd.DataFrame, pd.Series, np.ndarray],
                         y_data: Union[pd.DataFrame, pd.Series, np.ndarray],
                         span: Union[float, np.ndarray] = None,
                         ewm_lambda: Union[float, np.ndarray] = 0.94,
                         cross_xy_type: CrossXyType = CrossXyType.COVAR,
                         mean_adj_type: MeanAdjType = MeanAdjType.NONE,
                         init_type: InitType = InitType.ZERO,
                         var_init_type: InitType = InitType.MEAN,  # to avoid overflows
                         nan_backfill: NanBackfill = NanBackfill.FFILL
                         ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
    """
    EWM cross moment, beta or correlation of y on x, pair by pair.

    Runs ``M^{xy}_t = (1-λ) x_t y_t + λ M^{xy}_{t-1}`` and, for a ratio, the same recursion on
    ``x_t^2`` (and ``y_t^2``), each seeded before the first finite product, which then updates
    the seed. ``BETA`` is ``M^{xy} / M^{xx}`` and ``CORR`` is ``M^{xy} / sqrt(M^{xx} M^{yy})``;
    a ratio is NaN where its denominator is not strictly positive, a test that does not depend
    on the units of the data.

    Supported inputs:
        1. x and y DataFrames of the same shape: columns are paired by position and the output
           is labelled like y
        2. x a Series and y a DataFrame: x is paired with every column of y after an inner join
           of the indices; the output is a DataFrame labelled like y
        3. x and y Series: paired after an inner join; the output is a Series named like y
        4. x and y ndarrays of the same shape, 1-d or 2-d; the output is an ndarray

    Args:
        x_data: factor observations
        y_data: asset observations
        span: if given, overrides ``ewm_lambda`` via ``lambda = 1 - 2 / (span + 1)``
        ewm_lambda: decay in [0, 1)
        cross_xy_type: ``COVAR``, ``BETA`` or ``CORR``
        mean_adj_type: mean removed from x and y first, each on its own index
        init_type: seed of ``M^{xy}`` (and of an EWMA mean adjustment, with ``VAR`` read as
            ``MEAN`` there). ``ZERO`` by default
        var_init_type: seed of ``M^{xx}`` and ``M^{yy}``. The default ``MEAN`` is the
            full-sample mean square, a look-ahead that damps the ratios in the first
            ``1.5 N`` rows; ``X0`` or ``ZERO`` is point in time
        nan_backfill: missing-observation policy of every recursion

    Returns:
        the EWM cross statistic, in the container described above

    Raises:
        TypeError: for an unsupported pair of containers, arrays of different shapes or an
            unknown ``cross_xy_type``
    """

    # 1 - adjust by mean
    if mean_adj_type != MeanAdjType.NONE:

        x_data = compute_rolling_mean_adj(data=x_data,
                                          mean_adj_type=mean_adj_type,
                                          span=span,
                                          ewm_lambda=ewm_lambda,
                                          init_type=_mean_init_type(init_type),
                                          nan_backfill=nan_backfill)

        y_data = compute_rolling_mean_adj(data=y_data,
                                          mean_adj_type=mean_adj_type,
                                          span=span,
                                          ewm_lambda=ewm_lambda,
                                          init_type=_mean_init_type(init_type),
                                          nan_backfill=nan_backfill)

    # 2  take gen arrays and convert to ndarray to use with numbas
    if isinstance(x_data, pd.DataFrame) and isinstance(y_data, pd.DataFrame):
        # should be same dimensions
        x = npo.to_finite_np(data=x_data, fill_value=np.nan)
        y = npo.to_finite_np(data=y_data, fill_value=np.nan)
        if x.shape != y.shape:
            raise TypeError(f"x_data and y_data must have the same shape, "
                            f"got {x.shape} and {y.shape}")
        wrap = pd.DataFrame(index=y_data.index, columns=y_data.columns)

    elif isinstance(x_data, pd.Series) and isinstance(y_data, (pd.DataFrame, pd.Series)):
        joint = pd.concat([x_data, y_data], axis=1, sort=True, join='inner')
        # position 0 is x even if x_data.name is also a column of y_data
        x = npo.to_finite_np(data=joint.iloc[:, 0], fill_value=np.nan)
        if isinstance(y_data, pd.DataFrame):
            y = npo.to_finite_np(data=joint.iloc[:, 1:], fill_value=np.nan)
            x = np.tile(x.reshape(-1, 1), (1, y.shape[1]))
            wrap = pd.DataFrame(index=joint.index, columns=y_data.columns)
        else:
            y = npo.to_finite_np(data=joint.iloc[:, 1], fill_value=np.nan)
            wrap = pd.Series(index=joint.index, name=y_data.name, dtype=float)

    elif isinstance(x_data, np.ndarray) and isinstance(y_data, np.ndarray):
        if x_data.shape != y_data.shape:
            raise TypeError(f"ndarray x_data and y_data must have the same shape, "
                            f"got {x_data.shape} and {y_data.shape}")
        x = npo.to_finite_np(data=x_data.astype(float), fill_value=np.nan)
        y = npo.to_finite_np(data=y_data.astype(float), fill_value=np.nan)
        wrap = None

    else:
        raise TypeError(f"x_data of type {type(x_data)} with y_data of type {type(y_data)} is not "
                        f"supported: pass two DataFrames of the same shape, a Series with a "
                        f"DataFrame or a Series, or two ndarrays of the same shape")

    ewm_lambda = _to_decay(span=span, ewm_lambda=ewm_lambda)
    xy_covar = _run_ewm(a=np.multiply(x, y),
                        init_value=_second_moment_init(x=x, y=y, init_type=init_type),
                        ewm_lambda=ewm_lambda,
                        nan_backfill=nan_backfill)

    if cross_xy_type == CrossXyType.COVAR:
        cross_xy = xy_covar

    elif cross_xy_type in (CrossXyType.BETA, CrossXyType.CORR):
        x_var = _run_ewm(a=np.square(x),
                         init_value=_second_moment_init(x=x, y=x, init_type=var_init_type),
                         ewm_lambda=ewm_lambda,
                         nan_backfill=nan_backfill)
        if cross_xy_type == CrossXyType.BETA:
            divisor = x_var
        else:
            y_var = _run_ewm(a=np.square(y),
                             init_value=_second_moment_init(x=y, y=y, init_type=var_init_type),
                             ewm_lambda=ewm_lambda,
                             nan_backfill=nan_backfill)
            divisor = np.sqrt(np.multiply(x_var, y_var))
        # NumPy 2.x: explicit out= so masked positions are deterministic nan; the test is
        # strictly positive, not close to zero, so it does not depend on the units of the data
        cross_xy = np.divide(
            xy_covar, divisor,
            out=np.full_like(xy_covar, np.nan, dtype=float),
            where=np.greater(np.nan_to_num(divisor, nan=0.0), 0.0),
        )
    else:
        raise TypeError(f"unknown cross_xy_type = {cross_xy_type}")

    if isinstance(wrap, pd.Series):
        cross_xy = pd.Series(data=cross_xy, index=wrap.index, name=wrap.name)
    elif isinstance(wrap, pd.DataFrame):
        cross_xy = pd.DataFrame(data=cross_xy, index=wrap.index, columns=wrap.columns)

    return cross_xy


def compute_ewm_beta_alpha_forecast(x_data: Union[pd.DataFrame, pd.Series],
                                    y_data: pd.DataFrame,
                                    span: Union[float, np.ndarray] = None,
                                    ewm_lambda: Union[float, np.ndarray] = 0.94,
                                    mean_adj_type: MeanAdjType = MeanAdjType.NONE,
                                    init_type: InitType = InitType.X0,
                                    beta_init_value: Optional[
                                        Union[float, np.ndarray]
                                    ] = None,
                                    annualize: bool = False,
                                    nan_backfill: NanBackfill = NanBackfill.FFILL
                                    ) -> Tuple[
                                        pd.DataFrame,
                                        pd.DataFrame,
                                        pd.DataFrame,
                                        pd.DataFrame,
                                        pd.DataFrame,
                                        pd.DataFrame,
                                    ]:
    """Compute one-factor EWMA beta, alpha, one-step-ahead prediction and diagnostics.

    Beta is the ratio of exponentially weighted cross-moments,
    ``beta_t = E[x y]_t / E[x^2]_t``, and alpha the EWMA of the fitted residual
    ``y_t - beta_t x_t``; both are dated ``t`` and use the observation at ``t``. The prediction
    is the forecast of ``y_t`` from information up to ``t-1`` and the factor return at ``t``,
    ``beta_{t-1} x_t + alpha_{t-1}``, and is NaN on the first row. The residual variance and the
    R-squared are in-sample diagnostics of the fitted residual ``y_t - beta_t x_t - alpha_t``.

    With the default ``InitType.X0`` every recursion is seeded with its first observation, so
    every output is point in time. ``InitType.MEAN`` and ``InitType.VAR`` seed with full-sample
    statistics (look-ahead in the first ``1.5 N`` rows).

    When ``beta_init_value`` is supplied, the first jointly finite, nonzero factor observation
    ``x_f`` is replaced by a one-observation beta prior: both moments are seeded with it and the
    pair ``(x_f, y_f)`` is replaced by ``(x_f, beta_init_value * x_f)``, so the observed ``y_f``
    does not enter. The first finite beta equals the prior, which then keeps weight
    ``lambda^(t-f)`` in both moments and fades as later observations update the recursion,
    without look-ahead.

    Args:
        x_data: Factor-return Series, or one factor-return column per asset.
        y_data: Asset-return columns.
        span: EWMA span. Overrides ``ewm_lambda`` when supplied.
        ewm_lambda: EWMA decay used when ``span`` is None.
        mean_adj_type: Mean-adjustment convention; ``INSAMPLE`` is forward-looking.
        init_type: Seed of every recursion when no beta prior is given; see :class:`InitType`.
        beta_init_value: Optional scalar or per-asset initial beta prior.
        annualize: Whether to annualize factor and residual variances.
        nan_backfill: Missing-observation convention of every recursion.

    Returns:
        Beta, alpha, prediction, factor variance (one column per asset), residual variance and
        R-squared frames, all with the index and columns of ``y_data``.

    Raises:
        ValueError: If ``beta_init_value`` cannot broadcast to the asset columns
            or contains a non-finite value.
        IndexError: If paired factor and asset frames have different column counts.
        NotImplementedError: If the input container combination is unsupported.
    """
    # adjust indices in case
    if not x_data.index.equals(y_data.index):
        y_data = y_data.reindex(index=x_data.index, method='ffill')

    # 1 - adjust by mean
    if mean_adj_type != MeanAdjType.NONE:
        x_data = compute_rolling_mean_adj(data=x_data,
                                          mean_adj_type=mean_adj_type,
                                          span=span,
                                          ewm_lambda=ewm_lambda,
                                          init_type=_mean_init_type(init_type),
                                          nan_backfill=nan_backfill)

        y_data = compute_rolling_mean_adj(data=y_data,
                                          mean_adj_type=mean_adj_type,
                                          span=span,
                                          ewm_lambda=ewm_lambda,
                                          init_type=_mean_init_type(init_type),
                                          nan_backfill=nan_backfill)

    # 2  take gen arrays and convert to ndarray to use with numbdas
    if isinstance(x_data, pd.Series) and isinstance(y_data, pd.DataFrame):  # extend to df
        x_data = x_data.to_frame()
        x = npo.np_array_to_n_column_array(
            a=npo.to_finite_np(data=x_data, fill_value=np.nan),
            ncols=len(y_data.columns),
        )
        y = npo.to_finite_np(data=y_data, fill_value=np.nan)
    elif isinstance(x_data, pd.DataFrame) and isinstance(y_data, pd.DataFrame):
        if not len(x_data.columns) == len(y_data.columns):
            raise IndexError("x_data and y_data must have the same number of columns")
        # should be same dimensions
        x = npo.to_finite_np(data=x_data, fill_value=np.nan)
        y = npo.to_finite_np(data=y_data, fill_value=np.nan)
    else:
        raise NotImplementedError(
            'in compute_ewm_beta_resid: not implemented types '
            f'{type(x_data)} and {type(y_data)}'
        )
    x = x.astype(float)
    y = y.astype(float)
    ewm_lambda = _to_decay(span=span, ewm_lambda=ewm_lambda)

    # compute covar
    xy = np.multiply(x, y)
    x2 = np.square(x)
    xy_init = _second_moment_init(x=x, y=y, init_type=init_type)
    x2_init = _second_moment_init(x=x, y=x, init_type=init_type)
    if beta_init_value is not None:
        try:
            beta_init = np.broadcast_to(
                np.asarray(beta_init_value, dtype=float),
                (y.shape[1],),
            ).copy()
        except ValueError as error:
            raise ValueError(
                'beta_init_value must be scalar or match y_data columns'
            ) from error
        if not np.isfinite(beta_init).all():
            raise ValueError('beta_init_value must contain only finite values')
        xy_init = np.asarray(xy_init, dtype=float).copy()
        x2_init = np.asarray(x2_init, dtype=float).copy()
        for column in range(y.shape[1]):
            informative = np.flatnonzero(
                np.isfinite(x[:, column])
                & np.isfinite(y[:, column])
                & ~np.isclose(x[:, column], 0.0)
            )
            if informative.size == 0:
                continue
            first = informative[0]
            prior_variance = x2[first, column]
            x2[:first, column] = np.nan
            xy[:first, column] = np.nan
            x2[first, column] = prior_variance
            xy[first, column] = beta_init[column] * prior_variance
            x2_init[column] = prior_variance
            xy_init[column] = beta_init[column] * prior_variance
    xy_covar = _run_ewm(a=xy, init_value=xy_init, ewm_lambda=ewm_lambda,
                        nan_backfill=nan_backfill)

    # compute x var
    x_var = _run_ewm(a=x2, init_value=x2_init, ewm_lambda=ewm_lambda, nan_backfill=nan_backfill)

    # compute beta
    # NumPy 2.x: explicit out= so masked positions are deterministic nan, not uninitialized
    # memory. The test is strictly positive, so it does not depend on the units of the data.
    beta_xy = np.divide(
        xy_covar, x_var,
        out=np.full_like(xy_covar, np.nan, dtype=float),
        where=np.greater(np.nan_to_num(x_var, nan=0.0), 0.0),
    )

    # alpha: the EWMA of the fitted residual, point in time given the seeds
    y_fit0 = beta_xy * x
    resid = y - y_fit0
    ewm_alpha = _run_ewm(a=resid,
                         init_value=set_init_dim1(data=resid, init_type=_mean_init_type(init_type)),
                         ewm_lambda=ewm_lambda,
                         nan_backfill=nan_backfill)

    # one-step-ahead forecast of y_t: beta and alpha estimated to t-1, applied to x_t
    beta_lagged = np.vstack([np.full((1, beta_xy.shape[1]), np.nan), beta_xy[:-1]])
    alpha_lagged = np.vstack([np.full((1, ewm_alpha.shape[1]), np.nan), ewm_alpha[:-1]])
    y_prediction = beta_lagged * x + alpha_lagged

    # in-sample residual of the fit dated t
    resid = y - (y_fit0 + ewm_alpha)
    resid_var = _run_ewm(a=np.square(resid),
                         init_value=_second_moment_init(x=resid, y=resid, init_type=init_type),
                         ewm_lambda=ewm_lambda,
                         nan_backfill=nan_backfill)

    if annualize:
        an = infer_annualisation_factor_from_df(data=x_data)
        resid_var = an * resid_var
        x_var = an * x_var
    else:
        an = 1.0
    beta_xy = pd.DataFrame(data=beta_xy, index=y_data.index, columns=y_data.columns)
    alpha = pd.DataFrame(data=ewm_alpha, index=y_data.index, columns=y_data.columns)
    y_prediction = pd.DataFrame(data=y_prediction, index=y_data.index, columns=y_data.columns)
    resid_var = pd.DataFrame(data=resid_var, index=y_data.index, columns=y_data.columns)
    x_var = pd.DataFrame(data=x_var, index=y_data.index, columns=y_data.columns)

    # compute r2
    y_var0 = y_data.subtract(compute_ewm(
        data=y_data,
        span=span,
        ewm_lambda=ewm_lambda,
        nan_backfill=nan_backfill,
    ))
    y_var = an * _run_ewm(
        a=np.square(y_var0.to_numpy(dtype=float)),
        init_value=np.zeros(len(y_data.columns)),
        ewm_lambda=ewm_lambda,
        nan_backfill=nan_backfill,
    )
    # NumPy 2.x: work on ndarray with explicit out=; rebuild frame afterwards.
    resid_var_np = resid_var.to_numpy(dtype=float)
    ewm_r2_np = 1.0 - np.divide(
        resid_var_np, y_var,
        out=np.full_like(resid_var_np, np.nan),
        where=np.greater(np.nan_to_num(y_var, nan=0.0), 0.0),
    )
    ewm_r2 = np.clip(ewm_r2_np, a_min=0.0, a_max=1.0)
    ewm_r2 = pd.DataFrame(data=ewm_r2, index=y_data.index, columns=y_data.columns)

    return beta_xy, alpha, y_prediction, x_var, resid_var, ewm_r2


def compute_ewm_alpha_r2_given_prediction(y_data: pd.DataFrame,
                                          y_prediction: pd.DataFrame,
                                          span: Union[float, np.ndarray] = None,
                                          ewm_lambda: Union[float, np.ndarray] = 0.94,
                                          nan_backfill: NanBackfill = NanBackfill.FFILL
                                          ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """EWM alpha and R-squared of a given prediction.

    With ``e_t = y_t - prediction_t``, alpha is ``alpha_t = EWM(e)_t`` (``X0`` seed) and
    ``R2_t = 1 - EWM((e - alpha)^2)_t / EWM((y - EWM(y))^2)_t``, clipped to [0, 1] and NaN
    where the denominator is not positive. Both second moments start from a zero seed and are
    centred on contemporaneous EWMAs, so the ratio is an in-sample fit diagnostic, not an
    out-of-sample R-squared.

    Args:
        y_data: realised values, one column per series
        y_prediction: predictions with the index and columns of ``y_data``
        span: if given, overrides ``ewm_lambda`` via ``lambda = 1 - 2 / (span + 1)``
        ewm_lambda: decay in [0, 1)
        nan_backfill: missing-observation policy of every recursion

    Returns:
        the EWM alpha and the EWM R-squared, both with the index and columns of ``y_data``
    """
    ewm_lambda = _to_decay(span=span, ewm_lambda=ewm_lambda)
    # 1 - adjust by mean
    resid = y_data - y_prediction
    ewm_alpha = compute_ewm(data=resid, ewm_lambda=ewm_lambda, nan_backfill=nan_backfill)
    resid0 = resid.subtract(ewm_alpha)
    resid_var = _run_ewm(a=np.square(resid0.to_numpy(dtype=float)),
                         init_value=np.zeros(len(y_data.columns)), ewm_lambda=ewm_lambda,
                         nan_backfill=nan_backfill)

    y_var0 = y_data.subtract(compute_ewm(data=y_data, ewm_lambda=ewm_lambda,
                                         nan_backfill=nan_backfill))
    y_var = _run_ewm(a=np.square(y_var0.to_numpy(dtype=float)),
                     init_value=np.zeros(len(y_data.columns)), ewm_lambda=ewm_lambda,
                     nan_backfill=nan_backfill)

    # NumPy 2.x: explicit out= so masked positions (y_var<=0) are deterministic nan.
    ewm_r2 = 1.0 - np.divide(
        resid_var, y_var,
        out=np.full_like(resid_var, np.nan, dtype=float),
        where=np.greater(np.nan_to_num(y_var, nan=0.0), 0.0),
    )
    ewm_r2 = np.clip(ewm_r2, a_min=0.0, a_max=1.0)
    ewm_r2 = pd.DataFrame(data=ewm_r2, index=y_data.index, columns=y_data.columns)

    return ewm_alpha, ewm_r2


def compute_ewm_sharpe(returns: pd.DataFrame,
                       span: Union[float, np.ndarray] = 260,
                       norm_type: int = 1,
                       initial_sharpes: np.ndarray = None
                       ) -> pd.DataFrame:
    """Annualised EWM Sharpe-ratio paths of return columns.

    Missing returns are set to zero and the annualisation factor ``AN`` is inferred from the
    index. The EWM mean ``m_t`` and the EWM second moment run from zero seeds (or from the
    ``initial_sharpes`` prior) as the state before row 0, so the first return enters with weight
    ``1 - lambda``. The ratio is NaN where its denominator is zero.

    Args:
        returns: periodic returns, one column per strategy, with a frequency-bearing index
        span: EWM span of both moments
        norm_type: ``0`` the annualised EWM mean ``AN m_t``, not a ratio; ``1`` (default)
            ``sqrt(AN) m_t / sqrt(EWM(r^2)_t)``, the mean over the root mean square, which is
            the second moment about zero and compresses the Sharpe ratio by
            ``1 / sqrt(1 + SR^2 / AN)``; ``2`` ``sqrt(AN) m_t / sqrt(EWM((r - m)^2)_t)``, the
            mean over the EWM deviation from the running mean
        initial_sharpes: optional annualised Sharpe prior per column; seeds the mean with
            ``0.1 SR / AN`` and the second moment with ``0.01 / AN``, a 10% annual volatility

    Returns:
        the EWM Sharpe (or mean, for ``norm_type=0``) paths with the index and columns of
        ``returns``

    Raises:
        ValueError: if ``norm_type`` is not 0, 1 or 2
    """
    x = npo.to_finite_np(data=returns, fill_value=0.0).astype(float)
    ewm_lambda = _to_decay(span=span, ewm_lambda=0.94)
    an = infer_annualisation_factor_from_df(data=returns)
    san = np.sqrt(an)
    if initial_sharpes is not None:
        initial_vol = 0.1
        initial_mean = initial_vol * initial_sharpes / np.square(san)
        initial_var = np.square(initial_vol / san) * np.ones(len(returns.columns))
    else:
        initial_mean = np.zeros(len(returns.columns))
        initial_var = np.zeros(len(returns.columns))

    if norm_type == 0:
        ewm_mean = _run_ewm(a=x, init_value=initial_mean, ewm_lambda=ewm_lambda,
                            nan_backfill=NanBackfill.ZERO_FILL)
        sharpe = pd.DataFrame(data=an*ewm_mean, index=returns.index, columns=returns.columns)

    elif norm_type == 1 or norm_type == 2:
        ewm_mean = _run_ewm(a=x, init_value=initial_mean, ewm_lambda=ewm_lambda,
                            nan_backfill=NanBackfill.ZERO_FILL)
        if norm_type == 2:
            v = np.square(x-ewm_mean)
        else:
            v = np.square(x)

        ewm_var = _run_ewm(a=v, init_value=initial_var, ewm_lambda=ewm_lambda,
                           nan_backfill=NanBackfill.ZERO_FILL)
        ewm_vol = np.sqrt(ewm_var)
        # NumPy 2.x: explicit out= so masked positions (ewm_vol<=0) are deterministic nan.
        sharpe_np = san * np.divide(
            ewm_mean, ewm_vol,
            out=np.full_like(ewm_mean, np.nan, dtype=float),
            where=np.greater(ewm_vol, 0.0),
        )
        sharpe = pd.DataFrame(data=sharpe_np, index=returns.index, columns=returns.columns)
    else:
        raise ValueError(f"norm_type={norm_type} not implemented")

    return sharpe


def compute_ewm_sharpe_from_prices(prices: pd.DataFrame,
                                   freq: str = 'QE',
                                   span: int = 40,
                                   initial_sharpes: np.ndarray = None,
                                   norm_type: int = 2
                                   ) -> pd.DataFrame:

    prices = prices.asfreq(freq=freq, method='ffill')
    returns = np.log(prices.divide(prices.shift(1)))
    sharpe = compute_ewm_sharpe(returns=returns,
                                span=span,
                                initial_sharpes=initial_sharpes,
                                norm_type=norm_type)

    return sharpe


def compute_ewm_std1_norm(data: Union[pd.DataFrame, pd.Series],
                          span: Union[float, np.ndarray] = 260,
                          mean_adj_type: MeanAdjType = MeanAdjType.EWMA,
                          is_demean: bool = True,
                          is_nans_to_zero: bool = True
                          ) -> Union[pd.DataFrame, pd.Series]:
    """
    smoothed, volatility-normalised signal with unit standard deviation for IID input.

    With ``x~_t`` the (optionally demeaned) data and ``sigma_t`` its EWM volatility about zero,
    the output is ``c sqrt(N) EWM(x~ / sigma)_t`` with ``N = (1 + lambda) / (1 - lambda)``, the
    final EWM starting from a zero seed.
    Demeaning by the same-span EWMA removes the low-frequency content the final EWM keeps and
    leaves a standard deviation of ``1 / sqrt(1 + lambda)`` (0.71 at span 260) for IID input, so
    with ``is_demean=True`` and ``MeanAdjType.EWMA`` the output is multiplied by
    ``c = sqrt(1 + lambda)``; otherwise ``c = 1``. The output then has unit standard deviation
    for IID input in the stationary limit, with either setting.

    Args:
        data: observations, time along the first axis
        span: EWM span of the mean, the volatility and the smoother
        mean_adj_type: mean removed when ``is_demean`` is True
        is_demean: remove the mean before normalising
        is_nans_to_zero: replace missing outputs, including the warm-up, by zero

    Returns:
        the normalised signal, same container as ``data``
    """

    data_np = npo.to_finite_np(data=data, fill_value=np.nan)

    if is_demean:
        x_mean = compute_roll_mean(data=data_np, mean_adj_type=mean_adj_type, span=span)
        x_demean = data_np-x_mean
    else:
        x_demean = data_np

    ewm_var = compute_ewm(data=np.square(x_demean),
                          span=span,
                          is_unit_vol_scaling=False)
    ewm_vol = np.sqrt(ewm_var)

    x_std1_norm = npo.to_finite_ratio(x=x_demean, y=ewm_vol, fill_value=np.nan)
    # the unit-variance scaling sqrt(N) holds for a zero seed: seeding the smoother with its first
    # observation would give that one normalised value weight 1 and variance N at the start
    ewm_std1_norm = compute_ewm(data=x_std1_norm, span=span, init_type=InitType.ZERO,
                                is_unit_vol_scaling=True)
    if is_demean and mean_adj_type == MeanAdjType.EWMA:
        # the same-span EWMA demeaning leaves variance 1 / (1 + lambda) for IID input
        ewm_lambda = _to_decay(span=span, ewm_lambda=0.94)
        ewm_std1_norm = np.sqrt(1.0 + ewm_lambda) * ewm_std1_norm

    if isinstance(data, pd.Series):
        ewm_std1_norm = pd.Series(data=ewm_std1_norm, index=data.index, name=data.name)
    else:
        ewm_std1_norm = pd.DataFrame(data=ewm_std1_norm, index=data.index, columns=data.columns)

    if is_nans_to_zero:
        ewm_std1_norm = ewm_std1_norm.fillna(0.0)

    return ewm_std1_norm


# @njit
def ewm_vol_assymetric_np(returns: np.ndarray,
                          ewm_lambda: Union[float, np.ndarray] = 0.94,
                          annualization_factor: float = 1.0
                          ) ->Tuple[np.ndarray, np.ndarray]:
    """
    applies strictly to numpy arrays with utilization of numbda
    data: numpy with dimension = t*n
    ewm_lambda: float or ndarray of dimension n
    init_value: initial value of dimension n
    """
    ewm_lambda_1 = 1.0 - ewm_lambda

    # initialize all
    ewm_m, ewm_p = np.zeros_like(returns), np.zeros_like(returns)
    returns2 = np.square(returns)
    return2_m = np.where(np.less(returns, 0.0), returns2, np.nan)
    return2_p = np.where(np.greater(returns, 0.0), returns2, np.nan)
    ewm_m[0], ewm_p[0] = np.mean(return2_m[~np.isnan(return2_m)], axis=0), np.mean(return2_p[~np.isnan(return2_p)], axis=0)
    ewm_m0 = ewm_m[0]
    ewm_p0 = ewm_p[0]

    nt = returns.shape[0]
    for t in np.arange(1, nt):  # for x_t in x[1:]: # got by rows in x

        ewm_m1_ = ewm_lambda * ewm_m0 + ewm_lambda_1 * return2_m[t]
        ewm_m1 = np.where(np.isnan(return2_m[t]), ewm_m0, ewm_m1_)

        ewm_p1_ = ewm_lambda * ewm_p0 + ewm_lambda_1 * return2_p[t]
        ewm_p1 = np.where(np.isnan(return2_p[t]), ewm_p0, ewm_p1_)

        ewm_m[t] = ewm_m0 = ewm_m1
        ewm_p[t] = ewm_p0 = ewm_p1

    ewm_m = np.sqrt(annualization_factor*ewm_m)
    ewm_p = np.sqrt(annualization_factor*ewm_p)

    return ewm_m, ewm_p


def ewm_vol_assymetric(returns: Union[pd.Series, pd.DataFrame],
                        ewm_lambda: Union[float, np.ndarray] = 0.94,
                        annualization_factor: float = 1.0
                        ) -> Tuple[Union[pd.Series, pd.DataFrame], Union[pd.Series, pd.DataFrame]]:

    ewm_m, ewm_p = ewm_vol_assymetric_np(returns=returns.to_numpy(),
                                         ewm_lambda=ewm_lambda,
                                         annualization_factor=annualization_factor)
    if isinstance(returns, pd.DataFrame):
        ewm_m = pd.DataFrame(ewm_m, index=returns.index, columns=returns.columns)
        ewm_p = pd.DataFrame(ewm_p, index=returns.index, columns=returns.columns)
    elif isinstance(returns, pd.Series):
        ewm_m = pd.Series(ewm_m, index=returns.index, name=returns.name)
        ewm_p = pd.Series(ewm_p, index=returns.index, name=returns.name)
    else:
        raise TypeError
    return ewm_m, ewm_p

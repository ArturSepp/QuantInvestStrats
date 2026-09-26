"""
outlier filtering and winsorising of a time series driven by the score (x_t - m_t) / v_t.

``filter_outliers`` applies an ``OutlierPolicy`` in a fixed order: absolute ceiling and floor, the
optional log transform, a cut on the full-sample standard deviation, then a cut on the EWM score;
``OutlierPolicyTypes`` holds the ready-made policies. ``ewm_insample_winsorising`` cuts that score
at ``quantile_cut`` from each tail instead, with ``ReplacementType`` deciding what a rejected point
becomes - the EWM mean, NaN, or the corresponding quantile. ``compute_ewm_score`` is the shared
scoring step, clipping ``ewm_vol`` from below at its own ``clip_quantile``, column by column.

The score is contemporaneous: m_t and v_t include x_t, so |score| <= sqrt(λ/(1-λ)), 3.96 at
λ = 0.94, however extreme x_t is, and a move of k standard deviations scores about
λ k / sqrt(λ + (1-λ) k^2) (see ``score_of_move``). A score cut must sit below that bound to act.

Those three read the whole sample - the mean and volatility are contemporaneous and the quantiles
full-sample - so they clean a descriptive exhibit, not a backtest path.
``ewm_winsdor_markovian_score`` is the non-anticipating alternative: it scores x_t against the EWM
state at t-1 and returns the cleaned series alongside that state and the score.
"""
# packages
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
from typing import Union, NamedTuple, Optional, Tuple
from qis.models.linear.ewm import compute_ewm, compute_ewm_vol


class ReplacementType(Enum):
    """
    what a rejected observation becomes.

    Attributes:
        EWMA_MEAN: the EWM mean of the cleaned data at that date
        NAN: a missing value
        QUANTILES: the full-sample quantile of the data on the side it was cut
    """
    EWMA_MEAN = 1
    NAN = 2
    QUANTILES = 3


class OutlierPolicy(NamedTuple):
    """
    specify filtering policy params
    """
    abs_ceil: Optional[float] = None  # remove all above
    abs_floor: Optional[float] = None  # remove all below
    std_abs_ceil: Optional[float] = None  # > 0
    std_abs_floor: Optional[float] = None  # < 0
    std_ewm_ceil: Optional[float] = None  # >0
    std_ewm_floor: Optional[float] = None  # <0
    ewm_lambda: Union[float, np.ndarray] = 0.94
    is_log_transform: bool = False
    nan_replacement_type: ReplacementType = ReplacementType.NAN


def score_of_move(num_std: float, ewm_lambda: float = 0.94) -> float:
    """EWM score of an observation ``num_std`` standard deviations from a zero-mean state.

    ``compute_ewm_score`` measures x_t against an EWM mean and volatility that already contain
    x_t. From a state with mean 0 and second moment sigma^2, an observation k sigma away moves the
    mean to (1-λ) k sigma and the second moment to (λ + (1-λ) k^2) sigma^2, so it scores
    ``λ k / sqrt(λ + (1-λ) k^2)``. The score increases with k towards ``λ / sqrt(1-λ)``, 3.84 at
    λ = 0.94, which is how a threshold in standard deviations translates into a threshold on
    the score.

    Args:
        num_std: size of the move in standard deviations of the state
        ewm_lambda: EWM decay of the score

    Returns:
        the score of that move
    """
    return ewm_lambda * num_std / np.sqrt(ewm_lambda + (1.0 - ewm_lambda) * num_std ** 2)


# the EWM score cut of the soft presets: the score of a 10-standard-deviation move at λ = 0.94
_TEN_STD_SCORE = score_of_move(num_std=10.0, ewm_lambda=0.94)


class OutlierPolicyTypes(OutlierPolicy, Enum):
    """
    ready-made outlier policies.

    The EWM score is contemporaneous and bounded by sqrt(λ/(1-λ)) = 3.96 at λ = 0.94, so the soft
    presets cut it at the score of a 10-standard-deviation move, ``score_of_move(10) = 3.57``,
    rather than at 10, a level the score can never reach.

    Attributes:
        HARD_CEIL_POLICY: drop values below 1e-4 and above the full-sample mean plus 10
            full-sample standard deviations
        RANGE_CEIL_POLICY: the same cuts as HARD_CEIL_POLICY
        SOFT_RANGE_CEIL_POLICY: drop values below 1e-8, above the mean plus 10 standard
            deviations, and with an EWM score above that of a 10-standard-deviation move
        SOFT_POSITIVE_LOG_POLICY: drop values below 1e-8, then apply the EWM score cut of
            SOFT_RANGE_CEIL_POLICY to the log of the data
        NONE: no policy
    """
    HARD_CEIL_POLICY = OutlierPolicy(abs_floor=0.0001,
                                     std_abs_ceil=10.0)

    RANGE_CEIL_POLICY = OutlierPolicy(abs_floor=0.0001,
                                      std_abs_ceil=10.0)

    SOFT_RANGE_CEIL_POLICY = OutlierPolicy(abs_floor=1e-8,
                                           std_ewm_ceil=_TEN_STD_SCORE,
                                           std_ewm_floor=None,
                                           std_abs_ceil=10.0)

    SOFT_POSITIVE_LOG_POLICY = OutlierPolicy(abs_floor=1e-8,
                                             std_ewm_ceil=_TEN_STD_SCORE,
                                             std_ewm_floor=None,
                                             is_log_transform=True)
    NONE = None


def filter_outliers(data: Union[pd.DataFrame, pd.Series, np.ndarray],
                    outlier_policy: OutlierPolicy
                    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
    """Remove outliers by the cuts of an ``OutlierPolicy``, applied in a fixed order.

    The order is: absolute ceiling and floor; the optional log transform; a cut at the
    full-sample mean plus ``std_abs_ceil`` (or ``std_abs_floor``) full-sample standard
    deviations; a cut of the contemporaneous EWM score of the cleaned data at ``std_ewm_ceil``
    and ``std_ewm_floor``; the inverse log. Rejected points become NaN, or the EWM mean with
    ``ReplacementType.EWMA_MEAN``. Invalid-value warnings are silenced for the duration of the
    call only. The cuts use the whole sample: descriptive cleaning, not a backtest path.

    Args:
        data: observations, time along the first axis
        outlier_policy: the cuts; see ``OutlierPolicyTypes`` for presets

    Returns:
        the cleaned data, same container as ``data``

    Raises:
        TypeError: for an unsupported container, or a log transform without ``abs_floor``
    """
    with np.errstate(invalid='ignore'):  # local: numpy's error state is restored on exit
        return _filter_outliers(data=data, outlier_policy=outlier_policy)


def _filter_outliers(data: Union[pd.DataFrame, pd.Series, np.ndarray],
                     outlier_policy: OutlierPolicy
                     ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
    """The body of :func:`filter_outliers`."""
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        orig_data = data.to_numpy()
    elif isinstance(data, np.ndarray):
        orig_data = data
    else:
        raise TypeError('filter_outliers: unsupported data type')

    clean_data = orig_data.copy()
    #  keep track of nans - nans will be put back to output data
    non_nan_cond = np.isfinite(orig_data)

    # imnitial replacement is using nans
    nan_replacement = np.full_like(orig_data, np.nan, dtype=np.float64)

    # NumPy 2.x: comparison/arithmetic ufuncs with `where=` need an explicit `out=` buffer,
    # otherwise masked positions contain uninitialized memory (random bools for comparisons).
    # For comparisons used inside np.where(...) we want masked positions to be False so the
    # outer np.where selects `clean_data` (which already carries nan in those positions).
    def _greater_masked(a, b, mask):
        return np.greater(a, b, out=np.zeros_like(mask, dtype=bool), where=mask)

    def _less_masked(a, b, mask):
        return np.less(a, b, out=np.zeros_like(mask, dtype=bool), where=mask)

    # remove absolute outliers
    if outlier_policy.abs_ceil is not None:
        clean_data = np.where(_greater_masked(clean_data, outlier_policy.abs_ceil, non_nan_cond),
                              nan_replacement, clean_data)
    if outlier_policy.abs_floor is not None:
        clean_data = np.where(_less_masked(clean_data, outlier_policy.abs_floor, non_nan_cond),
                              nan_replacement, clean_data)

    # now apply log transform
    if outlier_policy.is_log_transform:
        if outlier_policy.abs_floor is None:
            raise TypeError('is_log_transform must be applied with abs_floor > 0')
        log_cond = _greater_masked(clean_data, 0.0, non_nan_cond)
        # np.log with explicit nan-filled out= for masked positions.
        clean_data = np.log(clean_data,
                            out=np.full_like(clean_data, np.nan, dtype=float),
                            where=log_cond)
    else:
        log_cond = None

    # remove relative outliers to in-sample std
    if outlier_policy.std_abs_ceil is not None or outlier_policy.std_abs_floor is not None:
        nan_mean = np.nanmean(clean_data, axis=0)
        nan_std = np.nanstd(clean_data, axis=0)

        if outlier_policy.std_abs_ceil is not None:
            # nan_mean/nan_std are 1-D; broadcast with 2-D mask produces 2-D result — keep existing semantics.
            ceil_broadcast = np.broadcast_to(nan_mean + nan_std * outlier_policy.std_abs_ceil, non_nan_cond.shape)
            ceil = np.where(non_nan_cond, ceil_broadcast, np.nan)
            clean_data = np.where(_greater_masked(clean_data, ceil, non_nan_cond), nan_replacement, clean_data)

        if outlier_policy.std_abs_floor is not None:
            floor_broadcast = np.broadcast_to(nan_mean + nan_std * outlier_policy.std_abs_floor, non_nan_cond.shape)
            floor = np.where(non_nan_cond, floor_broadcast, np.nan)
            clean_data = np.where(_less_masked(clean_data, floor, non_nan_cond), nan_replacement, clean_data)

    # now rolling ewm outliers
    if outlier_policy.std_ewm_ceil is not None or outlier_policy.std_ewm_floor is not None:

        ewm_mean, score = compute_ewm_score(data=clean_data, ewm_lambda=outlier_policy.ewm_lambda)
        if outlier_policy.std_ewm_ceil is not None:
            clean_data = np.where(_greater_masked(score, outlier_policy.std_ewm_ceil, non_nan_cond),
                                  nan_replacement, clean_data)

        if outlier_policy.std_ewm_floor is not None:
            clean_data = np.where(_less_masked(score, outlier_policy.std_ewm_floor, non_nan_cond),
                                  nan_replacement, clean_data)
    if outlier_policy.is_log_transform:
        # Inverse of the earlier log: exp on the same mask, with explicit out=.
        clean_data = np.exp(clean_data,
                            out=np.full_like(clean_data, np.nan, dtype=float),
                            where=log_cond)

    # implemented replacement type is EWMA mean
    if outlier_policy.nan_replacement_type == ReplacementType.EWMA_MEAN:
        ewm_mean, _ = compute_ewm_score(data=clean_data, ewm_lambda=outlier_policy.ewm_lambda)
        filtered_data = np.where(np.isfinite(clean_data), clean_data, ewm_mean)
    else:
        filtered_data = np.where(non_nan_cond, clean_data, nan_replacement)

    if isinstance(data, pd.DataFrame):
        filtered_data = pd.DataFrame(data=filtered_data, columns=data.columns, index=data.index)
    elif isinstance(data, pd.Series):
        filtered_data = pd.Series(data=filtered_data, name=data.name, index=data.index)

    return filtered_data


def ewm_insample_winsorising(data: Union[pd.DataFrame, pd.Series, np.ndarray],
                             ewm_lambda: Union[float, np.ndarray] = 0.94,
                             quantile_cut: float = 0.025,
                             nan_replacement_type: ReplacementType = ReplacementType.EWMA_MEAN
                             ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
    """Winsorise each column at full-sample quantiles of its EWM score.

    Points whose score lies below the ``quantile_cut`` or above the ``1 - quantile_cut``
    quantile of the column's scores are replaced as ``nan_replacement_type`` says. The quantiles
    ignore missing values, so a column with gaps is winsorised like any other. The quantiles use
    the whole sample: descriptive cleaning, not a backtest path.

    Args:
        data: observations, time along the first axis
        ewm_lambda: EWM decay of the score
        quantile_cut: tail probability cut on each side
        nan_replacement_type: ``EWMA_MEAN`` (default), ``NAN``, or ``QUANTILES`` for the
            full-sample quantile of the data on the side that was cut

    Returns:
        the winsorised data, same container as ``data``

    Raises:
        TypeError: for an unsupported container or replacement type
    """
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        np_data = data.to_numpy()
    elif isinstance(data, np.ndarray):
        np_data = data.copy()
    else:
        raise TypeError('ewm_winsorising: unsupported data type')

    # 1 compute ewm score
    ewm_mean, score = compute_ewm_score(data=np_data, ewm_lambda=ewm_lambda)

    with warnings.catch_warnings():  # an all-nan column has nan quantiles and is left as is
        warnings.simplefilter('ignore', RuntimeWarning)
        lower_quantile = np.nanquantile(score, quantile_cut, axis=0)
        upper_quantile = np.nanquantile(score, 1.0-quantile_cut, axis=0)
        data_lower = np.nanquantile(np_data, quantile_cut, axis=0)
        data_upper = np.nanquantile(np_data, 1.0-quantile_cut, axis=0)

    if nan_replacement_type == ReplacementType.EWMA_MEAN:
        replacement_cond = np.logical_or(score < lower_quantile, score > upper_quantile)
        winsor_data = np.where(replacement_cond, ewm_mean, np_data)

    elif nan_replacement_type == ReplacementType.NAN:
        replacement_cond = np.logical_or(score < lower_quantile, score > upper_quantile)
        winsor_data = np.where(replacement_cond, np.full_like(np_data, np.nan), np_data)

    elif nan_replacement_type == ReplacementType.QUANTILES:
        winsor_data = np.where(score < lower_quantile, data_lower, np_data)
        winsor_data = np.where(score > upper_quantile, data_upper, winsor_data)
    else:
        raise TypeError('replacement_type not implemented')

    if isinstance(data, pd.DataFrame):
        winsor_data = pd.DataFrame(data=winsor_data, columns=data.columns, index=data.index)
    elif isinstance(data, pd.Series):
        winsor_data = pd.Series(data=winsor_data, name=data.name, index=data.index)

    return winsor_data


def compute_ewm_score(data: np.ndarray,
                      ewm_lambda: Union[float, np.ndarray] = 0.94,
                      is_clip: bool = True,
                      clip_quantile: float = 0.16
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Contemporaneous EWM score ``(x_t - m_t) / max(sigma_t, c)`` of each column.

    ``m_t`` is the ``X0``-seeded EWM mean and ``sigma_t`` the EWM volatility about zero of
    :func:`compute_ewm_vol`; both include x_t, so the score is bounded by sqrt(λ/(1-λ)). ``c`` is
    the full-sample ``clip_quantile`` quantile of the column's own ``sigma``, a look-ahead floor.

    Args:
        data: observations, shape (t,) or (t, n)
        ewm_lambda: EWM decay of the mean and the volatility
        is_clip: floor the volatility of each column at its ``clip_quantile`` quantile
        clip_quantile: quantile level of that floor

    Returns:
        the EWM mean and the score, each shaped like ``data``; the score is NaN where x_t is
    """
    ewm_mean = compute_ewm(data=data, ewm_lambda=ewm_lambda)
    ewm_vol = compute_ewm_vol(data=data, ewm_lambda=ewm_lambda)
    if is_clip:  # floor each column's vol at its own clip_quantile quantile
        with warnings.catch_warnings():  # an all-nan column has no floor
            warnings.simplefilter('ignore', RuntimeWarning)
            vol_floor = np.nanquantile(ewm_vol, clip_quantile, axis=0)
        ewm_vol = np.fmax(ewm_vol, vol_floor)
    non_nan_cond = np.isfinite(data)
    # NumPy 2.x: explicit out= so masked positions are deterministic nan.
    diff = np.subtract(data, ewm_mean)
    score = np.divide(
        diff, ewm_vol,
        out=np.full_like(diff, np.nan, dtype=float),
        where=non_nan_cond,
    )
    return ewm_mean, score


# @njit
def ewm_winsdor_markovian_score(a: np.ndarray,
                                init_value: Union[float, np.ndarray],
                                init_var: Union[float, np.ndarray] = None,
                                score_threshold: float = 5.0,
                                span: Union[int, np.ndarray] = 31,
                                ewm_lambda: Union[float, np.ndarray] = None,
                                is_start_from_first_nonan: bool = True
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """
    use ewma score to filter out outliers in non-anticipating markovian way
    data: numpy with dimension = t*n

    score_t is defined as non-anticipating:
    score_t = (x[t]-ewm[t-1] / np.sqrt(ewm2[t-1])
    outlier x[t] is defined when:
    np.abs(score_t) > threshold

    if x[t] is outlier, it is ignored for ewm:
    ewm[t] = ewm[t-1]
    ewm2[t] = ewm2[t-1]
    else ewm is computed using recursion:
    ewm[t] = (1-lambda) * x[t] + lambda*ewm[t-1]
    ewm2[t] = (1-lambda) * (x[t]-ewm[t])^2 + lambda*ewm2[t-1]

    if x[t] is nan:
    ewm[t] = ewm[t-1]
    ewm2[t] = ewm2[t-1]
    and the cleaned value is nan; an outlier's cleaned value is the previous cleaned value

    assumption is that no np.nan value is returned from the function

    ewm_lambda: float or ndarray of dimension n
    init_value: initial value of dimension n
    start_from_first_nonan: start filling nans only from the first non-nan in underlying data: recomended because
                            it avoids backfilling of init_value
    """
    if span is not None:
        ewm_lambda = 1.0 - 2.0 / (span + 1.0)
    ewm_lambda_1 = 1.0 - ewm_lambda

    is_1d = (a.ndim == 1)  # or a.shape[1] == 1)

    # initialize all
    ewm = np.full_like(a, fill_value=np.nan, dtype=np.double)
    ewm2 = np.full_like(a, fill_value=np.nan, dtype=np.double)
    score = np.full_like(a, fill_value=np.nan, dtype=np.double)
    clean_a = np.full_like(a, fill_value=np.nan, dtype=np.double)

    if init_var is None:
        if is_1d:
            init_var = 0.1
        else:
            init_var = 0.1*np.ones(a.shape[1])

    if is_start_from_first_nonan:
        if is_1d:  # cannot use np.where
            last_ewm = init_value if np.isfinite(a[0]) else np.nan
        else:
            last_ewm = np.where(np.isfinite(a[0]), init_value, np.nan)
    else:
        last_ewm = init_value

    last_ewm2 = np.maximum(last_ewm * last_ewm, init_var)

    ewm[0] = last_ewm
    ewm2[0] = last_ewm2
    score[0] = 0.0

    # recurse from 1
    for t in np.arange(1, a.shape[0]):
        a_t = a[t]

        if is_start_from_first_nonan:
            # detect starting nonnans for when last ewma was np.nan and a_t is finite
            if is_1d:  # cannot use np.where
                if np.isfinite(last_ewm) == False and np.isfinite(a_t) == True:  # trick: if last_ewm is nan
                    last_ewm = init_value
                    last_ewm2 = np.maximum(init_value*init_value, init_var)
            else:
                new_nonnans = np.logical_and(np.isfinite(last_ewm) == False, np.isfinite(a_t) == True)
                if np.any(new_nonnans):
                    last_ewm = np.where(new_nonnans, init_value, last_ewm)
                    last_ewm2 = np.where(new_nonnans, np.maximum(init_value*init_value, init_var), last_ewm2)

        # fill nan-values
        current_ewm_ = ewm_lambda * last_ewm + ewm_lambda_1 * a_t
        current_ewm2_ = ewm_lambda * last_ewm2 + ewm_lambda_1 * np.square(a_t-current_ewm_)

        score_vol = np.sqrt(np.where(np.greater(last_ewm2, 0.0), last_ewm2, np.nan))
        score_t = (a_t - last_ewm) / score_vol
        is_outlier = np.abs(score_t) >= score_threshold
        # an outlier or a missing observation leaves the state unchanged
        is_hold = np.logical_or(is_outlier, np.logical_not(np.isfinite(a_t)))

        if is_1d:   # np.where cannot be used
            if is_hold:
                current_ewm = last_ewm
                current_ewm2 = last_ewm2
                clean_a_ = clean_a[t-1] if is_outlier else a_t
            else:
                current_ewm = current_ewm_
                current_ewm2 = current_ewm2_
                clean_a_ = a_t
        else:
            current_ewm = np.where(is_hold, last_ewm, current_ewm_)
            current_ewm2 = np.where(is_hold, last_ewm2, current_ewm2_)
            clean_a_ = np.where(is_outlier, clean_a[t-1], a_t)

        ewm[t] = last_ewm = current_ewm
        ewm2[t] = last_ewm2 = current_ewm2
        score[t] = score_t
        clean_a[t] = clean_a_

    return clean_a, ewm, ewm2, score

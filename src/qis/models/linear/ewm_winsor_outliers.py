"""
outlier filtering and winsorising of a time series driven by the score (x_t - m_t) / v_t.

``filter_outliers`` applies an ``OutlierPolicy`` in a fixed order: absolute ceiling and floor, the
optional log transform, a cut on the full-sample standard deviation, then a cut on the EWM score;
``OutlierPolicyTypes`` holds the ready-made policies. ``ewm_insample_winsorising`` cuts that score
at ``quantile_cut`` from each tail instead, with ``ReplacementType`` deciding what a rejected point
becomes - the EWM mean, NaN, or the corresponding quantile. ``compute_ewm_score`` is the shared
scoring step, clipping ``ewm_vol`` from below at its own ``clip_quantile``.

Those three read the whole sample - the mean and volatility are contemporaneous and the quantiles
full-sample - so they clean a descriptive exhibit, not a backtest path.
``ewm_winsdor_markovian_score`` is the non-anticipating alternative: it scores x_t against the EWM
state at t-1 and returns the cleaned series alongside that state and the score.
"""
# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
from typing import Union, NamedTuple, Optional, Tuple
from qis.models.linear.ewm import compute_ewm, compute_ewm_vol


class ReplacementType(Enum):
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


class OutlierPolicyTypes(OutlierPolicy, Enum):
    """
    defined policy type
    """
    HARD_CEIL_POLICY = OutlierPolicy(abs_floor=0.0001,
                                     std_abs_ceil=10.0)

    RANGE_CEIL_POLICY = OutlierPolicy(abs_floor=0.0001,
                                      std_abs_ceil=10.0)

    SOFT_RANGE_CEIL_POLICY = OutlierPolicy(abs_floor=1e-8,
                                           std_ewm_ceil=10.0,
                                           std_ewm_floor=None,
                                           std_abs_ceil=10.0)

    SOFT_POSITIVE_LOG_POLICY = OutlierPolicy(abs_floor=1e-8,
                                             std_ewm_ceil=10.0,
                                             std_ewm_floor=None,
                                             is_log_transform=True)
    NONE = None


def filter_outliers(data: Union[pd.DataFrame, pd.Series, np.ndarray],
                    outlier_policy: OutlierPolicy
                    ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

    np.seterr(invalid='ignore')  # off warnings

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

    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        np_data = data.to_numpy()
    elif isinstance(data, np.ndarray):
        np_data = data.copy()
    else:
        raise TypeError('ewm_winsorising: unsupported data type')

    # 1 compute ewm score
    ewm_mean, score = compute_ewm_score(data=np_data, ewm_lambda=ewm_lambda)

    lower_quantile = np.quantile(score, quantile_cut, axis=0)
    upper_quantile = np.quantile(score, 1.0-quantile_cut, axis=0)
    # print(f"lower_quantile={lower_quantile}, upper_quantile={upper_quantile}")

    if nan_replacement_type == ReplacementType.EWMA_MEAN:
        replacement_cond = np.logical_or(score < lower_quantile, score > upper_quantile)
        winsor_data = np.where(replacement_cond, ewm_mean, np_data)

    elif nan_replacement_type == ReplacementType.NAN:
        replacement_cond = np.logical_or(score < lower_quantile, score > upper_quantile)
        winsor_data = np.where(replacement_cond, np.full_like(np_data, np.nan), np_data)

    elif nan_replacement_type == ReplacementType.QUANTILES:
        winsor_data = np.where(score < lower_quantile, np.quantile(np_data, quantile_cut, axis=0), np_data)
        winsor_data = np.where(score > upper_quantile, np.quantile(np_data, 1.0-quantile_cut, axis=0), winsor_data)
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
                      ) -> (np.ndarray, np.ndarray):

    ewm_mean = compute_ewm(data=data, ewm_lambda=ewm_lambda)
    ewm_vol = compute_ewm_vol(data=data, ewm_lambda=ewm_lambda)
    if is_clip:  # remove small values below 1 _ std quantile
        ewm_vol = np.clip(a=ewm_vol, a_min=np.nanquantile(ewm_vol, clip_quantile), a_max=None)
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

        # score_t = np.divide(a_t - last_ewm, np.sqrt(last_ewm2), where=np.greater(last_ewm2, 0.0))
        score_t = np.where(np.greater(last_ewm2, 0.0), (a_t - last_ewm)/np.sqrt(last_ewm2), np.nan)
        is_outlier = np.abs(score_t) >= score_threshold

        if is_1d:   # np.where cannot be used
            if is_outlier:
                current_ewm = last_ewm
                current_ewm2 = last_ewm2
                clean_a_ = clean_a[t-1]
            else:
                current_ewm = current_ewm_
                current_ewm2 = current_ewm2_
                clean_a_ = a_t
        else:
            current_ewm = np.where(is_outlier, current_ewm_, last_ewm)
            current_ewm2 = np.where(is_outlier, current_ewm2_, last_ewm2)
            clean_a_ = np.where(is_outlier, clean_a[t-1], a_t)

        ewm[t] = last_ewm = current_ewm
        ewm2[t] = last_ewm2 = current_ewm2
        score[t] = score_t
        clean_a[t] = clean_a_

    return clean_a, ewm, ewm2, score

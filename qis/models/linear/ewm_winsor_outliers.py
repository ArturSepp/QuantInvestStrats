"""
implement winsorizing of time series data using ewm
1. compute mean_t and vol_t
2. select x% of outliers defined by normalized score (x_t-mean_t) / vol_t
3. replace or trim outliers as specified
"""
# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit
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

    # remove absolute outliers
    if outlier_policy.abs_ceil is not None:
        clean_data = np.where(np.greater(clean_data, outlier_policy.abs_ceil, where=non_nan_cond),
                              nan_replacement, clean_data)
    if outlier_policy.abs_floor is not None:
        clean_data = np.where(np.less(clean_data, outlier_policy.abs_floor, where=non_nan_cond),
                              nan_replacement, clean_data)

    # now apply log transform
    if outlier_policy.is_log_transform:
        if outlier_policy.abs_floor is None:
            raise TypeError('is_log_transform must be applied with abs_floor > 0')
        log_cond = np.greater(clean_data, 0.0, where=non_nan_cond)
        clean_data = np.log(clean_data, where=log_cond)
    else:
        log_cond = None

    # remove relative outliers to in-sample std
    if outlier_policy.std_abs_ceil is not None or outlier_policy.std_abs_floor is not None:
        nan_mean = np.nanmean(clean_data, axis=0)
        nan_std = np.nanstd(clean_data, axis=0)

        if outlier_policy.std_abs_ceil is not None:
            ceil = np.add(nan_mean, nan_std*outlier_policy.std_abs_ceil, where=non_nan_cond)
            clean_data = np.where(np.greater(clean_data, ceil, where=non_nan_cond), nan_replacement, clean_data)

        if outlier_policy.std_abs_floor is not None:
            floor = np.add(nan_mean, nan_std*outlier_policy.std_abs_floor, where=non_nan_cond)
            clean_data = np.where(np.less(clean_data, floor, where=non_nan_cond), nan_replacement, clean_data)

    # now rolling ewm outliers
    if outlier_policy.std_ewm_ceil is not None or outlier_policy.std_ewm_floor is not None:

        ewm_mean, score = compute_ewm_score(data=clean_data, ewm_lambda=outlier_policy.ewm_lambda)
        if outlier_policy.std_ewm_ceil is not None:
            clean_data = np.where(np.greater(score, outlier_policy.std_ewm_ceil, where=non_nan_cond),
                                  nan_replacement, clean_data)

        if outlier_policy.std_ewm_floor is not None:
            clean_data = np.where(np.less(score, outlier_policy.std_ewm_floor, where=non_nan_cond),
                                  nan_replacement, clean_data)
    if outlier_policy.is_log_transform:
        clean_data = np.exp(clean_data, where=log_cond)

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
    #print(f"lower_quantile={lower_quantile}, upper_quantile={upper_quantile}")

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
    score = np.divide(np.subtract(data, ewm_mean), ewm_vol, where=non_nan_cond)
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
                new_nonnans = np.logical_and(np.isfinite(last_ewm)==False, np.isfinite(a_t)==True)
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


class UnitTests(Enum):
    TEST1 = 1
    MARKOVIAN_WINDSOR = 2


def run_unit_test(unit_test: UnitTests):

    np.random.seed(2) #freeze seed

    import qis as qis

    dates = pd.date_range(start='12/31/2018', end='12/31/2019', freq='B')
    n = 2

    data = pd.DataFrame(data=np.random.standard_t(df=2, size=(len(dates), n)),
                        index=dates,
                        columns=['x'+str(m+1) for m in range(n)])
    # data = pd.Series(data=data_np, index=dates, name='data')

    if unit_test == UnitTests.TEST1:
        winsor_data = ewm_insample_winsorising(data=data,
                                               ewm_lambda=0.94,
                                               nan_replacement_type=ReplacementType.EWMA_MEAN,
                                               quantile_cut=0.05)

        winsor_data.columns = [x + ' winsor' for x in data.columns]

        plot_data = pd.concat([data, winsor_data], axis=1)
        title = 'Data Winsor'

        qis.plot_time_series(df=plot_data,
                             title=title,
                             legend_loc='upper left',
                             legend_stats=qis.LegendStats.AVG,
                             last_label=qis.LastLabel.AVERAGE_VALUE,
                             trend_line=qis.TrendLine.AVERAGE,
                             var_format='{:.2f}')

    elif unit_test == UnitTests.MARKOVIAN_WINDSOR:
        clean_x, ewm, ewm2, score = ewm_winsdor_markovian_score(a=data.to_numpy(), span=7,
                                                                init_value=0.0,
                                                                init_var=0.1*np.ones(len(data.columns)))
        clean_x = pd.DataFrame(clean_x, index=dates, columns=[x + ' winsor' for x in data.columns])
        plot_data = pd.concat([data, clean_x], axis=1)
        qis.plot_time_series(df=plot_data,
                             title='Windsor',
                             legend_loc='upper left',
                             legend_stats=qis.LegendStats.AVG_STD_LAST,
                             var_format='{:.2f}')

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.MARKOVIAN_WINDSOR

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

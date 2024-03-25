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
from enum import Enum
from typing import Union, NamedTuple, Optional
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

    np.seterr(invalid='ignore') # off warnings

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


def ewm_winsorising(data: Union[pd.DataFrame, pd.Series, np.ndarray],
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
    ewm_mean, score = compute_ewm_score(data=np_data,
                               ewm_lambda=ewm_lambda)

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


class UnitTests(Enum):
    TEST1 = 1


def run_unit_test(unit_test: UnitTests):

    np.random.seed(2) #freeze seed
    ewm_lambda = 0.94
    import qis as qis
    #simulate t*n data
    dates = pd.date_range(start='12/31/2018', end='12/31/2019', freq='B')
    n = 1

    data = pd.DataFrame(data=np.random.standard_t(df=2, size=(len(dates), n)),
                        index=dates,
                        columns=['x'+str(m+1) for m in range(n)])

    if unit_test == UnitTests.TEST1:

        winsor_data = ewm_winsorising(data=data,
                                      ewm_lambda=ewm_lambda,
                                      nan_replacement_type=ReplacementType.EWMA_MEAN,
                                      quantile_cut=0.05)

        winsor_data.columns = [x + ' winsor' for x in data.columns]

        plot_data = pd.concat([data, winsor_data], axis=1)
        title = 'Data Winsor'
        # plot_data.to_clipboard()

    else:
        return

    qis.plot_time_series(df=plot_data,
                         title=title,
                         legend_loc='upper left',
                         legend_stats=qis.LegendStats.AVG,
                         last_label=qis.LastLabel.AVERAGE_VALUE,
                         trend_line=qis.TrendLine.AVERAGE,
                         var_format='{:.2f}')

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.TEST1

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)


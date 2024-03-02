"""
common numpy operations
"""
import time
import numpy as np
import pandas as pd
from enum import Enum
from typing import Union, Optional, Tuple, Callable
from numba import njit


@njit
def np_apply_along_axis(func, axis: int, a: np.ndarray) -> np.ndarray:
    """
    numba function to apply func for 2-d array
    https://github.com/numba/numba/issues/1269
    """
    assert a.ndim == 2
    assert axis in [0, 1]

    if axis == 0:
        result = np.zeros((1, a.shape[1]))
        for i in range(a.shape[1]):
            result[0, i] = func(a[:, i])
    else:  # axis=1 with keep dims
        result = np.zeros((a.shape[0], 1))
        for i in range(a.shape[0]):
            result[i] = func(a[i, :])
    return result


@njit
def np_min(a: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    min of 2-d array along axis
    """
    return np_apply_along_axis(func=np.min, axis=axis, a=a)


@njit
def np_nanmean(a: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    nanmean of 2-d array along axis
    """
    return np_apply_along_axis(func=np.nanmean, axis=axis, a=a)


@njit
def np_nansum(a: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    nansum of 2-d array along axis
    """
    return np_apply_along_axis(func=np.nansum, axis=axis, a=a)


@njit
def np_nanstd(a: np.ndarray, axis: int = 1, ddof: int = 1) -> np.ndarray:
    """
    nan std of 2-d array along axis
    """
    avg_std = np_apply_along_axis(func=np.nanstd, axis=axis, a=a)
    if ddof == 1:  # numbda does not recognise ddof as param to nanstd
        n = np_apply_along_axis(func=np.count_nonzero, axis=axis, a=~np.isnan(a))
        avg_std = np.where(n > 1, avg_std*np.sqrt(n / (n - 1.0)), np.nan)
    return avg_std


@njit
def np_cumsum(a: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    canot do  return np_apply_along_axis(func=np.cumsum, axis=axis, a=a)
    because the axes is not reduced
    """
    result = np.zeros_like(a)
    if axis == 0:
        result[0] = a[0]
        for k in range(1, a.shape[0]):
            result[k] = result[k-1]+a[k]
    elif axis == 1:
        result[:, 0] = a[:, 0]
        if axis == 0:
            for k in range(1, a.shape[1]):
                result[:, k] = result[:, k - 1] + a[:, k]
    return result


@njit
def repeat_by_columns(a: np.ndarray, n: int) -> np.ndarray:
    return a.repeat(n).reshape((-1, n))


@njit
def repeat_by_rows(a: np.ndarray, n: int) -> np.ndarray:
    return a.repeat(n).reshape((-1, n)).T


@njit
def a_rank(a: np.ndarray, axis: int = 1) -> np.ndarray:
    order = a.argsort(axis=axis)  # get indices that sort array
    rank = order.argsort(axis=axis)
    return rank


@njit
def nan_func_to_data(a: np.ndarray,
                     func: Callable[[np.ndarray], np.ndarray] = np.nanmean,
                     axis: int = 0
                     ) -> np.ndarray:
    """
    apply nan sensitive function to data
    for pandas:  a.dim=2, a.shape = (n_rows, n_col)
    row or columns wise operation equivalent to
    out = np.where(ind_all_nans, nans, np.nanmean(a=a, axis=axis))
    avoiding RuntimeWarning: Mean of empty slice
    """

    # simple case: a is one dimensional, axis do not matter then
    if a.ndim == 1:
        if axis == 0:
            if not np.all(np.isnan(a)):
                out = func(a)
            else:
                out = np.nan
        else:  # for 1-d array, we assume that func[a_n] = a_n
            raise ValueError(f"axis=1 not defined for 1-d array")

    else:
        n_rows = a.shape[0]
        n_col = a.shape[1]

        if axis == 0:
            out = np.full(n_col, np.nan, dtype=np.double)
        else:
            out = np.full(n_rows, np.nan, dtype=np.double)

        if axis == 0:  # column wise
            for col_idx in np.arange(n_col):
                col_a = a[:, col_idx]
                if not np.all(np.isnan(col_a)):
                    out[col_idx] = func(col_a)

        else:  # row wise
            for row_idx in np.arange(n_rows):
                row_a = a[row_idx]
                if not np.all(np.isnan(row_a)):
                    out[row_idx] = func(row_a)

    return out


@njit
def to_finite(a: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    return np.where(np.isfinite(a), a, fill_value)


def to_finite_np(data: Union[pd.Series, pd.DataFrame, np.ndarray],
                 fill_value: float = 0.0,
                 a_min: Optional[float] = None,
                 a_max: Optional[float] = None,
                 is_min_max_clip_fill: bool = True  # will be filled at min or max
                 ) -> np.ndarray:
    """
    map to finite numbers
    note that pandas treans +/- inf differently than nan
    """
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        data_np = data.to_numpy()
    elif isinstance(data, np.ndarray):
        data_np = data.copy()
    else:
        raise TypeError(f"unsuported {type(data)}")

    data_np = to_finite(data_np, fill_value)

    if a_min is not None or a_max is not None:
        if is_min_max_clip_fill:
            data_np = np.clip(a=data_np, a_min=a_min, a_max=a_max)
        else:
            if a_max is not None:
                data_np = np.where(np.greater(data_np, a_max), np.nan, data_np)
            if a_min is not None:
                data_np = np.where(np.less(data_np, a_min), np.nan, data_np)

    return data_np


def to_finite_reciprocal(data: Union[pd.Series, pd.DataFrame, np.ndarray],
                         fill_value: float = 0.0,
                         is_gt_zero: bool = True,
                         a_min: float = None,
                         a_max: float = None
                         ) -> Union[pd.Series, pd.DataFrame, np.ndarray]:
    """
    finite repciprocal with inf fills
    """
    x_min, x_max = None, None
    if a_min is not None:
        x_max = 1.0 / a_min
    if a_max is not None:
        x_min = 1.0 / a_max

    x = to_finite_np(data, fill_value=fill_value, a_min=x_min, a_max=x_max)
    cond = None
    if is_gt_zero:
        cond = np.greater(x, 0.0)
    rec_x = np.reciprocal(x, where=cond)
    rec_x = to_finite(rec_x, fill_value)

    if isinstance(data, pd.DataFrame):
        rec_x = pd.DataFrame(rec_x, columns=data.columns, index=data.index)
    elif isinstance(data, pd.Series):
        rec_x = pd.Series(rec_x, name=data.name, index=data.index)

    return rec_x


def to_finite_ratio(x: Union[pd.Series, pd.DataFrame, np.ndarray],
                    y: Union[pd.Series, pd.DataFrame, np.ndarray],
                    fill_value: float = np.nan
                    ) -> Union[pd.Series, pd.DataFrame, np.ndarray]:
    """
    finite ratio x / y
    """
    x_np = to_finite_np(x, fill_value=fill_value)
    y_np = to_finite_np(y, fill_value=fill_value)
    x_y = np.divide(x_np, y_np, where=np.isclose(y_np, 0.0)==False)

    if isinstance(x, pd.DataFrame):
        x_y = pd.DataFrame(x_y, columns=x.columns, index=x.index)
    elif isinstance(x, pd.Series):
        x_y = pd.Series(x_y, name=x.name, index=x.index)

    return x_y


@njit
def covar_to_corr(covar: np.ndarray) -> np.ndarray:
    """
    compute correlation out of covariance
    """
    inv_vol = np.reciprocal(np.sqrt(np.diag(covar)))
    norm = np.outer(inv_vol, inv_vol)
    covar *= norm
    return covar


def np_get_sorted_idx(a: np.ndarray) -> np.ndarray:
    """
    return indexes of sorted array
    """
    sorted_idx = a.argsort().argsort()  # need to apply double argsort
    return sorted_idx


def np_array_to_df_columns(a: np.ndarray, n_col: int) -> np.ndarray:
    """
    in case operation of np array is needed to apply to pandas column wise,
    the numpy array is broadcast to pandas columns
    the lenth = pandas.index
    """
    return np.tile(a, (n_col, 1)).T


def np_array_to_df_index(a: np.ndarray, n_index: int) -> np.ndarray:
    """
    in case operation of np array is needed to apply to pandas row wise,
    the numpy array is broadcast to pandas index
    the lenth = pandas.index
    """
    return np.tile(a, (n_index, 1))


def np_array_to_n_column_array(a: np.ndarray, n_col: int) -> np.ndarray:
    """
    for numpy broadcast of t-dim array to t-n dim array
    """
    return np.tile(a, (1, n_col))


def np_array_to_t_rows_array(a, t_rows: int) -> np.ndarray:
    """
    for numpy broadcast of t-dim array to t-n dim array
    """
    return np.tile(a, (t_rows, 1))


def np_array_to_matrix(a: np.ndarray, n_col: int) -> np.ndarray:
    """
    convert series too pandas dimension
    """
    matrix = np.broadcast_to(a, (n_col, len(a))).T
    return matrix


def np_matrix_add_array(matrix: np.ndarray,  # dim = T*N
                        array: np.ndarray,  # dim[a] = N or T(todo)
                        axis: int = 1  # 1 if dim[a] = N, 0 if dim[a] = T(todo)
                        ) -> np.ndarray:
    if axis == 1:
        m_array = np_array_to_t_rows_array(a=array, t_rows=matrix.shape[1])
        matrix_add = matrix + m_array
    else:
        raise ValueError(f"axis=0 is not imlemneted")
    return matrix_add


class RollFillType(Enum):
    NAN = 1  # empty is filled by nan
    ROLLOVER = 2  # corresponds to np.roll
    LAST_FILL = 3 # replace last left or right values


def np_shift(a: np.ndarray,
             shift: int,
             roll_fill_type: RollFillType = RollFillType.NAN
             ) -> np.ndarray:
    """
    shift numpy array
    """
    if a.ndim > 1:
        raise TypeError('only 1-d arrays are supported')
    n = a.shape[0]

    result = np.empty_like(a)

    if shift > 0:
        result[shift:] = a[:-shift]

        if roll_fill_type == RollFillType.NAN:
            result[:shift] = np.nan
        elif roll_fill_type == RollFillType.ROLLOVER:
            result[:shift] = a[n-shift:]
        elif roll_fill_type == RollFillType.LAST_FILL:
            result[:shift] = a[0]

    elif shift < 0:
        result[:shift] = a[-shift:]

        if roll_fill_type == RollFillType.NAN:
            result[shift:] = np.nan
        elif roll_fill_type == RollFillType.ROLLOVER:
            result[shift:] = a[:-shift]
        elif roll_fill_type == RollFillType.LAST_FILL:
            result[shift:] = a[-1]

    else:
        result[:] = a

    return result


def compute_expanding_power(n: int, power_lambda: float, reverse_columns: bool = False) -> np.ndarray:
    """
    compute expanding power = [1, lambda, lambda^2, ...]
    """
    a = np.log(power_lambda) * np.ones(n)
    a[0] = 0.0
    b = np.exp(np.cumsum(a))
    if reverse_columns:
        b = b[::-1]
    return b


def running_mean(x: np.ndarray, n: int, fill_value: float = np.nan) -> np.ndarray:
    x = to_finite_np(data=x, fill_value=fill_value)
    rolling_s = pd.Series(x).rolling(n, min_periods=0).apply(lambda x: np.nanmean(x))
    rolling_s = rolling_s.ffill()  # when x has sequence of nans longer than n
    return rolling_s.to_numpy()


def compute_paired_signs(x: np.ndarray,
                         y: np.ndarray,
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    compute signs equality of x and y excluding 0.0
    """
    joint_cond = np.logical_not(np.logical_and(np.isclose(x, 0.0), np.isclose(y, 0.0)))
    trend_cond = np.equal(np.sign(x), np.sign(y), where=joint_cond)
    rev_cond = np.not_equal(np.sign(x), np.sign(y), where=joint_cond)
    return joint_cond, trend_cond, rev_cond


@njit
def tensor_mean(a: np.ndarray) -> np.ndarray:
    """
    tensor mean in 1st direction
    typical application tensor of model params in t to compute the time average as (n1,n2)
    """
    nt = a.shape[0]
    n1 = a.shape[1]
    n2 = a.shape[2]
    output = np.zeros((n1, n2))
    for n1_ in range(n1):
        for n2_ in range(n2):
            output[n1_, n2_] = np.nanmean(a[:, n1_, n2_])
    return output


def find_nearest(a: np.ndarray,
                 value: float,
                 is_sorted: bool = True
                 ) -> float:
    """
    find closes element
    https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    """
    if is_sorted:
        idx = np.searchsorted(a, value, side="left")
        if idx > 0 and (idx == len(a) or np.abs(value - a[idx - 1]) < np.abs(value - a[idx])):
            return a[idx - 1]
        else:
            return a[idx]
    else:
        a = np.asarray(a)
        idx = (np.abs(a - value)).argmin()
    return a[idx]


def to_nearest_values(a: np.ndarray,
                      values: np.ndarray,
                      is_sorted: bool = True
                      ) -> np.ndarray:
    """
    map array of values to nearest elements
    """
    values_ = np.zeros_like(values)
    for idx, value in enumerate(values):
        values_[idx] = find_nearest(a=a, value=value, is_sorted=is_sorted)
    return values_


def compute_histogram_data(a: np.ndarray,
                           x_grid: np.ndarray,
                           name: str = 'Histogram'
                           ) -> pd.Series:
    """
    compute histogram on defined discrete grid
    """
    hist_data, bin_edges = np.histogram(a=a,
                                        bins=len(x_grid)-1,
                                        range=(x_grid[0], x_grid[-1]))
    hist_data = np.append(np.array(x_grid[0]), hist_data)
    hist_data = hist_data / len(a)
    hist_data = pd.Series(hist_data, index=bin_edges, name=name)
    return hist_data


def np_nonan_weighted_avg(a: np.ndarray, weights: np.ndarray) -> float:
    """
    compute weighted average of a using weights by masking nans in a
    """
    a = np.where(np.isfinite(a), a, np.nan)
    if np.all(np.isnan(a)):
        va = np.nan
    elif np.all(np.isclose(weights, 0.0)):
        va = 0.0
    else:
        ma = np.ma.MaskedArray(a, mask=np.isnan(a))
        va = np.ma.average(ma, weights=weights)
    return va


class UnitTests(Enum):
    SHIFT_TEST = 1
    CUM_POWER = 2
    ROLLING = 3
    WA = 4


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.SHIFT_TEST:
        test_array = np.array([str(n) for n in range(10)])
        print('test_array')
        print(test_array)

        print('shift=2')
        for roll_fill_type in RollFillType:
            print(roll_fill_type)
            print(np_shift(a=test_array, shift=2, roll_fill_type=roll_fill_type))

        print('shift=-2')
        for roll_fill_type in RollFillType:
            print(roll_fill_type)
            print(np_shift(a=test_array, shift=-2, roll_fill_type=roll_fill_type))

    elif unit_test == UnitTests.CUM_POWER:
        tic = time.perf_counter()
        b = compute_expanding_power(n=10000000, power_lambda=0.97, reverse_columns=True)
        toc = time.perf_counter()
        print(f"{toc - tic} secs to run")
        print(b)

    elif unit_test == UnitTests.ROLLING:
        x = np.array([1.0, 2.0, np.nan, np.nan, 3.0, 4.0, 5.0])
        xx = running_mean(x=x, n=2)
        print(xx)

    elif unit_test == UnitTests.WA:
        x = np.array([1.0, 2.0, np.nan, np.nan, 3.0, 4.0, 5.0])
        weights = np.arange(len(x))
        print(x)
        print(weights)
        xx = np_nonan_weighted_avg(a=x, weights=weights)
        print(xx)


if __name__ == '__main__':

    unit_test = UnitTests.CUM_POWER

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

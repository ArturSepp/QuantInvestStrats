"""
implementation of exponentially weighted mean (ewm) filter using numpy and lambda
"""
# packages
import warnings
import numpy as np
import pandas as pd
from numba import njit
from typing import Union, Tuple, Optional
from enum import Enum

# qis
import qis.utils.dates as da
import qis.utils.np_ops as npo


class NanBackfill(Enum):
    """
    when you sung ewm recursion we need to treat nans
    base case: when we assume conitnuous time series with occasional gaps we use FFILL method
    case where outputs needs a flag that given data are nans we need to use nan fill
    """
    FFILL = 1   # use last nonnan value
    DEFLATED_FFILL = 2  #  use last nonnan value * lambda
    ZERO_FILL = 3  # use zero value: nans must be filled by zero otherwise the recursion cannot start
    NAN_FILL = 4  # use nan value: for recursion we use ZERO_FILL then we substitute zeros with nans:
    # it corresponds to DEFLATED_FFILL for subsequent non nans


class InitType(Enum):
    ZERO = 1
    X0 = 2
    MEAN = 3
    VAR = 4


class MeanAdjType(Enum):
    NONE = 1
    INSAMPLE = 2
    EXPANDING = 3
    EWMA = 4


class CrossXyType(Enum):
    COVAR = 1
    BETA = 2
    CORR = 3


def set_init_dim1(data: Union[pd.DataFrame, pd.Series, np.ndarray],
                  init_type: InitType = InitType.X0
                  ) -> Union[float, np.ndarray]:

    x = npo.to_finite_np(data=data, fill_value=np.nan)

    if init_type == InitType.ZERO:
        init_value = np.zeros_like(x[0])
    elif init_type == InitType.X0:
        init_value = np.where(np.isnan(x[0])==False, x[0], 0.0)
    elif init_type == InitType.MEAN:
        init_value = np.nanmean(x, axis=0)
        init_value = np.where(np.isnan(init_value) == False, init_value, 0.0)
    elif init_type == InitType.VAR:
        init_value = np.nanvar(x, axis=0)
    else:
        raise TypeError(f"in set_initial_condition: unsuported init_type")

    return init_value


def set_init_dim2(data: Union[pd.DataFrame, pd.Series, np.ndarray],
                  init_type: InitType = InitType.X0
                  ) -> np.ndarray:

    x = npo.to_finite_np(data=data, fill_value=np.nan)
    cross = np.transpose(x) @ x
    if init_type == InitType.ZERO:
        init_value = np.zeros((cross.shape[0], cross.shape[1]))
    elif init_type == InitType.X0:
        init_value = np.zeros((cross.shape[0], cross.shape[1]))
    else:
        raise TypeError(f"in set_initial_condition_dim2: unsuported init_type")

    return init_value


@njit
def ewm_recursion(a: np.ndarray,
                  init_value: Union[float, np.ndarray],
                  span: Union[float, np.ndarray] = None,
                  ewm_lambda: Union[float, np.ndarray] = 0.94,
                  is_start_from_first_nonan: bool = True,
                  is_unit_vol_scaling: bool = False,
                  nan_backfill: NanBackfill = NanBackfill.FFILL
                  ) -> np.ndarray:

    """
    compute ewm using recursion:
    ewm[t] = (1-lambda) * x[t] + lambda*ewm[t-1]

    assumption is that non np.nan value is returned from the function

    data: numpy with dimension = t*n
    ewm_lambda: float or ndarray of dimension n
    init_value: initial value of dimension n
    start_from_first_nonan: start filling nans only from the first non-nan in underlying data: recomended because
                            it avoids backfilling of init_value
    is_unit_vol_scaling: outputs are scaled to have var(ewm)=1 (for gaussian data with zero corrs)
    """
    if span is not None:
        ewm_lambda = 1.0 - 2.0 / (span + 1.0)

    ewm_lambda_1 = 1.0 - ewm_lambda

    is_1d = (a.ndim == 1)  # or a.shape[1] == 1)

    # initialize all
    ewm = np.full_like(a, fill_value=np.nan, dtype=np.double)

    if is_start_from_first_nonan:
        if is_1d:  # cannot use np.where
            last_ewm = init_value if np.isfinite(a[0]) else np.nan
        else:
            last_ewm = np.where(np.isfinite(a[0]), init_value, np.nan)
    else:
        last_ewm = init_value
    ewm[0] = last_ewm

    # recurse from 1
    for t in np.arange(1, a.shape[0]):
        a_t = a[t]

        if is_start_from_first_nonan:
            # detect starting nonnans for when last ewma was np.nan and a_t is finite
            if is_1d:  # cannot use np.where
                if np.isfinite(last_ewm)==False and np.isfinite(a_t)==True:  # trick: if last_ewm is nan
                    last_ewm = init_value
            else:
                new_nonnans = np.logical_and(np.isfinite(last_ewm)==False, np.isfinite(a_t)==True)
                if np.any(new_nonnans):
                    last_ewm = np.where(new_nonnans, init_value, last_ewm)

        # do the step
        current_ewm = ewm_lambda * last_ewm + ewm_lambda_1 * a_t

        # fill nan-values
        if is_1d:   # np.where cannot be used
            if not np.isfinite(current_ewm):
                if nan_backfill == NanBackfill.FFILL:
                    current_ewm = last_ewm
                elif nan_backfill == NanBackfill.DEFLATED_FFILL:
                    current_ewm = ewm_lambda*last_ewm
                else:  # use zero fill
                    current_ewm = 0.0
        else:
            if nan_backfill == NanBackfill.FFILL:
                fill_value = last_ewm
            elif nan_backfill == NanBackfill.DEFLATED_FFILL:
                fill_value = ewm_lambda*last_ewm
            else:  # use zero fill
                fill_value = np.zeros_like(last_ewm)

            current_ewm = np.where(np.isfinite(current_ewm), current_ewm, fill_value)

        ewm[t] = last_ewm = current_ewm

    if is_unit_vol_scaling:
        vol_ratio = np.sqrt((1 + ewm_lambda) / (1 - ewm_lambda))
        ewm = vol_ratio * ewm

    return ewm


@njit
def compute_ewm_long_short(a: np.ndarray,
                           init_value: Union[float, np.ndarray],
                           long_span: Union[float, np.ndarray] = 63,
                           short_span: Optional[Union[float, np.ndarray]] = 5
                           ) -> np.ndarray:
    """
    long short ewm filter with unit variance
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
    signal smoother for pd.Dataframe and pd.Series data
    """

    if isinstance(data, pd.DataFrame):
        data_np = data.to_numpy()
        init_value = np.zeros(data_np.shape[1])
    elif isinstance(data, pd.Series):
        data_np = data.to_numpy()
        init_value = 0.0
    else:
        data_np = data
        init_value = np.zeros_like(data[0])

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


@njit
def compute_ewm_covar(a: np.ndarray,
                      b: np.ndarray = None,
                      span: Union[int, np.ndarray] = None,
                      ewm_lambda: float = 0.94,
                      covar0: np.ndarray = None,
                      is_corr: bool = False,
                      nan_backfill: NanBackfill = NanBackfill.FFILL
                      ) -> np.ndarray:
    """
    compute ewm covariance matrix
    b is optional, when given the covar is cross product a and b
    """
    if b is None:
        b = a
    else:  # must use numba >= 0.61.2
        assert a.shape[0] == b.shape[0]
        assert a.shape[1] == b.shape[1]

    if span is not None:
        ewm_lambda = 1.0 - 2.0 / (span + 1.0)
    ewm_lambda_1 = 1.0 - ewm_lambda

    if a.ndim == 1:  # ndarry
        n = a.shape[0]
    else:
        n = a.shape[1]  # array of ndarray

    if covar0 is None:
        covar = np.zeros((n, n))
    else:
        covar = np.where(np.isfinite(covar0), covar0, 0.0)

    last_covar = covar
    if a.ndim == 1:  # ndarry array
        r_ij = np.outer(a, b)
        covar = ewm_lambda_1 * r_ij + ewm_lambda * last_covar
        if nan_backfill == NanBackfill.FFILL:
            fill_value = last_covar
        elif nan_backfill == NanBackfill.DEFLATED_FFILL:
            fill_value = ewm_lambda * last_covar
        else:  # use zero fill
            fill_value = np.zeros_like(last_covar)

        covar = last_covar = np.where(np.isfinite(covar), covar, fill_value)

    else:  # loop over rows
        t = a.shape[0]
        for idx in range(0, t):  # row in x:
            r_ij = np.outer(a[idx], b[idx])
            covar = ewm_lambda_1 * r_ij + ewm_lambda * last_covar

            if nan_backfill == NanBackfill.FFILL:
                fill_value = last_covar
            elif nan_backfill == NanBackfill.DEFLATED_FFILL:
                fill_value = ewm_lambda*last_covar
            else:  # use zero fill
                fill_value = np.zeros_like(last_covar)

            last_covar = np.where(np.isfinite(covar), covar, fill_value)

        # for covar normalise
        if is_corr:
            if np.nansum(np.diag(last_covar)) > 1e-10:
                inv_vol = np.reciprocal(np.sqrt(np.diag(last_covar)))
                norm = np.outer(inv_vol, inv_vol)
            else:
                norm = np.identity(n)
            covar = norm * last_covar
        else:
            covar = last_covar

    return covar


@njit
def compute_ewm_covar_newey_west(a: np.ndarray,
                                 num_lags: int = 2,
                                 span: Union[int, np.ndarray] = None,
                                 ewm_lambda: float = 0.94,
                                 covar0: np.ndarray = None,
                                 is_corr: bool = False,
                                 nan_backfill: NanBackfill = NanBackfill.FFILL
                                 ) -> np.ndarray:
    """
    implementation of newey west covar estimator
    """
    ewm0 = compute_ewm_covar(a=a, span=span, ewm_lambda=ewm_lambda, covar0=covar0, is_corr=False, nan_backfill=nan_backfill)
    # compute m recursions
    if num_lags > 0:
        nw_adjustment = np.zeros_like(ewm0)
        for m in np.arange(1, num_lags+1):
            # lagged value
            a_m = np.empty_like(a)
            a_m[m:] = a[:-m]
            a_m[:m] = np.nan
            ewm_m1 = compute_ewm_covar(a=a, b=a_m, span=span)
            # ewm_m2 = compute_ewm_covar(a=a_m, b=a, span=span)
            nw_adjustment += (1.0-m/(num_lags+1))*(ewm_m1 + np.transpose(ewm_m1))
        ewm_nw = ewm0 + nw_adjustment
    else:
        ewm_nw = ewm0

    if is_corr:
        if np.nansum(np.diag(ewm_nw)) > 1e-10:
            inv_vol = np.reciprocal(np.sqrt(np.diag(ewm_nw)))
            norm = np.outer(inv_vol, inv_vol)
        else:
            norm = np.identity(a.shape[1])
        ewm_nw = norm * ewm_nw

    return ewm_nw


@njit
def compute_ewm_covar_tensor(a: np.ndarray,
                             span: Union[int, np.ndarray] = None,
                             ewm_lambda: float = 0.94,
                             covar0: np.ndarray = None,
                             is_corr: bool = False,
                             nan_backfill: NanBackfill = NanBackfill.FFILL
                             ) -> np.ndarray:
    """
    compute ewm covariance matrix time series as 3-d tensor [t, x, x]
    """
    if span is not None:
        ewm_lambda = 1.0 - 2.0 / (span + 1.0)
    ewm_lambda_1 = 1.0 - ewm_lambda

    if not a.ndim == 2:
        raise ValueError(f"only 2-d arrays are supported")

    t = a.shape[0]
    n = a.shape[1]  # array of ndarray
    zero_covar = np.zeros((n, n))

    if covar0 is None:
        covar = zero_covar
    else:
        covar = np.where(np.isfinite(covar0), covar0, zero_covar)

    output_covar = np.empty((t, n, n))
    last_covar = covar
    # loop over rows
    for idx in range(0, t):  # row in x:
        row = a[idx]
        r_ij = np.outer(row, row)
        covar = ewm_lambda_1 * r_ij + ewm_lambda * last_covar
        if nan_backfill == NanBackfill.FFILL:
            last_covar = np.where(np.isfinite(covar), covar, last_covar)
        elif nan_backfill == NanBackfill.DEFLATED_FFILL:
            last_covar = np.where(np.isfinite(covar), covar, ewm_lambda * last_covar)
        else:  # use zero fill
            last_covar = np.where(np.isfinite(covar), covar, zero_covar)

        if is_corr:
            if np.nansum(np.diag(last_covar)) > 1e-10:
                inv_vol = np.reciprocal(np.sqrt(np.diag(last_covar)))
                norm = np.outer(inv_vol, inv_vol)
            else:
                norm = np.identity(n)
            last_covar_ = norm * last_covar
        else:
            last_covar_ = last_covar

        if nan_backfill == NanBackfill.NAN_FILL:  # fill zeros with nans
            last_covar_ = np.where(np.equal(last_covar_, zero_covar), np.nan, last_covar_)

        output_covar[idx] = last_covar_

    return output_covar


@njit
def compute_ewm_covar_tensor_vol_norm_returns(a: np.ndarray,
                                              span: Union[int, np.ndarray] = None,
                                              ewm_lambda: float = 0.94,
                                              covar0: np.ndarray = None,
                                              is_corr: bool = False,
                                              nan_backfill: NanBackfill = NanBackfill.FFILL
                                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    compute ewm covariance matrix time series as 3-d tensor [t, x, x]
    returns vector a is normalised by vol
    compute normalised covar, covar for natural vars and vols
    """
    if span is not None:
        ewm_lambda = 1.0 - 2.0 / (span + 1.0)
    ewm_lambda_1 = 1.0 - ewm_lambda

    if not a.ndim == 2:
        raise ValueError(f"only 2-d arrays are supported")

    t = a.shape[0]
    n = a.shape[1]  # array of ndarray
    zero_covar = np.zeros((n, n))

    if covar0 is None:
        covar = zero_covar
    else:
        covar = np.where(np.isfinite(covar0), covar0, zero_covar)

    output_covar_norm = np.empty((t, n, n))
    output_covar = np.empty((t, n, n))
    last_covar = covar

    # compute vols
    a_var = np.square(a)
    ewm_vol = np.sqrt(ewm_recursion(a=a_var, ewm_lambda=ewm_lambda,
                                    init_value=npo.nan_func_to_data(a=a_var, func=np.nanmean, axis=0),
                                    nan_backfill=nan_backfill))
    a_norm = np.divide(a, ewm_vol)

    # loop over rows
    for idx in range(0, t):  # row in x:
        row = a_norm[idx]
        r_ij = np.outer(row, row)
        covar = ewm_lambda_1 * r_ij + ewm_lambda * last_covar
        if nan_backfill == NanBackfill.FFILL:
            last_covar = np.where(np.isfinite(covar), covar, last_covar)
        elif nan_backfill == NanBackfill.DEFLATED_FFILL:
            last_covar = np.where(np.isfinite(covar), covar, ewm_lambda * last_covar)
        else:  # use zero fill
            last_covar = np.where(np.isfinite(covar), covar, zero_covar)

        if is_corr:
            if np.nansum(np.diag(last_covar)) > 1e-10:
                inv_vol = np.reciprocal(np.sqrt(np.diag(last_covar)))
                norm = np.outer(inv_vol, inv_vol)
            else:
                norm = np.identity(n)
            last_covar_ = norm * last_covar
        else:
            last_covar_ = last_covar

        if nan_backfill == NanBackfill.NAN_FILL:  # fill zeros with nans
            last_covar_ = np.where(np.equal(last_covar_, zero_covar), np.nan, last_covar_)

        # normalise to preserve vols for output_covar
        norm_to_ewm_vols = ewm_vol[idx] / np.sqrt(np.diag(last_covar_))
        output_covar[idx] = last_covar_ * np.outer(norm_to_ewm_vols, norm_to_ewm_vols)
        output_covar_norm[idx] = last_covar_

    return output_covar, output_covar_norm, ewm_vol


# @njit
def compute_ewm_xy_beta_tensor(x: np.ndarray,  # factor returns
                               y: np.ndarray,  # asset returns
                               span: Union[int, np.ndarray] = None,
                               ewm_lambda: float = 0.94,
                               warmup_period: int = 20,  # to avoid excessive betas at start,
                               is_x_correlated: bool = True,  # computation of [x,x]
                               nan_backfill: NanBackfill = NanBackfill.FFILL
                               ) -> np.ndarray:
    """
    compute ewm cross matrices with x*y using outer product = dim[x] * dim[y]
    the dimension of tensor is [t, x, y]
    njit
    """
    if not x.ndim in [1, 2] or not y.ndim in [1, 2]:
        raise TypeError("Expected 1- or 2-dimensional NumPy array for x and y")
    if x.shape[0] != y.shape[0]:
        raise TypeError("first time series dimension of x and y must be equal")

    if x.ndim == 1:  # numba is sensetive to how dimensions and initial value
        nx = 1
        is_x_correlated = False  # for 1-d factor no need to compute outer product
    else:
        nx = x.shape[1]
    inv_t1 = np.diag(np.ones(nx))
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
    ewm_lambda_1 = 1.0 - ewm_lambda
    for t in range(nt):  # over time index
        row_x = x[t]  # time series row
        row_y = y[t]
        cross_xy = ewm_lambda_1 * np.outer(row_x, row_y) + ewm_lambda * last_cross_xy

        if nan_backfill == NanBackfill.FFILL:
            fill_value = last_cross_xy
        elif nan_backfill == NanBackfill.DEFLATED_FFILL:
            fill_value = ewm_lambda * last_cross_xy
        else:  # use zero fill
            fill_value = np.zeros_like(last_cross_xy)

        cross_xy = np.where(np.isfinite(cross_xy), cross_xy, fill_value)
        last_cross_xy = cross_xy

        # covar matrix
        covar_xx = ewm_lambda_1 * np.outer(row_x, row_x) + ewm_lambda * last_covar_xx
        if nan_backfill:
            covar_xx = np.where(np.isfinite(covar_xx), covar_xx, ewm_lambda * last_covar_xx)
        else:
            covar_xx = np.where(np.isfinite(covar_xx), covar_xx, last_covar_xx)
        last_covar_xx = covar_xx

        if t > warmup_period:
            # if np.trace(covar_xx) > 1e-8:
            if np.min(np.diag(covar_xx)) > 1e-8:
                if is_x_correlated:  # use inversion
                    try:
                        inv_t = np.linalg.inv(covar_xx)
                    except:  # "Singular matrix": #LinAlgError("Singular matrix")
                        inv_t = np.diag(np.reciprocal(np.diag(covar_xx)))
                    inv_t = np.ascontiguousarray(inv_t)  # to remove numpy warning
                else:
                    # reciprocal of diagonal elements
                    inv_t = np.diag(np.reciprocal(np.diag(covar_xx)))
            else:
                inv_t = inv_t1

            inv_t = np.where(np.isfinite(inv_t), inv_t, inv_t1)
            betas_t = inv_t @ cross_xy
        else:
            betas_t = beta_nan

        betas_ts[t] = betas_t

    return betas_ts


def compute_one_factor_ewm_betas(x: pd.Series,
                                 y: pd.DataFrame,
                                 span: Union[int, np.ndarray] = None,
                                 ewm_lambda: float = 0.94,
                                 nan_backfill: NanBackfill = NanBackfill.FFILL
                                 ) -> pd.DataFrame:
    """
    ewm betas of y wrt factor 1-d x
    """
    if not x.index.equals(y.index):
        raise ValueError("x.index={x.index} is not equal to y.index={y.index}")

    x_np = npo.to_finite_np(data=x, fill_value=np.nan)
    y_np = npo.to_finite_np(data=y, fill_value=np.nan)

    betas_ts = compute_ewm_xy_beta_tensor(x=x_np, y=y_np, span=span,
                                          ewm_lambda=ewm_lambda,
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
    ewm for pandas or series
    data dimension = t*n
    use gen data of pandas and np and call ewm_np with numa
    """
    a = npo.to_finite_np(data=data, fill_value=np.nan)

    if init_value is None:
        init_value = set_init_dim1(data=a, init_type=init_type)

    # important for numba to have uniform data
    if isinstance(data, pd.Series) or (isinstance(data, np.ndarray) and data.ndim == 1):
        ewm_lambda = float(ewm_lambda)
        if isinstance(init_value, np.ndarray):
            init_value = float(init_value)

    if span is not None:
        ewm_lambda = 1.0 - 2.0 / (span + 1.0)

    ewm = ewm_recursion(a=a,
                        span=None,
                        ewm_lambda=ewm_lambda,
                        init_value=init_value,
                        is_unit_vol_scaling=is_unit_vol_scaling,
                        nan_backfill=nan_backfill)

    if isinstance(data, pd.DataFrame):  # return of data type
        ewm = pd.DataFrame(data=ewm, index=data.index, columns=data.columns)

    elif isinstance(data, pd.Series):  # return of data type
        ewm = pd.Series(data=ewm, index=data.index, name=data.name)

    return ewm


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
    implementation of ewm recursion for variance/volatility computation
    vol_floor_quantile_roll_period will replace ewma estimate with quantile vol if vol < quantile vol
    """
    a = npo.to_finite_np(data=data, fill_value=np.nan)

    if span is not None:
        ewm_lambda = 1.0 - 2.0 / (span + 1.0)

    if mean_adj_type != MeanAdjType.NONE:
        a = compute_rolling_mean_adj(data=a,
                                     mean_adj_type=mean_adj_type,
                                     ewm_lambda=ewm_lambda,
                                     init_type=init_type,
                                     nan_backfill=nan_backfill)

    # initial conditions
    a = np.square(a)
    if init_value is None:
        init_value = set_init_dim1(data=a, init_type=init_type)

    if isinstance(data, pd.Series) or (isinstance(data, np.ndarray) and data.ndim == 1):
        ewm_lambda = float(ewm_lambda)
        if isinstance(init_value, np.ndarray):
            init_value = float(init_value)

    ewm = ewm_recursion(a=a, ewm_lambda=ewm_lambda, init_value=init_value, nan_backfill=nan_backfill)

    # apply quantile
    if vol_floor_quantile is not None:
        ewm_pd = pd.DataFrame(ewm)
        ewm_quantiles = ewm_pd.rolling(vol_floor_quantile_roll_period,
                                       min_periods=int(0.2*vol_floor_quantile_roll_period)
                                       ).quantile(vol_floor_quantile, interpolation="lower")
        vol_floor = ewm_quantiles.to_numpy()
        ewm = np.where(np.less(ewm, vol_floor), vol_floor, ewm)

    if warmup_period is not None:   # set to nan first nonnan in warmup_period
        ewm = npo.set_nans_for_warmup_period(a=ewm, warmup_period=warmup_period)

    if annualize or annualization_factor is not None:
        if annualization_factor is None:
            if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
                annualization_factor = da.infer_an_from_data(data=data)
            else:
                warnings.warn(f"in compute_ewm  annualization_factor for np array default is 1")
                annualization_factor = 1.0
        ewm = annualization_factor * ewm

    if apply_sqrt:
        ewm = np.sqrt(ewm)

    if isinstance(data, pd.DataFrame):
        ewm = pd.DataFrame(data=ewm, index=data.index, columns=data.columns)
    elif isinstance(data, pd.Series):
        ewm = pd.Series(data=ewm, index=data.index, name=data.name)
    return ewm


@njit
def matrix_recursion(a: np.ndarray,
                     a_m: np.ndarray,
                     span: Optional[Union[float, np.ndarray]] = None,
                     ewm_lambda: Union[float, np.ndarray] = 0.94
                     ) -> np.ndarray:
    if span is not None:
        ewm_lambda = 1.0 - 2.0 / (span + 1.0)
    ewm_lambda_1 = 1.0 - ewm_lambda
    t = a.shape[0]
    last_covar = np.zeros((a.shape[1], a.shape[1]))
    ewm_m = np.zeros_like(a_m)
    for idx in range(0, t):
        r_ij = np.outer(a[idx], a_m[idx])
        covar = ewm_lambda_1 * r_ij + ewm_lambda * last_covar
        fill_value = last_covar
        last_covar = np.where(np.isfinite(covar), covar, fill_value)
        ewm_m[idx, :] = np.diag(last_covar) + np.diag(np.transpose(last_covar))
    return ewm_m


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
    implementation of newey west vol estimator
    implementation of ewm recursion for variance/volatility computation
    vol_floor_quantile_roll_period will replace ewma estimate with quantile vol if vol < quantile vol
    """
    a = npo.to_finite_np(data=data, fill_value=np.nan)

    if span is not None:
        ewm_lambda = 1.0 - 2.0 / (span + 1.0)

    if mean_adj_type != MeanAdjType.NONE:
        a = compute_rolling_mean_adj(data=a,
                                     mean_adj_type=mean_adj_type,
                                     span=span,
                                     ewm_lambda=ewm_lambda,
                                     init_type=init_type,
                                     nan_backfill=nan_backfill)

    # initial conditions
    if init_value is None:
        init_value = set_init_dim1(data=a, init_type=init_type)

    if isinstance(data, pd.Series) or (isinstance(data, np.ndarray) and data.ndim == 1):
        ewm_lambda = float(ewm_lambda)
        if isinstance(init_value, np.ndarray):
            init_value = float(init_value)

    ewm0 = ewm_recursion(a=np.square(a), ewm_lambda=ewm_lambda, init_value=init_value, nan_backfill=nan_backfill)

    if num_lags == 0:
        ewm_nw = ewm0
        nw_ratio = np.ones_like(ewm0)
    else:
        nw_adjustment = np.zeros_like(ewm0)
        # compute m recursions
        for m in np.arange(1, num_lags+1):
            # lagged value
            a_m = np.empty_like(a)
            a_m[m:] = a[:-m]
            a_m[:m] = np.nan
            ewm_m = matrix_recursion(a=a, a_m=a_m, span=span)
            nw_adjustment += (1.0-m/(num_lags+1))*ewm_m

        ewm_nw = ewm0 + nw_adjustment
        nw_ratio = np.divide(ewm_nw, ewm0, where=ewm0 > 0.0)
        nw_ratio = np.where(nw_ratio > 0.0, nw_ratio, 1.0)

    if warmup_period is not None:   # set to nan first nonnan in warmup_period
        ewm_nw = npo.set_nans_for_warmup_period(a=ewm_nw, warmup_period=warmup_period)
        nw_ratio = npo.set_nans_for_warmup_period(a=nw_ratio, warmup_period=warmup_period)

    if annualize or annualization_factor is not None:
        if annualization_factor is None:
            if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
                annualization_factor = da.infer_an_from_data(data=data)
            else:
                warnings.warn(f"in compute_ewm  annualization_factor for np array default is 1")
                annualization_factor = 1.0
        ewm_nw = annualization_factor * ewm_nw

    if apply_sqrt:
        ewm_nw = np.sqrt(ewm_nw)

    if isinstance(data, pd.DataFrame):
        ewm_nw = pd.DataFrame(data=ewm_nw, index=data.index, columns=data.columns)
        nw_ratio = pd.DataFrame(data=nw_ratio, index=data.index, columns=data.columns)
    elif isinstance(data, pd.Series):
        ewm_nw = pd.Series(data=ewm_nw, index=data.index, name=data.name)
        nw_ratio = pd.Series(data=nw_ratio, index=data.index, name=data.name)
    return ewm_nw, nw_ratio


def compute_roll_mean(data: Union[pd.DataFrame, pd.Series, np.ndarray],
                      mean_adj_type: MeanAdjType = MeanAdjType.EWMA,
                      span: Union[float, np.ndarray] = None,
                      ewm_lambda: Union[float, np.ndarray] = 0.94,
                      init_value: Union[float, np.ndarray] = None,
                      nan_backfill: NanBackfill = NanBackfill.FFILL
                      ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:

    """
    compute rolling mean by columns
    the output has the same dimension as input
    """
    if not (isinstance(data, pd.DataFrame) or isinstance(data, pd.Series) or isinstance(data, np.ndarray)):
        raise TypeError(f"unsupported type {type(data)}")

    if mean_adj_type == MeanAdjType.NONE:
        if isinstance(data, np.ndarray):
            mean = np.zeros_like(data)
        else:
            mean_nd = np.zeros_like(data.to_numpy())
            if isinstance(data, pd.DataFrame):
                mean = pd.DataFrame(data=mean_nd, index=data.index, columns=data.columns)
            else:
                mean = pd.Series(data=mean_nd, index=data.index, name=data.name)

    elif mean_adj_type == MeanAdjType.INSAMPLE:
        if isinstance(data, np.ndarray):
            mean = np.mean(data, axis=0, keepdims=True)
        else:
            mean_nd = np.mean(data.to_numpy(), axis=0, keepdims=True)
            if isinstance(data, pd.DataFrame):
                mean = pd.DataFrame(data=mean_nd, index=data.index, columns=data.columns)
            else:
                mean = pd.Series(data=mean_nd, index=data.index, name=data.name)

    elif mean_adj_type == MeanAdjType.EXPANDING:  # use pandas core
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            x = data
        else:  # convert to pandas
            x = pd.DataFrame(data=data)
        mean = x.expanding(min_periods=1, axis=0).mean()  # apply pandas expanding
        if isinstance(data, np.ndarray):  # return of np.ndarray data type
            mean = mean.to_numpy()

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

    if mean_adj_type == MeanAdjType.NONE:
        x_mean = data
    else:
        if init_value is None:
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
    compute cross ewm for 1-d arrays x and y
    cross_xy[t] = (1-lambda)*x[t]*y[t] + lambda*cross_xy[t-1]

    three supported cases:
    1: x and y are pd.DataFrame with same dimensions: z = x*y
    2: x is pd.Series, y is pd.DataFrame: ml y=betas*x -> output pandas.dim = pandas.dim y
    3: both pd.Series -> output series.dim = series.dim y
    4: both np.nd arrays with same dimension
    """

    # 1 - adjust by mean
    if mean_adj_type != MeanAdjType.NONE:

        x_data = compute_rolling_mean_adj(data=x_data,
                                          mean_adj_type=mean_adj_type,
                                          span=span,
                                          ewm_lambda=ewm_lambda,
                                          init_type=init_type,
                                          nan_backfill=nan_backfill)

        y_data = compute_rolling_mean_adj(data=y_data,
                                          mean_adj_type=mean_adj_type,
                                          span=span,
                                          ewm_lambda=ewm_lambda,
                                          init_type=init_type,
                                          nan_backfill=nan_backfill)

    # 2  take gen arrays and convert to ndarray to use with numbas
    if isinstance(x_data, pd.DataFrame) and isinstance(y_data, pd.DataFrame):
        # should be same dimensions
        x = npo.to_finite_np(data=x_data, fill_value=np.nan)
        y = npo.to_finite_np(data=y_data, fill_value=np.nan)
        xy = np.multiply(x, y)

    elif isinstance(x_data, pd.DataFrame) and isinstance(y_data, pd.Series):
        xy = pd.concat([x_data, y_data], axis=1, join='inner')

        # it will work even if x_data.name is in y_data.columns
        x = npo.to_finite_np(data=xy.iloc[:, 0], fill_value=np.nan)
        y = npo.to_finite_np(data=xy.iloc[:, 1:], fill_value=np.nan)

        # x is array: tile by rows and transpose
        xn = np.transpose(np.tile(x, (len(y_data.columns), 1)))
        xy = np.multiply(xn, y)

    elif isinstance(x_data, pd.Series) and isinstance(y_data, pd.Series):
        xy = pd.concat([x_data, y_data], axis=1, join='inner')
        x = npo.to_finite_np(data=xy.iloc[:, 0], fill_value=np.nan)
        y = npo.to_finite_np(data=xy.iloc[:, 1], fill_value=np.nan)
        xy = np.multiply(x, y)

    elif isinstance(x_data, np.ndarray) and isinstance(y_data, np.ndarray):
        if x_data.shape[1] != y_data.shape[1]:
            raise TypeError(f"ndarray data must have same number of column")
        if x_data.shape[0] != y_data.shape[0]:
            raise TypeError(f"ndarray data must have same number of rows")

        x = x_data
        y = y_data
        xy = np.multiply(x_data, y_data)

    else:
        raise TypeError(f"{type(x_data)}, {type(y_data)}  should be of the same type")

    init_value_xy = set_init_dim1(data=xy, init_type=init_type)
    xy_covar = ewm_recursion(a=xy,
                             span=span,
                             ewm_lambda=ewm_lambda,
                             init_value=init_value_xy,
                             nan_backfill=nan_backfill)

    if cross_xy_type == CrossXyType.COVAR:
        cross_xy = xy_covar

    elif cross_xy_type == CrossXyType.BETA:
        x2 = np.square(x)
        init_value_x2 = set_init_dim1(data=x2, init_type=var_init_type)
        x_var = ewm_recursion(a=x2,
                              span=span,
                              ewm_lambda=ewm_lambda,
                              init_value=init_value_x2,
                              nan_backfill=nan_backfill)
        divisor = x_var
        cross_xy = np.divide(xy_covar, divisor, where=np.isclose(divisor, 0.0) == False)

    elif cross_xy_type == CrossXyType.CORR:
        x2 = np.square(x)
        init_value_x2 = set_init_dim1(data=x2, init_type=var_init_type)
        x_var = ewm_recursion(a=x2, span=span, ewm_lambda=ewm_lambda, init_value=init_value_x2, nan_backfill=nan_backfill)
        y2 = np.square(y)
        init_value_y2 = set_init_dim1(data=y2, init_type=var_init_type)
        y_var = ewm_recursion(a=y2, span=span, ewm_lambda=ewm_lambda, init_value=init_value_y2, nan_backfill=nan_backfill)
        divisor = np.sqrt(np.multiply(x_var, y_var))
        cross_xy = np.divide(xy_covar, divisor, where=np.isclose(divisor, 0.0) == False)
    else:
        raise TypeError(f"unknown cross_xy_type = {cross_xy_type}")

    if isinstance(y_data, pd.Series):
        cross_xy = pd.Series(data=cross_xy, index=y_data.index, name=y_data.name)

    elif isinstance(y_data, pd.DataFrame):
        cross_xy = pd.DataFrame(data=cross_xy, index=y_data.index, columns=y_data.columns)

    return cross_xy


def compute_ewm_beta_alpha_forecast(x_data: Union[pd.DataFrame, pd.Series],
                                    y_data: pd.DataFrame,
                                    span: Union[float, np.ndarray] = None,
                                    ewm_lambda: Union[float, np.ndarray] = 0.94,
                                    mean_adj_type: MeanAdjType = MeanAdjType.NONE,
                                    init_type: InitType = InitType.MEAN,
                                    annualize: bool = False,
                                    nan_backfill: NanBackfill = NanBackfill.FFILL
                                    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    """
    compute 1-d ewm beta for x and y
    x can be either series or dataframe
    cross_xy[t] = (1-lambda)*x[t]*y[t] + lambda*cross_xy[t-1]
    returns betas, ewm_alpha and annualized vars of resid and beta
    """
    # adjust indices in case
    if not x_data.index.equals(y_data.index):
        y_data = y_data.reindex(index=x_data, method='ffill')

    # 1 - adjust by mean
    if mean_adj_type != MeanAdjType.NONE:
        x_data = compute_rolling_mean_adj(data=x_data,
                                          mean_adj_type=mean_adj_type,
                                          span=span,
                                          ewm_lambda=ewm_lambda,
                                          init_type=init_type,
                                          nan_backfill=nan_backfill)

        y_data = compute_rolling_mean_adj(data=y_data,
                                          mean_adj_type=mean_adj_type,
                                          span=span,
                                          ewm_lambda=ewm_lambda,
                                          init_type=init_type,
                                          nan_backfill=nan_backfill)

    # 2  take gen arrays and convert to ndarray to use with numbdas
    if isinstance(x_data, pd.Series) and isinstance(y_data, pd.DataFrame):  # extend to df
        x = npo.np_array_to_n_column_array(a=npo.to_finite_np(data=x_data, fill_value=np.nan), ncols=len(y_data.columns))
        y = npo.to_finite_np(data=y_data, fill_value=np.nan)
    elif isinstance(x_data, pd.DataFrame) and isinstance(y_data, pd.DataFrame):
        # should be same dimensions
        x = npo.to_finite_np(data=x_data, fill_value=np.nan)
        y = npo.to_finite_np(data=y_data, fill_value=np.nan)
    else:
        raise NotImplementedError(f"in compute_ewm_beta_resid: not implemented types {type(x_data)} and {type(y_data)}")

    # compute covar
    xy = np.multiply(x, y)
    xy_covar = ewm_recursion(a=xy, span=span, ewm_lambda=ewm_lambda,
                             init_value=set_init_dim1(data=xy, init_type=init_type))

    # compute x var
    x2 = np.square(x)
    x_var = ewm_recursion(a=x2, span=span, ewm_lambda=ewm_lambda,
                          init_value=set_init_dim1(data=x2, init_type=init_type))

    # compute beta
    beta_xy = np.divide(xy_covar, x_var, where=np.isclose(x_var, 0.0) == False)

    # alpha and prediction assuming 1-d factor model
    y_prediction0 = beta_xy * x
    resid = y - y_prediction0
    ewm_alpha = ewm_recursion(a=resid, span=span, ewm_lambda=ewm_lambda, nan_backfill=nan_backfill,
                              init_value=set_init_dim1(data=resid, init_type=init_type))
    y_prediction = y_prediction0 + ewm_alpha

    # residual
    resid = y - y_prediction
    resid2 = np.square(resid)
    resid_var = ewm_recursion(a=resid2, span=span, ewm_lambda=ewm_lambda,
                              init_value=set_init_dim1(data=resid2, init_type=init_type))

    if annualize:
        an = da.infer_an_from_data(data=x_data)
        resid_var = an * resid_var
        x_var = an * x_var

    beta_xy = pd.DataFrame(data=beta_xy, index=y_data.index, columns=y_data.columns)
    alpha = pd.DataFrame(data=ewm_alpha, index=y_data.index, columns=y_data.columns)
    y_prediction = pd.DataFrame(data=y_prediction, index=y_data.index, columns=y_data.columns)
    resid_var = pd.DataFrame(data=resid_var, index=y_data.index, columns=y_data.columns)
    x_var = pd.DataFrame(data=x_var, index=y_data.index)

    # compute r2
    y_var0 = y_data.subtract(compute_ewm(data=y_data, span=span, ewm_lambda=ewm_lambda, nan_backfill=nan_backfill))
    y_var = ewm_recursion(a=np.square(y_var0.to_numpy()), span=span, ewm_lambda=ewm_lambda,
                          init_value=np.zeros(len(y_data.columns)), nan_backfill=nan_backfill)
    ewm_r2 = 1.0 - np.divide(resid_var, y_var, where=np.greater(y_var, 0.0))
    ewm_r2 = np.clip(ewm_r2, a_min=0.0, a_max=1.0)
    ewm_r2 = pd.DataFrame(data=ewm_r2, index=y_data.index, columns=y_data.columns)

    return beta_xy, alpha, y_prediction, x_var, resid_var, ewm_r2


def compute_ewm_alpha_r2_given_prediction(y_data: pd.DataFrame,
                                          y_prediction: pd.DataFrame,
                                          span: Union[float, np.ndarray] = None,
                                          ewm_lambda: Union[float, np.ndarray] = 0.94,
                                          nan_backfill: NanBackfill = NanBackfill.FFILL
                                          ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    """
    # 1 - adjust by mean
    resid = y_data - y_prediction
    ewm_alpha = compute_ewm(data=resid, span=span, ewm_lambda=ewm_lambda, nan_backfill=nan_backfill)
    resid0 = resid.subtract(ewm_alpha)
    resid_var = ewm_recursion(a=np.square(resid0.to_numpy()), span=span, ewm_lambda=ewm_lambda,
                              init_value=np.zeros(len(y_data.columns)), nan_backfill=nan_backfill)

    y_var0 = y_data.subtract(compute_ewm(data=y_data, span=span, ewm_lambda=ewm_lambda, nan_backfill=nan_backfill))
    y_var = ewm_recursion(a=np.square(y_var0.to_numpy()), span=span, ewm_lambda=ewm_lambda,
                          init_value=np.zeros(len(y_data.columns)), nan_backfill=nan_backfill)

    ewm_r2 = 1.0 - np.divide(resid_var, y_var, where=np.greater(y_var, 0.0))
    ewm_r2 = np.clip(ewm_r2, a_min=0.0, a_max=1.0)
    ewm_r2 = pd.DataFrame(data=ewm_r2, index=y_data.index, columns=y_data.columns)

    return ewm_alpha, ewm_r2


def compute_ewm_sharpe(returns: pd.DataFrame,
                       span: Union[float, np.ndarray] = 260,
                       norm_type: int = 1,
                       initial_sharpes: np.ndarray = None
                       ) -> pd.DataFrame:

    x = npo.to_finite_np(data=returns, fill_value=0.0)
    an = da.infer_an_from_data(data=returns)
    san = np.sqrt(an)
    if initial_sharpes is not None:
        initial_vol = 0.1
        initial_mean = initial_vol * initial_sharpes / np.square(san)
        initial_var = np.square(initial_vol / san) * np.ones(len(returns.columns))
    else:
        initial_mean = np.zeros(len(returns.columns))
        initial_var = np.zeros(len(returns.columns))

    if norm_type == 0:
        ewm_mean = ewm_recursion(a=x,
                                 span=span,
                                 init_value=initial_mean,
                                 nan_backfill=NanBackfill.ZERO_FILL)
        sharpe = pd.DataFrame(data=an*ewm_mean, index=returns.index, columns=returns.columns)

    elif norm_type == 1 or norm_type == 2:
        ewm_mean = ewm_recursion(a=x,
                                 span=span,
                                 init_value=initial_mean,
                                 nan_backfill=NanBackfill.ZERO_FILL)
        if norm_type == 2:
            v = np.square(x-ewm_mean)
        else:
            v = np.square(x)

        ewm_var = ewm_recursion(a=v, span=span,
                                init_value=initial_var,
                                nan_backfill=NanBackfill.ZERO_FILL)
        ewm_vol = np.sqrt(ewm_var)
        sharpe = pd.DataFrame(data=san * np.divide(ewm_mean, ewm_vol, where=np.greater(ewm_vol, 0.0)),
                              index=returns.index,
                              columns=returns.columns)
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
    given time serise var X_t
    compute Y_t = ewm(X_t / ewm_vol(X_t))
    expected std(Y_t) = 1
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
    ewm_std1_norm = compute_ewm(data=x_std1_norm, span=span, is_unit_vol_scaling=True)

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

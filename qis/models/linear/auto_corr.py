# packages
import numpy as np
import pandas as pd
from typing import Union, Tuple
from numba import njit
from statsmodels.tsa.stattools import pacf, acf

# qis
from qis.models.linear.ewm import MeanAdjType, compute_rolling_mean_adj, compute_ewm, NanBackfill
from qis.utils.np_ops import np_apply_along_axis


def estimate_acf_from_path(path: pd.Series, nlags: int = 10) -> Tuple[pd.Series, pd.Series]:
    """
    compute path act and pacfs
    """
    if isinstance(path, pd.Series):
        data = path.to_numpy()
    else:
        data = path.copy()
    data = data[np.isnan(data) == False]
    index = np.arange(1, nlags+1)
    if len(data) > 2.0 * nlags:
        pacfs = pd.Series(pacf(data, nlags=nlags)[1:], index=index)
        acfs = pd.Series(acf(data, nlags=nlags)[1:], index=index)
    else:
        pacfs = pd.Series(np.nan, index=index)
        acfs = pd.Series(np.nan, index=index)
    return acfs, pacfs


def estimate_acf_from_paths(paths: Union[np.ndarray, pd.DataFrame],
                            nlags: int = 10,
                            is_pacf: bool = True
                            ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    paths data are contained in columns of dataframe
    outputs are df with index=lags, columns = data columns
    use statsmodels.tsa.stattools
    """
    if isinstance(paths, pd.DataFrame):
        columns = paths.columns
        paths = paths.to_numpy()
    else:
        columns = None

    nb_path = paths.shape[1]
    acfs = np.zeros((nlags + 1, nb_path))
    for path in np.arange(nb_path):
        data = paths[:, path]
        data = data[np.isnan(data) == False]
        if len(data) > 2.0*nlags:
            if is_pacf:
                acfs[:, path] = pacf(data, nlags=nlags)
            else:
                acfs[:, path] = acf(data, nlags=nlags)
        else:
            acfs[:, path] = np.nan
    acfs = pd.DataFrame(acfs, columns=columns)
    m_acf = pd.Series(np.nanmean(acfs, axis=1), name='mean')
    std_acf = pd.Series(np.nanstd(acfs, axis=1), name='str')
    return acfs, m_acf, std_acf


def compute_autocorr_df(df: Union[pd.Series, pd.DataFrame],
                        num_lags: int = 20,
                        axis: int = 0
                        ) -> pd.DataFrame:
    """
    compute auto correlation columns wise and return df with lags = index
    default is axis = 0
    """
    acf = compute_path_autocorr(a=df.to_numpy())
    if isinstance(df, pd.Series):
        df = pd.Series(data=acf, index=np.arange(0, num_lags), name=df.name)
    else:
        df = pd.DataFrame(data=acf, index=np.arange(0, num_lags), columns=df.columns)
    return df


@njit
def compute_path_lagged_corr(a1: np.ndarray,
                             a2: np.ndarray,
                             num_lags: int = 20
                             ) -> np.ndarray:
    """
    compute correlation between a1 and a2 at num_lags
    """
    acorr = np.ones(num_lags)
    for idx in range(1, num_lags):
        acorr[idx] = np.corrcoef(a1[idx:], a2[:-idx], rowvar=False)[0][1]
    return acorr


@njit
def compute_path_autocorr(a: np.ndarray,
                          num_lags: int = 20
                          ) -> np.ndarray:
    """
    given data = df.to_numpy() of a compute column vise
    """
    is_1d = (a.ndim == 1)
    if is_1d:
        acfs = compute_path_lagged_corr(a1=a, a2=a, num_lags=num_lags)
    else:
        nb_path = a.shape[1]
        acfs = np.zeros((num_lags, nb_path))
        for path in np.arange(nb_path):
            a_ = a[:, path]
            acfs[:, path] = compute_path_lagged_corr(a1=a_, a2=a_, num_lags=num_lags)
    return acfs


@njit
def compute_ewm_matrix_autocorr(a: np.ndarray,
                                ewm_lambda: float = 0.94,
                                covar0: np.ndarray = None,
                                lag: int = 1,
                                aggregation_type: str = 'mean',
                                is_normalize: bool = True
                                ) -> (np.ndarray, np.ndarray):
    """
    x is T*N arrays
    """
    ewm_lambda_1 = 1.0 - ewm_lambda

    if a.ndim == 1:  # ndarry
        raise TypeError(f"time dimension must be higher than one")
    else:
        n = a.shape[1]  # array of ndarray
        num_off_diag = n * (n - 1)
        t = a.shape[0]

    if covar0 is None:
        covar = np.zeros((n, n))
        auto_covar = np.zeros((n, n))
    else:
        covar = covar0
        auto_covar = covar0

    trace_cov = np.zeros(t)
    trace_off = np.zeros(t)
    for idx in range(lag, t):  # row in x:
        x_t = a[idx]
        auto_covar = ewm_lambda_1 * np.outer(a[idx - lag], x_t) + ewm_lambda * auto_covar
        covar = ewm_lambda_1 * np.outer(x_t, x_t) + ewm_lambda * covar

        if is_normalize:
            auto_covar_t = auto_covar / covar
        else:
            auto_covar_t = auto_covar

        if aggregation_type == 'mean':
            trace_cov[idx] = np.nansum(np.diag(auto_covar_t)) / n
            trace_off[idx] = (np.nansum(auto_covar_t) - np.nansum(np.diag(auto_covar_t)))/num_off_diag

        elif aggregation_type == 'median':
            trace_cov[idx] = np.nanmedian(np.diag(auto_covar_t))
            trace_off[idx] = np.nanmedian(auto_covar_t)

    return trace_cov, trace_off


def compute_ewm_matrix_autocorr_df(data: pd.DataFrame,
                                   ewm_lambda: float = 0.94,
                                   mean_adj_type: MeanAdjType = MeanAdjType.EWMA,
                                   lag: int = 1,
                                   aggregation_type: str = 'mean',
                                   is_normalize: bool = True
                                   ) -> pd.DataFrame:
    """
    compute auto-correlation columns wise
    """
    if len(data.index) == 1:
        raise TypeError('data must be time series')

    data = data.ffill().dropna()
    x = compute_rolling_mean_adj(data=data.to_numpy(),
                                 mean_adj_type=mean_adj_type,
                                 ewm_lambda=ewm_lambda)

    trace_cov, trace_corr = compute_ewm_matrix_autocorr(a=x,
                                                        ewm_lambda=ewm_lambda,
                                                        lag=lag,
                                                        aggregation_type=aggregation_type,
                                                        is_normalize=is_normalize)

    data = pd.DataFrame(data=np.column_stack((trace_cov, trace_corr)),
                        index=data.index,
                        columns=['diagonal', 'off-diag'])
    return data


# @njit
def compute_ewm_vector_autocorr(a: np.ndarray,
                                span: Union[int, np.ndarray] = None,
                                ewm_lambda: float = 0.94,
                                lag: int = 1,
                                is_normalize: bool = True,
                                nan_backfill: NanBackfill = NanBackfill.FFILL
                                ) -> np.ndarray:
    """
    x is T*N arrays
    """
    if span is not None:
        ewm_lambda = 1.0 - 2.0 / (span + 1.0)
    ewm_lambda_1 = 1.0 - ewm_lambda

    if a.ndim == 1:  # ndarry
        is_1d = True
        n = 1
        last_auto_covar = 0.0
        last_covar = np.nanvar(a)
    else:
        is_1d = False
        n = a.shape[1]  # array of ndarray
        last_auto_covar = np.zeros(n)  # initialise at zero
        last_covar = np.nanvar(a, axis=0) # np_apply_along_axis(func=np.nanstd, axis=axis, a=a)**2  # initialised at insample variance
    t = a.shape[0]
    autocorr = np.zeros((t, n))

    for idx in range(lag, t):  # row in x:
        x_t = a[idx]
        current_auto_covar = ewm_lambda_1 * (a[idx-lag]*x_t) + ewm_lambda * last_auto_covar
        current_covar = ewm_lambda_1 * (x_t*x_t) + ewm_lambda * last_covar

        # fill nan-values
        if is_1d:   # np.where cannot be used
            if not np.isfinite(current_auto_covar):
                if nan_backfill == NanBackfill.FFILL:
                    current_auto_covar = last_auto_covar
                    current_covar = last_covar
                elif nan_backfill == NanBackfill.DEFLATED_FFILL:
                    current_auto_covar = ewm_lambda*last_auto_covar
                    current_covar = ewm_lambda*last_covar
                else:  # use zero fill
                    current_auto_covar = 0.0
                    current_covar = 0.0
        else:
            if nan_backfill == NanBackfill.FFILL:
                fill_value = last_auto_covar
                fill_covar = last_covar
            elif nan_backfill == NanBackfill.DEFLATED_FFILL:
                fill_value = ewm_lambda*last_auto_covar
                fill_covar = ewm_lambda*last_covar
            else:  # use zero fill
                fill_value = np.zeros_like(last_auto_covar)
                fill_covar = np.zeros_like(last_covar)

            current_auto_covar = np.where(np.isfinite(current_auto_covar), current_auto_covar, fill_value)
            current_covar = np.where(np.isfinite(current_covar), current_covar, fill_covar)

        last_auto_covar = current_auto_covar
        last_covar = current_covar

        if is_normalize:
            autocorr[idx, :] = current_auto_covar / current_covar
        else:
            autocorr[idx, :] = current_auto_covar

    return autocorr


def compute_ewm_vector_autocorr_df(data: Union[pd.DataFrame, pd.Series],
                                   span: Union[int, np.ndarray] = 30,
                                   lag: int = 1,
                                   is_normalize: bool = True
                                   ) -> Union[pd.DataFrame, pd.Series]:
    x = data - compute_ewm(data=data, span=span)
    autocorr = compute_ewm_vector_autocorr(a=x.to_numpy(),
                                           span=span,
                                           lag=lag,
                                           is_normalize=is_normalize)
    if isinstance(data, pd.DataFrame):
        autocorr = pd.DataFrame(autocorr, index=data.index, columns=data.columns)
    else:
        autocorr = pd.Series(autocorr[:, 0], index=data.index, name=data.name)
    return autocorr

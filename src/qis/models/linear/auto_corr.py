"""
autocorrelation of a path, at fixed lags over the whole sample and as an EWM state through time.

``estimate_acf_from_path`` and ``estimate_acf_from_paths`` wrap the ``statsmodels`` ``acf`` and
``pacf``, dropping NaNs first and returning NaN for a column with no more than ``2 * nlags``
finite observations. ``compute_path_autocorr`` is the njit full-sample version, correlating a
column with itself at lags 0 to ``num_lags`` - 1, entry 0 being one by definition;
``compute_path_lagged_corr`` correlates two arrays, entry 0 being their contemporaneous
correlation. ``compute_ewm_vector_autocorr`` runs the EWM recursion on the lagged product column
by column; ``compute_ewm_matrix_autocorr`` does the same for the matrix, aggregating diagonal and
off-diagonal under ``aggregation_type='mean'``; under ``'median'`` the second output is the median
over the whole matrix, diagonal included.

``lag`` counts rows of the index, so it inherits the frequency of the input rather than setting
one. The EWM path is normalised by the contemporaneous second moment when ``is_normalize``. Both
EWM kernels seed their states at zero by default, so they are point in time, and report NaN for
the first ``lag`` rows; the vector version takes ``var_init_type=InitType.VAR`` for the former
full-sample variance seed, the matrix version ``covar0``.
"""
# packages
import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, List
from numba import njit
from statsmodels.tsa.stattools import pacf, acf

# qis
from qis.models.linear.ewm import (MeanAdjType, compute_rolling_mean_adj, compute_ewm, NanBackfill,
                                   InitType)
from qis.utils.df_freq import df_resample_at_int_index


def estimate_acf_from_path(path: Union[pd.Series, np.ndarray],
                           nlags: int = 10
                           ) -> Tuple[pd.Series, pd.Series]:
    """
    standard sample autocorrelations and partial autocorrelations of one series.

    Calls ``statsmodels`` ``acf`` (full-sample mean and variance, ``adjusted=False``) and
    ``pacf`` (its default Yule-Walker method) at lags 1 to ``nlags``. NaNs are dropped first, so
    the observations on either side of a gap become adjacent: a lag then counts non-missing
    observations, not rows of the index.

    Args:
        path: one series, as a Series or a 1-d ndarray
        nlags: number of lags K

    Returns:
        (acf, pacf), each indexed by lag 1 to ``nlags``; all NaN unless more than ``2 * nlags``
        finite observations remain
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
    partial (default) or ordinary sample autocorrelations of many paths, one column per path.

    Each column is treated as :func:`estimate_acf_from_path` treats a series: NaNs are dropped
    and a column with no more than ``2 * nlags`` finite observations is NaN. Note the default:
    ``is_pacf=True`` returns the partial autocorrelations; pass ``is_pacf=False`` for the
    ordinary ACF.

    Args:
        paths: paths in columns, as a DataFrame or a 2-d ndarray
        nlags: number of lags K
        is_pacf: return the ``statsmodels`` ``pacf`` (default) rather than ``acf``

    Returns:
        (table, mean, std): the table indexed by lag 0 to ``nlags``, lag 0 included (one), with
        the columns of ``paths``; its mean across columns, named ``'mean'``; and its standard
        deviation across columns with ``ddof=0``, named ``'std'``
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
    std_acf = pd.Series(np.nanstd(acfs, axis=1), name='std')
    return acfs, m_acf, std_acf


def compute_autocorr_df(df: Union[pd.Series, pd.DataFrame],
                        num_lags: int = 20,
                        axis: int = 0
                        ) -> pd.DataFrame:
    """
    lagged Pearson autocorrelation of each column, with the lag as the index.

    Lag ``k`` correlates ``x[k:]`` with ``x[:-k]`` over the overlapping observations, each segment
    with its own mean; lag 0 is one by definition. Missing values are not handled.

    Args:
        df: observations in rows, one series per column
        num_lags: number of lags returned, from 0 to ``num_lags - 1``
        axis: retained for compatibility; the computation is always along the rows

    Returns:
        autocorrelations indexed by lag, as a Series for a Series input
    """
    acf = compute_path_autocorr(a=df.to_numpy(), num_lags=num_lags)
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
    lead-lag Pearson correlation of two equal-length arrays at lags 0 to ``num_lags - 1``.

    Entry ``k`` is ``corr(a1[k:], a2[:-k])``, the correlation of ``a1`` at t with ``a2`` at
    t - k over the overlapping observations, each segment about its own mean; entry 0 is the
    contemporaneous correlation of ``a1`` and ``a2``. NaNs are not handled.

    Args:
        a1: leading series, shape (t,)
        a2: lagged series, shape (t,)
        num_lags: number of lags returned

    Returns:
        the correlations, shape (num_lags,)
    """
    acorr = np.ones(num_lags)
    if num_lags > 0:
        acorr[0] = np.corrcoef(a1, a2, rowvar=False)[0][1]
    for idx in range(1, num_lags):
        acorr[idx] = np.corrcoef(a1[idx:], a2[:-idx], rowvar=False)[0][1]
    return acorr


@njit
def compute_path_lagged_corr_given_lags(a1: np.ndarray,
                                        a2: np.ndarray,
                                        lags: List[int] = (1, 5, 10, )
                                        ) -> np.ndarray:
    """
    lead-lag Pearson correlation of two equal-length arrays at the given lags.

    Entry ``i`` is ``corr(a1[k:], a2[:-k])`` for ``k = lags[i]``, and the contemporaneous
    correlation of ``a1`` and ``a2`` for a lag of 0. NaNs are not handled.

    Args:
        a1: leading series, shape (t,)
        a2: lagged series, shape (t,)
        lags: non-negative lags, in rows

    Returns:
        the correlations, shape (len(lags),)
    """
    acorr = np.zeros(len(lags))
    for idx, lag in enumerate(lags):
        if lag == 0:
            acorr[idx] = np.corrcoef(a1, a2, rowvar=False)[0][1]
        else:
            acorr[idx] = np.corrcoef(a1[lag:], a2[:-lag], rowvar=False)[0][1]
    return acorr


@njit
def compute_path_autocorr(a: np.ndarray,
                          num_lags: int = 20
                          ) -> np.ndarray:
    """
    lagged Pearson autocorrelation of each column at lags 0 to ``num_lags - 1``.

    Column by column :func:`compute_path_lagged_corr` of a series with itself; lag 0 is one by
    definition. NaNs are not handled.

    Args:
        a: observations, shape (t,) or (t, n)
        num_lags: number of lags returned

    Returns:
        the autocorrelations, shape (num_lags,) for 1-d input, else (num_lags, n)
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
    if num_lags > 0:  # autocorrelation at lag 0 is one by definition, also for a constant column
        acfs[0] = 1.0
    return acfs


@njit
def compute_path_autocorr_given_lags(a: np.ndarray,
                                     lags: List[int] = (1, 5, 10, )
                                     ) -> np.ndarray:
    """
    lagged Pearson autocorrelation of each column at the given lags.

    Column by column :func:`compute_path_lagged_corr_given_lags` of a series with itself; a lag
    of 0 gives one by definition. NaNs are not handled.

    Args:
        a: observations, shape (t,) or (t, n)
        lags: non-negative lags, in rows

    Returns:
        the autocorrelations, shape (len(lags),) for 1-d input, else (n, len(lags)), transposed
        relative to :func:`compute_path_autocorr`
    """
    is_1d = (a.ndim == 1)
    if is_1d:
        acfs = compute_path_lagged_corr_given_lags(a1=a, a2=a, lags=lags)
        for idx, lag in enumerate(lags):
            if lag == 0:
                acfs[idx] = 1.0
    else:
        nb_path = a.shape[1]
        acfs = np.zeros((nb_path, len(lags)))
        for path in np.arange(nb_path):
            a_ = a[:, path]
            acfs[path, :] = compute_path_lagged_corr_given_lags(a1=a_, a2=a_, lags=lags)
        for idx, lag in enumerate(lags):
            if lag == 0:
                acfs[:, idx] = 1.0
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
    EWM lagged cross moments of a panel, aggregated to a diagonal and an off-diagonal number.

    Runs ``G_t = lambda G_{t-1} + (1 - lambda) x_{t-lag} x_t'`` and
    ``C_t = lambda C_{t-1} + (1 - lambda) x_t x_t'`` from ``t = lag``, both seeded at zero
    (point in time) or both at ``covar0``, and divides them elementwise when ``is_normalize``.
    Off the diagonal the ratio is a lead-lag ratio, not a correlation, and is unbounded.

    Args:
        a: demeaned observations, shape (t, n)
        ewm_lambda: EWM decay
        covar0: seed of both states, shape (n, n). None seeds at zero
        lag: lag in rows
        aggregation_type: ``'mean'`` gives the mean of the diagonal and the mean of the
            off-diagonal entries (NaN entries count as zero; NaN when n is 1); ``'median'`` the
            median of the diagonal and the median of the whole matrix, diagonal included
        is_normalize: divide the lagged by the contemporaneous moments elementwise

    Returns:
        (diagonal, off_diagonal), each of shape (t,), NaN for the first ``lag`` rows

    Raises:
        TypeError: if ``a`` is 1-d
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

    # rows before the first lagged pair have no estimate
    trace_cov = np.full(t, np.nan)
    trace_off = np.full(t, np.nan)
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
            if num_off_diag > 0:
                trace_off[idx] = ((np.nansum(auto_covar_t) - np.nansum(np.diag(auto_covar_t)))
                                  / num_off_diag)
            else:  # a single column has no off-diagonal entry
                trace_off[idx] = np.nan

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
    EWM lagged cross moments of a panel as two columns, ``diagonal`` and ``off-diag``.

    Forward-fills, drops rows with any remaining NaN, removes the mean chosen by
    ``mean_adj_type`` (the point-in-time EWM mean with the same decay by default) and runs
    :func:`compute_ewm_matrix_autocorr` from a zero seed.

    Args:
        data: observations, rows are dates and columns are assets
        ewm_lambda: EWM decay of the mean and of the moments
        mean_adj_type: mean removed first; ``MeanAdjType.INSAMPLE`` looks ahead
        lag: lag in rows
        aggregation_type: ``'mean'`` or ``'median'``; see :func:`compute_ewm_matrix_autocorr`
        is_normalize: divide the lagged by the contemporaneous moments elementwise

    Returns:
        columns ``diagonal`` and ``off-diag`` on the retained rows, NaN for the first ``lag``

    Raises:
        TypeError: if ``data`` has a single row
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
                                nan_backfill: NanBackfill = NanBackfill.FFILL,
                                var_init_type: InitType = InitType.ZERO
                                ) -> np.ndarray:
    """
    EWM autocorrelation ratio of each column: EWM lagged moment over EWM second moment.

    Runs ``g_t = lambda g_{t-1} + (1 - lambda) x_{t-lag} x_t`` and
    ``v_t = lambda v_{t-1} + (1 - lambda) x_t^2`` from ``t = lag`` and returns ``g_t / v_t``.
    The input is used as given (demean it first). Both states are seeded at zero by default,
    so the estimate dated t uses rows up to t only, the warm-up factor cancels in the ratio, and
    the result is the diagonal of :func:`compute_ewm_matrix_autocorr`. The ratio divides by the
    current second moment only, so it is not bounded by one.

    Args:
        a: demeaned observations, shape (t,) or (t, n)
        span: if given, overrides ``ewm_lambda`` via ``lambda = 1 - 2 / (span + 1)``
        ewm_lambda: EWM decay, used when ``span`` is None
        lag: lag in rows
        is_normalize: return the ratio; False returns ``g_t``
        nan_backfill: how a non-finite update is handled; by default both states carry forward
        var_init_type: seed of ``v``. ``InitType.ZERO`` (default) is point in time;
            ``InitType.VAR`` seeds with the full-sample ``np.nanvar`` of each column, which
            looks ahead with a weight that decays like ``lambda^t``

    Returns:
        shape (t, n), n being 1 for 1-d input; NaN for the first ``lag`` rows and wherever the
        second moment is not positive

    Raises:
        ValueError: if ``var_init_type`` is neither ``InitType.ZERO`` nor ``InitType.VAR``
    """
    if span is not None:
        ewm_lambda = 1.0 - 2.0 / (span + 1.0)
    ewm_lambda_1 = 1.0 - ewm_lambda
    if var_init_type not in (InitType.ZERO, InitType.VAR):
        raise ValueError(f"var_init_type must be InitType.ZERO or InitType.VAR, "
                         f"got {var_init_type}")
    is_insample_seed = var_init_type == InitType.VAR

    if a.ndim == 1:  # ndarry
        is_1d = True
        n = 1
        last_auto_covar = 0.0
        last_covar = np.nanvar(a) if is_insample_seed else 0.0
    else:
        is_1d = False
        n = a.shape[1]  # array of ndarray
        last_auto_covar = np.zeros(n)  # initialise at zero
        last_covar = np.nanvar(a, axis=0) if is_insample_seed else np.zeros(n)
    t = a.shape[0]
    autocorr = np.full((t, n), np.nan)  # rows before the first lagged pair have no estimate

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

        if is_normalize:  # NaN, not inf, where the second moment is zero (e.g. at a zero seed)
            if is_1d:
                autocorr[idx, :] = (current_auto_covar / current_covar if current_covar > 0.0
                                    else np.nan)
            else:
                autocorr[idx, :] = np.divide(current_auto_covar, current_covar,
                                             out=np.full(n, np.nan),
                                             where=current_covar > 0.0)
        else:
            autocorr[idx, :] = current_auto_covar

    return autocorr


def compute_ewm_vector_autocorr_df(data: Union[pd.DataFrame, pd.Series],
                                   span: Union[int, np.ndarray] = 30,
                                   lag: int = 1,
                                   is_normalize: bool = True,
                                   var_init_type: InitType = InitType.ZERO
                                   ) -> Union[pd.DataFrame, pd.Series]:
    """
    EWM autocorrelation ratio of each column after removing its EWM mean.

    Sets ``z_t = x_t - m_t`` with ``m_t`` the EWM mean of :func:`qis.compute_ewm` at the same
    span (seeded at the first observation and including ``x_t``, so point in time) and runs
    :func:`compute_ewm_vector_autocorr` on ``z``.

    Args:
        data: observations, rows are dates
        span: EWM span of the mean and of the moments, in rows
        lag: lag in rows
        is_normalize: return the ratio; False returns the EWM lagged moment
        var_init_type: seed of the second moment; ``InitType.ZERO`` (default) is point in
            time, ``InitType.VAR`` the full-sample variance seed, which looks ahead

    Returns:
        the estimates in the container of ``data``, NaN for the first ``lag`` rows
    """
    x = data - compute_ewm(data=data, span=span)
    autocorr = compute_ewm_vector_autocorr(a=x.to_numpy(),
                                           span=span,
                                           lag=lag,
                                           is_normalize=is_normalize,
                                           var_init_type=var_init_type)
    if isinstance(data, pd.DataFrame):
        autocorr = pd.DataFrame(autocorr, index=data.index, columns=data.columns)
    else:
        autocorr = pd.Series(autocorr[:, 0], index=data.index, name=data.name)
    return autocorr


def compute_autocorrelation_at_int_periods(data: pd.DataFrame,
                                           span: int = 30,
                                           is_returns: bool = True,
                                           demean: bool = True,
                                           ewma_smoothin_span: Optional[int] = None
                                           ) -> pd.Series:
    """
    lag-one Pearson autocorrelation of non-overlapping blocks of ``span`` rows, per column.

    Blocks are formed by ``qis.utils.df_freq.df_resample_at_int_index`` and correlated with
    :func:`compute_path_autocorr` at lag one, each segment about its own mean. The result is a
    full-sample, descriptive statistic.

    Args:
        data: observations, rows are dates and columns are series
        span: block length in rows (not an EWM span)
        is_returns: a block value is the sum of its rows with NaNs counted as zero; False takes
            the last value of the block, for levels
        demean: subtract the full-sample mean of the block values first. A Pearson correlation
            removes the segment means itself, so this has no effect on the result; retained
            for compatibility
        ewma_smoothin_span: reserved for an EWM-smoothed variant that is not implemented; must
            be None

    Returns:
        the block autocorrelations, indexed by the columns of ``data``

    Raises:
        NotImplementedError: if ``ewma_smoothin_span`` is not None
    """
    if ewma_smoothin_span is not None:
        raise NotImplementedError(f"ewma_smoothin_span={ewma_smoothin_span} is not implemented; "
                                  f"pass None for the full-sample block autocorrelation")
    if is_returns:
        # resample with sums
        resampled_data = df_resample_at_int_index(df=data, func=np.nansum, sample_size=span)
    else:
        # resample at last value
        resampled_data = df_resample_at_int_index(df=data, func=lambda x: x.iloc[-1], sample_size=span)

    if demean:
        resampled_data = resampled_data-np.nanmean(resampled_data, axis=0, keepdims=True)

    # compute autocorr for lags = [0, 1] and report the last value
    autocorr = compute_path_autocorr(a=resampled_data.to_numpy(), num_lags=2)
    autocorr = pd.Series(autocorr[-1], index=data.columns)

    return autocorr


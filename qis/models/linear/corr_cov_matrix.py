"""
correlation and covariance estimation on the EWM engine, in the shapes a caller needs them in.

``estimate_rolling_ewma_covar`` is the backtest-facing entry point: the EWM covariance of log
returns taken at ``returns_freq``, sampled on a ``rebalancing_freq`` schedule, one matrix per
rebalancing date and annualised unless ``apply_an_factor`` is False. ``compute_masked_covar_corr``
is the single-matrix path for a ragged panel - each pair is computed on the observations both
series have, which uses all the data and is not guaranteed positive semi-definite.
``compute_ewm_corr_df`` unstacks the correlation tensor into one column per pair, with
``CorrMatrixOutput`` choosing which pairs come back.

``span`` is in units of ``returns_freq``, not days, and the estimation and rebalancing frequencies
are separate arguments because one sets the sampling error and the other the turnover. The
recursion lives in ``ewm.py``, the heatmaps and time-series exhibits in ``plot_correlations.py``.
"""
# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
from typing import Tuple, List, Union, Optional, Dict
from numba import njit

# qis
import qis.utils.dates as da
import qis.plots.time_series as pts
import qis.models.linear.ewm as ewm
import qis.perfstats.returns as ret
from qis.utils.annualisation import infer_annualisation_factor_from_df


def estimate_rolling_ewma_covar(prices: pd.DataFrame,
                                time_period: da.TimePeriod = None,  # when we start estimation
                                returns_freq: str = 'W-WED',
                                rebalancing_freq: str = 'QE',
                                span: int = 52,
                                is_apply_vol_normalised_returns: bool = False,
                                demean: bool = True,
                                apply_an_factor: bool = True
                                ) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    EWM covariance matrix sampled on a rebalancing schedule, ready for a rolling backtest.

    Returns one matrix per rebalancing date rather than a single matrix, so a backtest can look up
    the covariance it would have had at each rebalancing without recomputing. Returns are estimated
    at ``returns_freq`` and the matrices are taken at ``rebalancing_freq``: the two are separate
    because the estimation frequency sets the sampling error and the rebalancing frequency sets the
    turnover.

    Args:
        prices: price levels, one column per asset. NaN is tolerated
        time_period: restrict the output dates. None uses the full sample
        returns_freq: frequency the returns are computed at
        rebalancing_freq: frequency the covariance is sampled at
        span: EWM span in units of ``returns_freq``
        is_apply_vol_normalised_returns: estimate the correlation on vol-normalised returns and
            rebuild the covariance from it, which stops a single volatile asset dominating
        demean: subtract the EWM mean before estimating. False takes the second moment about zero
        apply_an_factor: annualise, so the matrix is in annual units

    Returns:
        rebalancing date to the covariance matrix estimated at that date, indexed and labelled by
        the columns of ``prices``
    """
    returns = ret.to_returns(prices=prices, is_log_returns=True, drop_first=True, freq=returns_freq)
    returns_np = returns.to_numpy()
    if demean:
        x = returns_np - ewm.compute_ewm(returns_np, span=span)
    else:
        x = returns_np

    if is_apply_vol_normalised_returns:
        covar_tensor_txy, _, _ = ewm.compute_ewm_covar_tensor_vol_norm_returns(a=x, span=span, nan_backfill=ewm.NanBackfill.ZERO_FILL)
    else:
        covar_tensor_txy = ewm.compute_ewm_covar_tensor(a=x, span=span, nan_backfill=ewm.NanBackfill.ZERO_FILL)

    # create rebalancing schedule
    rebalancing_schedule = da.generate_rebalancing_indicators(df=returns, freq=rebalancing_freq)

    tickers = prices.columns.to_list()
    covars = {}
    if apply_an_factor:
        an_factor = infer_annualisation_factor_from_df(data=returns)
    else:
        an_factor = 1.0
    if time_period is not None:
        start_date = time_period.start.tz_localize(tz=returns.index.tz)  # make sure tz is alined with rebalancing_schedule
    else:
        start_date = rebalancing_schedule.index[0]
    for idx, (date, value) in enumerate(rebalancing_schedule.items()):
        if value and date >= start_date:
            covar_t = pd.DataFrame(covar_tensor_txy[idx], index=tickers, columns=tickers)
            covars[date] = an_factor*covar_t
    return covars


@njit
def compute_path_corr(a1: np.ndarray,
                      a2: np.ndarray
                      ) -> np.ndarray:
    """
    compute paths correlation between columns of a1 and a2
    """
    is_1d = (a1.ndim == 1)
    if not is_1d:
        ncols = a1.shape[1]
        acorr = np.zeros(ncols)
        for idx in range(ncols):
            acorr[idx] = np.corrcoef(a1[:, idx], a2[:, idx], rowvar=False)[0][1]
    else:
        acorr = np.corrcoef(a1, a2, rowvar=False)[0][1]

    return acorr


def compute_masked_covar_corr(data: Union[np.ndarray, pd.DataFrame],
                              is_covar: bool = True,
                              bias: bool = False
                              ) -> Union[np.ndarray, pd.DataFrame]:
    """
    covariance or correlation of a returns panel, computed pairwise over the observed entries.

    A ragged panel has no common sample: dropping rows with any missing value can discard most of
    the history, and filling with zero biases the estimate towards zero. Masked arrays compute each
    pair on the observations both series have, which uses all the data at the cost of a matrix that
    is not guaranteed positive semi-definite. Check before feeding it to an optimiser.

    Args:
        data: returns, rows are dates and columns are assets
        is_covar: return the covariance. False returns the correlation
        bias: normalise by ``n`` rather than ``n - 1``. Ignored for the correlation, where the two
            normalisations cancel

    Returns:
        the matrix, in the same type as the input

    Raises:
        ValueError: if ``data`` is neither a DataFrame nor an ndarray
    """
    if isinstance(data, pd.DataFrame):
        data_np = data.to_numpy()
    elif isinstance(data, np.ndarray):
        data_np = data
    else:
        raise ValueError(f"unsuported type {type(data)}")

    if np.any(np.isnan(data_np)):  # applay masked arrays
        if is_covar:
            covar = np.ma.cov(np.ma.masked_invalid(data_np), rowvar=False, bias=bias, allow_masked=True).data
        else:
            # NumPy 2.x removed `bias` from np.ma.corrcoef (and np.corrcoef). It was
            # deprecation-warned since NumPy 1.10 because bias/ddof cancel in correlation
            # normalization — the argument had no effect on the result.
            covar = np.ma.corrcoef(np.ma.masked_invalid(data_np), rowvar=False, allow_masked=True).data
    else:
        if is_covar:
            covar = np.cov(data_np, rowvar=False, bias=bias)
        else:
            # NumPy 2.x: `bias` removed from np.corrcoef (see note above).
            covar = np.corrcoef(data_np, rowvar=False)

    if isinstance(data, pd.DataFrame):
        covar = pd.DataFrame(data=covar, index=data.columns, columns=data.columns)

    return covar


def corr_to_pivot_row(pivot: np.ndarray,
                      data: np.ndarray,
                      is_normalized: bool = True,
                      vol_scalers: List[Tuple[float, float]] = None  # [0] is pivot vol, [1] vol of asset
                      ) -> np.ndarray:
    """
    compute correlation row between pivot row of returns and return columns in data
    pivot returns are row data (r1,r2)
    data columns are column data ([c1, c2])
    output is correlation row of pivot to columns
    columns may have nans but pivot must be non-nan
    """
    n = len(data[0])
    corrs = np.zeros(n)

    # split column data into columns arrays
    column_data = np.hsplit(data, n)
    for idx, column_data in enumerate(column_data):

        # columnn data may have nans different from pivot
        # cross wil get nans from column_data
        # need to transpose column_data back to rows data
        cross = pivot * column_data.T

        if vol_scalers is not None:
            # multiply by n of non nans
            num = np.count_nonzero(np.isnan(cross) == False)
            std2 = num*vol_scalers[idx][0]*vol_scalers[idx][1]

        else:
            if is_normalized:
                cond = np.isnan(cross) == False  # cond will be [[]] array
                if np.any(cond == True):
                    clean_pivot = pivot[cond[0]]
                    clean_column = column_data[cond[0]]
                else:
                    clean_pivot = pivot
                    clean_column = column_data
                std2 = np.sqrt(np.nansum(clean_pivot*clean_pivot) * np.nansum(clean_column*clean_column))

            else:
                std2 = 1.0

        num_sum = np.nansum(cross)
        if not np.isnan(std2) and not np.isclose(std2, 0.0) and not np.isnan(num_sum):
            corrs[idx] = num_sum / std2
        else:
            corrs[idx] = np.nan

    return corrs


class CorrMatrixOutput(Enum):
    FULL = 1
    TOP_ROW = 2
    SUB_TOP = 3


def compute_ewm_corr_df(df: pd.DataFrame,
                        corr_matrix_output: CorrMatrixOutput = CorrMatrixOutput.FULL,
                        span: Union[int, np.ndarray] = None,
                        ewm_lambda: float = 0.94,
                        init_value: np.ndarray = None,
                        init_type: ewm.InitType = ewm.InitType.ZERO
                        ) -> pd.DataFrame:
    """
    compute ewm corr as and output as xi-xj pandas j>i, i = 0,..
    """
    if init_value is None:
        init_value = ewm.set_init_dim2(data=df.to_numpy(), init_type=init_type)

    corr = ewm.compute_ewm_covar_tensor(a=df.to_numpy(),
                                        span=span,
                                        ewm_lambda=ewm_lambda,
                                        is_corr=True,
                                        covar0=init_value)
    corr_ijs = []
    for idx_i, column_i in enumerate(df.columns):
        if corr_matrix_output == CorrMatrixOutput.SUB_TOP and idx_i == 0:  # skip for idx_i = 0
            continue

        for idx_j, column_j in enumerate(df.columns):
            if corr_matrix_output == CorrMatrixOutput.TOP_ROW:  # get j after i
                if idx_j > idx_i:
                    corr_ij = pd.Series(corr[:, idx_i, idx_j], name=f"{column_i} - {column_j}")
                    corr_ijs.append(corr_ij)
            else:  # get j before i
                if idx_j < idx_i:
                    corr_ij = pd.Series(corr[:, idx_i, idx_j], name=f"{column_i} - {column_j}")
                    corr_ijs.append(corr_ij)
        if corr_matrix_output == CorrMatrixOutput.TOP_ROW:  # stop after idx_i = 0
            break

    corrs_by_column = pd.concat(corr_ijs, axis=1)
    corrs_by_column = corrs_by_column.set_index(df.index)

    return corrs_by_column


def compute_ewm_corr_single(returns: pd.DataFrame,
                            ewm_lambda: float = 0.94,
                            span: Optional[int] = None,
                            time_period: da.TimePeriod = None
                            ) -> pd.Series:
    """
    plot correlation all time series in correlation matrix  as row
    """
    if len(returns.columns) != 2:
        raise ValueError("should be two columns {returns.columns}")

    if span is not None:
        ewm_lambda = 1.0 - 2.0 / (1.0 + span)

    corr = compute_ewm_corr_df(df=returns,
                               corr_matrix_output=CorrMatrixOutput.SUB_TOP,
                               ewm_lambda=ewm_lambda)

    if time_period is not None:
        corr = time_period.locate(corr)

    return corr.iloc[:, 0]


def matrix_regularization(covar: np.ndarray, cut: float = 1e-5) -> np.ndarray:
    eig_vals, eig_vecs = np.linalg.eigh(covar)
    eig_vals_alpha = np.where(np.greater(eig_vals, cut), eig_vals, 0.0)
    covar_a = eig_vecs @ np.diag(eig_vals_alpha) @ eig_vecs.T
    return covar_a

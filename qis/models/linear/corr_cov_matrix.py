"""
correlation related core
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
    compute ewma covar matrix: supporting for nans in prices
    covar data frequency is rebalancing_freq for period time_period
    output is dict[estimation timestamp, pd.Dataframe(estimated_covar)
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
    given returns: time * assets
    compute covar by masking nans
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
            covar = np.ma.corrcoef(np.ma.masked_invalid(data_np), rowvar=False, bias=bias, allow_masked=True).data
    else:
        if is_covar:
            covar = np.cov(data_np, rowvar=False, bias=bias)
        else:
            covar = np.corrcoef(data_np, rowvar=False, bias=bias)

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


class LocalTests(Enum):
    CORR = 1
    EWMA_CORR_MATRIX = 2
    PLOT_CORR_MATRIX = 3
    MATRIX_REGULARIZATION = 4


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    if local_test == LocalTests.CORR:
        t = 100
        n = 4
        data = np.random.normal(0, 1.0, (t, n))
        pivot = data[:, 0]
        corrs = corr_to_pivot_row(pivot=pivot, data=data, is_normalized=True)
        print(corrs)

    elif local_test == LocalTests.EWMA_CORR_MATRIX:
        dates = pd.date_range(start='12/31/2018', end='12/31/2019', freq='B')
        n = 3
        mean = [-2.0, -1.0, 0.0]
        returns = pd.DataFrame(data=np.random.normal(mean, 1.0, (len(dates), n)),
                               index=dates,
                               columns=['x' + str(m + 1) for m in range(n)])

        corr = ewm.compute_ewm_covar_tensor(a=returns.to_numpy(), is_corr=True)
        print('corr')
        print(corr)
        print('corr_00')
        print(corr[0, 0, :])
        print('corr_01')
        print(corr[0, 1, :])
        print('corr_02')
        print(corr[0, 2, :])
        print('corr_last')
        print(corr[:, :, -1])

    elif local_test == LocalTests.PLOT_CORR_MATRIX:

        dates = pd.date_range(start='31Dec2020', end='31Dec2021', freq='B')
        n = 3
        mean = [-2.0, -1.0, 0.0]
        returns = pd.DataFrame(data=np.random.normal(mean, 1.0, (len(dates), n)),
                               index=dates,
                               columns=['x' + str(m + 1) for m in range(n)])
        corrs = compute_ewm_corr_df(df=returns)
        print(corrs)

        pts.plot_time_series(df=corrs,
                             legend_stats=pts.LegendStats.AVG_LAST,
                             trend_line=pts.TrendLine.AVERAGE)

    elif local_test == LocalTests.MATRIX_REGULARIZATION:
        covar = np.array([[1.0, -0.01, 0.01],
                         [-0.01, 0.5, 0.005],
                         [0.01, 0.005, 0.0001]])
        covar_a = matrix_regularization(covar=covar)
        print(covar_a)

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.MATRIX_REGULARIZATION)

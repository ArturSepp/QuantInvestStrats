"""
correlation and covariance estimation on the EWM engine, in the shapes a caller needs them in.

``estimate_rolling_ewma_covar`` is the backtest-facing entry point: the EWM covariance of log
returns taken at ``returns_freq``, sampled on a ``rebalancing_freq`` schedule, one matrix per
rebalancing date and annualised unless ``apply_an_factor`` is False. ``compute_masked_covar_corr``
is the single-matrix path for a ragged panel - each pair is computed on the observations both
series have, about the means of that overlap, which uses all the data and is not guaranteed
positive semi-definite. ``compute_ewm_corr_df`` unstacks the correlation tensor into one column
per pair, with ``CorrMatrixOutput`` choosing which pairs come back.

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
                                apply_an_factor: bool = True,
                                warmup_period: Optional[int] = None
                                ) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    EWM covariance matrix sampled on a rebalancing schedule, ready for a rolling backtest.

    Returns one matrix per rebalancing date rather than a single matrix, so a backtest can look up
    the covariance it would have had at each rebalancing without recomputing. Returns are estimated
    at ``returns_freq`` and the matrices are taken at ``rebalancing_freq``: the two are separate
    because the estimation frequency sets the sampling error and the rebalancing frequency sets the
    turnover.

    The recursion starts from a zero matrix, so the first matrices are scaled down by the
    warm-up factor ``1 - lambda^K`` after ``K`` returns; pass ``warmup_period`` to mask each
    asset until it has enough returns, or a ``time_period`` that starts a few spans after the
    first price. An asset's row and column are NaN until its first return, in both estimators.
    On that date the direct recursion gives ``(1 - lambda) x x'`` (times ``(1 + lambda) / 2``
    when demeaned); the vol-normalised estimator rebuilds the covariance from EWM volatilities
    seeded with the first squared residual, so its first variances carry no ``1 - lambda``
    warm-up factor. Missing returns after an asset's first return
    reset the affected entries to zero (``NanBackfill.ZERO_FILL``), which keeps every matrix
    positive semi-definite.

    Args:
        prices: price levels, one column per asset. NaN is tolerated
        time_period: restrict the output to rebalancing dates on or after its start and on or
            before its end; a missing bound is not applied. None uses the full sample. The
            estimation always runs from the first price
        returns_freq: frequency the returns are computed at
        rebalancing_freq: frequency the covariance is sampled at
        span: EWM span in units of ``returns_freq``
        is_apply_vol_normalised_returns: estimate the correlation on vol-normalised returns and
            rebuild the covariance from it, which stops a single volatile asset dominating
        demean: remove the EWM mean before estimating. The residual is the one-step forecast
            error ``x_t - m_{t-1}`` against the EWM mean of the previous date, so it is point in
            time; the mean starts from zero, so the first residual of an asset is its first
            return. The matrix is multiplied by ``(1 + lambda) / 2``, which makes it unbiased for
            iid returns once the seed is forgotten. False takes the second moment about zero
        apply_an_factor: annualise, so the matrix is in annual units
        warmup_period: None masks an asset only before its first return. An integer ``k``
            also masks it for its first ``k`` returns, counted from its own first return, so
            its row and column are NaN until it has ``k + 1`` returns

    Returns:
        rebalancing date to the covariance matrix estimated at that date, indexed and labelled by
        the columns of ``prices``
    """
    returns = ret.to_returns(prices=prices, is_log_returns=True, drop_first=True, freq=returns_freq)
    returns_np = returns.to_numpy()
    if demean:
        # the EWM mean m_t includes x_t, so x_t - m_t = lambda (x_t - m_{t-1}): dividing by lambda
        # gives the one-step forecast error against the prior mean m_{t-1}. The mean is seeded at
        # zero, so an asset's first residual is its first return rather than exactly zero, which
        # would give a zero first volatility and a NaN in the vol-normalised estimator.
        # Its steady-state covariance for iid returns is 2 / (1 + lambda) Sigma, so the residual is
        # scaled by sqrt((1 + lambda) / 2) to make the covariance unbiased.
        ewm_lambda = 1.0 - 2.0 / (span + 1.0)
        prior_mean = ewm.compute_ewm(returns_np, span=span, init_type=ewm.InitType.ZERO)
        prior_mean_residual = (returns_np - prior_mean) / ewm_lambda
        x = np.sqrt(0.5 * (1.0 + ewm_lambda)) * prior_mean_residual
    else:
        x = returns_np

    if is_apply_vol_normalised_returns:
        covar_tensor_txy, _, _ = ewm.compute_ewm_covar_tensor_vol_norm_returns(a=x, span=span, nan_backfill=ewm.NanBackfill.ZERO_FILL)
    else:
        covar_tensor_txy = ewm.compute_ewm_covar_tensor(a=x, span=span, nan_backfill=ewm.NanBackfill.ZERO_FILL)

    # an asset without data, or still in its warm-up, has no covariance: NaN in its row and
    # column rather than the zero that the reset recursion carries, which reads as no risk
    num_returns = np.cumsum(np.isfinite(returns_np), axis=0)
    is_available = num_returns > (0 if warmup_period is None else int(warmup_period))
    pair_available = is_available[:, :, None] & is_available[:, None, :]
    covar_tensor_txy = np.where(pair_available, covar_tensor_txy, np.nan)

    # create rebalancing schedule
    rebalancing_schedule = da.generate_rebalancing_indicators(df=returns, freq=rebalancing_freq)

    tickers = prices.columns.to_list()
    covars = {}
    if apply_an_factor:
        an_factor = infer_annualisation_factor_from_df(data=returns)
    else:
        an_factor = 1.0
    start_date = rebalancing_schedule.index[0]
    end_date = rebalancing_schedule.index[-1]
    if time_period is not None:  # make sure tz is aligned with rebalancing_schedule
        if time_period.start is not None:
            start_date = time_period.start.tz_localize(tz=returns.index.tz)
        if time_period.end is not None:
            end_date = time_period.end.tz_localize(tz=returns.index.tz)
    for idx, (date, value) in enumerate(rebalancing_schedule.items()):
        if value and start_date <= date <= end_date:
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
    the history, and filling with zero biases the estimate towards zero. Each pair is computed on
    the observations both series have, about the means of that overlap, which uses all the data at
    the cost of a matrix that is not guaranteed positive semi-definite. Check before feeding it to
    an optimiser. On a panel with NaN the covariance equals pandas ``DataFrame.cov`` (and, with
    ``bias=True``, the same sums divided by the overlap count ``n_ij`` rather than ``n_ij - 1``)
    and the correlation equals ``DataFrame.corr``; a pair without enough common observations is
    NaN. Without NaN the common-sample ``np.cov`` or ``np.corrcoef`` is used.

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

    if np.any(np.isnan(data_np)):  # pairwise-complete estimation
        if is_covar:
            # pandas centres each pair on its overlap means and divides by n_ij - 1; a pair with
            # fewer than two common observations is NaN. np.ma.cov centred each series on its
            # own full-history mean and dropped the mask, returning 0 for pairs with no overlap.
            covar = pd.DataFrame(data_np).cov().to_numpy()
            if bias:
                observed = np.isfinite(data_np).astype(float)
                n_ij = observed.T @ observed
                covar = covar * np.divide(n_ij - 1.0, n_ij, out=np.full_like(n_ij, np.nan),
                                          where=n_ij > 0.0)
        else:
            # Masked corrcoef can normalise a pairwise covariance with full-history variances,
            # producing correlations outside [-1, 1] for ragged histories.
            covar = pd.DataFrame(data_np).corr().to_numpy()
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
    """
    which pairs of a correlation matrix :func:`compute_ewm_corr_df` returns as columns.

    Attributes:
        FULL: every pair below the diagonal, (i, j) with j < i, named ``"<column i> - <column j>"``
            and ordered by i, then j
        TOP_ROW: the pairs of the first column with every later column, named
            ``"<column 0> - <column j>"``
        SUB_TOP: the same pairs, names and order as ``FULL``. The first row it skips has no pair
            below the diagonal, so the two members coincide; retained for compatibility, and
            used by :func:`compute_ewm_corr_single`
    """
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
    uncentred EWM correlation of every requested pair of columns, one column per pair.

    Runs ``S_t = lambda S_{t-1} + (1 - lambda) x_t x_t'`` on the rows of ``df`` without removing
    a mean, from the seed ``init_value``, and normalises each matrix to a correlation. With the
    default zero seed the first row is the sign of ``x_i x_j``, so the path needs a warm-up.

    Args:
        df: returns, rows are dates and columns are assets
        corr_matrix_output: which pairs are returned: ``FULL`` gives (i, j) with j < i, named
            ``"<column i> - <column j>"``; see :class:`CorrMatrixOutput`
        span: if given, overrides ``ewm_lambda`` via ``lambda = 1 - 2 / (span + 1)``
        ewm_lambda: EWM decay, used when ``span`` is None
        init_value: seed matrix, shape (n, n). None uses ``init_type``
        init_type: seed when ``init_value`` is None; ``InitType.ZERO`` and ``InitType.X0``
            both give a zero matrix

    Returns:
        the correlation paths, indexed like ``df``
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

    corrs_by_column = pd.concat(corr_ijs, axis=1, sort=False)
    corrs_by_column = corrs_by_column.set_index(df.index)

    return corrs_by_column


def compute_ewm_corr_single(returns: pd.DataFrame,
                            ewm_lambda: float = 0.94,
                            span: Optional[int] = None,
                            time_period: da.TimePeriod = None
                            ) -> pd.Series:
    """
    uncentred EWM correlation path of a two-column return panel.

    The two-column case of :func:`compute_ewm_corr_df`, with its zero seed.

    Args:
        returns: returns with exactly two columns
        ewm_lambda: EWM decay, used when ``span`` is None
        span: if given, overrides ``ewm_lambda`` via ``lambda = 1 - 2 / (span + 1)``
        time_period: restrict the output dates; the estimation runs over the whole sample

    Returns:
        the correlation path, named ``"<second column> - <first column>"``

    Raises:
        ValueError: if ``returns`` does not have exactly two columns
    """
    if len(returns.columns) != 2:
        raise ValueError(f"should be two columns {returns.columns}")

    if span is not None:
        ewm_lambda = 1.0 - 2.0 / (1.0 + span)

    corr = compute_ewm_corr_df(df=returns,
                               corr_matrix_output=CorrMatrixOutput.SUB_TOP,
                               ewm_lambda=ewm_lambda)

    if time_period is not None:
        corr = time_period.locate(corr)

    return corr.iloc[:, 0]


def matrix_regularization(covar: np.ndarray, cut: float = 1e-5) -> np.ndarray:
    """
    eigenvalue clipping of a symmetric matrix: eigenvalues at or below ``cut`` are set to zero.

    Rebuilds ``Q diag(nu_j 1{nu_j > cut}) Q'`` from ``np.linalg.eigh``, which reads only the
    lower triangle. With ``cut=0`` this is the Frobenius-nearest positive semi-definite matrix.
    The result is singular whenever an eigenvalue was clipped, and clipping raises the
    diagonal, so renormalise a clipped correlation matrix with ``qis.covar_to_corr``.

    Args:
        covar: symmetric matrix, shape (n, n)
        cut: absolute eigenvalue threshold, in the units of ``covar``. The default ``1e-5`` is
            small for an annualised covariance but comparable to daily variances

    Returns:
        the clipped matrix, shape (n, n)
    """
    eig_vals, eig_vecs = np.linalg.eigh(covar)
    eig_vals_alpha = np.where(np.greater(eig_vals, cut), eig_vals, 0.0)
    covar_a = eig_vecs @ np.diag(eig_vals_alpha) @ eig_vecs.T
    return covar_a

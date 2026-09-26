"""
portfolio volatility and 99% value-at-risk from an EWM covariance of the instrument returns.

Two names sit close together. ``compute_portfolio_var_np`` returns *variance*, w' Σ_t w rolled
forward date by date by the EWM covariance recursion, seeded at zero (or at an explicit
``covar0``) so that the estimate on a date uses no later observation;
``compute_portfolio_correlated_var_by_groups`` and ``compute_portfolio_independent_var_by_ac``
return *value-at-risk*, the volatility scaled by ``VAR99`` = 2.3263, the normal 99% quantile -
the first keeping correlations, the second summing |w| σ across instruments for the
undiversified figure. Both pair the weights of date t with the same covariance Σ_t, so the
undiversified figure is never below the correlated one. ``limit_weights_to_max_var_limit`` caps
a weight at a VaR budget in bp.

``compute_portfolio_vol`` lags the weights one period before contracting them with the
covariance (``weight_lag``), and leaves the result per period unless ``annualize`` is set.
Decomposing a covariance already in hand is ``contributions.py``; the recursion is in
``qis/models/linear/``.
"""

import warnings
import numpy as np
import pandas as pd
from numba import njit
from typing import Union, Optional, List, Tuple

import qis.utils.dates as da
import qis.utils.np_ops as npo
import qis.utils.df_groups as dfg
import qis.utils.df_agg as dfa
import qis.perfstats.returns as ret
from qis.models.linear.ewm import MeanAdjType, InitType, NanBackfill, compute_rolling_mean_adj
from qis.utils.annualisation import infer_annualisation_factor_from_df


VAR99 = 2.3263
VAR99_SCALER_BP = VAR99 * 10000


@njit
def limit_weights_to_max_var_limit(weights: np.ndarray,
                                   vols: np.ndarray,
                                   max_var_limit_bp: Union[np.ndarray, float] = 25.00,
                                   annualization_factor: float = 252.0
                                   ) -> np.ndarray:
    """Limits portfolio weights to stay within maximum VaR limits per instrument.

    The one-period 99% VaR of instrument i is 10^4 z |w_i| σ_i / sqrt(annualization_factor) in
    basis points, with z = ``VAR99``; a weight whose VaR exceeds the limit is cut to the limit,
    keeping its sign. The cap is per instrument and ignores correlation.

    Args:
        weights: Portfolio weights array.
        vols: Annualized volatilities for each instrument.
        max_var_limit_bp: Maximum VaR limit in basis points (default 25bp).
        annualization_factor: Periods per year with which ``vols`` were annualised; the VaR is
            computed over one such period. The default 252 is the factor qis applies to
            business-day returns (``get_annualization_factor('B')``); pass 260 to reproduce the
            former default.

    Returns:
        Adjusted weights that satisfy VaR constraints.
    """
    saf = np.sqrt(annualization_factor)
    instrument_var = VAR99_SCALER_BP * np.abs(weights) * vols / saf
    cond = instrument_var > max_var_limit_bp
    if np.any(cond):
        weight_limit = max_var_limit_bp / (VAR99_SCALER_BP * vols / saf)
        up_breach = np.logical_and(cond, np.greater(weights, 0.0))
        down_breach = np.logical_and(cond, np.less(weights, 0.0))
        weights1 = np.where(up_breach, weight_limit, np.where(down_breach, -weight_limit, weights))
    else:
        weights1 = weights
    return weights1


def compute_portfolio_vol(returns: pd.DataFrame,
                          weights: pd.DataFrame,
                          span: Union[int, np.ndarray] = None,
                          ewm_lambda: Union[float, np.ndarray] = 0.94,
                          is_return_vol: bool = True,
                          mean_adj_type: MeanAdjType = MeanAdjType.NONE,
                          init_type: InitType = InitType.ZERO,
                          annualize: bool = False,
                          annualization_factor: float = None,
                          nan_backfill: NanBackfill = NanBackfill.FFILL,
                          weight_lag: int = 1
                          ) -> pd.Series:
    """Computes portfolio volatility from a point-in-time EWM covariance of the returns.

    On each date t of the common index of ``returns`` and ``weights`` the covariance
    Σ_t = (1 - λ) r_t r_t' + λ Σ_{t-1} is updated with that date's returns, starting from a
    zero matrix before the first date, and contracted with the weights ``weight_lag`` rows
    earlier: σ²_t = w_{t-k}' Σ_t w_{t-k}. With the default k = 1 the weights held over
    (t-1, t] meet a covariance that includes the return they earn, so the result is the EWM
    variance of the held portfolio as of t; k = 0 pairs the weights of date t with the
    covariance known at t, the one-period-ahead estimate the VaR functions use. No observation
    after t enters the estimate at t, so the early values carry the warm-up of the zero seed
    (its weight at the n-th date is λ^n) rather than information from the end of the sample.

    Args:
        returns: Asset returns DataFrame. Missing returns are set to zero, so a gap decays the
            covariance by λ per period.
        weights: Portfolio weights DataFrame, aligned with ``returns`` on their common dates and
            columns; missing weights are zero.
        span: EWM span for covariance estimation; overrides ``ewm_lambda`` via
            λ = 1 - 2 / (span + 1).
        ewm_lambda: EWM decay parameter (default 0.94), used when ``span`` is None.
        is_return_vol: If True, returns volatility; if False, returns variance.
        mean_adj_type: Mean subtracted from the returns before the recursion.
            ``MeanAdjType.INSAMPLE`` subtracts the full-sample mean and is forward-looking.
        init_type: Seed of the running mean of the optional mean adjustment. It does not
            affect the covariance recursion, which always starts from a zero matrix.
        annualize: Whether to annualize: the variance is multiplied by the annualisation
            factor, so the volatility scales by its square root.
        annualization_factor: Factor for annualizing (inferred from the index if None).
        nan_backfill: Forwarded to the optional mean adjustment. It has no effect, because
            missing returns are set to zero before the mean adjustment and the recursion.
        weight_lag: Number of rows by which the weights are lagged before they meet the
            covariance (default 1).

    Returns:
        Time series of portfolio volatility or variance on the common index.

    Raises:
        ValueError: If ``weight_lag`` is negative, which would pair each covariance with
            weights decided after it.
    """
    if weight_lag < 0:
        raise ValueError(f"weight_lag must be non-negative, got {weight_lag}")
    # align index and columns
    weights, returns = weights.align(other=returns, join='inner')
    if weight_lag != 0:
        weights = weights.shift(weight_lag)

    returns_np = _to_recursion_returns(returns=returns,
                                       span=span,
                                       ewm_lambda=ewm_lambda,
                                       mean_adj_type=mean_adj_type,
                                       init_type=init_type,
                                       nan_backfill=nan_backfill)
    weights_np = npo.to_finite_np(data=weights, fill_value=0.0)

    if span is not None:
        ewm_lambda = 1.0 - 2.0 / (span + 1.0)

    portfolio_vol = compute_portfolio_var_np(returns=returns_np,
                                             weights=weights_np,
                                             ewm_lambda=ewm_lambda)

    if annualize:
        if annualization_factor is None:
            if isinstance(weights, pd.DataFrame):
                annualization_factor = infer_annualisation_factor_from_df(data=weights)
            else:
                warnings.warn(f"in compute_ewm: annualization_factor for np array, default is 1")
                annualization_factor = 1.0

        portfolio_vol = annualization_factor * portfolio_vol

    if is_return_vol:
        portfolio_vol = np.sqrt(portfolio_vol)

    portfolio_vol = pd.Series(data=portfolio_vol, index=weights.index)

    return portfolio_vol


def _to_recursion_returns(returns: pd.DataFrame,
                          span: Optional[Union[int, float]] = None,
                          ewm_lambda: float = 0.94,
                          mean_adj_type: MeanAdjType = MeanAdjType.NONE,
                          init_type: InitType = InitType.ZERO,
                          nan_backfill: NanBackfill = NanBackfill.FFILL
                          ) -> np.ndarray:
    """Returns as they enter the EWM covariance recursion: zero-filled, optionally demeaned.

    Shared by ``compute_portfolio_vol`` and ``compute_portfolio_independent_var_by_ac``, so the
    correlated and the undiversified VaR run on identical inputs.

    Args:
        returns: Asset returns DataFrame.
        span: EWM span of the optional mean adjustment.
        ewm_lambda: EWM decay of the optional mean adjustment when ``span`` is None.
        mean_adj_type: Mean subtracted before the recursion.
        init_type: Seed of the running mean.
        nan_backfill: Forwarded to the mean adjustment.

    Returns:
        Array of shape (T, N) with finite values.
    """
    returns_np = npo.to_finite_np(data=returns, fill_value=0.0)
    if mean_adj_type != MeanAdjType.NONE:
        returns_np = compute_rolling_mean_adj(data=returns_np,
                                              mean_adj_type=mean_adj_type,
                                              span=span,
                                              ewm_lambda=ewm_lambda,
                                              init_type=init_type,
                                              nan_backfill=nan_backfill)
    return returns_np


@njit
def _compute_ewm_var_np(returns: np.ndarray, ewm_lambda: float) -> np.ndarray:
    """Diagonal of the ``compute_portfolio_var_np`` covariance recursion, zero-seeded.

    Args:
        returns: Asset returns array of shape (T, N).
        ewm_lambda: EWM decay.

    Returns:
        Array of shape (T, N) with the EWM variance of each asset on each row; element by
        element the same arithmetic as the diagonal of Σ_t in ``compute_portfolio_var_np``.
    """
    t = returns.shape[0]
    n = returns.shape[1]
    ewm_lambda_1 = 1.0 - ewm_lambda
    last_var = np.zeros(n)
    variances = np.zeros((t, n))
    for idx in range(0, t):
        row = returns[idx]
        var = ewm_lambda_1 * (row * row) + ewm_lambda * last_var
        var = np.where(np.isfinite(var), var, ewm_lambda * last_var)
        variances[idx] = var
        last_var = var
    return variances


@njit
def compute_portfolio_var_np(returns: np.ndarray,
                             weights: np.ndarray,
                             span: Union[int, float, np.ndarray] = None,
                             ewm_lambda: Union[float, np.ndarray] = 0.94,
                             covar0: np.ndarray = None
                             ) -> np.ndarray:
    """Computes the path of portfolio variances from an EWM covariance recursion (numba).

    Σ_t = (1 - λ) r_t r_t' + λ Σ_{t-1}, started from Σ_{-1} = ``covar0`` (a zero matrix when
    None), and the output on row t is w_t' Σ_t w_t. Weights and returns on the same row are
    paired, so lag the weights before the call when they should be the ones held over the
    period. The estimate on row t uses rows 0, ..., t only: it is point in time. A non-finite
    return makes the covariance entries it touches decay by λ, as a zero return would.

    Args:
        returns: Asset returns array of shape (T, N).
        weights: Portfolio weights array of shape (T, N); non-finite weights count as zero.
        span: EWM span; overrides ``ewm_lambda`` via λ = 1 - 2 / (span + 1).
        ewm_lambda: EWM decay parameter (default 0.94), used when ``span`` is None.
        covar0: Seed covariance before the first row, shape (N, N); non-finite entries count
            as zero. Its weight on row t is λ^(t+1). A seed estimated on the same sample, such
            as ``compute_ewm_covar(a=returns)``, puts later observations into early estimates.

    Returns:
        Array of portfolio variances over time.
    """
    t = returns.shape[0]  # time dimension
    n = returns.shape[1]  # space dimension

    # important to replace nans for @ operator
    weights = np.where(np.isfinite(weights), weights, 0.0)
    if covar0 is None:  # point in time: nothing is known before the first row
        last_covar = np.zeros((n, n))
    else:
        last_covar = np.where(np.isfinite(covar0), covar0, 0.0)
    portfolio_vol = np.zeros(t)
    if span is not None:
        ewm_lambda = 1.0 - 2.0 / (span + 1.0)
    ewm_lambda_1 = 1.0 - ewm_lambda
    for idx in range(0, t):  # row in x
        row = returns[idx]
        r_ij = np.outer(row, row)
        covar = ewm_lambda_1 * r_ij + ewm_lambda * last_covar
        covar = np.where(np.isfinite(covar), covar, ewm_lambda*last_covar)
        last_covar = covar
        weights_t = np.ascontiguousarray(weights[idx])  # to remove NumbaPerformanceWarning warning
        portfolio_vol[idx] = weights_t.T @ covar @ weights_t

    return portfolio_vol


def compute_portfolio_correlated_var_by_groups(prices: pd.DataFrame,
                                               weights: pd.DataFrame,
                                               group_data: Optional[pd.Series] = None,
                                               group_order: List[str] = None,
                                               total_column: Optional[str] = 'Total',
                                               freq: Optional[str] = 'B',
                                               vol_span: int = 33,  # span in number of freq-retunrs
                                               time_period: da.TimePeriod = None,
                                               mean_adj_type: MeanAdjType = MeanAdjType.NONE
                                               ) -> pd.DataFrame:
    """Computes portfolio VaR accounting for correlations, optionally grouped by categories.

    The one-period 99% VaR on date t is VAR99 * sqrt(w_t' Σ_t w_t): the weights of date t
    with the zero-seeded EWM covariance of the log returns up to and including t, both known
    at t, so the figure is the VaR of the current positions over the next period. A group's
    VaR uses the group's own weights and covariance block; group figures are standalone and do
    not add up to the total.

    Args:
        prices: Asset price DataFrame.
        weights: Portfolio weights DataFrame, aligned with the returns on common dates.
        group_data: Optional Series for grouping assets by categories.
        group_order: Order of groups in output.
        total_column: Name for total portfolio column.
        freq: Frequency for return calculation (default 'B' for business daily).
        vol_span: EWM span for volatility estimation (default 33).
        time_period: Optional time period filter, applied after estimation.
        mean_adj_type: Type of mean adjustment to apply.

    Returns:
        DataFrame of VaR values by group over time, as decimal fractions of NAV per period.
    """
    returns = ret.to_returns(prices=prices, freq=freq, is_log_returns=True)

    weights, returns = weights.align(other=returns, join='inner')

    if group_data is not None:
        ac_exposures_dict = dfg.split_df_by_groups(df=weights, group_data=group_data, group_order=group_order,
                                                   total_column=total_column)
        ac_returns_dict = dfg.split_df_by_groups(df=returns, group_data=group_data, group_order=group_order,
                                                 total_column=total_column)
    else:
        ac_exposures_dict = {'Total VAR': weights}
        ac_returns_dict = {'Total VAR': returns}

    portfolio_vars = {}
    for (ac, ac_exposure), (ac, ac_returns) in zip(ac_exposures_dict.items(), ac_returns_dict.items()):
        portfolio_vars[ac] = compute_portfolio_vol(returns=ac_returns,
                                                   weights=ac_exposure,
                                                   span=vol_span,
                                                   mean_adj_type=mean_adj_type,
                                                   annualize=False,
                                                   weight_lag=0)
    portfolio_vars = pd.DataFrame.from_dict(portfolio_vars, orient='columns')
    portfolio_vars = VAR99*portfolio_vars
    if time_period is not None:
        portfolio_vars = time_period.locate(portfolio_vars)
    return portfolio_vars


def compute_portfolio_independent_var_by_ac(prices: pd.DataFrame,
                                            weights: pd.DataFrame,
                                            group_data: Optional[pd.Series] = None,
                                            group_order: List[str] = None,
                                            total_column: Optional[str] = 'Total',
                                            freq: Optional[str] = 'B',
                                            vol_span: int = 33,  # span in number of freq-retunrs
                                            time_period: da.TimePeriod = None,
                                            mean_adj_type: MeanAdjType = MeanAdjType.NONE
                                            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Computes the undiversified portfolio VaR: the sum of standalone instrument VaRs.

    Instrument i contributes VAR99 * |w_{i,t}| σ_{i,t}, where σ²_{i,t} is the diagonal of the
    same zero-seeded EWM covariance Σ_t that ``compute_portfolio_correlated_var_by_groups``
    uses, with the same weights of date t. Adding standalone VaRs assumes that every pair of
    positions is perfectly aligned (correlation times the sign of the weight product equal to
    one), not that assets are independent: the result is the undiversified upper bound, and on
    every date it is at least the correlated VaR of the same instruments and group.

    Args:
        prices: Asset price DataFrame.
        weights: Portfolio weights DataFrame, aligned with the returns on common dates.
        group_data: Optional Series for grouping assets by categories.
        group_order: Order of groups in output.
        total_column: Name for total portfolio column.
        freq: Frequency for return calculation (default 'B' for business daily).
        vol_span: EWM span for volatility estimation (default 33).
        time_period: Optional time period filter, applied after estimation.
        mean_adj_type: Type of mean adjustment to apply.

    Returns:
        Tuple of (instrument-level VaR DataFrame, aggregated VaR by group DataFrame, or the
        total as a Series when ``group_data`` is None), as decimal fractions of NAV per period.
    """
    returns = ret.to_returns(prices=prices, freq=freq, is_log_returns=True)
    weights, returns = weights.align(other=returns, join='inner')

    ewm_lambda = 0.94 if vol_span is None else 1.0 - 2.0 / (vol_span + 1.0)
    returns_np = _to_recursion_returns(returns=returns,
                                       span=vol_span,
                                       ewm_lambda=ewm_lambda,
                                       mean_adj_type=mean_adj_type)
    vols = pd.DataFrame(np.sqrt(_compute_ewm_var_np(returns=returns_np, ewm_lambda=ewm_lambda)),
                        index=returns.index, columns=returns.columns)

    instrument_vars = VAR99 * vols.mul(np.abs(weights))

    if group_data is not None:
        ac_vars = dfg.agg_df_by_groups(df=instrument_vars,
                                       group_data=group_data,
                                       group_order=group_order,
                                       agg_func=dfa.df_nansum,
                                       total_column=total_column)
    else:
        ac_vars = instrument_vars.sum(axis=1)

    if time_period is not None:
        instrument_vars = time_period.locate(instrument_vars)
        ac_vars = time_period.locate(ac_vars)
    return instrument_vars, ac_vars

"""
analytics for computing portfolio risk with ewm filter
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
from qis.models.linear.ewm import MeanAdjType, InitType, NanBackfill, compute_rolling_mean_adj, compute_ewm_vol


VAR99 = 2.3263
VAR99_SCALER_BP = VAR99 * 10000


def compute_portfolio_vol(returns: pd.DataFrame,
                          weights: pd.DataFrame,
                          span: Union[int, np.ndarray] = None,
                          ewm_lambda: Union[float, np.ndarray] = 0.94,
                          is_return_vol: bool = True,
                          mean_adj_type: MeanAdjType = MeanAdjType.NONE,
                          init_type: InitType = InitType.ZERO,
                          annualize: bool = False,
                          annualization_factor: float = None,
                          nan_backfill: NanBackfill = NanBackfill.FFILL
                          ) -> pd.Series:
    """
    compute portfolio vol using ewm filter
    """
    # align index and columns
    weights, returns = weights.align(other=returns, join='inner')
    weights = weights.shift(1)

    returns_np = npo.to_finite_np(data=returns, fill_value=0.0)
    weights_np = npo.to_finite_np(data=weights, fill_value=0.0)

    if mean_adj_type != MeanAdjType.NONE:
        returns_np = compute_rolling_mean_adj(data=returns_np,
                                              mean_adj_type=mean_adj_type,
                                              span=span,
                                              ewm_lambda=ewm_lambda,
                                              init_type=init_type,
                                              nan_backfill=nan_backfill)

    portfolio_vol = compute_portfolio_vol_np(returns=returns_np,
                                             weights=weights_np,
                                             span=span,
                                             ewm_lambda=ewm_lambda)

    if annualize:
        if annualization_factor is None:
            if isinstance(weights, pd.DataFrame):
                annualization_factor = da.infer_an_from_data(data=weights)
            else:
                warnings.warn(f"in compute_ewm: annualization_factor for np array, default is 1")
                annualization_factor = 1.0

        portfolio_vol = annualization_factor * portfolio_vol

    if is_return_vol:
        portfolio_vol = np.sqrt(portfolio_vol)

    portfolio_vol = pd.Series(data=portfolio_vol, index=weights.index)

    return portfolio_vol


@njit
def compute_portfolio_vol_np(returns: np.ndarray,
                             weights: np.ndarray,
                             span: Union[int, np.ndarray] = None,
                             ewm_lambda: Union[float, np.ndarray] = 0.94
                             ) -> np.ndarray:

    t = returns.shape[0]  # time dimension
    n = returns.shape[1]  # space dimension

    # important to replace nans for @ operator
    weights = np.where(np.isfinite(weights), weights, 0.0)
    last_covar = np.zeros((n, n))
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


def compute_portfolio_correlated_var_by_ac(prices: pd.DataFrame,
                                           weights: pd.DataFrame,
                                           group_data: Optional[pd.Series] = None,
                                           group_order: List[str] = None,
                                           total_column: Optional[str] = 'Total',
                                           freq: Optional[str] = 'B',
                                           vol_span: int = 31,  # span in number of freq-retunrs
                                           time_period: da.TimePeriod = None,
                                           mean_adj_type: MeanAdjType = MeanAdjType.NONE
                                           ) -> pd.DataFrame:
    """
    portfolio VAR accounting for correlations
    by the default var is max daily loss in bp
    if group_data is provided the var is split into groups
    """
    returns = ret.to_returns(prices=prices, freq=freq, is_log_returns=True)

    weights, returns = weights.align(other=returns, join='inner')
    # weights = weights.shift(1)

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
                                                   annualize=False)
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
                                            vol_span: int = 31,  # span in number of freq-retunrs
                                            time_period: da.TimePeriod = None,
                                            mean_adj_type: MeanAdjType = MeanAdjType.NONE
                                            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    portfolio VAR accounting for correlations
    by the default var is max daily loss in bp
    if group_data is provided the var is split into groups
    """
    returns = ret.to_returns(prices=prices, freq=freq, is_log_returns=True)
    vols = compute_ewm_vol(data=returns,
                           span=vol_span,
                           mean_adj_type=mean_adj_type,
                           annualize=False)

    weights, vols = weights.align(other=vols, join='inner')

    instrument_vars = VAR99 * vols.mul(np.abs(weights))

    if group_data is not None:
        ac_vars = dfg.agg_df_by_groups(df=instrument_vars,
                                       group_data=group_data,
                                       group_order=group_order,
                                       agg_func=dfa.nansum,
                                       total_column=total_column)
    else:
        ac_vars = instrument_vars.sum(1)

    if time_period is not None:
        instrument_vars = time_period.locate(instrument_vars)
        ac_vars = time_period.locate(ac_vars)
    return instrument_vars, ac_vars
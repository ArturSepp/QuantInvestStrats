"""
implementation of multi factor ewm model
"""
# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional
from enum import Enum

# qis
import qis.utils.df_ops as dfo
import qis.plots.time_series as pts
import qis.models.linear.ewm as ewm
from qis.models.linear.ewm import MeanAdjType, InitType
from qis import TimePeriod


class LinearModel:
    """
    core class to store and process of data for different linear signals of shape:
    y =  loadings^T @ x
    """
    def __init__(self,
                 x: pd.DataFrame,  # t, x_n factors
                 y: pd.DataFrame,  # t, y_m factors
                 loadings: Dict[str, pd.DataFrame],
                 ):

        str_factors = list(loadings.keys())
        assert str_factors == x.columns.to_list()

        self.x = x
        self.y = y
        self.loadings = loadings

    def print(self):
        print(f"x:\n{self.x}")
        print(f"y:\n{self.y}")
        for factor, loading in self.loadings.items():
            print(f"{factor}:\n{loading}")

    def compute_agg_factor_exposures(self,
                                     asset_exposures: pd.DataFrame
                                     ) -> pd.DataFrame:
        factor_exposures = {}
        for factor, loading in self.loadings.items():
            factor_exposures[factor] = loading.multiply(asset_exposures).sum(axis=1)
        factor_exposures = pd.DataFrame.from_dict(factor_exposures)
        return factor_exposures

    def plot_factor_loadings(self,
                             factor: str,
                             var_format: str = '{:,.2f}',
                             time_period: TimePeriod = None,
                             ax: plt.Subplot = None,
                             **kwargs):
        df = self.loadings[factor]
        if time_period is not None:
            df = time_period.locate(df)
        pts.plot_time_series(df=df,
                             var_format=var_format,
                             ax=ax, **kwargs)


def estimate_ewm_linear_model(x: pd.DataFrame,
                              y: pd.DataFrame,
                              span: Optional[int] = 31,
                              ewm_lambda: float = 0.94,
                              is_x_correlated: bool = True,
                              mean_adj_type: MeanAdjType = MeanAdjType.NONE,
                              init_type: InitType = InitType.MEAN
                              ) -> LinearModel:

    """
    estimate time series ewm betas
    factor_returns is factors returns data: T*M
    asset_returns is asset returns data: T*N
    return type is {factor_returns.columns: factor of asset_returns T*N}

    is_independent specifies the asset_returns covar structure
    is_x_correlated:
    True = x covariance is orthogonal with ewm_covar being diagonal matrix
    False = x covariance is ewm covariance with ewm_covar being full matrix

    ewm betas = (ewm_covar[x, x]^{-1}) @ (ewm_covar[x, y])
    """
    if span is not None:
        ewm_lambda = 1.0 - 2.0/(span+1.0)
    if mean_adj_type != MeanAdjType.NONE:
        x = ewm.compute_rolling_mean_adj(data=x,
                                         mean_adj_type=mean_adj_type,
                                         ewm_lambda=ewm_lambda,
                                         init_type=init_type)

        y = ewm.compute_rolling_mean_adj(data=y,
                                         mean_adj_type=mean_adj_type,
                                         ewm_lambda=ewm_lambda,
                                         init_type=init_type)

    # compute list of betas using ewm numba recursion for cross product of x y and covariance of x
    # output is tensor of betas per date = [t, factors, assets]
    betas_ts = ewm.compute_ewm_xy_beta_tensor(x=x.to_numpy(),
                                              y=y.to_numpy(),
                                              ewm_lambda=ewm_lambda,
                                              is_x_correlated=is_x_correlated)
    # factor_loadings = {factor_id: pd.DataFrame(factor loadings)}
    loadings = dfo.np_txy_tensor_to_pd_dict(np_tensor_txy=betas_ts,
                                            dateindex=x.index,
                                            factor_names=x.columns.to_list(),
                                            asset_names=y.columns.to_list())

    ewm_lm = LinearModel(x=x, y=y, loadings=loadings)

    return ewm_lm


class UnitTests(Enum):
    MODEL_TEST = 1


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.MODEL_TEST:

        from qis.test_data import load_etf_data
        prices = load_etf_data().dropna()

        returns = np.log(prices.divide(prices.shift(1)))

        # factors
        factors = ['SPY', 'TLT', 'GLD']
        factors = ['SPY']
        factor_returns = returns[factors]

        # assets
        is_check = False
        if is_check:
            asset_returns = returns[factors]
            asset_returns.columns = [f"{x.split('_')[0]}_asset" for x in factors]
        else:
            assets = ['QQQ', 'HYG']
            asset_returns = returns[assets]

        ewm_linear_model = estimate_ewm_linear_model(x=factor_returns,
                                                     y=asset_returns,
                                                     ewm_lambda=0.94,
                                                     is_x_correlated=True)

        ewm_linear_model.print()
        ewm_linear_model.plot_factor_loadings(factor='SPY')

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.MODEL_TEST

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

"""
implementation of multi factor ewm model
"""
# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Literal
from enum import Enum

# qis
import qis as qis
import qis.utils.df_ops as dfo
import qis.perfstats.returns as ret
import qis.plots.time_series as pts
import qis.models.linear.ewm as ewm
from qis.models.linear.ewm import MeanAdjType, InitType
from qis import TimePeriod


@dataclass
class LinearModel:
    """
    core class to store and process of data for different linear signals of shape:
    y =  loadings^T @ x
    """
    x: pd.DataFrame  # t, x_n factors
    y: pd.DataFrame  # t, y_m factors
    loadings: Dict[str, pd.DataFrame] = None  # estimated factor loadings

    def __post_init__(self):
        if self.loadings is not None:
            str_factors = list(self.loadings.keys())
            assert str_factors == self.x.columns.to_list()

    def print(self):
        print(f"x:\n{self.x}")
        print(f"y:\n{self.y}")
        for factor, loading in self.loadings.items():
            print(f"{factor}:\n{loading}")

    def get_factor_loadings(self, factor: str) -> pd.DataFrame:
        return self.loadings[factor]

    def compute_agg_factor_exposures(self,
                                     exposures: pd.DataFrame
                                     ) -> pd.DataFrame:
        factor_exposures = {}
        for factor, loading in self.loadings.items():
            factor_exposures[factor] = loading.multiply(exposures).sum(axis=1)
        factor_exposures = pd.DataFrame.from_dict(factor_exposures)
        return factor_exposures

    def get_asset_factor_betas(self,
                               time_period: TimePeriod = None,
                               asset: str = None
                               ) -> pd.DataFrame:
        """
        return df of asset exposures to factors
        """
        if asset is None:
            asset = self.y.columns[0]
        exps = {}
        for factor, factor_exp in self.loadings.items():
            exps[factor] = factor_exp[asset]
        exps = pd.DataFrame.from_dict(exps)
        if time_period is not None:
            exps = time_period.locate(exps)
        return exps

    def get_asset_factor_attribution(self, asset: str = None, add_total: bool = True) -> pd.DataFrame:
        """
        return df of asset exposures to factors
        """
        factor_betas = self.get_asset_factor_betas(asset=asset)
        exposures = self.x
        attribution = exposures.multiply(factor_betas.shift(1))
        if add_total:
            total = attribution.sum(1).rename('Total')
            attribution = pd.concat([total, attribution], axis=1)
        return attribution

    def get_factor_alpha(self, lag: Literal[0, 1] = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        factor_alpha = y - sum(factor_beta_{t-shift}*x)
        lag = 1 for step-1 forecast
        lag = 0 for insample forecast
        """
        explained_returns = pd.DataFrame(0.0, index=self.y.index, columns=self.y.columns)
        for factor in self.x.columns:
            factor_betas = self.loadings[factor]
            explained_return = (factor_betas.shift(lag)).multiply(self.x[factor].to_numpy(), axis=0)
            explained_returns = explained_returns.add(explained_return)
        factor_alpha = self.y.subtract(explained_returns)
        return factor_alpha, explained_returns

    def get_model_ewm_r2(self, span: int = 52, lag: Literal[0, 1] = 0) -> pd.DataFrame:
        """
        ss_res = ewm (y - sum(factor_beta_{t-1}*x)) ^ 2
        """
        residuals, explained_returns = self.get_factor_alpha(lag=lag)
        ewm_residuals_2 = ewm.compute_ewm(data=np.square(residuals), span=span)
        y_demean = self.y  #- ewm.compute_ewm(data=self.y, span=span)
        ewm_variance = ewm.compute_ewm(data=np.square(y_demean), span=span)
        r_2 = 1.0 - ewm_residuals_2.divide(ewm_variance)
        return r_2

    def get_model_residuals_corrs(self, span: int = 52) -> Tuple[pd.DataFrame, pd.Series]:
        residuals, explained_returns = self.get_factor_alpha(lag=0)
        corr = ewm.compute_ewm_covar(residuals.to_numpy(), span=span, is_corr=True)
        avg_corr = pd.Series(0.5*np.nanmean(corr - np.eye(corr.shape[0]), axis=1), index=self.y.columns)
        corr_pd = pd.DataFrame(corr, index=self.y.columns, columns=self.y.columns)
        return corr_pd, avg_corr

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


class EwmLinearModel(LinearModel):
    """
    implementation for ewma model
    """
    def fit(self,
            span: Optional[int] = 31,
            ewm_lambda: float = 0.94,
            is_x_correlated: bool = True,
            mean_adj_type: MeanAdjType = MeanAdjType.NONE,
            init_type: InitType = InitType.MEAN
            ) -> None:
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
        x = self.x
        y = self.y
        if span is not None:
            ewm_lambda = 1.0 - 2.0 / (span + 1.0)
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
        self.x = x
        self.y = y
        self.loadings = loadings


def compute_portfolio_benchmark_betas(instrument_prices: pd.DataFrame,
                                      exposures: pd.DataFrame,
                                      benchmark_prices: pd.DataFrame,
                                      time_period: TimePeriod = None,
                                      freq_beta: str = None,
                                      factor_beta_span: int = 63,  # quarter
                                      mean_adj_type: MeanAdjType = MeanAdjType.EWMA
                                      ) -> pd.DataFrame:
    """
    compute benchmark betas of instruments
    portfolio_beta_i = sum(instrument_beta_i*exposure)
    """
    benchmark_prices = benchmark_prices.reindex(index=instrument_prices.index, method='ffill')
    ewm_linear_model = EwmLinearModel(x=ret.to_returns(prices=benchmark_prices, freq=freq_beta, is_log_returns=True),
                                      y=ret.to_returns(prices=instrument_prices, freq=freq_beta, is_log_returns=True))
    ewm_linear_model.fit(span=factor_beta_span, is_x_correlated=True, mean_adj_type=mean_adj_type)
    exposures = exposures.reindex(index=instrument_prices.index, method='ffill')
    benchmark_betas = ewm_linear_model.compute_agg_factor_exposures(exposures=exposures)
    benchmark_betas = benchmark_betas.replace({0.0: np.nan}).ffill()  # fillholidays
    if time_period is not None:
        benchmark_betas = time_period.locate(benchmark_betas)
    return benchmark_betas


def compute_portfolio_benchmark_beta_alpha_attribution(instrument_prices: pd.DataFrame,
                                                       exposures: pd.DataFrame,
                                                       benchmark_prices: pd.DataFrame,
                                                       portfolio_nav: pd.Series,
                                                       time_period: TimePeriod = None,
                                                       freq_beta: str = None,
                                                       factor_beta_span: int = 63,  # quarter
                                                       residual_name: str = 'Alpha'
                                                       ) -> pd.DataFrame:
    """
    attribution:=alpha_{t} = portfolio_return_{t} - benchmark_return_{t}*beta_{t-1}
    using compounded returns
    portfolio_nav is the gross/net portfolio nav
    """
    portfolio_benchmark_betas = compute_portfolio_benchmark_betas(instrument_prices=instrument_prices,
                                                                  exposures=exposures,
                                                                  benchmark_prices=benchmark_prices,
                                                                  time_period=None,
                                                                  freq_beta=freq_beta,
                                                                  factor_beta_span=factor_beta_span)
    joint_attrib = compute_benchmarks_beta_attribution(portfolio_nav=portfolio_nav,
                                                       benchmark_prices=benchmark_prices,
                                                       portfolio_benchmark_betas=portfolio_benchmark_betas,
                                                       residual_name=residual_name,
                                                       time_period=time_period)
    return joint_attrib


def compute_benchmarks_beta_attribution(portfolio_nav: pd.Series,
                                        benchmark_prices: pd.DataFrame,
                                        portfolio_benchmark_betas: pd.DataFrame,
                                        residual_name: str = 'Alpha',
                                        time_period: TimePeriod = None
                                        ) -> pd.DataFrame:
    benchmark_prices = benchmark_prices.reindex(index=portfolio_benchmark_betas.index, method='ffill')
    portfolio_nav = portfolio_nav.reindex(index=portfolio_benchmark_betas.index, method='ffill')
    x = ret.to_returns(prices=benchmark_prices, freq=None)
    x_attribution = (portfolio_benchmark_betas.shift(1)).multiply(x)
    total_attrib = x_attribution.sum(axis=1)
    total = portfolio_nav.pct_change()
    residual = np.subtract(total, total_attrib)
    joint_attrib = pd.concat([x_attribution, residual.rename(residual_name)], axis=1)
    if time_period is not None:
        joint_attrib = time_period.locate(joint_attrib)
    return joint_attrib


def estimate_linear_model(price: pd.Series, hedges: pd.DataFrame,
                          freq: str = 'W-WED',
                          span: int = 26,
                          mean_adj_type: MeanAdjType = MeanAdjType.NONE
                          ) -> EwmLinearModel:
    y = qis.to_returns(price.to_frame(), freq=freq, is_log_returns=True, drop_first=True)
    x = qis.to_returns(hedges, freq=freq, is_log_returns=True, drop_first=True)
    ewm_linear_model = EwmLinearModel(x=x.reindex(index=y.index), y=y)
    ewm_linear_model.fit(span=span, is_x_correlated=True, mean_adj_type=mean_adj_type)
    return ewm_linear_model


class UnitTests(Enum):
    MODEL = 1
    ATTRIBUTION = 2


def run_unit_test(unit_test: UnitTests):

    from qis.test_data import load_etf_data
    prices = load_etf_data().dropna()

    if unit_test == UnitTests.MODEL:
        returns = np.log(prices.divide(prices.shift(1)))

        # factors
        factors = ['SPY', 'TLT', 'GLD']
        # factors = ['SPY']
        factor_returns = returns[factors]

        # assets
        is_check = False
        if is_check:
            asset_returns = returns[factors]
            asset_returns.columns = [f"{x.split('_')[0]}_asset" for x in factors]
        else:
            assets = ['QQQ', 'HYG']
            asset_returns = returns[assets]
        ewm_linear_model = EwmLinearModel(x=factor_returns, y=asset_returns)
        ewm_linear_model.fit(ewm_lambda=0.94, is_x_correlated=True)

        ewm_linear_model.print()
        ewm_linear_model.plot_factor_loadings(factor='SPY')

        factor_alpha, explained_returns = ewm_linear_model.get_factor_alpha()
        pts.plot_time_series(df=factor_alpha.cumsum(0), title='Cumulative alpha')
        pts.plot_time_series(df=explained_returns.cumsum(0), title='Cumulative explained return')

    elif unit_test == UnitTests.ATTRIBUTION:
        benchmark_prices = prices[['SPY', 'TLT']]
        instrument_prices = prices[['QQQ', 'HYG', 'GLD']]
        exposures = pd.DataFrame(1.0/3.0, index=instrument_prices.index, columns=instrument_prices.columns)
        portfolio_nav = ret.returns_to_nav(returns=(exposures.shift(1)).multiply(instrument_prices.pct_change()).sum(axis=1))
        print(portfolio_nav)

        attribution = compute_portfolio_benchmark_beta_alpha_attribution(instrument_prices=instrument_prices,
                                                                         exposures=exposures,
                                                                         benchmark_prices=benchmark_prices,
                                                                         portfolio_nav=portfolio_nav,
                                                                         time_period=None,
                                                                         freq_beta='W-WED',
                                                                         factor_beta_span=52,  # quarter
                                                                         residual_name='Alpha')
        pts.plot_time_series(df=attribution.cumsum(0))

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.MODEL

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)



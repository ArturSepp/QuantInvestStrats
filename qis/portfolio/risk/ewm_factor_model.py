"""
implementation of multi factor ewm model
"""
# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Union
from enum import Enum

# qis
import qis as qis
from qis import TimePeriod
import qis.utils.df_ops as dfo
import qis.perfstats.returns as ret
import qis.plots.time_series as pts
import qis.models.linear.ewm as ewm
from qis.models.linear.ewm import MeanAdjType, InitType
from qis.portfolio.risk.factor_model import LinearModel, compute_benchmarks_beta_attribution_from_prices


class EwmLinearModel(LinearModel):
    """Exponentially weighted moving average implementation of LinearModel."""

    def fit(self,
            span: Optional[int] = 31,
            ewm_lambda: float = 0.94,
            is_x_correlated: bool = True,
            mean_adj_type: MeanAdjType = MeanAdjType.NONE,
            init_type: InitType = InitType.MEAN,
            warmup_period: int = 20  # to avoid excessive betas at start,
            ) -> None:
        """Estimate time series EWM betas using exponential weighting.

        Args:
            span: Span for EWM calculation.
            ewm_lambda: Decay parameter for EWM.
            is_x_correlated: Whether to use diagonal (True) or full covariance matrix.
            mean_adj_type: Type of mean adjustment to apply.
            init_type: Initialization method for EWM.
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
                                                  is_x_correlated=is_x_correlated,
                                                  warmup_period=warmup_period)
        # factor_loadings = {factor_id: pd.DataFrame(factor loadings)}
        loadings = dfo.np_txy_tensor_to_pd_dict(np_tensor_txy=betas_ts,
                                                dateindex=x.index,
                                                factor_names=x.columns.to_list(),
                                                asset_names=y.columns.to_list())
        self.x = x
        self.y = y
        self.loadings = loadings


def compute_portfolio_ewm_benchmark_betas(instrument_prices: pd.DataFrame,
                                          weights: pd.DataFrame,
                                          benchmark_prices: pd.DataFrame,
                                          time_period: TimePeriod = None,
                                          freq_beta: str = None,
                                          factor_beta_span: int = 63,  # quarter
                                          mean_adj_type: MeanAdjType = MeanAdjType.EWMA
                                          ) -> pd.DataFrame:
    """Compute portfolio benchmark betas using instrument exposures.

    Args:
        instrument_prices: Individual instrument price data.
        weights: Portfolio exposures to instruments.
        benchmark_prices: Benchmark price data.
        time_period: Optional time period filter.
        freq_beta: Frequency for return calculation.
        factor_beta_span: Span for EWM beta estimation.
        mean_adj_type: Mean adjustment method.

    Returns:
        Portfolio benchmark betas over time.
    """
    benchmark_prices = benchmark_prices.reindex(index=instrument_prices.index, method='ffill')
    ewm_linear_model = EwmLinearModel(x=ret.to_returns(prices=benchmark_prices, freq=freq_beta, is_log_returns=True),
                                      y=ret.to_returns(prices=instrument_prices, freq=freq_beta, is_log_returns=True))
    ewm_linear_model.fit(span=factor_beta_span, is_x_correlated=True, mean_adj_type=mean_adj_type)
    weights = weights.reindex(index=instrument_prices.index, method='ffill')
    benchmark_betas = ewm_linear_model.compute_agg_factor_exposures(weights=weights)
    benchmark_betas = benchmark_betas.replace({0.0: np.nan}).ffill()  # fillholidays
    if time_period is not None:
        benchmark_betas = time_period.locate(benchmark_betas)
    return benchmark_betas


def compute_portfolio_benchmark_ewm_beta_alpha_attribution(instrument_prices: pd.DataFrame,
                                                           weights: pd.DataFrame,
                                                           benchmark_prices: pd.DataFrame,
                                                           portfolio_nav: pd.Series,
                                                           time_period: TimePeriod = None,
                                                           freq_beta: str = None,
                                                           factor_beta_span: int = 63,  # quarter
                                                           residual_name: str = 'Alpha'
                                                           ) -> pd.DataFrame:
    """Compute portfolio beta-alpha attribution using benchmark decomposition.

    Args:
        instrument_prices: Individual instrument prices.
        weights: Portfolio weights to instruments.
        benchmark_prices: Benchmark prices for attribution.
        portfolio_nav: Portfolio NAV time series.
        time_period: Optional time period filter.
        freq_beta: Frequency for beta estimation.
        factor_beta_span: EWM span for beta calculation.
        residual_name: Name for alpha/residual component.

    Returns:
        Attribution breakdown including alpha component.
    """
    portfolio_benchmark_betas = compute_portfolio_ewm_benchmark_betas(instrument_prices=instrument_prices,
                                                                      weights=weights,
                                                                      benchmark_prices=benchmark_prices,
                                                                      time_period=None,
                                                                      freq_beta=freq_beta,
                                                                      factor_beta_span=factor_beta_span)
    joint_attrib = compute_benchmarks_beta_attribution_from_prices(portfolio_nav=portfolio_nav,
                                                                   benchmark_prices=benchmark_prices,
                                                                   portfolio_benchmark_betas=portfolio_benchmark_betas,
                                                                   residual_name=residual_name,
                                                                   time_period=time_period)
    return joint_attrib


def estimate_ewm_factor_model(asset_prices: Union[pd.Series, pd.DataFrame],
                              factor_prices: Union[pd.Series, pd.DataFrame],
                              freq: str = 'W-WED',
                              span: int = 26,
                              mean_adj_type: MeanAdjType = MeanAdjType.NONE
                              ) -> EwmLinearModel:
    """Estimate linear factor model from price data.

    Args:
        asset_prices: Asset price time series.
        factor_prices: Factor price time series.
        freq: Frequency for return calculation.
        span: EWM span for model estimation.
        mean_adj_type: Mean adjustment method.

    Returns:
        Fitted EWM linear model.
    """
    y = qis.to_returns(asset_prices, freq=freq, is_log_returns=True, drop_first=True)
    x = qis.to_returns(factor_prices, freq=freq, is_log_returns=True, drop_first=True)
    ewm_linear_model = EwmLinearModel(x=x.reindex(index=y.index), y=y)
    ewm_linear_model.fit(span=span, is_x_correlated=True, mean_adj_type=mean_adj_type)
    return ewm_linear_model


class LocalTests(Enum):
    MODEL = 1
    ATTRIBUTION = 2


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    from qis.test_data import load_etf_data
    prices = load_etf_data().dropna()

    if local_test == LocalTests.MODEL:
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

    elif local_test == LocalTests.ATTRIBUTION:
        benchmark_prices = prices[['SPY', 'TLT']]
        instrument_prices = prices[['QQQ', 'HYG', 'GLD']]
        exposures = pd.DataFrame(1.0/3.0, index=instrument_prices.index, columns=instrument_prices.columns)
        portfolio_nav = ret.returns_to_nav(returns=(exposures.shift(1)).multiply(instrument_prices.pct_change()).sum(axis=1))
        print(portfolio_nav)

        attribution = compute_portfolio_benchmark_ewm_beta_alpha_attribution(instrument_prices=instrument_prices,
                                                                             weights=exposures,
                                                                             benchmark_prices=benchmark_prices,
                                                                             portfolio_nav=portfolio_nav,
                                                                             time_period=None,
                                                                             freq_beta='W-WED',
                                                                             factor_beta_span=52,  # quarter
                                                                             residual_name='Alpha')
        pts.plot_time_series(df=attribution.cumsum(0))

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.MODEL)

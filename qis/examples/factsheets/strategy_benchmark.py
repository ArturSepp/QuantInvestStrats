"""
example of using strategy construction and reporting vs benchmark
for implementation see portfolio.report.strategy_factsheet
"""
# packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from enum import Enum
import yfinance as yf
import qis as qis
from qis import TimePeriod, MultiPortfolioData

# reporting
from qis.portfolio.reports.config import fetch_default_report_kwargs
from qis.portfolio.reports.strategy_benchmark_factsheet import (generate_strategy_benchmark_factsheet_plt,
                                                                generate_strategy_benchmark_active_perf_plt,
                                                                generate_performance_attribution_report,
                                                                weights_tracking_error_report_by_ac_subac)
from qis.test_data import load_etf_data


def fetch_universe_data(live_prices: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    define custom universe with asset class grouping
    """
    universe_data = dict(SPY='Equities',
                         QQQ='Equities',
                         EEM='Equities',
                         TLT='Bonds',
                         IEF='Bonds',
                         LQD='Credit',
                         HYG='HighYield',
                         GLD='Gold')
    tickers = list(universe_data.keys())
    group_data = pd.Series(universe_data)  # for portfolio reporting
    if live_prices:
        prices = yf.download(tickers=tickers, start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True)['Close'][tickers]
    else:
        prices = load_etf_data()[tickers]
        print(prices)

    prices = prices.asfreq('B', method='ffill')
    benchmark_prices = prices[['SPY', 'TLT']]
    return prices, benchmark_prices, group_data


def generate_volparity_multiportfolio(prices: pd.DataFrame,
                                      benchmark_prices: pd.DataFrame,
                                      group_data: pd.Series,
                                      time_period: TimePeriod = None,
                                      span: int = 60,
                                      vol_target: float = 0.15,
                                      rebalancing_costs: float = 0.0010
                                      ) -> MultiPortfolioData:

    ra_returns, weights, ewm_vol = qis.compute_ra_returns(returns=qis.to_returns(prices=prices, is_log_returns=True),
                                                          span=span,
                                                          vol_target=vol_target)
    weights = weights.divide(weights.sum(1), axis=0)

    if time_period is not None:
        weights = time_period.locate(weights)

    volparity_portfolio = qis.backtest_model_portfolio(prices=prices,
                                                       weights=weights,
                                                       rebalancing_costs=rebalancing_costs,
                                                       ticker='VolParity')
    volparity_portfolio.set_group_data(group_data=group_data, group_order=list(group_data.unique()))

    ew_portfolio = qis.backtest_model_portfolio(prices=prices,
                                                weights=np.ones(len(prices.columns)) / len(prices.columns),
                                                rebalancing_costs=rebalancing_costs,
                                                ticker='EqualWeight')
    ew_portfolio.set_group_data(group_data=group_data, group_order=list(group_data.unique()))

    multi_portfolio_data = MultiPortfolioData(portfolio_datas=[volparity_portfolio, ew_portfolio],
                                              benchmark_prices=benchmark_prices)
    return multi_portfolio_data


class UnitTests(Enum):
    STRATEGY_BENCHMARK_PLT = 1
    PERFORMANCE_ATTRIBUTION = 2
    ACTIVE_PERFORMANCE = 3
    TRACKING_ERROR = 4


@qis.timer
def run_unit_test(unit_test: UnitTests):

    # time period for portfolio reporting
    time_period = qis.TimePeriod('31Dec2006', '21Apr2025')
    prices, benchmark_prices, group_data = fetch_universe_data()

    multi_portfolio_data = generate_volparity_multiportfolio(prices=prices,
                                                             benchmark_prices=benchmark_prices,
                                                             group_data=group_data,
                                                             time_period=time_period,
                                                             span=30,
                                                             vol_target=0.15,
                                                             rebalancing_costs=0.0010  # per traded volume
                                                             )

    if unit_test == UnitTests.STRATEGY_BENCHMARK_PLT:
        pnl_attribution = True
        figs = generate_strategy_benchmark_factsheet_plt(multi_portfolio_data=multi_portfolio_data,
                                                         backtest_name='Vol Parity Portfolio vs Equal Weight',
                                                         time_period=time_period,
                                                         add_brinson_attribution=pnl_attribution,
                                                         add_exposures_pnl_attribution=pnl_attribution,
                                                         add_exposures_comp=pnl_attribution,
                                                         add_strategy_factsheet=True,  # for strategy factsheet
                                                         add_grouped_exposures=False,  # for strategy factsheet
                                                         add_grouped_cum_pnl=False,  # for strategy factsheet
                                                         is_grouped=True,
                                                         **fetch_default_report_kwargs(time_period=time_period,
                                                                                       add_rates_data=True))
        qis.save_figs_to_pdf(figs=figs,
                             file_name=f"strategy_benchmark_factsheet", orientation='landscape',
                             local_path=qis.local_path.get_output_path())
        qis.save_fig(fig=figs[0], file_name=f"strategy_benchmark", local_path=qis.local_path.get_output_path())
        if pnl_attribution:
            qis.save_fig(fig=figs[1], file_name=f"brinson_attribution", local_path=qis.local_path.get_output_path())
            qis.save_fig(fig=figs[2], file_name=f"pnl_attribution", local_path=qis.local_path.get_output_path())

    elif unit_test == UnitTests.PERFORMANCE_ATTRIBUTION:
        figs = generate_performance_attribution_report(multi_portfolio_data=multi_portfolio_data,
                                                       time_period=TimePeriod('31Dec2021', None),
                                                       **fetch_default_report_kwargs(time_period=time_period))

    elif unit_test == UnitTests.ACTIVE_PERFORMANCE:
        figs = generate_strategy_benchmark_active_perf_plt(multi_portfolio_data=multi_portfolio_data,
                                                           time_period=time_period,
                                                           figsize=(11, 6),
                                                           is_long_only=True,
                                                           **fetch_default_report_kwargs(time_period=time_period))

    elif unit_test == UnitTests.TRACKING_ERROR:
        # compute pd_covras
        covar_dict = qis.estimate_rolling_ewma_covar(prices=prices,
                                                     time_period=time_period,
                                                     returns_freq='W-WED',
                                                     rebalancing_freq='ME',
                                                     span=52)
        multi_portfolio_data.covar_dict = covar_dict
        ac_group_data = multi_portfolio_data.portfolio_datas[0].group_data
        asset_tickers = multi_portfolio_data.portfolio_datas[0].weights.columns
        sub_ac_group_data = pd.Series(asset_tickers, index=asset_tickers)
        turnover_groups = multi_portfolio_data.portfolio_datas[0].group_data
        multi_portfolio_data.portfolio_datas[0].benchmark_prices = multi_portfolio_data.benchmark_prices
        figs, dfs = weights_tracking_error_report_by_ac_subac(multi_portfolio_data=multi_portfolio_data,
                                                              ac_group_data=ac_group_data,
                                                              sub_ac_group_data=sub_ac_group_data,
                                                              turnover_groups=turnover_groups,
                                                              time_period=time_period)
        qis.save_figs_to_pdf(figs=figs,
                             file_name=f"strategy_benchmark_tracking_error", orientation='landscape',
                             local_path=qis.local_path.get_output_path())
    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.STRATEGY_BENCHMARK_PLT

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

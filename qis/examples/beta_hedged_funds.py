"""
performance of beta hedged funds
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import qis as qis
from qis import EwmLinearModel, MultiPortfolioData
from typing import Tuple
from enum import Enum

from bbg_fetch import fetch_field_timeseries_per_tickers


def get_price_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    tickers = {'HYG US Equity': 'HYG', 'FALN US Equity': 'HY Fallen Angels', 'PIMCMEI ID Equity': 'FI Alpha Fund'}
    prices = fetch_field_timeseries_per_tickers(tickers=tickers, freq='B')
    hedge_tickers = {'ES1 Index': 'SPY', 'US1 Comdty': 'UST'}
    hedges = fetch_field_timeseries_per_tickers(tickers=hedge_tickers, freq='B')
    return prices, hedges


def estimate_linear_model(price: pd.Series, hedges: pd.DataFrame,
                          freq: str = 'W-WED',
                          span: int = 26,
                          mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.NONE
                          ) -> EwmLinearModel:
    y = qis.to_returns(price.to_frame(), freq=freq, is_log_returns=True, drop_first=True)
    x = qis.to_returns(hedges, freq=freq, is_log_returns=True, drop_first=True)
    ewm_linear_model = qis.EwmLinearModel(x=x.reindex(index=y.index), y=y)
    ewm_linear_model.fit(span=span, is_x_correlated=True, mean_adj_type=mean_adj_type)
    return ewm_linear_model


class BetaHedgeType(Enum):
    STATIC = 1
    EWM_MOMENTUM = 2


def compute_beta_to_hedges(price: pd.Series, hedges: pd.DataFrame,
                           freq: str = 'W-WED',
                           beta_span: int = 52,
                           momentum_span: int = 13,
                           short_span: int = 4,
                           mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.NONE
                           ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    linear_model = estimate_linear_model(price=price, hedges=hedges, freq=freq, span=beta_span, mean_adj_type=mean_adj_type)
    betas = linear_model.get_asset_factor_betas()
    hedge_weights = -1.0*qis.compute_ewm(betas, span=beta_span)
    beta_hedge_weights = hedge_weights
    beta_hedge_weights1 = hedge_weights.where(hedge_weights < 0.0, other=0.0)

    returns = qis.to_returns(hedges, is_log_returns=True, freq=freq)
    momentum = qis.compute_ewm_long_short_filtered_ra_returns(returns=returns,
                                                              vol_span=momentum_span,
                                                              long_span=momentum_span,
                                                              short_span=short_span,
                                                              warmup_period=2,
                                                              mean_adj_type=mean_adj_type)
    # hedge only when momentum is negative
    momentum = momentum.reindex(index=beta_hedge_weights.index, method='ffill')
    neg_momentum_hedge_weights = beta_hedge_weights1.where(momentum < 0.0, other=0.0)

    return beta_hedge_weights, neg_momentum_hedge_weights, betas


def backtest_beta_hedged_portfolios(price: pd.Series, hedges: pd.DataFrame,
                                    freq: str = 'W-WED',
                                    beta_span: int = 52,
                                    momentum_span: int = 13,
                                    mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.NONE,
                                    rebalancing_costs: float = 0.0005
                                    ) -> Tuple[MultiPortfolioData, pd.DataFrame]:
    """
    run both beta and neg momentum hedges
    """
    prices = pd.concat([price, hedges], axis=1).dropna()
    beta_hedge_weights, neg_momentum_hedge_weights, betas = compute_beta_to_hedges(price=price, hedges=hedges, freq=freq,
                                                                                   beta_span=beta_span,
                                                                                   momentum_span=momentum_span,
                                                                                   mean_adj_type=mean_adj_type)
    hedges_weightss = {'Beta-Hedge': beta_hedge_weights,
                      'Neg-Mom-Beta-Hedge': neg_momentum_hedge_weights}
    portfolio_datas = []
    for key, hedge_weights in hedges_weightss.items():
        hedge_weights = hedge_weights.dropna()
        portfolio_weights = pd.Series(1.0, index=hedge_weights.index, name=price.name)
        weights = pd.concat([portfolio_weights, hedge_weights], axis=1)
        time_period = qis.TimePeriod(weights.index[0], prices.index[-1])
        portfolio_data = qis.backtest_model_portfolio(prices=prices,
                                                      weights=weights,
                                                      rebalancing_costs=rebalancing_costs,
                                                      weight_implementation_lag=1,
                                                      ticker=f"{key}")
        portfolio_datas.append(portfolio_data)
    # benchmark_prices = price.to_frame(f"Benchmark: {price.name}")
    # benchmark_prices = pd.concat([price, hedges], axis=1)
    benchmark_prices = price.to_frame()
    multi_portfolio_data = MultiPortfolioData(portfolio_datas, benchmark_prices=benchmark_prices)
    return multi_portfolio_data, betas


class UnitTests(Enum):
    BACKTEST_BETA_HEDGE_MULTI = 1


def run_unit_test(unit_test: UnitTests):

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    # local_path = "C://Users//uarts//Python//quant_strats//resources//"
    local_path = "C://Users//artur//OneDrive//analytics//qdev//resources//"
    import quant_strats.local_path as lp

    if unit_test == UnitTests.BACKTEST_BETA_HEDGE_MULTI:
        prices, hedges = get_price_data()

        report_figs = []
        for asset in prices.columns:
            price = prices[asset].dropna()
            multi_portfolio_data, betas = backtest_beta_hedged_portfolios(price=price,
                                                                          hedges=hedges.reindex(index=price.index, method='ffill'),
                                                                          beta_span=13,
                                                                          momentum_span=13,
                                                                          mean_adj_type=qis.MeanAdjType.EWMA)
            time_period = qis.get_time_period(df=multi_portfolio_data.get_navs())
            kwargs = qis.fetch_default_report_kwargs(time_period=time_period)
            figs = qis.generate_multi_portfolio_factsheet(multi_portfolio_data=multi_portfolio_data,
                                                          time_period=time_period,
                                                          add_benchmarks_to_navs=True,
                                                          backtest_name=f"{asset} beta hedged",
                                                          **kwargs)

            qis.save_figs_to_pdf(figs=figs,
                                 file_name=f"{asset.split(' ')[0]} beta_hedged", orientation='landscape',
                                 local_path=lp.get_output_path())
            # create report figures
            kwargs = qis.update_kwargs(kwargs, dict(perf_stats_labels=[qis.PerfStat.PA_RETURN, qis.PerfStat.VOL,
                                                                       qis.PerfStat.SHARPE_RF0, qis.PerfStat.MAX_DD],
                                                    fontsize=8, framealpha=0.9))
            with sns.axes_style("darkgrid"):
                fig, axs = plt.subplots(3, 2, figsize=(11.7, 8.3), tight_layout=True)
                qis.set_suptitle(fig, title=f"{asset} beta-hedged")
                qis.plot_time_series(df=time_period.locate(betas),
                                     title='(A) Exposure Betas',
                                     var_format='{:,.2f}',
                                     ax=axs[0][0],
                                     **kwargs)
                joint_attrib = qis.compute_benchmarks_beta_attribution(portfolio_nav=price,
                                                                       benchmark_prices=hedges,
                                                                       portfolio_benchmark_betas=betas.dropna(),
                                                                       time_period=time_period)
                qis.plot_time_series(df=joint_attrib.cumsum(0),
                                     title='(B) Cumulative attribution to betas and Alpha',
                                     var_format='{:,.0%}',
                                     legend_stats=qis.LegendStats.LAST,
                                     ax=axs[1][0],
                                     **kwargs)

                multi_portfolio_data.plot_nav(time_period=time_period,
                                              regime_benchmark=asset,
                                              add_benchmarks_to_navs=True,
                                              title='(D) Cumulative Performance',
                                              ax=axs[0][1],
                                              **kwargs)
                multi_portfolio_data.plot_periodic_returns(time_period=time_period,
                                                           title='(E) Annual performance',
                                                           add_benchmarks_to_navs=True,
                                                           fontsize=7,
                                                           ax=axs[1][1])

                freq = 'W-WED'
                benchmark = hedges.columns[0]
                prices1 = pd.concat([hedges[benchmark], multi_portfolio_data.get_navs(add_benchmarks_to_navs=True, time_period=time_period)], axis=1).dropna()
                colors = qis.get_n_colors(n=len(prices1.columns))
                kwargs1 = qis.update_kwargs(kwargs, dict(alpha_an_factor=52, markersize=2, freq=freq, colors=colors))
                qis.plot_returns_scatter(prices=prices1,
                                         benchmark=benchmark,
                                         order=2,
                                         title=f"(C) Scatterplot of {freq}-freq returns vs {benchmark}",
                                         ax=axs[2][0],
                                         **kwargs1)

                benchmark = hedges.columns[1]
                prices1 = pd.concat([hedges[benchmark], multi_portfolio_data.get_navs(add_benchmarks_to_navs=True, time_period=time_period)], axis=1).dropna()
                qis.plot_returns_scatter(prices=prices1,
                                         benchmark=benchmark,
                                         order=2,
                                         title=f"(F) Scatterplot of {freq}-freq returns vs {benchmark}",
                                         ax=axs[2][1],
                                         **kwargs1)

                report_figs.append(fig)

        qis.save_figs_to_pdf(figs=report_figs,
                             file_name=f"report_beta_hedged", orientation='landscape',
                             local_path=lp.get_output_path())
    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.BACKTEST_BETA_HEDGE_MULTI

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

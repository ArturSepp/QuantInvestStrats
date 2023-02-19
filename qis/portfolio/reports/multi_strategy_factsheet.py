"""
factsheet for multi strategy report for cross sectional comparision of strategies
and generating sensetivities to param reports
"""
# packages
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
from enum import Enum

# qis
import qis
import qis.file_utils as fu
import qis.utils.dates as da
import qis.utils.struct_ops as sop
from qis.utils.dates import TimePeriod
from qis.perfstats.config import PerfParams, PerfStat
from qis.perfstats.regime_classifier import BenchmarkReturnsQuantileRegimeSpecs

# portfolio
import qis.portfolio.backtester as bp
from qis.portfolio.multi_portfolio_data import MultiPortfolioData

PERF_PARAMS = PerfParams(freq='W-WED')
REGIME_PARAMS = BenchmarkReturnsQuantileRegimeSpecs(freq='Q')

# use for number years > 5
KWARG_LONG = dict(perf_params=PerfParams(freq='W-WED', freq_reg='Q'),
                  regime_params=BenchmarkReturnsQuantileRegimeSpecs(freq='Q'),
                  x_date_freq='A',
                  date_format='%b-%y')

# use for number years < 3
KWARG_SHORT = dict(perf_params=PerfParams(freq='W-WED', freq_reg='M'),
                   regime_params=BenchmarkReturnsQuantileRegimeSpecs(freq='M'),
                   x_date_freq='Q',
                   date_format='%b-%y')


def generate_multi_portfolio_factsheet(multi_portfolio_data: MultiPortfolioData,
                                       time_period: da.TimePeriod = None,
                                       perf_params: PerfParams = PERF_PARAMS,
                                       regime_params: BenchmarkReturnsQuantileRegimeSpecs = REGIME_PARAMS,
                                       backtest_name: str = None,
                                       figsize: Tuple[float, float] = (8.3, 11.7),  # A4 for portrait
                                       **kwargs
                                       ):
    """
    for portfolio data with structurally different strategies
    """
    regime_benchmark = multi_portfolio_data.benchmark_prices.columns[0]
    benchmark_price = multi_portfolio_data.benchmark_prices[regime_benchmark]

    plot_kwargs = dict(fontsize=5,
                       linewidth=0.5,
                       digits_to_show=1, sharpe_digits=2,
                       weight='normal',
                       markersize=1,
                       framealpha=0.75)
    kwargs = sop.update_kwargs(kwargs, plot_kwargs)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(nrows=7, ncols=4, wspace=0.0, hspace=0.0)

    if backtest_name is not None:
        fig.suptitle(backtest_name, fontweight="bold", fontsize=8, color='blue')

    multi_portfolio_data.plot_nav(ax=fig.add_subplot(gs[0, :2]),
                                  time_period=time_period,
                                  benchmark=regime_benchmark,
                                  perf_params=perf_params,
                                  regime_params=regime_params,
                                  title='Cumulative performance',
                                  **kwargs)

    multi_portfolio_data.plot_drawdowns(ax=fig.add_subplot(gs[1, :2]),
                                        time_period=time_period,
                                        benchmark=regime_benchmark,
                                        regime_params=regime_params,
                                        title='Running Drawdowns',
                                        **kwargs)

    multi_portfolio_data.plot_rolling_time_under_water(ax=fig.add_subplot(gs[2, :2]),
                                                       time_period=time_period,
                                                       benchmark=regime_benchmark,
                                                       regime_params=regime_params,
                                                       title='Rolling time under water',
                                                       **kwargs)

    multi_portfolio_data.plot_exposures(ax=fig.add_subplot(gs[3, :2]),
                                        portfolio_idx=0,
                                        time_period=time_period,
                                        benchmark=regime_benchmark,
                                        regime_params=regime_params,
                                        **kwargs)

    multi_portfolio_data.plot_turnover(ax=fig.add_subplot(gs[4, :2]),
                                       time_period=time_period,
                                       benchmark=regime_benchmark,
                                       regime_params=regime_params,
                                       **kwargs)

    multi_portfolio_data.plot_costs(ax=fig.add_subplot(gs[5, :2]),
                                    time_period=time_period,
                                    benchmark=regime_benchmark,
                                    regime_params=regime_params,
                                    **kwargs)

    multi_portfolio_data.plot_factor_betas(ax=fig.add_subplot(gs[6, :2]),
                                           benchmark_prices=multi_portfolio_data.benchmark_prices,
                                           time_period=time_period,
                                           benchmark=regime_benchmark,
                                           regime_params=regime_params,
                                           **kwargs)

    multi_portfolio_data.plot_performance_bars(ax=fig.add_subplot(gs[0, 2]),
                                               perf_params=perf_params,
                                               perf_column=PerfStat.SHARPE,
                                               time_period=time_period,
                                               **sop.update_kwargs(kwargs, dict(fontsize=5)))

    multi_portfolio_data.plot_performance_bars(ax=fig.add_subplot(gs[0, 3]),
                                               perf_params=perf_params,
                                               perf_column=PerfStat.MAX_DD,
                                               time_period=time_period,
                                               **sop.update_kwargs(kwargs, dict(fontsize=5)))

    multi_portfolio_data.plot_periodic_returns(ax=fig.add_subplot(gs[1, 2:]),
                                               heatmap_freq='A',
                                               time_period=time_period,
                                               **sop.update_kwargs(kwargs, dict(date_format='%Y', fontsize=5)))

    multi_portfolio_data.plot_ra_perf_table(ax=fig.add_subplot(gs[2, 2:]),
                                            perf_params=perf_params,
                                            time_period=time_period,
                                            **sop.update_kwargs(kwargs, dict(fontsize=5)))

    multi_portfolio_data.plot_ra_perf_table(ax=fig.add_subplot(gs[3, 2:]),
                                            perf_params=perf_params,
                                            time_period=da.get_time_period_shifted_by_years(time_period=time_period),
                                            **sop.update_kwargs(kwargs, dict(fontsize=5)))

    multi_portfolio_data.plot_regime_data(ax=fig.add_subplot(gs[4, 2:]),
                                          is_grouped=False,
                                          time_period=time_period,
                                          perf_params=perf_params,
                                          regime_params=regime_params,
                                          benchmark=regime_benchmark,
                                          **kwargs)

    multi_portfolio_data.plot_returns_scatter(ax=fig.add_subplot(gs[5, 2:]),
                                              time_period=time_period,
                                              benchmark=regime_benchmark,
                                              **kwargs)

    multi_portfolio_data.plot_corr_table(ax=fig.add_subplot(gs[6, 2:]),
                                         time_period=time_period,
                                         freq='W-WED',
                                         **sop.update_kwargs(kwargs, dict(fontsize=4)))

    return fig


class UnitTests(Enum):
    PERF_ATTRIBUTION = 1
    FACTSHEET = 2


def run_unit_test(unit_test: UnitTests):

    from qis.test_data import load_etf_data
    import qis.models.linear.ra_returns as rar
    prices = load_etf_data()

    time_period = TimePeriod('31Dec2003', '31Dec2022')
    prices = time_period.locate(prices)
    group_data = pd.Series(
        dict(SPY='Equities', QQQ='Equities', EEM='Equities', TLT='Bonds', IEF='Bonds', SHY='Bonds', LQD='Credit',
             HYG='HighYield', GLD='Gold'))

    benchmark_prices = prices[['SPY']]

    spans = [10, 20, 30, 60, 120, 260]
    portfolio_datas = []
    for span in spans:
        ra_returns, weights, ewm_vol = rar.compute_ra_returns(returns=prices.pct_change(), span=span, vol_target=0.15)
        weights = weights.divide(weights.sum(1), axis=0)
        portfolio_data = bp.backtest_model_portfolio(prices=prices,
                                                     weights=time_period.locate(weights),
                                                     is_output_portfolio_data=True,
                                                     ticker=f"VP span-{span}")
        portfolio_data._set_group_data(group_data=group_data, group_order=list(group_data.unique()))
        portfolio_datas.append(portfolio_data)

    multi_portfolio_data = MultiPortfolioData(portfolio_datas,
                                              benchmark_prices=benchmark_prices)

    if unit_test == UnitTests.FACTSHEET:
        fig = generate_multi_portfolio_factsheet(multi_portfolio_data=multi_portfolio_data,
                                                 backtest_name='Vol Parity Portfolio vs Equal Weight',
                                                 time_period=TimePeriod('31Dec2006', '31Dec2022'),
                                                 **KWARG_LONG)

        fu.save_figs_to_pdf(figs=[fig],
                            file_name=f"multistrategy_factsheet",
                            orientation='landscape',
                            local_path=qis.local_path.get_output_path())

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.FACTSHEET

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

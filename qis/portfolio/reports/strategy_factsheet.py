"""
generate strategy factsheet report using MultiPortfolioData data object
with comparision to 1-2 cash benchmarks
"""
# built in
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum

# qis
import qis.file_utils as fu
import qis.utils.struct_ops as sop
import qis.utils.dates as da
from qis.utils.dates import TimePeriod
from qis.perfstats.config import PerfParams
from qis.perfstats.regime_classifier import BenchmarkReturnsQuantileRegimeSpecs

# portfolio
import qis.portfolio.backtester as bp
from qis.portfolio.portfolio_data import AttributionMetric
from qis.portfolio.multi_portfolio_data import MultiPortfolioData
from qis.portfolio.reports.portfolio_factsheet import generate_portfolio_factsheet

PERF_PARAMS = PerfParams(freq='W-WED')
REGIME_PARAMS = BenchmarkReturnsQuantileRegimeSpecs(freq='Q')

FIG_SIZE = (8.3, 11.7)  # A4 for portrait

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


def generate_strategy_factsheet(multi_portfolio_data: MultiPortfolioData,
                                time_period: da.TimePeriod = None,
                                perf_params: PerfParams = PERF_PARAMS,
                                regime_params: BenchmarkReturnsQuantileRegimeSpecs = REGIME_PARAMS,
                                backtest_name: str = None,
                                file_name_to_save: str = None,
                                add_strategy_factsheets: bool = False,
                                **kwargs
                                ) -> plt.Figure:
    """
    designed for 1 strategy and 1 benchmark report
    multi_portfolio_data = [stragegy portfolio, benchmark strategy portfolio]
    """
    if len(multi_portfolio_data.portfolio_datas) == 1:
        raise ValueError(f"must be at least two strategieg")

    plot_kwargs = dict(fontsize=5,
                       linewidth=0.5,
                       digits_to_show=1, sharpe_digits=2,
                       weight='normal',
                       markersize=1,
                       framealpha=0.75)
    kwargs = sop.update_kwargs(kwargs, plot_kwargs)
    fig = plt.figure(figsize=FIG_SIZE, constrained_layout=True)
    gs = fig.add_gridspec(nrows=14, ncols=4, wspace=0.0, hspace=0.0)

    if backtest_name is not None:
        fig.suptitle(backtest_name, fontweight="bold", fontsize=8, color='blue')

    regime_benchmark = multi_portfolio_data.benchmark_prices.columns[0]
    benchmark_price = multi_portfolio_data.benchmark_prices[regime_benchmark]

    ax = fig.add_subplot(gs[:2, :2])
    multi_portfolio_data.plot_nav(ax=ax,
                                  time_period=time_period,
                                  benchmark=regime_benchmark,
                                  perf_params=perf_params,
                                  regime_params=regime_params,
                                  title='Cumulative performance',
                                  **kwargs)
    ax = fig.add_subplot(gs[2:4, :2])
    multi_portfolio_data.plot_drawdowns(ax=ax,
                                        time_period=time_period,
                                        benchmark=regime_benchmark,
                                        regime_params=regime_params,
                                        title='Running Drawdowns',
                                        **kwargs)

    ax = fig.add_subplot(gs[4:6, :2])
    multi_portfolio_data.plot_rolling_time_under_water(ax=ax,
                                                       time_period=time_period,
                                                       benchmark=regime_benchmark,
                                                       regime_params=regime_params,
                                                       title='Rolling time under water',
                                                       **kwargs)

    ax = fig.add_subplot(gs[6:8, :2])
    multi_portfolio_data.plot_exposures(ax=ax,
                                        time_period=time_period,
                                        benchmark=regime_benchmark,
                                        regime_params=regime_params,
                                        **kwargs)
    ax = fig.add_subplot(gs[8:10, :2])
    multi_portfolio_data.plot_exposures_diff(ax=ax,
                                             time_period=time_period,
                                             benchmark=regime_benchmark,
                                             regime_params=regime_params,
                                             **kwargs)

    ax = fig.add_subplot(gs[10:12, :2])
    multi_portfolio_data.plot_turnover(ax=ax,
                                       time_period=time_period,
                                       benchmark=regime_benchmark,
                                       regime_params=regime_params,
                                       **kwargs)

    ax = fig.add_subplot(gs[12:14, :2])
    multi_portfolio_data.plot_costs(ax=ax,
                                    time_period=time_period,
                                    benchmark=regime_benchmark,
                                    regime_params=regime_params,
                                    **kwargs)

    ax = fig.add_subplot(gs[0:2, 2:])
    multi_portfolio_data.plot_ac_ra_perf_table(ax=ax,
                                               benchmark_price=benchmark_price,
                                               time_period=time_period,
                                               perf_params=perf_params,
                                               **sop.update_kwargs(kwargs, dict(fontsize=4)))
    ax = fig.add_subplot(gs[2:4, 2:])
    multi_portfolio_data.plot_ac_ra_perf_table(ax=ax,
                                               benchmark_price=benchmark_price,
                                               time_period=da.get_time_period_shifted_by_years(time_period=time_period),
                                               perf_params=perf_params,
                                               **sop.update_kwargs(kwargs, dict(fontsize=4)))

    # periodic returns
    ax = fig.add_subplot(gs[4:6, 2])
    local_kwargs = sop.update_kwargs(kwargs=kwargs,
                                     new_kwargs=dict(fontsize=4, square=False, x_rotation=90, is_transpose=False))
    multi_portfolio_data.portfolio_datas[0].plot_periodic_returns(ax=ax,
                                                                  heatmap_freq='A',
                                                                  time_period=time_period,
                                                                  **sop.update_kwargs(local_kwargs, dict(date_format='%Y')))
    ax = fig.add_subplot(gs[4:6, 3])
    multi_portfolio_data.portfolio_datas[1].plot_periodic_returns(ax=ax,
                                                                  heatmap_freq='A',
                                                                  time_period=time_period,
                                                                  **sop.update_kwargs(local_kwargs, dict(date_format='%Y')))

    ax = fig.add_subplot(gs[6:8, 2])
    multi_portfolio_data.portfolio_datas[0].plot_regime_data(ax=ax,
                                                             benchmark_price=benchmark_price,
                                                             time_period=time_period,
                                                             title=f"{multi_portfolio_data.portfolio_datas[0].nav.name}",
                                                             perf_params=perf_params,
                                                             regime_params=regime_params,
                                                             **sop.update_kwargs(kwargs, dict(fontsize=4, x_rotation=90)))
    ax = fig.add_subplot(gs[6:8, 3])
    multi_portfolio_data.portfolio_datas[1].plot_regime_data(ax=ax,
                                                             benchmark_price=benchmark_price,
                                                             time_period=time_period,
                                                             title=f"{multi_portfolio_data.portfolio_datas[1].nav.name}",
                                                             perf_params=perf_params,
                                                             regime_params=regime_params,
                                                             **sop.update_kwargs(kwargs, dict(fontsize=4, x_rotation=90)))

    # vol regimes
    """
    ax = fig.add_subplot(gs[8:10, 2])
    multi_portfolio_data.portfolio_datas[0].plot_vol_regimes(ax=ax,
                                                             benchmark_price=benchmark_price,
                                                             time_period=time_period,
                                                             perf_params=perf_params,
                                                             regime_params=regime_params,
                                                             **sop.update_kwargs(kwargs, dict(fontsize=4, x_rotation=90)))
    ax = fig.add_subplot(gs[8:10, 3])
    multi_portfolio_data.portfolio_datas[1].plot_vol_regimes(ax=ax,
                                                             benchmark_price=benchmark_price,
                                                             time_period=time_period,
                                                             perf_params=perf_params,
                                                             regime_params=regime_params,
                                                             **sop.update_kwargs(kwargs, dict(fontsize=4, x_rotation=90)))
    """
    ax = fig.add_subplot(gs[8:10, 2:])
    multi_portfolio_data.plot_instrument_pnl_diff(ax=ax,
                                                  time_period=time_period,
                                                  benchmark=regime_benchmark,
                                                  regime_params=regime_params,
                                                  **kwargs)

    ax = fig.add_subplot(gs[10:12, 2:])
    multi_portfolio_data.plot_factor_betas(ax=ax,
                                           benchmark_prices=multi_portfolio_data.benchmark_prices,
                                           time_period=time_period,
                                           benchmark=regime_benchmark,
                                           regime_params=regime_params,
                                           **kwargs)

    ax = fig.add_subplot(gs[12:, 2:])
    multi_portfolio_data.plot_returns_scatter(ax=ax,
                                              time_period=time_period,
                                              benchmark=regime_benchmark,
                                              **kwargs)

    figs = [fig]
    if add_strategy_factsheets:
        for portfolio_data in multi_portfolio_data.portfolio_datas:
            figs.append(generate_portfolio_factsheet(portfolio_data=portfolio_data,
                                                     benchmark_prices=multi_portfolio_data.benchmark_prices,
                                                     time_period=time_period,
                                                     perf_params=perf_params,
                                                     regime_params=regime_params,
                                                     file_name_to_save=None))

    if file_name_to_save is not None:
        fu.figs_to_pdf(figs=figs, file_name=file_name_to_save, orientation='landscape')

    return fig


def generate_performance_attribution_report(multi_portfolio_data: MultiPortfolioData,
                                            time_period: da.TimePeriod = None,
                                            file_name: str = 'paired_report',
                                            **kwargs
                                            ):
    figs = []
    with sns.axes_style('darkgrid'):
        fig = plt.figure(figsize=FIG_SIZE, constrained_layout=True)
        figs.append(fig)
        gs = fig.add_gridspec(nrows=3, ncols=2, wspace=0.0, hspace=0.0)

        multi_portfolio_data.plot_nav(ax=fig.add_subplot(gs[0, :]),
                                      time_period=time_period,
                                      **kwargs)

        multi_portfolio_data.plot_performance_attribution(portfolio_ids=[0],
                                                          time_period=time_period,
                                                          attribution_metric=AttributionMetric.PNL,
                                                          ax=fig.add_subplot(gs[1, :]),
                                                          **kwargs)

        multi_portfolio_data.plot_performance_attribution(portfolio_ids=[1],
                                                          time_period=time_period,
                                                          attribution_metric=AttributionMetric.PNL_RISK,
                                                          ax=fig.add_subplot(gs[2, :]),
                                                          **kwargs)

        fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE)
        multi_portfolio_data.plot_performance_attribution(portfolio_ids=[0],
                                                          time_period=time_period,
                                                          attribution_metric=AttributionMetric.PNL_RISK,
                                                          ax=ax,
                                                          **kwargs)

        fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE)
        figs.append(fig)
        multi_portfolio_data.plot_performance_periodic_table(portfolio_ids=[0],
                                                             time_period=time_period,
                                                             attribution_metric=AttributionMetric.INST_PNL,
                                                             freq='M',
                                                             ax=ax,
                                                             **kwargs)

        fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE)
        figs.append(fig)
        multi_portfolio_data.plot_composite_table(ax=ax, **kwargs)

    fu.figs_to_pdf(figs=figs, file_name=file_name, orientation='landscape', is_add_current_date=True)
    return figs


class UnitTests(Enum):
    PERF_ATTRIBUTION = 1
    FACTSHEET = 2


def run_unit_test(unit_test: UnitTests):
    from qis.data.yf_data import load_etf_data
    import qis.models.linear.ra_returns as rar

    prices = load_etf_data()  # .dropna()
    ra_returns, weights, ewm_vol = rar.compute_ra_returns(returns=prices.pct_change(), span=60, vol_target=0.15)
    weights = weights.divide(weights.sum(1), axis=0)

    time_period = TimePeriod('31Dec2003', '31Dec2022')
    prices = time_period.locate(prices)
    group_data = pd.Series(
        dict(SPY='Equities', QQQ='Equities', EEM='Equities', TLT='Bonds', IEF='Bonds', SHY='Bonds', LQD='Credit',
             HYG='HighYield', GLD='Gold'))

    benchmark_prices = prices[['SPY', 'TLT']]

    portfolio_data1 = bp.backtest_model_portfolio(prices=prices,
                                                  weights=time_period.locate(weights),
                                                  is_output_portfolio_data=True,
                                                  ticker='VolParity')
    portfolio_data1._set_group_data(group_data=group_data, group_order=list(group_data.unique()))

    portfolio_data2 = bp.backtest_model_portfolio(prices=prices,
                                                  weights=np.ones(len(prices.columns)) / len(prices.columns),
                                                  is_output_portfolio_data=True,
                                                  ticker='EW')
    portfolio_data2._set_group_data(group_data=group_data, group_order=list(group_data.unique()))

    multi_portfolio_data = MultiPortfolioData(portfolio_datas=[portfolio_data1, portfolio_data2],
                                              benchmark_prices=benchmark_prices)

    if unit_test == UnitTests.PERF_ATTRIBUTION:

        generate_performance_attribution_report(multi_portfolio_data=multi_portfolio_data,
                                                time_period=TimePeriod('31Dec2021', '31Dec2022'),
                                                **KWARG_SHORT)

    if unit_test == UnitTests.FACTSHEET:
        generate_strategy_factsheet(multi_portfolio_data=multi_portfolio_data,
                                    backtest_name='Vol Parity Portfolio vs Equal Weight',
                                    time_period=TimePeriod('31Dec2006', '31Dec2022'),
                                    file_name_to_save=f"strategy_factsheet",
                                    **KWARG_LONG)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.FACTSHEET

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

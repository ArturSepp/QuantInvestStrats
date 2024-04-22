"""
performance report for a universe of several assets
with comparison to 1-2 benchmarks
output is one-page figure with key numbers
see example in qis.examples.multi_asset.py
"""
# packages
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, List, Optional, Tuple
from enum import Enum

# qis
import qis as qis
from qis import TimePeriod, PerfStat, PerfParams, RegimeData, RollingPerfStat, LegendStats, BenchmarkReturnsQuantileRegimeSpecs
from qis.portfolio.reports.config import PERF_PARAMS, REGIME_PARAMS


PERF_COLUMNS = (
    # PerfStat.START_DATE,
    # PerfStat.END_DATE,
    PerfStat.TOTAL_RETURN,
    PerfStat.PA_RETURN,
    PerfStat.VOL,
    PerfStat.SHARPE_RF0,
    PerfStat.BEAR_SHARPE,
    PerfStat.NORMAL_SHARPE,
    PerfStat.BULL_SHARPE,
    PerfStat.MAX_DD,
    PerfStat.MAX_DD_VOL,
    PerfStat.WORST,
    PerfStat.BEST,
    PerfStat.SKEWNESS,
    PerfStat.KURTOSIS)


class MultiAssetsReport:

    def __init__(self,
                 prices: pd.DataFrame,
                 benchmark_prices: Union[pd.Series, pd.DataFrame],
                 perf_params: PerfParams = PERF_PARAMS,
                 regime_params: qis.BenchmarkReturnsQuantileRegimeSpecs = REGIME_PARAMS
                 ):

        # make sure it is consistent
        self.prices = prices
        self.benchmark_prices = benchmark_prices.reindex(index=prices.index, method='ffill')
        self.perf_params = perf_params
        self.regime_params = regime_params

    def get_prices(self, benchmark: str = None, time_period: TimePeriod = None) -> pd.DataFrame:
        if benchmark is not None and benchmark not in self.prices.columns:
            if isinstance(self.benchmark_prices, pd.Series):
                prices = pd.concat([self.benchmark_prices, self.prices], axis=1)
            else:
                prices = pd.concat([self.benchmark_prices[benchmark], self.prices], axis=1)
        else:
            prices = self.prices
        if time_period is not None:
            prices = time_period.locate(prices)
        return prices

    def add_regime_shadows(self, ax: plt.Subplot,
                           regime_benchmark: str,
                           data_df: pd.DataFrame,
                           time_period: TimePeriod = None,
                           regime_params: BenchmarkReturnsQuantileRegimeSpecs = REGIME_PARAMS
                           ) -> None:
        if isinstance(self.benchmark_prices, pd.Series):
            pivot_prices = self.benchmark_prices
        else:
            if regime_benchmark is None:
                regime_benchmark = self.benchmark_prices.columns[0]
            pivot_prices = self.benchmark_prices[regime_benchmark]
        if time_period is not None:
            data_df = time_period.locate(data_df)
        pivot_prices = pivot_prices.reindex(index=data_df.index, method='ffill')
        qis.add_bnb_regime_shadows(ax=ax,
                                   data_df=data_df,
                                   pivot_prices=pivot_prices,
                                   benchmark=regime_benchmark,
                                   regime_params=regime_params)

    def plot_ra_perf_table(self,
                           benchmark: str,
                           time_period: TimePeriod = None,
                           perf_columns: List[PerfStat] = qis.BENCHMARK_TABLE_COLUMNS,
                           ax: plt.Subplot = None,
                           **kwargs) -> None:
        prices = self.get_prices(benchmark, time_period=time_period)
        qis.plot_ra_perf_table_benchmark(prices=prices,
                                         benchmark=benchmark,
                                         perf_params=self.perf_params,
                                         perf_columns=perf_columns,
                                         title=f"RA performance table: {qis.get_time_period(prices).to_str()}",
                                         rotation_for_columns_headers=0,
                                         ax=ax,
                                         **kwargs)

    def plot_ra_regime_table(self,
                             regime_benchmark_str: str = None,
                             time_period: TimePeriod = None,
                             perf_columns: List[PerfStat] = PERF_COLUMNS,
                             columns_title: str = 'Programs',
                             first_column_width: float = 3.5,
                             ax: plt.Subplot = None,
                             **kwargs) -> None:

        prices = pd.concat([self.prices, self.benchmark_prices], axis=1)

        if time_period is not None:
            prices = time_period.locate(prices)

        cvar_table = qis.compute_bnb_regimes_pa_perf_table(prices=prices,
                                                          benchmark=regime_benchmark_str,
                                                          perf_params=self.perf_params,
                                                          regime_params=self.regime_params)
        table_data = pd.DataFrame(data=prices.columns, index=cvar_table.index, columns=[columns_title])

        for perf_column in perf_columns:
            table_data[perf_column.to_str()] = qis.series_to_str(ds=cvar_table[perf_column.to_str()],
                                                             var_format=perf_column.to_format(**kwargs))

        special_columns_colors = [(0, 'steelblue')]
        qis.plot_df_table(df=table_data,
                          first_column_width=first_column_width,
                          add_index_as_column=False,
                          index_column_name='Strategies',
                          special_columns_colors=special_columns_colors,
                          ax=ax,
                          **kwargs)

    def plot_nav(self,
                 regime_benchmark: str = None,
                 var_format: str = '{:.0%}',
                 sharpe_format: str = '{:.2f}',
                 title: str = 'Cumulative performance',
                 is_log: bool = False,
                 time_period: TimePeriod = None,
                 ax: plt.Subplot = None,
                 **kwargs) -> None:
        prices = self.get_prices(time_period=time_period, benchmark=regime_benchmark)
        qis.plot_prices(prices=prices,
                        perf_params=self.perf_params,
                        start_to_one=True,
                        is_log=is_log,
                        var_format=var_format,
                        sharpe_format=sharpe_format,
                        title=title,
                        ax=ax,
                        **kwargs)
        self.add_regime_shadows(ax=ax, regime_benchmark=regime_benchmark, data_df=prices)

    def plot_drawdowns(self,
                       regime_benchmark_str: str = None,
                       time_period: TimePeriod = None,
                       title: str = 'Running Drawdowns',
                       ax: plt.Subplot = None,
                       **kwargs) -> None:
        prices = self.get_prices(time_period=time_period, benchmark=regime_benchmark_str)
        qis.plot_rolling_drawdowns(prices=prices, title=title, ax=ax, **kwargs)
        self.add_regime_shadows(ax=ax, regime_benchmark=regime_benchmark_str, data_df=prices)

    def plot_rolling_time_under_water(self,
                                      regime_benchmark: str = None,
                                      time_period: TimePeriod = None,
                                      title: str = 'Running Time Under Water',
                                      ax: plt.Subplot = None,
                                      **kwargs) -> None:
        prices = self.get_prices(time_period=time_period, benchmark=regime_benchmark)
        qis.plot_rolling_time_under_water(prices=prices, title=title, ax=ax, **kwargs)
        self.add_regime_shadows(ax=ax, regime_benchmark=regime_benchmark, data_df=prices)

    def plot_annual_returns(self,
                            heatmap_freq: str = 'YE',
                            date_format: str = '%Y',
                            time_period: TimePeriod = None,
                            ax: plt.Subplot = None,
                            **kwargs) -> None:
        local_kwargs = qis.update_kwargs(kwargs=kwargs,
                                        new_kwargs=dict(fontsize=4,
                                                        square=False,
                                                        x_rotation=90))
        qis.plot_periodic_returns_table(prices=self.get_prices(time_period=time_period),
                                        freq=heatmap_freq,
                                        ax=ax,
                                        title=f"{heatmap_freq} Returns",
                                        date_format=date_format,
                                        **local_kwargs)

    def plot_corr_table(self,
                        freq: str = 'W-WED',
                        time_period: TimePeriod = None,
                        ax: plt.Subplot = None,
                        **kwargs) -> None:
        local_kwargs = qis.update_kwargs(kwargs=kwargs, new_kwargs=dict(fontsize=4))
        prices = self.get_prices(time_period=time_period)
        qis.plot_returns_corr_table(prices=prices,
                                    x_rotation=90,
                                    freq=freq,
                                    title=f"Correlation {freq} returns: {qis.get_time_period(prices).to_str()}",
                                    ax=ax,
                                    **local_kwargs)

    def plot_returns_scatter(self,
                             benchmark: str,
                             time_period: TimePeriod = None,
                             freq: str = 'QE',
                             order: int = 2,
                             ax: plt.Subplot = None,
                             **kwargs) -> None:
        local_kwargs = qis.update_kwargs(kwargs=kwargs,
                                         new_kwargs=dict(weight='bold',
                                                         markersize=8,
                                                         x_rotation=0,
                                                         first_color_fixed=False,
                                                         ci=None))
        qis.plot_returns_scatter(prices=self.get_prices(benchmark=benchmark, time_period=time_period),
                                 benchmark=benchmark,
                                 order=order,
                                 freq=freq,
                                 title=f"Scatterplot of {self.perf_params.freq_reg}-returns vs {benchmark}",
                                 ax=ax,
                                 **local_kwargs)

    def plot_benchmark_beta(self,
                            benchmark: str,
                            freq: str = 'ME',
                            span: int = 12,
                            time_period: TimePeriod = None,
                            ax: plt.Subplot = None,
                            **kwargs) -> None:
        returns = qis.to_returns(prices=self.get_prices(benchmark=benchmark), freq=freq)
        ewm_linear_model = qis.estimate_ewm_linear_model(x=returns[benchmark].to_frame(),
                                                         y=returns.drop(benchmark, axis=1),
                                                         span=span,
                                                         is_x_correlated=True)
        ewm_linear_model.plot_factor_loadings(factor=benchmark,
                                              time_period=time_period,
                                              title=f"Rolling EWM span-{span:0.0f} beta to {benchmark}",
                                              ax=ax,
                                              **kwargs)
        self.add_regime_shadows(ax=ax, regime_benchmark=benchmark, time_period=time_period, data_df=self.prices)

    def plot_rolling_perf(self,
                          rolling_perf_stat: RollingPerfStat = RollingPerfStat.SHARPE,
                          regime_benchmark: str = None,
                          time_period: TimePeriod = None,
                          rolling_window: int = 3*252,
                          roll_freq: Optional[str] = None,
                          legend_stats: LegendStats = LegendStats.AVG_LAST,
                          title: Optional[str] = None,
                          var_format: str = '{:.2f}',
                          regime_params: BenchmarkReturnsQuantileRegimeSpecs = REGIME_PARAMS,
                          ax: plt.Subplot = None,
                          **kwargs
                          ) -> plt.Figure:

        # do not use start end dates here so the sharpe will be continuous with different time_period
        if ax is None:
            fig, ax = plt.subplots()

        prices = self.get_prices(time_period=time_period, benchmark=regime_benchmark)
        fig = qis.plot_rolling_perf_stat(prices=prices,
                                         rolling_perf_stat=rolling_perf_stat,
                                         time_period=time_period,
                                         roll_periods=rolling_window,
                                         roll_freq=roll_freq,
                                         legend_stats=legend_stats,
                                         var_format=var_format,
                                         title=title or f"3y rolling {rolling_perf_stat.value}",
                                         ax=ax,
                                         **kwargs)

        if regime_benchmark is not None:
            self.add_regime_shadows(ax=ax, regime_benchmark=regime_benchmark, data_df=prices, regime_params=regime_params)
        return fig

    def plot_regime_data(self,
                         benchmark: str,
                         regime_data_to_plot: RegimeData = RegimeData.REGIME_SHARPE,
                         time_period: TimePeriod = None,
                         var_format: Optional[str] = None,
                         is_conditional_sharpe: bool = True,
                         legend_loc: Optional[str] = 'upper center',
                         ax: plt.Subplot = None,
                         **kwargs) -> None:
        prices = self.get_prices(time_period=time_period)
        title = f"Sharpe ratio decomposition by Strategies to {benchmark} Bear/Normal/Bull regimes"
        regime_classifier = qis.BenchmarkReturnsQuantilesRegime(regime_params=self.regime_params)
        qis.plot_regime_data(regime_classifier=regime_classifier,
                             prices=prices,
                             benchmark=benchmark,
                             is_conditional_sharpe=is_conditional_sharpe,
                             regime_data_to_plot=regime_data_to_plot,
                             var_format=var_format or '{:.2f}',
                             legend_loc=legend_loc,
                             perf_params=self.perf_params,
                             title=title,
                             ax=ax,
                             **kwargs)


def generate_multi_asset_factsheet(prices: pd.DataFrame,
                                   benchmark_prices: Union[pd.Series, pd.DataFrame] = None,
                                   benchmark: str = None,
                                   perf_params: PerfParams = PERF_PARAMS,
                                   regime_params: qis.BenchmarkReturnsQuantileRegimeSpecs = REGIME_PARAMS,
                                   heatmap_freq: str = 'YE',
                                   corr_freq: str = 'ME',
                                   time_period: TimePeriod = None,  # time period for reporting
                                   figsize: Tuple[float, float] = (8.3, 11.7),  # A4 for portrait
                                   **kwargs
                                   ) -> plt.Figure:

    # use passed benchmark
    if benchmark is None and benchmark_prices is not None:
        if benchmark_prices is None:
            raise ValueError(f"pass either benchmark or benchmark_prices")
        else:
            if isinstance(benchmark_prices, pd.Series):
                benchmark = benchmark_prices.name
            else:
                benchmark = benchmark_prices.columns[0]
    # use benchmark from prices
    if benchmark is not None and benchmark_prices is None:
        if benchmark in prices.columns:
            benchmark_prices = prices[benchmark]
        else:
            raise ValueError(f"benchmark must be in prices")

    # report data
    report = MultiAssetsReport(prices=prices,
                               benchmark_prices=benchmark_prices,
                               perf_params=perf_params,
                               regime_params=regime_params)

    local_kwargs = dict(fontsize=5,
                        linewidth=0.5,
                        weight='normal',
                        markersize=1,
                        framealpha=0.75,
                        x_date_freq='YE',
                        time_period=time_period)
    # overrite local_kwargs with kwargs is they are provided
    kwargs = qis.update_kwargs(local_kwargs, kwargs)

    # figure
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(nrows=10, ncols=4, wspace=0.0, hspace=0.0)

    report.plot_nav(regime_benchmark=benchmark,
                    ax=fig.add_subplot(gs[:2, :2]),
                    **kwargs)

    report.plot_drawdowns(regime_benchmark_str=benchmark,
                          ax=fig.add_subplot(gs[2:4, :2]),
                          **kwargs)

    report.plot_rolling_time_under_water(regime_benchmark=benchmark,
                                         ax=fig.add_subplot(gs[4:6, :2]),
                                         **kwargs)

    report.plot_rolling_perf(regime_benchmark=benchmark,
                             rolling_perf_stat=RollingPerfStat.SHARPE,
                             ax=fig.add_subplot(gs[6:8, :2]),
                             **kwargs)

    report.plot_benchmark_beta(benchmark=benchmark,
                               ax=fig.add_subplot(gs[8:10, :2]),
                               **kwargs)

    report.plot_ra_perf_table(benchmark=benchmark,
                              ax=fig.add_subplot(gs[0, 2:]),
                              **kwargs)

    time_period1 = qis.get_time_period_shifted_by_years(time_period=qis.get_time_period(df=prices))
    # change regression to weekly
    report.plot_ra_perf_table(benchmark=benchmark,
                              ax=fig.add_subplot(gs[1, 2:]),
                              **qis.update_kwargs(kwargs, dict(time_period=time_period1,
                                                               alpha_an_factor=52,
                                                               freq_reg='W-WED')))

    report.plot_annual_returns(ax=fig.add_subplot(gs[2:4, 2:]),
                               heatmap_freq=heatmap_freq,
                               **kwargs)

    report.plot_corr_table(freq=corr_freq,
                           ax=fig.add_subplot(gs[4, 2]),
                           **kwargs)
    report.plot_corr_table(freq=corr_freq,
                           ax=fig.add_subplot(gs[4, 3]),
                           **qis.update_kwargs(kwargs, dict(time_period=time_period1)))

    report.plot_regime_data(benchmark=benchmark,
                            ax=fig.add_subplot(gs[5, 2:]),
                            **kwargs)

    report.plot_rolling_perf(regime_benchmark=benchmark,
                             rolling_perf_stat=RollingPerfStat.VOL,
                             var_format='{:.1%}',
                             ax=fig.add_subplot(gs[6:8, 2:]),
                             **kwargs)

    report.plot_returns_scatter(benchmark=benchmark,
                                ax=fig.add_subplot(gs[8:10, 2:]),
                                freq=perf_params.freq_reg,
                                **kwargs)
    return fig

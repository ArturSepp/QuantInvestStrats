"""
generic to generate strategy factsheet report
with comparision to 1-2 benchmarks
"""
# built in
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, List, Optional
from enum import Enum

# qis
import qis.file_utils as fu
import qis.utils.dates as da
import qis.utils.struct_ops as sop
from qis.utils.df_to_str import series_to_str

import qis.perfstats.perf_table as rpt
import qis.perfstats.returns as ret
import qis.perfstats.drawdowns as cdr
from qis.perfstats.config import PerfStat, PerfParams, RegimeData
import qis.plots.derived.returns_scatter as prs
import qis.perfstats.regime_classifier as rcl
from qis.perfstats.regime_classifier import BenchmarkReturnsQuantileRegimeSpecs
import qis.models.linear.ewm_factors as ef

# plots
import qis.plots.derived.prices as ppd
import qis.plots.derived.perf_table as ppt
import qis.plots.table as ptb
import qis.plots.derived.returns_heatmap as rhe
import qis.plots.derived.regime_scatter as drc
import qis.plots.derived.correlations as pco


FIG_SIZE = (8.3, 11.7)  # A4 for portrait


PERF_COLUMNS = (
    # PerfStat.START_DATE,
    # PerfStat.END_DATE,
    PerfStat.TOTAL_RETURN,
    PerfStat.PA_RETURN,
    PerfStat.VOL,
    PerfStat.SHARPE,
    PerfStat.BEAR_SHARPE,
    PerfStat.NORMAL_SHARPE,
    PerfStat.BULL_SHARPE,
    PerfStat.MAX_DD,
    PerfStat.MAX_DD_VOL,
    PerfStat.WORST,
    PerfStat.BEST,
    PerfStat.SKEWNESS,
    PerfStat.KURTOSIS)


PERF_PARAMS = PerfParams(freq='W-WED')
REGIME_PARAMS = BenchmarkReturnsQuantileRegimeSpecs(freq='Q')


class MultiAssetReport:

    def __init__(self,
                 prices: pd.DataFrame,
                 benchmark_prices: Union[pd.Series, pd.DataFrame],
                 perf_params: PerfParams = PERF_PARAMS,
                 regime_params: BenchmarkReturnsQuantileRegimeSpecs = REGIME_PARAMS
                 ):

        # make sure it is consistent
        self.prices = prices
        self.benchmark_prices = benchmark_prices.reindex(index=prices.index, method='ffill')
        self.perf_params = perf_params
        self.regime_params = regime_params

    def get_prices(self, benchmark: str = None, time_period: da.TimePeriod = None) -> pd.DataFrame:
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

    def add_regime_shadows(self, ax: plt.Subplot, regime_benchmark_str: str, data_df: pd.DataFrame) -> None:
        if isinstance(self.benchmark_prices, pd.Series):
            pivot_prices = self.benchmark_prices
        else:
            if regime_benchmark_str is None:
                regime_benchmark_str = self.benchmark_prices.columns[0]
            pivot_prices = self.benchmark_prices[regime_benchmark_str]
        pivot_prices = pivot_prices.reindex(index=data_df.index, method='ffill')
        rcl.add_bnb_regime_shadows(ax=ax,
                                   data_df=data_df,
                                   pivot_prices=pivot_prices,
                                   benchmark=regime_benchmark_str,
                                   regime_params=self.regime_params)

    def plot_ra_perf_table(self,
                           benchmark: str,
                           time_period: da.TimePeriod = None,
                           ax: plt.Subplot = None,
                           **kwargs) -> None:
        prices = self.get_prices(benchmark, time_period=time_period)
        ppt.plot_ra_perf_table_benchmark(prices=prices,
                                         benchmark=benchmark,
                                         perf_params=self.perf_params,
                                         perf_columns=rpt.BENCHMARK_TABLE_COLUMNS,
                                         title=f"RA performance table: {da.get_time_period(prices).to_str()}",
                                         rotation_for_columns_headers=0,
                                         ax=ax,
                                         **kwargs)

    def plot_ra_regime_table(self,
                             regime_benchmark_str: str = None,
                             time_period: da.TimePeriod = None,
                             perf_columns: List[PerfStat] = PERF_COLUMNS,
                             columns_title: str = 'Programs',
                             first_column_width: float = 3.5,
                             ax: plt.Subplot = None,
                             **kwargs
                             ) -> plt.Figure:

        prices = pd.concat([self.prices, self.benchmark_prices], axis=1)

        if time_period is not None:
            prices = time_period.locate(prices)

        cvar_table = rcl.compute_bnb_regimes_pa_perf_table(prices=prices,
                                                           benchmark=regime_benchmark_str,
                                                           perf_params=self.perf_params,
                                                           regime_params=self.regime_params)
        table_data = pd.DataFrame(data=prices.columns, index=cvar_table.index, columns=[columns_title])

        for perf_column in perf_columns:
            table_data[perf_column.to_str()] = series_to_str(ds=cvar_table[perf_column.to_str()],
                                                             var_format=perf_column.to_format(**kwargs))

        special_columns_colors = [(0, 'steelblue')]
        fig = ptb.plot_df_table(df=table_data,
                                first_column_width=first_column_width,
                                add_index_as_column=False,
                                index_column_name='Strategies',
                                special_columns_colors=special_columns_colors,
                                ax=ax,
                                **kwargs)
        return fig

    def plot_nav(self,
                 regime_benchmark_str: str = None,
                 var_format: str = '{:.0%}',
                 sharpe_format: str = '{:.2f}',
                 title: str = 'Cumulative performance',
                 is_log: bool = True,
                 ax: plt.Subplot = None,
                 **kwargs
                 ) -> plt.Figure:
        fig = ppd.plot_prices(prices=self.prices,
                              perf_params=self.perf_params,
                              start_to_one=True,
                              is_log=is_log,
                              var_format=var_format,
                              sharpe_format=sharpe_format,
                              title=title,
                              ax=ax,
                              **kwargs)
        self.add_regime_shadows(ax=ax, regime_benchmark_str=regime_benchmark_str, data_df=self.prices)

        return fig

    def plot_drawdowns(self,
                       regime_benchmark_str: str = None,
                       ax: plt.Subplot = None,
                       **kwargs) -> None:
        cdr.plot_drawdown(prices=self.prices, ax=ax, **kwargs)
        self.add_regime_shadows(ax=ax, regime_benchmark_str=regime_benchmark_str, data_df=self.prices)

    def plot_rolling_time_under_water(self,
                                      regime_benchmark_str: str = None,
                                      ax: plt.Subplot = None,
                                      **kwargs) -> None:
        cdr.plot_rolling_time_under_water(prices=self.prices, ax=ax, **kwargs)
        self.add_regime_shadows(ax=ax, regime_benchmark_str=regime_benchmark_str, data_df=self.prices)

    def plot_annual_returns(self,
                            heatmap_freq: str = 'A',
                            date_format: str = '%Y',
                            ax: plt.Subplot = None,
                            **kwargs
                            ) -> plt.Figure:
        local_kwargs = sop.update_kwargs(kwargs=kwargs,
                                         new_kwargs=dict(fontsize=4,
                                                         square=False,
                                                         x_rotation=90))
        fig = rhe.plot_periodic_returns_table_from_prices(prices=self.prices,
                                                          freq=heatmap_freq,
                                                          ax=ax,
                                                          title=f"Annual Returns",
                                                          date_format=date_format,
                                                          **local_kwargs)
        return fig

    def plot_corr_table(self, freq: str = 'W-WED',
                        time_period: da.TimePeriod = None,
                        ax: plt.Figure = None,
                        **kwargs
                        ) -> plt.Figure:
        local_kwargs = sop.update_kwargs(kwargs=kwargs, new_kwargs=dict(fontsize=4))
        prices = self.get_prices(time_period=time_period)
        fig = pco.plot_corr_table(prices=prices,
                                  x_rotation=90,
                                  freq=freq,
                                  title=f"Correelation {freq} returns: {da.get_time_period(prices).to_str()}",
                                  ax=ax,
                                  **local_kwargs)
        return fig

    def plot_returns_scatter(self, benchmark: str, ax: plt.Figure, **kwargs):
        local_kwargs = sop.update_kwargs(kwargs=kwargs,
                                         new_kwargs={'weight': 'bold',
                                                     # 'alpha_an_factor': 52.0,
                                                     'x_rotation': 0,
                                                     'first_color_fixed': False,
                                                     'ci': None})
        fig = prs.plot_returns_scatter(prices=self.get_prices(benchmark=benchmark),
                                       benchmark=benchmark,
                                       freq=self.perf_params.freq_reg,
                                       order=2,
                                       title=f"Scatterplot of {self.perf_params.freq_reg}-returns vs {benchmark}",
                                       ax=ax,
                                       **local_kwargs)
        return fig

    def plot_benchmark_beta(self,
                            benchmark: str,
                            freq: str = 'B',
                            ewm_lambda: float = 0.97,
                            ax: plt.Subplot = None,
                            **kwargs):
        returns = ret.to_returns(prices=self.get_prices(benchmark=benchmark), freq=freq)
        ewm_linear_model = ef.estimate_ewm_linear_model(x=returns[benchmark].to_frame(),
                                                        y=returns.drop(benchmark, axis=1),
                                                        ewm_lambda=ewm_lambda,
                                                        is_x_correlated=True)
        ewm_linear_model.plot_factor_loadings(factor=benchmark,
                                              title=f"Rolling EWM-{ewm_lambda} beta to {benchmark}",
                                              ax=ax,
                                              **kwargs)
        self.add_regime_shadows(ax=ax, regime_benchmark_str=benchmark, data_df=self.prices)

    def plot_regime_data(self,
                         benchmark: str,
                         regime_data_to_plot: RegimeData = RegimeData.REGIME_SHARPE,
                         time_period: da.TimePeriod = None,
                         var_format: Optional[str] = None,
                         is_conditional_sharpe: bool = True,
                         legend_loc: Optional[str] = 'upper center',
                         ax: plt.Subplot = None,
                         **kwargs
                         ) -> plt.Figure:
        prices = self.get_prices()
        title = f"Sharpe ratio decomposition by Strategies to {benchmark} Bear/Normal/Bull regimes"
        regime_classifier = rcl.BenchmarkReturnsQuantilesRegime(regime_params=REGIME_PARAMS)
        fig = rcl.plot_regime_data(regime_classifier=regime_classifier,
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
        return fig


def generate_multi_asset_factsheet(prices: pd.DataFrame,
                                   benchmark_prices: Union[pd.Series, pd.DataFrame] = None,
                                   benchmark: str = None,
                                   perf_params: PerfParams = PERF_PARAMS,
                                   regime_params: BenchmarkReturnsQuantileRegimeSpecs = REGIME_PARAMS,
                                   heatmap_freq: str = 'A',
                                   time_period: da.TimePeriod = None,
                                   file_name: str = 'multi_asset_report',
                                   **kwargs
                                   ) -> None:

    # add default benchmark str
    if benchmark is None and benchmark_prices is not None:
        if benchmark_prices is None:
            raise ValueError(f"pass either benchmark or benchmark_prices")
        else:
            if isinstance(benchmark_prices, pd.Series):
                benchmark = benchmark_prices.name
            else:
                benchmark = benchmark_prices.columns[0]
    if benchmark is not None and benchmark_prices is None:
        if benchmark in prices.columns:
            benchmark_prices = prices[benchmark]
        else:
            raise ValueError(f"benchmark must be in prices")

    # report data
    report = MultiAssetReport(prices=prices,
                              benchmark_prices=benchmark_prices,
                              perf_params=perf_params,
                              regime_params=regime_params)

    local_kwargs = dict(fontsize=5,
                        linewidth=0.5,
                        weight='normal',
                        markersize=1,
                        legend_alpha=0.75)
    kwargs = sop.update_kwargs(local_kwargs, kwargs)

    # create 5*2 figure
    fig = plt.figure(figsize=FIG_SIZE, constrained_layout=True)
    gs = fig.add_gridspec(nrows=8, ncols=4, wspace=0.0, hspace=0.0)

    ax11 = fig.add_subplot(gs[:2, :2])
    report.plot_nav(regime_benchmark_str=benchmark,
                    ax=ax11,
                    **kwargs)

    ax21 = fig.add_subplot(gs[2:4, :2])
    report.plot_drawdowns(regime_benchmark_str=benchmark,
                          ax=ax21,
                          **kwargs)

    ax31 = fig.add_subplot(gs[4:6, :2])
    report.plot_rolling_time_under_water(regime_benchmark_str=benchmark,
                                         ax=ax31,
                                         **kwargs)

    ax41 = fig.add_subplot(gs[6:8, :2])
    report.plot_benchmark_beta(benchmark=benchmark,
                               ax=ax41,
                               **kwargs)

    ax12 = fig.add_subplot(gs[0, 2:])
    report.plot_ra_perf_table(benchmark=benchmark, ax=ax12, **kwargs)

    ax22 = fig.add_subplot(gs[1, 2:])
    time_period1 = da.get_time_period_shifted_by_years(time_period=da.get_time_period(df=prices))
    report.plot_ra_perf_table(benchmark=benchmark, time_period=time_period1, ax=ax22, **kwargs)

    ax23 = fig.add_subplot(gs[2:4, 2:])
    report.plot_annual_returns(ax=ax23,
                               **kwargs)

    ax24a = fig.add_subplot(gs[4, 2])
    report.plot_corr_table(freq='W-WED', ax=ax24a, **kwargs)
    ax24b = fig.add_subplot(gs[4, 3])
    report.plot_corr_table(freq='W-WED',
                           time_period=da.get_time_period_shifted_by_years(time_period=da.get_time_period(df=prices)),
                           ax=ax24b, **kwargs)

    ax25 = fig.add_subplot(gs[5, 2:])
    report.plot_regime_data(ax=ax25, benchmark=benchmark, **kwargs)

    ax26 = fig.add_subplot(gs[6:8, 2:])
    report.plot_returns_scatter(benchmark=benchmark,
                                ax=ax26,
                                **kwargs)

    fu.figs_to_pdf(figs=[fig], file_name=f"{file_name}", orientation='landscape')


class UnitTests(Enum):
    FACTSHEET = 1


def run_unit_test(unit_test: UnitTests):

    from qis.data.yf_data import load_etf_data
    prices = load_etf_data().dropna()

    if unit_test == UnitTests.FACTSHEET:
        generate_multi_asset_factsheet(prices=prices,
                                       benchmark='SPY',
                                       heatmap_freq='A',
                                       perf_params=PERF_PARAMS,
                                       regime_params=BenchmarkReturnsQuantileRegimeSpecs(freq='Q'))

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.FACTSHEET

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

# biult in
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Tuple
from enum import Enum

# qis
import qis.utils.dates as da
import qis.utils.df_ops as dfo
import qis.utils.struct_ops as sop
import qis.perfstats.perf_stats as pt
from qis.perfstats.rolling_stats import RollingPerfStat, compute_rolling_perf_stat
from qis.perfstats.config import PerfStat, PerfParams
from qis.perfstats.regime_classifier import BenchmarkReturnsQuantileRegimeSpecs

import qis.plots.derived.drawdowns as dra
import qis.plots.time_series as pts
import qis.plots.utils as put
from qis.plots.derived.regime_data import add_bnb_regime_shadows


class PerfStatsLabels(Enum):
    """
    enumerate some combinations for perf stat labels
    can refer through value
    perf_stats_labels=qis.PerfStatsLabels.DETAILED_WITH_DD.value
    perf_stats_labels=qis.PerfStatsLabels.DETAILED_WITH_DDVOL.value
    perf_stats_labels=qis.PerfStatsLabels.TOTAL_DETAILED.value
    """
    SHARPE = (PerfStat.SHARPE_RF0, )
    DETAILED_EXCESS_SHARPE = (PerfStat.PA_RETURN, PerfStat.VOL, PerfStat.SHARPE_EXCESS, )
    DETAILED_SHARPE_RF0 = (PerfStat.PA_RETURN, PerfStat.VOL, PerfStat.SHARPE_RF0,)
    DETAILED_WITH_DD = (PerfStat.PA_RETURN, PerfStat.VOL, PerfStat.SHARPE_RF0, PerfStat.MAX_DD, )
    DETAILED_WITH_SKEW = (PerfStat.PA_RETURN, PerfStat.VOL, PerfStat.SHARPE_RF0, PerfStat.SKEWNESS, )
    DETAILED_LOG_SHARPE = (PerfStat.AN_LOG_RETURN, PerfStat.VOL, PerfStat.SHARPE_LOG_AN, )
    DETAILED_WITH_DDVOL = (PerfStat.PA_RETURN, PerfStat.VOL, PerfStat.SHARPE_EXCESS, PerfStat.MAX_DD, PerfStat.MAX_DD_VOL,)
    TOTAL = (PerfStat.TOTAL_RETURN, PerfStat.MAX_DD, )
    TOTAL_DETAILED = (PerfStat.TOTAL_RETURN, PerfStat.PA_RETURN, PerfStat.VOL, PerfStat.SHARPE_EXCESS, PerfStat.MAX_DD, )


def get_performance_labels_for_stats(prices: Union[pd.DataFrame, pd.Series],
                                     perf_stats_labels: List[PerfStat] = (PerfStat.PA_RETURN, PerfStat.VOL,
                                                                          PerfStat.SHARPE_RF0, PerfStat.MAX_DD,),
                                     perf_params: PerfParams = None,
                                     **kwargs
                                     ) -> List[str]:

    if any(prices.columns.duplicated()):
        raise ValueError(f"dublicated columns:\n{prices.columns[prices.columns.duplicated()]}")

    ra_perf_table = pt.compute_ra_perf_table(prices=prices, perf_params=perf_params)
    legend_labels = []
    for index in ra_perf_table.index:
        name = index if isinstance(index, str) else str(index)
        label = f"{name}: "
        for perf_stat in perf_stats_labels:
            # pefromance strat is always defines using perf_stat.to_str()
            label += f"{perf_stat.to_str(**kwargs)}={perf_stat.to_format(**kwargs).format(ra_perf_table.loc[index, perf_stat.to_str()])}, "
        legend_labels.append(label[:-2])  # remove last ", "

    return legend_labels


def plot_prices(prices: Union[pd.DataFrame, pd.Series],
                perf_stats_labels: List[PerfStat] = (PerfStat.PA_RETURN, PerfStat.VOL, PerfStat.SHARPE_RF0, ),
                perf_params: PerfParams = None,
                regime_benchmark_str: str = None,  # to add regimes
                pivot_prices: pd.Series = None,
                regime_params: BenchmarkReturnsQuantileRegimeSpecs = BenchmarkReturnsQuantileRegimeSpecs(),
                var_format: str = '{:,.1f}',
                digits_to_show: int = 1,
                sharpe_format: str = '{:.2f}',
                x_date_freq: str = 'QE',
                trend_line: put.TrendLine = put.TrendLine.NONE,
                is_log: bool = False,
                resample_freq: str = None,
                start_to_one: bool = True,
                end_to_one: bool = False,
                title: str = None,
                ax: plt.Subplot = None,
                **kwargs
                ) -> plt.Figure:

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    legend_labels = get_performance_labels_for_stats(prices=prices,
                                                     perf_stats_labels=perf_stats_labels,
                                                     perf_params=perf_params,
                                                     digits_to_show=digits_to_show,
                                                     sharpe_format=sharpe_format,
                                                     **kwargs)

    if resample_freq is not None:
        prices = prices.asfreq(resample_freq, method='ffill')

    if end_to_one:
        scaler = 1.0 / dfo.get_last_nonnan_values(df=prices)
        prices = prices.multiply(scaler)
    elif start_to_one:
        prices = prices.divide(dfo.get_first_nonnan_values(df=prices))

    fig = pts.plot_time_series(df=prices,
                               trend_line=trend_line,
                               var_format=var_format,
                               title=title,
                               legend_labels=legend_labels,
                               is_log=is_log,
                               x_date_freq=x_date_freq,
                               ax=ax,
                               **kwargs)

    if regime_benchmark_str is not None and regime_params is not None:
        add_bnb_regime_shadows(ax=ax,
                               data_df=prices,
                               pivot_prices=pivot_prices,
                               benchmark=regime_benchmark_str,
                               regime_params=regime_params)
    return fig


def plot_prices_with_dd(prices: Union[pd.DataFrame, pd.Series],
                        perf_stats_labels: List[PerfStat] = (PerfStat.PA_RETURN, PerfStat.VOL,
                                                             PerfStat.SHARPE_RF0),
                        perf_params: PerfParams = None,
                        regime_benchmark_str: str = None,  # to add regimes
                        pivot_prices: pd.Series = None,
                        regime_params: BenchmarkReturnsQuantileRegimeSpecs = BenchmarkReturnsQuantileRegimeSpecs(),
                        var_format: str = '{:,.1f}',
                        dd_format: str = '{:.0%}',
                        digits_to_show: int = 1,
                        sharpe_format: str = '{:.2f}',
                        x_date_freq: str = 'QE',
                        is_log: bool = False,
                        remove_xticklabels_ax1: bool = True,
                        title: str = 'Performance',
                        dd_title: str = 'Running Drawdown',
                        dd_legend_type: dra.DdLegendType = dra.DdLegendType.NONE,
                        axs: List[plt.Subplot] = None,
                        **kwargs
                        ) -> plt.Figure:

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    if axs is None:
        fig, axs = plt.subplots(2, 1)
    else:
        fig = None

    plot_prices(prices=prices,
                perf_params=perf_params,
                var_format=var_format,
                digits_to_show=digits_to_show,
                sharpe_format=sharpe_format,
                perf_stats_labels=perf_stats_labels,
                x_date_freq=x_date_freq,
                title=title,
                is_log=is_log,
                ax=axs[0],
                **kwargs)

    dra.plot_rolling_drawdowns(prices=prices,
                               perf_params=perf_params,
                               dd_legend_type=dd_legend_type,
                               x_date_freq=x_date_freq,
                               var_format=dd_format,
                               title=dd_title,
                               ax=axs[1],
                               **kwargs)

    if remove_xticklabels_ax1:
        axs[0].set_xticklabels('')

    if (regime_benchmark_str is not None or pivot_prices is not None) and regime_params is not None:
        for ax in axs:
            add_bnb_regime_shadows(ax=ax,
                                   data_df=prices,
                                   pivot_prices=pivot_prices,
                                   benchmark=regime_benchmark_str,
                                   regime_params=regime_params,
                                   perf_params=perf_params)
    return fig


def plot_prices_with_fundamentals(prices: Union[pd.DataFrame, pd.Series],
                                  volumes: Union[pd.DataFrame, pd.Series],
                                  mcap: Union[pd.DataFrame, pd.Series],
                                  perf_stats_labels: List[PerfStat] = (PerfStat.PA_RETURN, PerfStat.VOL, PerfStat.SHARPE_RF0, ),
                                  perf_params: PerfParams = None,
                                  regime_benchmark_str: str = None,  # to add regimes
                                  pivot_prices: pd.Series = None,
                                  regime_params: BenchmarkReturnsQuantileRegimeSpecs = BenchmarkReturnsQuantileRegimeSpecs(),
                                  trend_line: put.TrendLine = put.TrendLine.AVERAGE,
                                  var_format: str = '{:,.1f}',
                                  digits_to_show: int = 2,
                                  sharpe_format: str = '{:.2f}',
                                  is_log: bool = False,
                                  title: str = None,
                                  dd_title: str = 'Running Drawdown',
                                  dd_legend_type: dra.DdLegendType = dra.DdLegendType.NONE,
                                  axs: List[plt.Subplot] = None,
                                  **kwargs
                                  ) -> plt.Figure:

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    if axs is None:
        fig, axs = plt.subplots(3, 1)
    else:
        fig = None

    plot_prices(prices=prices,
                perf_params=perf_params,
                var_format=var_format,
                digits_to_show=digits_to_show,
                sharpe_format=sharpe_format,
                perf_stats_labels=perf_stats_labels,
                is_log=is_log,
                ax=axs[0],
                **kwargs)

    dra.plot_rolling_drawdowns(prices=prices,
                               perf_params=perf_params,
                               dd_legend_type=dd_legend_type,
                               title=dd_title,
                               ax=axs[1],
                               **kwargs)

    pts.plot_time_series_2ax(df1=volumes,
                             df2=mcap,
                             trend_line1=trend_line,
                             trend_line2=trend_line,
                             var_format=var_format,
                             title=title,
                             ax=axs[2],
                             **kwargs)

    axs[0].set_xticklabels('')
    axs[1].set_xticklabels('')

    if regime_benchmark_str is not None and regime_params is not None:
        for ax in axs:
            add_bnb_regime_shadows(ax=ax,
                                   data_df=prices,
                                   pivot_prices=pivot_prices,
                                   benchmark=regime_benchmark_str,
                                   regime_params=regime_params,
                                   perf_params=perf_params)
    return fig


def plot_prices_2ax(prices_ax1: Union[pd.DataFrame, pd.Series],
                    prices_ax2: Union[pd.DataFrame, pd.Series],
                    perf_stats_labels: List[PerfStat] = (PerfStat.PA_RETURN, PerfStat.VOL, PerfStat.SHARPE_RF0),
                    perf_params: PerfParams = None,
                    var_format: str = '{:,.1f}',
                    digits_to_show: int = 2,
                    sharpe_format: str = '{:.2f}',
                    trend_line: put.TrendLine = put.TrendLine.NONE,
                    is_logs: Tuple[bool, bool] = (False, False),
                    start_to_one: bool = True,
                    end_to_one: bool = False,
                    title: str = None,
                    ax: plt.Subplot = None,
                    **kwargs
                    ) -> plt.Figure:

    if isinstance(prices_ax1, pd.Series):
        prices_ax1 = prices_ax1.to_frame()
    if isinstance(prices_ax2, pd.Series):
        prices_ax2 = prices_ax2.to_frame()

    prices_ax1.columns = [f"{x} (left)" for x in prices_ax1.columns]
    prices_ax2.columns = [f"{x} (right)" for x in prices_ax2.columns]
    legend_labels1 = get_performance_labels_for_stats(prices=prices_ax1,
                                                      perf_stats_labels=perf_stats_labels,
                                                      perf_params=perf_params,
                                                      digits_to_show=digits_to_show,
                                                      sharpe_format=sharpe_format,
                                                      **kwargs)
    legend_labels2 = get_performance_labels_for_stats(prices=prices_ax2,
                                                      perf_stats_labels=perf_stats_labels,
                                                      perf_params=perf_params,
                                                      digits_to_show=digits_to_show,
                                                      sharpe_format=sharpe_format,
                                                      **kwargs)

    legend_labels = sop.to_flat_list(legend_labels1 + legend_labels2)

    if start_to_one:
        prices_ax1 = prices_ax1.divide(dfo.get_first_nonnan_values(df=prices_ax1))

    fig = pts.plot_time_series_2ax(df1=prices_ax1,
                                   df2=prices_ax2,
                                   trend_line=trend_line,
                                   var_format=var_format,
                                   title=title,
                                   legend_labels=legend_labels,
                                   is_logs=is_logs,
                                   ax=ax,
                                   **kwargs)
    return fig


def plot_rolling_perf_stat(prices: Union[pd.Series, pd.DataFrame],
                           rolling_perf_stat: RollingPerfStat = RollingPerfStat.SHARPE,
                           time_period: da.TimePeriod = None,
                           roll_periods: int = 260,
                           roll_freq: str = None,
                           legend_stats: pts.LegendStats = pts.LegendStats.AVG_LAST,
                           var_format: str = '{:.2f}',
                           title: Optional[str] = None,
                           regime_benchmark_str: str = None,
                           pivot_prices: pd.Series = None,
                           regime_params: BenchmarkReturnsQuantileRegimeSpecs = BenchmarkReturnsQuantileRegimeSpecs(),
                           perf_params: PerfParams = None,
                           ax: plt.Subplot = None,
                           **kwargs
                           ) -> plt.Figure:
    """
    plot rolling performance
    """
    df = compute_rolling_perf_stat(prices=prices,
                                   rolling_perf_stat=rolling_perf_stat,
                                   roll_periods=roll_periods,
                                   roll_freq=roll_freq)

    if time_period is not None:
        df = time_period.locate(df)

    fig = pts.plot_time_series(df=df,
                               legend_stats=legend_stats,
                               var_format=var_format,
                               title=title,
                               ax=ax,
                               **kwargs)

    if regime_benchmark_str is not None and regime_params is not None:
        add_bnb_regime_shadows(ax=ax,
                               data_df=prices.reindex(index=df.index, method='ffill'),
                               pivot_prices=pivot_prices,
                               benchmark=regime_benchmark_str,
                               regime_params=regime_params,
                               perf_params=perf_params)

    return fig


class UnitTests(Enum):
    PERFORMANCE_LABELS = 1
    PRICE = 2
    PRICE_WITH_DD = 3


def run_unit_test(unit_test: UnitTests):

    from qis.test_data import load_etf_data
    prices = load_etf_data().dropna()

    if unit_test == UnitTests.PERFORMANCE_LABELS:
        this = get_performance_labels_for_stats(prices=prices, perf_stats_labels=[PerfStat.PA_RETURN,
                                                                                  PerfStat.VOL,
                                                                                  PerfStat.SHARPE_RF0,
                                                                                  PerfStat.MAX_DD])
        print(this)

    elif unit_test == UnitTests.PRICE:
        perf_params = PerfParams(freq='B')
        plot_prices(prices=prices, perf_params=perf_params)

    elif unit_test == UnitTests.PRICE_WITH_DD:
        perf_params = PerfParams(freq='ME')
        plot_prices_with_dd(prices=prices,
                            regime_benchmark_str=prices.columns[0],
                            perf_params=perf_params)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.PERFORMANCE_LABELS

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

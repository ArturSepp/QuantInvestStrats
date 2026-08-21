"""
price and NAV panels, each labelled with the performance statistics of the window it draws.

``plot_prices`` is the base panel: series are rebased - to 1.0 at the first observation under
``start_to_one``, at the last under ``end_to_one`` - and every legend entry carries the
``PerfStat`` members listed in ``perf_stats_labels``, computed by ``compute_ra_perf_table`` over
the plotted window, so the chart and its numbers cannot disagree. ``plot_prices_with_dd`` stacks
that panel above the running drawdown, ``plot_prices_2ax`` splits two groups across a left and a
right axis, and ``plot_rolling_perf_stat`` draws a ``RollingPerfStat`` through time.

Passing ``regime_benchmark`` shades the axis by benchmark regime; ``plot_prices_with_dd`` also
accepts ``pivot_prices`` in its place, the other panels do not. The statistics are computed in
``qis/perfstats/``, apart from the rolling panel's ``compute_rolling_perf_stat``, taken from
``qis/models/stats/rolling_stats.py``; the unlabelled line primitive is
``qis/plots/time_series.py`` and the shared arguments are in ``qis/docs/plotting_kwargs.md``.
"""
# packages
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, List, Tuple, Optional
from enum import Enum

# qis
import qis.utils.dates as da
import qis.utils.df_ops as dfo
import qis.utils.struct_ops as sop
import qis.perfstats.perf_stats as pt
from qis.models.stats.rolling_stats import RollingPerfStat, compute_rolling_perf_stat
from qis.perfstats.config import PerfStat, PerfParams
from qis.perfstats.regime_classifier import BenchmarkReturnsQuantilesRegime

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
                perf_stats_labels: Optional[List[PerfStat]] = (PerfStat.PA_RETURN, PerfStat.VOL, PerfStat.SHARPE_RF0, ),
                perf_params: PerfParams = None,
                regime_benchmark: str = None,  # to add regimes
                pivot_prices: pd.Series = None,
                regime_classifier: BenchmarkReturnsQuantilesRegime = BenchmarkReturnsQuantilesRegime(),
                var_format: str = '{:,.1f}',
                digits_to_show: int = 1,
                sharpe_format: str = '{:.2f}',
                x_date_freq: str = 'YE',
                trend_line: put.TrendLine = put.TrendLine.NONE,
                is_log: bool = False,
                resample_freq: str = None,
                start_to_one: bool = True,
                end_to_one: bool = False,
                title: str = None,
                ax: plt.Subplot = None,
                **kwargs
                ) -> plt.Figure:
    """
    plot price or NAV series with performance statistics in the legend.

    The legend is the point of this plot: each series is labelled with the statistics in
    ``perf_stats_labels`` computed over the plotted window, so the chart and the numbers cannot
    disagree. Series are rebased so that levels are comparable across instruments.

    Arguments shared with every ``plot_*`` function — ``ax``, ``title``, ``var_format``,
    ``x_date_freq``, ``fontsize``, ``colors`` and the rest — are documented in
    ``qis/docs/plotting_kwargs.md``.

    Args:
        prices: price or NAV levels indexed by date, one column per instrument. A Series is
            promoted to a one-column frame
        perf_stats_labels: statistics to append to each legend entry, computed on the plotted
            window. None labels with the column name alone
        perf_params: annualisation, frequency and rate conventions for those statistics.
            None uses the defaults of :class:`PerfParams`
        regime_benchmark: column name whose returns classify the regime shading. None draws
            no shading
        pivot_prices: benchmark levels for the regime classification when the benchmark is not
            one of the plotted columns
        regime_classifier: how the benchmark return is mapped to a regime
        var_format: format for the price levels in the legend
        digits_to_show: significant digits for the performance statistics
        sharpe_format: format for Sharpe ratios, which conventionally carry more digits than
            the other statistics
        x_date_freq: tick frequency on the date axis
        trend_line: trend line drawn through each series
        is_log: plot the vertical axis on a log scale, so that equal vertical distances are
            equal relative moves
        resample_freq: resample the prices before plotting, forward-filling. None plots at the
            input frequency
        start_to_one: rebase every series to 1.0 at its first observation
        end_to_one: rebase every series to 1.0 at its last observation instead, which compares
            the paths that led to the same endpoint. Takes precedence over ``start_to_one``
        title: axis title
        ax: axis to draw on; None creates a figure

    Returns:
        the figure drawn on, whether or not ``ax`` was supplied
    """
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    if perf_stats_labels is not None:
        legend_labels = get_performance_labels_for_stats(prices=prices,
                                                         perf_stats_labels=perf_stats_labels,
                                                         perf_params=perf_params,
                                                         digits_to_show=digits_to_show,
                                                         sharpe_format=sharpe_format,
                                                         **kwargs)
    else:
        legend_labels = prices.columns.to_list()

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

    if regime_benchmark is not None and regime_classifier is not None:
        add_bnb_regime_shadows(ax=ax,
                               data_df=prices,
                               pivot_prices=pivot_prices,
                               benchmark=regime_benchmark,
                               regime_classifier=regime_classifier)
    return fig


def plot_prices_with_dd(prices: Union[pd.DataFrame, pd.Series],
                        perf_stats_labels: List[PerfStat] = (PerfStat.PA_RETURN, PerfStat.VOL,
                                                             PerfStat.SHARPE_RF0),
                        perf_params: PerfParams = None,
                        regime_benchmark: str = None,  # to add regimes
                        pivot_prices: pd.Series = None,
                        regime_classifier: BenchmarkReturnsQuantilesRegime = BenchmarkReturnsQuantilesRegime(),
                        var_format: str = '{:,.1f}',
                        dd_format: str = '{:.0%}',
                        digits_to_show: int = 1,
                        start_to_one: bool = True,
                        sharpe_format: str = '{:.2f}',
                        x_date_freq: str = 'YE',
                        is_log: bool = False,
                        remove_xticklabels_ax1: bool = True,
                        title: str = 'Performance',
                        dd_title: str = 'Running Drawdown',
                        dd_legend_type: dra.DdLegendType = dra.DdLegendType.SIMPLE,
                        axs: List[plt.Subplot] = None,
                        **kwargs
                        ) -> plt.Figure:
    """
    plot price series above their running drawdown, on a shared date axis.

    Two stacked panels: :func:`plot_prices` on top, the running drawdown from the prior peak
    below. The pairing is the point — a performance line alone hides the path, and the drawdown
    panel puts the worst of it directly under the level that produced it.

    Arguments shared with every ``plot_*`` function are documented in
    ``qis/docs/plotting_kwargs.md``. Arguments in common with :func:`plot_prices` mean the same
    thing here and are passed straight through.

    Args:
        prices: price or NAV levels indexed by date, one column per instrument
        perf_stats_labels: statistics appended to each legend entry of the upper panel
        perf_params: annualisation, frequency and rate conventions for those statistics
        regime_benchmark: column name whose returns classify the regime shading on both panels
        pivot_prices: benchmark levels when the benchmark is not one of the plotted columns
        regime_classifier: how the benchmark return is mapped to a regime
        var_format: format for the price levels in the upper legend
        dd_format: format for the drawdown, conventionally a percentage
        digits_to_show: significant digits for the performance statistics
        start_to_one: rebase every series to 1.0 at its first observation
        sharpe_format: format for Sharpe ratios
        x_date_freq: tick frequency on the shared date axis
        is_log: log scale on the price panel only; the drawdown panel is always linear
        remove_xticklabels_ax1: drop the date labels from the upper panel, since the two panels
            share an axis and one set of labels is enough
        title: title of the price panel
        dd_title: title of the drawdown panel
        dd_legend_type: what the drawdown legend reports - the maximum alone, or the full
            drawdown statistics
        axs: the two axes to draw on, price first. None creates a two-panel figure

    Returns:
        the figure drawn on, or None when ``axs`` was supplied
    """
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
                start_to_one=start_to_one,
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

    if (regime_benchmark is not None or pivot_prices is not None) and regime_classifier is not None:
        for ax in axs:
            add_bnb_regime_shadows(ax=ax,
                                   data_df=prices,
                                   pivot_prices=pivot_prices,
                                   benchmark=regime_benchmark,
                                   regime_classifier=regime_classifier,
                                   perf_params=perf_params)
    return fig


def plot_prices_with_fundamentals(prices: Union[pd.DataFrame, pd.Series],
                                  volumes: Union[pd.DataFrame, pd.Series],
                                  mcap: Union[pd.DataFrame, pd.Series],
                                  perf_stats_labels: List[PerfStat] = (PerfStat.PA_RETURN, PerfStat.VOL, PerfStat.SHARPE_RF0, ),
                                  perf_params: PerfParams = None,
                                  regime_benchmark: str = None,  # to add regimes
                                  pivot_prices: pd.Series = None,
                                  regime_classifier: BenchmarkReturnsQuantilesRegime = BenchmarkReturnsQuantilesRegime(),
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

    if regime_benchmark is not None and regime_classifier is not None:
        for ax in axs:
            add_bnb_regime_shadows(ax=ax,
                                   data_df=prices,
                                   pivot_prices=pivot_prices,
                                   benchmark=regime_benchmark,
                                   regime_classifier=regime_classifier,
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
                                   trend_line1=trend_line,
                                   trend_line2=trend_line,
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
                           title: Optional[str] = None,
                           regime_benchmark: str = None,
                           pivot_prices: pd.Series = None,
                           regime_classifier: BenchmarkReturnsQuantilesRegime = BenchmarkReturnsQuantilesRegime(),
                           perf_params: PerfParams = None,
                           ax: plt.Subplot = None,
                           **kwargs
                           ) -> plt.Figure:
    """
    plot rolling performance
    """
    df, title_ps = compute_rolling_perf_stat(prices=prices,
                                             rolling_perf_stat=rolling_perf_stat,
                                             roll_periods=roll_periods,
                                             roll_freq=roll_freq)

    if time_period is not None:
        df = time_period.locate(df)

    fig = pts.plot_time_series(df=df,
                               legend_stats=legend_stats,
                               title=title or title_ps,
                               ax=ax,
                               **sop.update_kwargs(kwargs, dict(var_format=rolling_perf_stat.value[1])))

    if regime_benchmark is not None and regime_classifier is not None:
        add_bnb_regime_shadows(ax=ax,
                               data_df=prices.reindex(index=df.index, method='ffill'),
                               pivot_prices=pivot_prices,
                               benchmark=regime_benchmark,
                               regime_classifier=regime_classifier,
                               perf_params=perf_params)

    return fig

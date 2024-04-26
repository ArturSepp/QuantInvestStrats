# packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Union, Tuple, Optional, Dict
from enum import Enum

# qis
import qis.utils.struct_ops as sop
import qis.plots.utils as put
from qis.plots.utils import LegendStats, TrendLine, LastLabel
from qis.perfstats.desc_table import compute_desc_table, DescTableType


def plot_time_series(df: Union[pd.Series, pd.DataFrame],
                     linestyles: List[str] = None,
                     linewidth: float = 1.0,
                     x_date_freq: Union[str, None] = 'QE',
                     date_format: str = '%d-%b-%y',
                     legend_title: str = None,
                     legend_loc: Optional[Union[str, bool]] = 'upper left',
                     last_label: LastLabel = LastLabel.NONE,
                     sort_by_value_stretch_factor: float = 1.0,
                     trend_line: Optional[TrendLine] = TrendLine.NONE,
                     trend_line_colors: List[str] = None,
                     legend_stats: LegendStats = LegendStats.AVG_LAST,
                     desc_table_type: DescTableType = DescTableType.NONE,
                     legend_labels: List[str] = None,
                     indices_for_shaded_areas: Dict[str, Tuple[int, int]] = None,
                     xlabel: str = None,
                     ylabel: str = None,
                     var_format: Optional[str] = '{:,.2f}',
                     markers: List[str] = False,
                     title: Union[str, bool] = None,
                     fontsize: int = 10,
                     markersize: int = None,
                     colors: List[str] = None,
                     x_limits: Tuple[Union[float, None], Union[float, None]] = None,
                     y_limits: Tuple[Optional[float], Optional[float]] = None,
                     is_log: bool = False,
                     ax: plt.Subplot = None,
                     **kwargs
                     ) -> Optional[plt.Figure]:

    data1 = df.copy()
    if isinstance(data1, pd.DataFrame):
        pass
    elif isinstance(data1, pd.Series):
        data1 = data1.to_frame()
    else:
        raise TypeError(f"unsuported data type {type(data1)}")
    columns = data1.columns

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    if colors is None:
        colors = put.get_n_colors(n=len(columns), **kwargs)

    sns.lineplot(data=data1, palette=colors, dashes=False, markers=markers,
                 markersize=markersize,
                 linewidth=linewidth, ax=ax)

    # add tredlines
    if trend_line_colors is None:
        trend_line_colors = colors
    if trend_line in [TrendLine.ZERO_SHADOWS, TrendLine.ABOVE_ZERO_SHADOWS]:
        for column, color in zip(columns, trend_line_colors):
            x0 = data1.index[0]
            x1 = data1.index[-1]
            y0 = 0
            y1 = 0
            x = [x0, x1]
            y = [y0, y1]
            ax.plot(x, y,
                    color='black',
                    linestyle='-',
                    transform=ax.transData,
                    lw=linewidth)

            x_ = (data1.index - data1.index[0]).days
            slope = (y1 - y0) / (x_[-1] - x_[0])
            y_line = [slope * x + y0 for x in x_]
            y = data1[column]
            if trend_line == TrendLine.ZERO_SHADOWS:
                ax.fill_between(data1.index, y, y_line, where=y_line >= y,
                            facecolor=color, interpolate=True, alpha=0.2, lw=linewidth)
            else:
                ax.fill_between(data1.index, y, y_line, where=y_line <= y,
                                facecolor=color, interpolate=True, alpha=0.2, lw=linewidth)

    elif trend_line in [TrendLine.AVERAGE, TrendLine.AVERAGE_SHADOWS]:
        for column, color in zip(columns, trend_line_colors):
            data2 = data1[column].dropna()  # exclude nans from showing the average lines
            if data2.empty:  # skip this columns from ere
                continue
            else:
                average = np.nanmean(data2.to_numpy())

            x0 = data2.index[0]
            x1 = data2.index[-1]
            y0 = average
            y1 = average
            x = [x0, x1]
            y = [y0, y1]

            ax.plot(x, y,
                    color=color,
                    linestyle=':',
                    transform=ax.transData,
                    linewidth=linewidth)

            if trend_line == TrendLine.AVERAGE_SHADOWS:
                x_ = (data2.index - data2.index[0]).days
                slope = (y1 - y0) / (x_[-1] - x_[0])
                y_line = [slope * x + y0 for x in x_]
                y = data2.to_numpy()
                ax.fill_between(data2.index, y, y_line, where=y_line >= y,
                                facecolor=color, interpolate=True, alpha=0.2, lw=linewidth)

    elif trend_line in [TrendLine.TREND_LINE, TrendLine.TREND_LINE_SHADOWS]:
        for column, color in zip(columns, trend_line_colors):
            y = data1[column].dropna()
            x0 = y.first_valid_index() or y.index[0]  # if all are nons
            x1 = y.index[-1]
            y0 = y[x0]
            y1 = y.iloc[-1]
            x = [x0, x1]
            y = [y0, y1]
            ax.plot(x, y,
                    color=color,
                    linestyle='--',
                    transform=ax.transData,
                    linewidth=linewidth)

            if trend_line == TrendLine.TREND_LINE_SHADOWS:
                x_ = (data1.index - data1.index[0]).days
                slope = (y1 - y0) / (x_[-1] - x_[0])
                y_line = [slope * x + y0 for x in x_]
                y = data1[column]
                ax.fill_between(data1.index, y, y_line, where=y_line >= y,
                                facecolor=color, interpolate=True, alpha=0.2, lw=linewidth)

    # add last labels
    if last_label in [LastLabel.AVERAGE_VALUE, LastLabel.AVERAGE_VALUE_SORTED]:
        average_dict = {}
        for column, color in zip(columns, colors):
            data2 = data1[column].dropna()  # exclude nans from showing the average lines

            if data2.empty:  # skip this columns from ere
                continue
            else:
                average = np.nanmean(data2.to_numpy())

            if var_format is not None:
                average_str = var_format.format(average)
            else:
                average_str = average
            y1 = average
            variable_label = f"{column}, average={average_str}"
            average_dict.update({y1: [variable_label, color]})

        x1 = data1.index[-1]
        ymin, ymax = ax.get_ylim()
        mid = sort_by_value_stretch_factor * (ymax - ymin)
        if last_label == LastLabel.AVERAGE_VALUE_SORTED:
            pivot_dict = sorted(average_dict)
            locs = np.linspace(pivot_dict[0], sort_by_value_stretch_factor*mid, len(pivot_dict), endpoint=True)
        else:
            pivot_dict = average_dict
            locs = [dict for dict in pivot_dict]

        for key, loc in zip(pivot_dict, locs):
            ax.annotate(average_dict[key][0],
                        xy=(x1, key), xytext=(x1, loc),
                        fontsize=fontsize, weight='normal', color=average_dict[key][1],
                        textcoords='data', ha='left', va='bottom',
                        bbox={'boxstyle': 'round,pad=0.5', 'fc': average_dict[key][1], 'alpha': 0.1},
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    elif last_label in [LastLabel.LAST_VALUE, LastLabel.LAST_VALUE_SORTED]:
        last_dict = {}
        for column, color in zip(columns, colors):
            y = data1[column].dropna()
            if last_label == LastLabel.LAST_VALUE:
                if var_format is not None:
                    if len(y.index) > 0:
                        variable_str = var_format.format(y.iloc[-1])
                    else:
                        variable_str = 'nan'
                else:
                    variable_str = y.iloc[-1]
                variable_label = f"{column}, last = {variable_str}"

                if len(y.index) > 0:
                    y1 = y.iloc[-1]
                else:
                    y1 = np.nan
                last_dict.update({y1: [variable_label, color]})

        # plot dicts sorted by last value
        x1 = data1.index[-1]
        ymin, ymax = ax.get_ylim()
        mid = sort_by_value_stretch_factor * (ymax - ymin)

        if last_label == LastLabel.LAST_VALUE_SORTED:
            pivot_dict = sorted(last_dict)
            locs = np.linspace(pivot_dict[0], sort_by_value_stretch_factor*mid, len(pivot_dict), endpoint=True)
        else:
            pivot_dict = last_dict
            locs = [dict for dict in pivot_dict]

        for key, loc in zip(pivot_dict,locs):
            ax.annotate(last_dict[key][0],
                        xy=(x1, key), xytext=(x1, loc),
                        fontsize=fontsize, weight ='normal', color = last_dict[key][1],
                        textcoords='data', ha='left', va='bottom',
                        bbox={'boxstyle': 'round,pad=0.5', 'fc': last_dict[key][1], 'alpha': 0.1},
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    if indices_for_shaded_areas is not None:
        for col, indxs in indices_for_shaded_areas.items():
            y0 = data1.iloc[:, indxs[0]]
            y1 = data1.iloc[:, indxs[1]]
            ax.fill_between(data1.index, y0, y1, where=y0 >= y1,
                            facecolor=col, alpha=0.2,
                            interpolate=True)

    if legend_loc is not None:
        if legend_labels is None:
            if desc_table_type != DescTableType.NONE:  # use get_legend_with_stats_table
                stats_table = compute_desc_table(df=data1,
                                                 desc_table_type=desc_table_type,
                                                 var_format=var_format)
                put.set_legend_with_stats_table(stats_table=stats_table,
                                                ax=ax,
                                                colors=colors,
                                                legend_loc=legend_loc,
                                                fontsize=fontsize,
                                                **kwargs)
            else:  # generate legend_labels
                legend_labels = put.get_legend_lines(data=data1, legend_stats=legend_stats, var_format=var_format)

        if legend_labels is not None:
            put.set_legend(ax=ax,
                           labels=legend_labels,
                           colors=colors,
                           legend_loc=legend_loc,
                           legend_title=legend_title,
                           fontsize=fontsize,
                           **kwargs)

    else:
        ax.legend().set_visible(False)

    if linestyles is not None:
        put.set_linestyles(ax=ax, linestyles=linestyles)

    if x_date_freq is not None and isinstance(data1.index, pd.DatetimeIndex):
        put.set_date_on_axis(data=data1,
                             ax=ax,
                             x_date_freq=x_date_freq,
                             date_format=date_format,
                             fontsize=fontsize,
                             **kwargs)

    if y_limits is not None:
        put.set_y_limits(ax=ax, y_limits=y_limits)

    if x_limits is not None:
        put.set_x_limits(ax=ax, x_limits=x_limits)

    if var_format is not None:
        put.set_ax_ticks_format(ax=ax, fontsize=fontsize, xvar_format=None, yvar_format=var_format, **kwargs)

    if isinstance(data1.index, pd.DatetimeIndex):
        put.set_ax_tick_labels(ax=ax, skip_y_axis=True, fontsize=fontsize, **kwargs)

    put.set_ax_xy_labels(ax=ax, xlabel=xlabel, ylabel=ylabel, fontsize=fontsize, **kwargs)
    put.set_spines(ax=ax, **kwargs)

    if is_log:
        ax.set_yscale('log')

    if title is not None:
        put.set_title(ax=ax, title=title, fontsize=fontsize, **kwargs)

    return fig


def plot_time_series_2ax(df1: Union[pd.Series, pd.DataFrame],
                         df2: Union[pd.Series, pd.DataFrame],
                         legend_loc: Optional[str] = 'upper left',
                         legend_stats: LegendStats = LegendStats.NONE,
                         legend_stats2: LegendStats = LegendStats.NONE,
                         title: Optional[str] = None,
                         var_format: str = '{:,.0f}',
                         var_format_yax2: str = '{:,.0f}',
                         ylabel1: str = None,
                         ylabel2: str = None,
                         x_date_freq: Union[str, None] = 'QE',
                         legend_labels: List[str] = None,
                         linestyles: List[str] = None,
                         linestyles_ax2: List[str] = None,
                         y_limits: Tuple[Optional[float], Optional[float]] = None,
                         y_limits_ax2: Tuple[Optional[float], Optional[float]] = None,
                         trend_line1: put.TrendLine = put.TrendLine.NONE,
                         trend_line2: put.TrendLine = put.TrendLine.NONE,
                         yvar_major_ticks1: np.ndarray = None,
                         yvar_major_ticks2: np.ndarray = None,
                         colors: List[str] = None,
                         fontsize: int = 10,
                         x_rotation: int = 90,
                         is_logs: Tuple[bool, bool] = (False, False),
                         ax: plt.Subplot = None,
                         **kwargs
                         ) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    if isinstance(df1, pd.Series):
        df1 = df1.to_frame()
    if isinstance(df2, pd.Series):
        df2 = df2.to_frame()

    ncol1 = len(df1.columns)
    ncol2 = len(df2.columns)
    if colors is None:
        colors = put.get_n_colors(n=ncol1 + ncol2, **kwargs)

    ax_twin = ax.twinx()
    plot_time_series(df=df1,
                     legend_loc=None,
                     colors=colors[:ncol1],
                     var_format=None,
                     linestyles=linestyles,
                     trend_line=trend_line1,
                     x_date_freq=x_date_freq,
                     is_log=is_logs[0],
                     y_limits=y_limits,
                     ylabel=ylabel1,
                     ax=ax,
                     **kwargs)

    plot_time_series(df=df2,
                     legend_loc=None,
                     colors=colors[ncol1:],
                     var_format=None,
                     linestyles=linestyles_ax2,
                     trend_line=trend_line2,
                     x_date_freq=x_date_freq,
                     is_log=is_logs[1],
                     y_limits=y_limits_ax2,
                     ylabel=ylabel2,
                     ax=ax_twin,
                     **kwargs)

    put.set_ax_ticks_format(ax=ax, fontsize=fontsize, xvar_format=None, yvar_format=var_format, set_ticks=False,
                            yvar_major_ticks=yvar_major_ticks1, x_rotation=x_rotation, **kwargs)
    put.set_ax_ticks_format(ax=ax_twin, fontsize=fontsize, xvar_format=None, yvar_format=var_format_yax2, set_ticks=False,
                            yvar_major_ticks=yvar_major_ticks2, x_rotation=x_rotation, **kwargs)

    ax.tick_params(axis='x', which='both', bottom=False)

    if legend_loc is not None:
        if legend_labels is None:
            df1.columns = [f"{x} (left)" for x in df1.columns]
            df2.columns = [f"{x} (right)" for x in df2.columns]
            legend_labels1 = put.get_legend_lines(data=df1,
                                                  legend_stats=legend_stats,
                                                  var_format=var_format)
            legend_labels2 = put.get_legend_lines(data=df2,
                                                  legend_stats=legend_stats2,
                                                  var_format=var_format_yax2)
            legend_labels = sop.to_flat_list(legend_labels1 + legend_labels2)
        put.set_legend(ax=ax,
                       labels=legend_labels,
                       colors=colors,
                       fontsize=fontsize,
                       legend_loc=legend_loc,
                       **kwargs)

    if title is not None:
        put.set_title(ax=ax, title=title, fontsize=fontsize)

    return fig


def plot_lines_list(xy_datas: List[pd.DataFrame],
                    data_labels: List[List[str]],
                    colors: List[str] = None,
                    is_fill_annotations: bool = False,
                    xvar_format: str = '{:.1f}',
                    yvar_format: str = '{:.1f}',
                    xlabel: str = None,
                    ylabel: str = None,
                    title: str = None,
                    fontsize: int = 10,
                    x_limits: Tuple[Union[float, None], Union[float, None]] = None,
                    y_limits: Tuple[Union[float, None], Union[float, None]] = None,
                    ax: plt.Subplot = None,
                    **kwargs
                    ) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    if colors is None:
        colors = put.get_n_colors(n=len(xy_datas))

    for xy, color, labels in zip(xy_datas, colors, data_labels):
        sns.lineplot(x=xy.columns[0], y=xy.columns[1], data=xy, marker='None', color=color, ax=ax)

        for label, x, y in zip(labels, xy.iloc[:, 0].to_numpy(), xy.iloc[:, 1].to_numpy()):

            if label != '':
                # very specific to blended portfolios
                port_weight = label[-4:]
                if port_weight == '100%':
                    x_offset = 40  # 100 portfolios are shifted to right
                    y_offset = 0
                elif port_weight in ['-50%', ' 60%']:
                    x_offset = 40
                    y_offset = 15
                elif port_weight == '0-0%':
                    x_offset = 40
                    y_offset = 30
                else:
                    x_offset = 1
                    y_offset = 1

                if is_fill_annotations:
                    ax.annotate(
                        label,
                        xy=(x, y), xytext=(x_offset, y_offset),
                        textcoords='offset points', ha='right', va='bottom',
                        bbox=dict(boxstyle='round, pad=0.5', fc='blue', alpha=0.75),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                        fontsize=fontsize)
                else:
                    ax.annotate(
                        label,
                        xy=(x, y), xytext=(x_offset, y_offset),
                        textcoords='offset points', ha='right', va='bottom',
                        fontsize=fontsize)

    if y_limits is not None:
        put.set_y_limits(ax=ax, y_limits=y_limits)

    if x_limits is not None:
        put.set_x_limits(ax=ax, x_limits=x_limits)

    put.set_ax_ticks_format(ax=ax, fontsize=fontsize, xvar_format=xvar_format, yvar_format=yvar_format)
    put.set_ax_xy_labels(ax=ax, xlabel=xlabel, ylabel=ylabel, fontsize=fontsize, **kwargs)

    if title is not None:
        put.set_title(ax=ax, title=title, fontsize=fontsize)

    put.set_spines(ax=ax, **kwargs)

    return fig


class UnitTests(Enum):
    PRICES = 1
    PRICES_2AX = 2


def run_unit_test(unit_test: UnitTests):

    from qis.test_data import load_etf_data
    prices = load_etf_data().dropna()

    if unit_test == UnitTests.PRICES:
        fig, axs = plt.subplots(2, 1, figsize=(8, 6))
        global_kwargs = {'fontsize': 8,
                         'linewidth': 0.5,
                         'weight': 'normal',
                         'markersize': 1}
        plot_time_series(df=prices,
                         legend_stats=put.LegendStats.AVG_LAST,
                         last_label=LastLabel.AVERAGE_VALUE_SORTED,
                         trend_line=TrendLine.AVERAGE_SHADOWS,
                         ax=axs[0],
                         **global_kwargs)
        plot_time_series(df=prices,
                         legend_stats=put.LegendStats.AVG_LAST,
                         last_label=LastLabel.LAST_VALUE,
                         trend_line=TrendLine.AVERAGE_SHADOWS,
                         ax=axs[1],
                         **global_kwargs)

    elif unit_test == UnitTests.PRICES_2AX:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        global_kwargs = {'fontsize': 6,
                         'linewidth': 0.5,
                         'weight': 'normal',
                         'markersize': 1}

        plot_time_series_2ax(df1=prices.iloc[:, -1],
                             df2=prices.iloc[:, :-1],
                             legend_stats=put.LegendStats.AVG_LAST,
                             var_format_yax2='{:.0f}',
                             ax=ax,
                             **global_kwargs)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.PRICES

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

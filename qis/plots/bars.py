"""
bar plots
"""

# packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from typing import List, Tuple, Optional, Union
from enum import Enum

# qis
import qis.utils.struct_ops as sop
import qis.plots.utils as put
from qis.plots.utils import LegendStats


def plot_bars(df: Union[pd.DataFrame, pd.Series],
              stacked: bool = True,
              date_format: str = '%d-%b-%y',
              x_date_freq: str = 'QE',
              title: str = None,
              fontsize: int = 10,
              add_bar_values: bool = False,
              add_top_bar_values: bool = False,
              legend_stats: LegendStats = LegendStats.NONE,
              xvar_format: str = '{:.1%}',
              yvar_format: str = '{:,.2f}',
              x_rotation: int = 0,
              skip_y_axis: bool = False,
              legend_loc: Optional[str] = 'upper center',
              bbox_to_anchor: Optional[Tuple[float, float]] = None,
              y_limits: Tuple[Optional[float], Optional[float]] = None,
              totals: List[float] = None,
              is_top_totals: bool = False,
              totals_offset: Tuple[float, float] = (2.55, 5),
              colors: List[str] = None,
              legend_labels: List[str] = None,
              legend_colors: List[str] = None,
              vline_columns: List[int] = None,
              xlabel: str = None,
              ylabel: str = None,
              reverse_columns: bool = False,
              is_sns: bool = True,
              add_avg_line: bool = False,
              is_horizontal: bool = False,
              ax: plt.Subplot = None,
              **kwargs
              ) -> Optional[plt.Figure]:
    """
    plot bars
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    # convert to series to avoid melting
    # if isinstance(df, pd.DataFrame) and len(df.columns) == 1:
    #    df = df.iloc[:, 0]

    if colors is None:
        if isinstance(df, pd.Series):
            n = 1
        else:
            if stacked:
                n = len(df.columns)
            else:
                n = len(df.index)
        colors = put.get_n_colors(n=n, **kwargs)

    # use str for dates with plot.bar
    if isinstance(df.index, pd.DatetimeIndex) and isinstance(df, pd.Series):
        df, datalables = put.map_dates_index_to_str(data=df,
                                                    x_date_freq=x_date_freq,
                                                    date_format=date_format)
        df.index = datalables
        if is_horizontal:
            df.plot.barh(stacked=stacked, color=colors, edgecolor='none', ax=ax)
        else:
            df.plot.bar(stacked=stacked, color=colors, edgecolor='none', ax=ax)

    elif isinstance(df, pd.Series):
        # sns.barplot(x=df.index, y=df, palette=colors, ax=ax)
        if is_horizontal:
            df.plot.barh(stacked=stacked, color=colors, edgecolor='none', ax=ax)
        else:
            df.plot.bar(stacked=stacked, color=colors, edgecolor='none', ax=ax)

    else:  # need to melt for barplot
        value_name = ylabel or 'y'
        var_name = xlabel or 'x'
        df1 = df.melt(ignore_index=False, var_name=var_name, value_name=value_name)
        if is_sns:
            sns.barplot(x=df1.index, y=value_name, data=df1, hue=var_name,
                        palette=colors, edgecolor='none',
                        orient='h' if is_horizontal else 'v',
                        ax=ax)
        else:
            if is_horizontal:
                df.plot.barh(stacked=stacked, color=colors, edgecolor='none', ax=ax)
            else:
                df.plot.bar(stacked=stacked, color=colors, edgecolor='none', ax=ax)

    # put totals to bar and store locations
    x_locs = []
    x_mins = []
    x_maxs = []
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        y = p.get_y()
        x = p.get_x()

        x_loc = x+.2*width
        y_loc = y+.3*height if height > 0.0 else y+0.8*height
        if add_bar_values:
            if height != 0:
                ax.annotate(text=yvar_format.format(height), xy=(x_loc, y_loc), fontsize=fontsize, weight='normal')
        elif add_top_bar_values:
            if height != 0:
                ymin, ymax = ax.get_ylim()
                ax.annotate(text=yvar_format.format(height), xy=(x_loc, 0.95*ymax), fontsize=fontsize, weight='normal')

        if x not in x_locs:
            x_locs.append(x)  # take only one location per asset
            x_mins.append(x)
            x_maxs.append(x + width)

    if totals is not None:
        if is_top_totals:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim([ymin, ymax * 1.1])

        for total, x_loc, x_min, x_max in zip(totals, x_locs, x_mins, x_maxs):
            label = yvar_format.format(total)
            if is_top_totals:
                trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
                ax.text(x_min + 0.2 * (x_max - x_min), 0.975, label,
                        transform=trans, fontsize=fontsize, weight='normal')
            else:
                ax.hlines(xmin=x_min, xmax=x_max, y=total, linestyle='-', color='black', linewidth=2)
                ax.annotate(text=label, xytext=totals_offset, textcoords='offset points',
                            xy=(x_max, total),
                            fontsize=fontsize,
                            ha='left', va='top')

    if vline_columns is not None:
        for vline_column in vline_columns:
            ax.vlines([vline_column-0.5], *ax.get_ylim(), lw=1) # shift by 0.5 for visibility

    if legend_labels is not None:
        labels = legend_labels
    else:
        # handles, labels = ax.get_legend_handles_labels()
        labels = put.get_legend_lines(data=df, legend_stats=legend_stats, var_format=yvar_format)

    if legend_colors is not None:
        colors = legend_colors

    put.set_legend(ax=ax,
                   labels=labels,
                   colors=colors,
                   reverse_columns=reverse_columns,
                   bbox_to_anchor=bbox_to_anchor,
                   legend_loc=legend_loc,
                   fontsize=fontsize,
                   **kwargs)

    for line in ax.get_legend().get_lines():
        line.set_linewidth(4.0)

    if add_avg_line:
        avg = np.nanmean(df)
        ax.axhline(avg, color='coral', linewidth=2, linestyle='--', label='Average')
        xmin, xmax = ax.get_xlim()
        ax.text(xmax, avg, f"Average", fontsize=fontsize, weight='normal', color='coral')

    put.set_ax_xy_labels(ax=ax,
                         fontsize=fontsize,
                         xlabel=xlabel,
                         ylabel=ylabel,
                         **kwargs)

    put.set_ax_tick_params(ax=ax)
    local_kwargs = sop.update_kwargs(dict(yvar_format=yvar_format, xvar_format=xvar_format,  fontsize=fontsize), kwargs)
    put.set_ax_ticks_format(ax=ax, x_rotation=x_rotation, **local_kwargs)
    put.set_ax_tick_labels(ax=ax, fontsize=fontsize, skip_y_axis=skip_y_axis, **kwargs)

    if is_horizontal:
        ax.set_yticks(np.arange(len(df.index)), labels=df.index.to_list())
        ax.xaxis.set_tick_params(rotation=x_rotation)
        ax.invert_yaxis()  # labels read top-to-bottom
    else:
        ax.set_xticks(np.arange(len(df.index)), labels=df.index.to_list())
        ax.axhline(0, color='black', lw=1)

    if y_limits is not None:
        put.set_y_limits(ax=ax, y_limits=y_limits)

    put.set_spines(ax=ax, **kwargs)

    if title is not None:
        put.set_title(ax=ax, title=title, fontsize=fontsize)

    return fig


def plot_vbars(df: pd.DataFrame,
               title: Optional[str] = None,
               fontsize: int = 10,
               add_bar_values: bool = True,
               add_bar_perc_values: bool = False,
               var_format: str = '{:.1%}',
               legend_loc: Optional[str] = 'upper center',
               bbox_to_anchor: Optional[Tuple[float, float]] = (0.5, 1.04),
               totals: List[float] = None,
               colors: Union[List[str], List[Tuple[float, float, float]]] = None,
               legend_labels: List[str] = None,
               legend_colors: List[str] = None,
               x_rotation: int = 0,
               xmin_shift: Optional[float] = None,  # add shift to x-axis to left
               add_bar_value_at_mid: bool = True,
               add_total_bar: bool = True,
               add_total_to_index: bool = False,
               add_total_to_left: bool = False,
               is_category_names_colors: bool = True,
               x_step: Optional[float] = None,  # specify x-step
               x_limits: Tuple[Union[float, None], Union[float, None]] = None,
               reverse_columns: bool = True,
               rows_edge_lines: List[int] = None,
               ax: plt.Subplot = None,
               **kwargs
               ) -> plt.Figure:

    category_names = df.columns.to_list()

    if add_total_to_index and totals is not None:
        df.index = [f"{x} {var_format.format(total)}" for x, total in zip(df.index, totals)]

    results = {rdata[0]: rdata[1].to_list() for rdata in df.iterrows()}

    labels = list(results.keys())
    np_data = np.array(list(results.values()))
    totals = np.sum(np_data, axis=1)

    if colors is None:
        if is_category_names_colors:
            if len(df.columns) == 1:
                colors = put.compute_heatmap_colors(a=df.to_numpy())
            else:
                colors = put.compute_heatmap_colors(a=np.sum(df.to_numpy(), axis=1))
        else:
            colors = put.get_n_colors(n=len(df.columns))
    else:
        legend_colors = colors
        colors = np.tile(colors, len(category_names))

    if ax is None:
        height = put.calc_table_height(num_rows=len(df.index), scale=0.30)
        fig, ax = plt.subplots(figsize=(9.2, height))
    else:
        fig = None

    if add_bar_value_at_mid:
        bar_value_at_max = None
    else:
        bar_value_at_max = np.max(np.cumsum(np_data, axis=1))

    ax.invert_yaxis()

    # negative
    last_starts = None
    initial_starts = np.sum(np.where(np_data < 0.0, np_data, 0.0), axis=1)
    for i, colname in enumerate(category_names):
        if is_category_names_colors:
            col_colors = colors[i]
        else:
            col_colors = colors
        widths = np.where(np_data[:, i] < 0.0, np_data[:, i], 0.0)
        if last_starts is None:
            starts = initial_starts
            last_starts = initial_starts
        else:
            starts = last_starts
        last_starts = last_starts + np.abs(widths)
        ax.barh(labels, np.abs(widths), left=starts, height=0.5, label=colname, color=col_colors)

        if add_bar_values:
            xcenters = starts + np.abs(widths) / 2
            text_color = 'black'
            for y, (x, c) in enumerate(zip(xcenters, widths)):
                if not np.isclose(c, 0.0):
                    if add_bar_value_at_mid:
                        x_loc = x
                    else:
                        x_loc = bar_value_at_max

                    if add_bar_perc_values:
                        label = f"{var_format.format(c)} / {'{:.0%}'.format(c/totals[y])}"
                    else:
                        label = var_format.format(c)
                    ax.text(x_loc, y, label, ha='center', va='center', color=text_color, fontsize=fontsize)

    # positive
    last_starts = 0*last_starts
    for i, colname in enumerate(category_names):
        if is_category_names_colors:
            col_colors = colors[i]
        else:
            col_colors = colors

        widths = np.where(np_data[:, i] > 0.0, np_data[:, i], 0.0)
        if last_starts is None:
            starts = 0
            last_starts = widths
        else:
            starts = last_starts
            last_starts = last_starts + widths
        ax.barh(labels, widths, left=starts, height=0.5, label=colname, color=col_colors)

        if add_bar_values:
            xcenters = starts + widths / 2
            text_color = 'black'
            for y, (x, c) in enumerate(zip(xcenters, widths)):
                if not np.isclose(c, 0.0):
                    if add_bar_value_at_mid:
                        x_loc = x
                    else:
                        x_loc = bar_value_at_max
                    if add_bar_perc_values:
                        label = f"{var_format.format(c)} / {'{:.0%}'.format(c/totals[y])}"
                    else:
                        label = var_format.format(c)

                    ax.text(x_loc, y, label, ha='center', va='center', color=text_color, fontsize=fontsize)

    if xmin_shift is not None:
        xmin, xmax = ax.get_xlim()
        xmin_ = xmin + xmin_shift
        ax.set_xlim([xmin_, xmax])

    if add_total_bar:
        for idx, total in enumerate(totals):
            ax.vlines(x=total, ymin=idx-0.25, ymax=idx+0.25, linestyle='-', color='black', linewidth=2)

    if add_total_to_left:
        widths = np.nansum(np.where(np_data > 0.0, np_data, 0.0), axis=1)
        shift = np.maximum(0.2 * np.max(widths), 0.2)
        for idx, total in enumerate(totals):
            label = f"total: {var_format.format(total)}"
            ax.text(widths[idx]+shift, idx, label, ha='center', va='center', fontsize=fontsize)

    # legend
    if legend_labels is None:
        legend_labels = category_names
        # handles, labels = ax.get_legend_handles_labels()

    if legend_colors is not None:
        legend_colors = legend_colors
    else:
        legend_colors = colors

    # reverse
    if reverse_columns:
        legend_labels = legend_labels[::-1]
        legend_colors = legend_colors[::-1]

    put.set_legend(ax=ax,
                   labels=legend_labels,
                   colors=legend_colors,
                   legend_loc=legend_loc,
                   reverse_columns=True,
                   bbox_to_anchor=bbox_to_anchor,
                   fontsize=fontsize,
                   **kwargs)
    # increase line width
    for line in ax.get_legend().get_lines():
        line.set_linewidth(5.0)

    if x_limits is not None:
        put.set_x_limits(ax=ax, x_limits=x_limits)

    # ax.xaxis.set_visible(False)
    ax.grid(zorder=0, axis='x')
    ax.axvline(x=0, linewidth=2, color='orange')

    x_labels = [var_format.format(x) for x in ax.get_xticks()]
    put.set_ax_tick_labels(ax=ax,
                           x_labels=x_labels,
                           y_labels=labels,
                           fontsize=fontsize,
                           x_rotation=x_rotation,
                           **kwargs)

    put.set_spines(ax=ax, **kwargs)

    if x_step is not None:
        x_limits = (x_step*np.floor(np.min(np.cumsum(np.where(np_data < 0.0, np_data, 0.0), axis=1))/x_step),
                    x_step*np.ceil(np.max(np.cumsum(np.where(np_data > 0.0, np_data, 0.0), axis=1))/x_step))

        put.set_x_limits(ax=ax, x_limits=x_limits)

    if rows_edge_lines is not None:
        for rows_edge_line in rows_edge_lines:
            ax.axhline(y=rows_edge_line-0.5, color='black', alpha=0.5)

    if title is not None:
        put.set_title(ax=ax, title=title, fontsize=fontsize, **kwargs)

    return fig


class UnitTests(Enum):
    BARS2 = 1
    TOP_BOTTOM_RETURNS = 2
    VBAR_WEIGHTS = 3
    MONTHLY_RETURNS_BARS = 4


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.BARS2:

        n = 11
        index = [f"id{x+1} {x**2}" for x in range(n)]
        df1 = pd.DataFrame(np.linspace(0.0, 1.0, 11), index=index, columns=['data1'])
        df2 = pd.DataFrame(np.linspace(0.0, 0.5, 11), index=index, columns=['data2'])

        fig, axs = plt.subplots(1, 2, figsize=(8, 6), tight_layout=True)
        datas = [df1, df2]
        titles = ['Group Lasso', 'Lasso']
        for data, ax, title in zip(datas, axs, titles):
            plot_bars(df=data,
                      stacked=False,
                      skip_y_axis=True,
                      title=title,
                      legend_loc=None,
                      x_rotation=90,
                      ax=ax)
        put.align_y_limits_ax12(ax1=axs[0], ax2=axs[1], is_invisible_y_ax2=True)

    elif unit_test == UnitTests.TOP_BOTTOM_RETURNS:

        from qis.test_data import load_etf_data
        import qis.perfstats.returns as ret

        prices = load_etf_data().dropna().loc['2021', :]
        returns = ret.to_total_returns(prices=prices).sort_values()
        print(returns)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)

        plot_bars(df=returns,
                  stacked=False,
                  skip_y_axis=True,
                  legend_loc=None,
                  x_rotation=90,
                  ax=ax)

    elif unit_test == UnitTests.VBAR_WEIGHTS:
        desc_dict = {'f1': (0.5, 0.5),
                     'f2': (0.7, 0.3),
                     'f3': (1.0, 0.0),
                     'f4': (0.8, 0.2),
                     'f5': (0.35, 0.65),
                     'f6': (0.0, 1.0)}

        df = pd.DataFrame.from_dict(desc_dict, orient='index', columns=['as1', 'as2'])
        print(df)
        plot_vbars(df=df,
                   colors=put.get_n_colors(n=len(df.columns)),
                   bbox_to_anchor=(0.5, 1.25),
                   add_bar_values=False,
                   add_bar_value_at_mid=False,
                   add_total_bar=False)

    elif unit_test == UnitTests.MONTHLY_RETURNS_BARS:
        from qis.test_data import load_etf_data
        import qis.perfstats.returns as ret

        prices = load_etf_data().dropna().loc['2020':, :].iloc[:, :3]
        returns = ret.to_returns(prices=prices, freq='ME', drop_first=True)
        print(returns)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)

        plot_bars(df=returns,
                  stacked=False,
                  skip_y_axis=True,
                  x_rotation=90,
                  yvar_format='{:,.0%}',
                  date_format='%b-%y',
                  fontsize=6,
                  ax=ax)
    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.MONTHLY_RETURNS_BARS

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

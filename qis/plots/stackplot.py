# packages
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple
from enum import Enum

# qis
import qis.plots.utils as put
from qis.plots.utils import LegendStats


def plot_stack(df: pd.DataFrame,
               use_bar_plot: bool = False,
               is_yaxis_limit_01: bool = False,
               add_mean_levels: bool = False,
               add_cum_levels: bool = False,
               add_total_line: bool = False,
               colors: List[str] = None,
               step: Optional[str] = None,  # 'mid
               title: Optional[str] = None,
               baseline: str = 'zero',  # "zero", "sym", "wiggle", "weighted_wiggle"
               ncol: int = 1,
               legend_loc: Optional[str] = 'upper center',
               legend_labels: Optional[List[str]] = None,
               legend_stats: LegendStats = LegendStats.NONE,
               var_format: str = '{:.0%}',
               fontsize: int = 10,
               linewidth: float = 1.5,
               x_rotation: int = 90,
               reverse_columns: bool = False,
               x_date_freq: str = 'YE',
               date_format: str = '%b-%y',
               skip_y_axis: bool = True,
               bbox_to_anchor: Optional[Tuple[float, float]] = None,
               xlabel: str = None,
               ylabel: str = None,
               ax: plt.Subplot = None,
               **kwargs
               ) -> plt.Figure:

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    if isinstance(df.index, pd.DatetimeIndex):
        re_indexed_data, datalables = put.map_dates_index_to_str(data=df,
                                                                 x_date_freq=x_date_freq,
                                                                 date_format=date_format)
    else:
        re_indexed_data = df
        datalables = None

    if colors is None:
        colors = put.get_n_colors(n=len(re_indexed_data.columns))

    if use_bar_plot:  # plot bar apperas to look better for unconstraint plots
        re_indexed_data.plot.bar(stacked=True,
                                 color=colors, width=1.0, alpha=1.0, edgecolor='none',
                                 linewidth=0, ax=ax)
    else:
        ax.stackplot(re_indexed_data.index, re_indexed_data.T,
                     labels=re_indexed_data.columns, step=step, colors=colors,
                     baseline=baseline, edgecolor='none')

    # set x axes to nearest years
    ax.set_xlim(re_indexed_data.index[0], re_indexed_data.index[-1])

    if add_total_line:  # add total as line
        totals = re_indexed_data.sum(1)
        sns.lineplot(x=re_indexed_data.index, y=totals, marker='None', color='black', ax=ax)
        # legend_labels.append('Total')
        colors.append('black')

    # change axes labels, positions of each tick, relative to the indices of the x-values
    if datalables is not None:
        current_ticks = ax.get_xticks()
        ax.set_xticks(np.linspace(current_ticks[0], current_ticks[-1], len(datalables)))
        ax.set_xticklabels(datalables, rotation=90, fontsize=fontsize)

    if is_yaxis_limit_01:
        ax.set_ylim(0, 1)

    if add_mean_levels or add_cum_levels:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        cum_mean = 0.0
        cum_mean0 = 0.0
        handles, labels = ax.get_legend_handles_labels()

        for (idx, column), handle, label in zip(enumerate(re_indexed_data.columns), handles, labels):
            mean = np.mean(re_indexed_data[column].values)
            cum_mean = cum_mean + mean

            if add_mean_levels:
                vlabel = var_format.format(mean)
            else:
                vlabel = var_format.format(cum_mean)

            if add_mean_levels and is_yaxis_limit_01 is False:  # show absolute effect
                y_loc = mean
            else:  # show cumulative effect
                if column == re_indexed_data.columns[-1]:  # make it vidsiblae
                    y_loc = 1.0 - cum_mean0 if cum_mean0 < 0.5 else cum_mean0 + 0.5 * mean
                else:
                    y_loc = cum_mean
            cum_mean0 = cum_mean

            ax.axhline(y_loc, color='black', linestyle='--', linewidth=linewidth)

            color = mpl.colors.to_rgb(handle.get_facecolors()[0])
            ax.annotate(text=f"{label}={vlabel}", xy=(xmax, y_loc), fontsize=fontsize, weight='normal', color=color)

        y_annotation = 'Avg' if add_mean_levels else 'Total'
        ax.annotate(y_annotation, xy=(xmax, ymax), xytext=(1, 2), fontsize=fontsize, weight='normal',
                    textcoords='offset points', ha='left', va='bottom')

    put.set_ax_ticks_format(ax=ax, fontsize=fontsize, xvar_format=None, yvar_format=var_format)
    put.set_ax_tick_labels(ax=ax, x_rotation=x_rotation, fontsize=fontsize, skip_y_axis=skip_y_axis, **kwargs)
    put.set_ax_xy_labels(ax=ax, xlabel=xlabel, ylabel=ylabel, **kwargs)

    if legend_loc is not None:
        legend_title = None
        if legend_labels is None:
            legend_labels = put.get_legend_lines(data=re_indexed_data,
                                                 legend_stats=legend_stats,
                                                 var_format=var_format)
            if legend_stats in [put.LegendStats.LAST, put.LegendStats.FIRST_LAST_NON_ZERO]:
                legend_title = f"Total: last={var_format.format(re_indexed_data.sum(axis=1).iloc[-1])}"

        put.set_legend(ax=ax,
                       labels=legend_labels,
                       colors=colors,
                       legend_loc=legend_loc,
                       reverse_columns=reverse_columns,
                       ncol=ncol,
                       bbox_to_anchor=bbox_to_anchor,
                       fontsize=fontsize,
                       legend_title=legend_title,
                       **kwargs)

    if title is not None:
        put.set_title(ax=ax, title=title, fontsize=fontsize, **kwargs)

    put.set_spines(ax=ax, **kwargs)

    return fig


class UnitTests(Enum):
    WEIGHTS = 1


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.WEIGHTS:
        from qis.test_data import load_etf_data

        prices = load_etf_data().dropna().loc['2020':, :]
        weights = prices.divide(np.sum(prices, axis=1), axis=0)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)

        plot_stack(df=weights,
                   stacked=False,
                   x_rotation=90,
                   yvar_format='{:,.0%}',
                   date_format='%b-%y',
                   fontsize=6,
                   ax=ax)
    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.WEIGHTS

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

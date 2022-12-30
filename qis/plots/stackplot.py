# built in
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple

# qis
import qis.plots.utils as put
from qis.plots.utils import LegendLineType


def stackplot_timeseries(df: pd.DataFrame,
                         is_use_bar_plot: bool = False,
                         is_yaxis_limit_01: bool = False,
                         is_add_mean_levels: bool = False,
                         is_add_cum_levels: bool = False,
                         is_add_total_line: bool = False,
                         colors: List[str] = None,
                         step: Optional[str] = None,  # 'mid
                         title: Optional[str] = None,
                         baseline: str = 'zero',  # "zero", "sym", "wiggle", "weighted_wiggle"
                         ncol: int = 1,
                         legend_loc: Optional[str] = 'upper center',
                         legend_labels: Optional[List[str]] = None,
                         legend_line_type: LegendLineType = LegendLineType.NONE,
                         var_format: str = '{:.0%}',
                         fontsize: int = 10,
                         linewidth: float = 1.5,
                         x_date_freq: str = 'A',
                         x_rotation: int = 90,
                         is_reversed: bool = False,
                         date_format: str = '%b-%y',
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
    re_indexed_data, datalables = put.map_dates_index_to_str(data=df,
                                                             x_date_freq=x_date_freq,
                                                             date_format=date_format)

    if colors is None:
        colors = put.get_n_colors(n=len(re_indexed_data.columns))

    if is_use_bar_plot:  # plot bar apperas to look better for unconstraint plots
        re_indexed_data.plot.bar(stacked=True,
                                 color=colors, width=1.0, alpha=1.0, edgecolor='none',
                                 linewidth=0, ax=ax)
    else:
        ax.stackplot(re_indexed_data.index, re_indexed_data.T,
                     labels=re_indexed_data.columns, step=step, colors=colors, baseline=baseline)

    # set x axes to nearest years
    ax.set_xlim(re_indexed_data.index[0], re_indexed_data.index[-1])

    if is_add_total_line:  # add total as line
        totals = re_indexed_data.sum(1)
        sns.lineplot(x=re_indexed_data.index, y=totals, marker='None', color='black', ax=ax)
        # legend_labels.append('Total')
        colors.append('black')

    # change axes labels, positions of each tick, relative to the indices of the x-values
    current_ticks = ax.get_xticks()
    ax.set_xticks(np.linspace(current_ticks[0], current_ticks[-1], len(datalables)))
    ax.set_xticklabels(datalables, rotation=90, fontsize=fontsize)

    if is_yaxis_limit_01:
        ax.set_ylim(0, 1)

    if is_add_mean_levels or is_add_cum_levels:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        cum_mean = 0.0
        cum_mean0 = 0.0
        handles, labels = ax.get_legend_handles_labels()

        for (idx, column), handle, label in zip(enumerate(re_indexed_data.columns), handles, labels):
            mean = np.mean(re_indexed_data[column].values)
            cum_mean = cum_mean + mean

            if is_add_mean_levels:
                vlabel = var_format.format(mean)
            else:
                vlabel = var_format.format(cum_mean)

            if is_add_mean_levels and is_yaxis_limit_01 is False:  # show absolute effect
                y_loc = mean
            else: # show cumulative effect
                if column == re_indexed_data.columns[-1]:  # make it vidsiblae
                    y_loc = 1.0 - cum_mean0 if cum_mean0 < 0.5 else cum_mean0 + 0.5 * mean
                else:
                    y_loc = cum_mean
            cum_mean0 = cum_mean

            ax.axhline(y_loc, color='black', linestyle='--', linewidth=linewidth)

            color = mpl.colors.to_rgb(handle.get_facecolors()[0])
            ax.annotate(text=f"{label}={vlabel}", xy=(xmax, y_loc), fontsize=fontsize, weight='normal', color=color)

        y_annotation = 'Avg' if is_add_mean_levels else 'Total'
        ax.annotate(y_annotation, xy=(xmax, ymax), xytext=(1, 2), fontsize=fontsize, weight='normal',
                    textcoords='offset points', ha='left', va='bottom')

    put.set_ax_ticks_format(ax=ax, fontsize=fontsize, xvar_format=None, yvar_format=var_format)
    put.set_ax_tick_labels(ax=ax, x_rotation=x_rotation, fontsize=fontsize, skip_y_axis=True, **kwargs)
    put.set_ax_xy_labels(ax=ax, xlabel=xlabel, ylabel=ylabel, **kwargs)

    if legend_loc is not None:
        legend_title = None
        if legend_labels is None:
            legend_labels = put.get_legend_lines(data=re_indexed_data,
                                                 legend_line_type=legend_line_type,
                                                 var_format=var_format)
            if legend_line_type in [put.LegendLineType.LAST, put.LegendLineType.FIRST_LAST_NON_ZERO]:
                legend_title = f"Total: last={var_format.format(re_indexed_data.sum(axis=1).iloc[-1])}"

        put.set_legend(ax=ax,
                       labels=legend_labels,
                       colors=colors,
                       legend_loc=legend_loc,
                       is_reversed=is_reversed,
                       ncol=ncol,
                       bbox_to_anchor=bbox_to_anchor,
                       fontsize=fontsize,
                       legend_title=legend_title,
                       **kwargs)

    if title is not None:
        put.set_title(ax=ax, title=title, fontsize=fontsize, **kwargs)

    put.set_spines(ax=ax, **kwargs)

    return fig

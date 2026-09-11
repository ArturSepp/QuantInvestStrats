"""
stacked area or stacked bar charts of the columns of a frame. ``plot_stack`` is the only entry
point: ``use_bar_plot`` switches from ``ax.stackplot`` to a stacked bar chart, and
``add_mean_levels`` or ``add_cum_levels`` annotate per-column horizontal levels.
"""
# packages
import warnings
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, cast
from numpy.typing import NDArray
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
               ncols: int = 1,
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
    """Plot DataFrame columns as stacked areas or stacked bars.

    Args:
        df: Numeric values to stack. Nullable floating columns are supported; stacked-area
            rendering represents their missing values as NumPy ``nan``.
        use_bar_plot: Use pandas stacked bars instead of Matplotlib stacked areas.
        add_mean_levels: Annotate each column's mean over its observed values.
        add_cum_levels: Annotate cumulative observed-value column means.
        add_total_line: Draw the per-row total as a black line.
        colors: One color per column. The caller-owned list is not modified.

    Returns:
        The created figure, or None when the caller supplies ``ax``.
    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    if df.empty:
        warnings.warn('df is empty: no data to plot')
        return fig

    if isinstance(df.index, pd.DatetimeIndex):
        re_indexed_data, datalables = put.map_dates_index_to_str(data=df,
                                                                 x_date_freq=x_date_freq,
                                                                 date_format=date_format)
    else:
        re_indexed_data = df
        datalables = None

    if colors is None:
        colors = put.get_n_colors(n=len(re_indexed_data.columns))
    else:
        # Work on a local palette because the total-line path appends its display color.
        colors = colors.copy()

    if use_bar_plot:  # plot bar apperas to look better for unconstraint plots
        re_indexed_data.plot.bar(stacked=True,
                                 color=colors, width=1.0, alpha=1.0, edgecolor='none',
                                 linewidth=0, ax=ax)
    else:
        # Translate extension scalars only at the renderer boundary and leave caller data intact.
        stack_values = re_indexed_data.to_numpy(dtype=float, na_value=np.nan).T
        ax.stackplot(re_indexed_data.index, stack_values,
                     labels=re_indexed_data.columns, step=step, colors=colors,
                     baseline=baseline, edgecolor='none')

    # set x axes to nearest years
    ax.set_xlim(re_indexed_data.index[0], re_indexed_data.index[-1])

    if add_total_line:  # add total as line
        totals = re_indexed_data.sum(axis=1)
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
        _, labels = ax.get_legend_handles_labels()

        for (idx, column), label in zip(enumerate(re_indexed_data.columns), labels):
            # Normalize ordinary and nullable missing values, then average observed samples only.
            column_values = cast(
                NDArray[np.float64],
                re_indexed_data.iloc[:, idx].to_numpy(dtype=float, na_value=np.nan),
            )
            observed_values = column_values[~np.isnan(column_values)]
            mean = np.nan if observed_values.size == 0 else float(np.mean(observed_values))
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

            # The resolved column palette is common to area and bar artists.
            color = mcolors.to_rgb(colors[idx])
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
                       ncols=ncols,
                       bbox_to_anchor=bbox_to_anchor,
                       fontsize=fontsize,
                       legend_title=legend_title,
                       **kwargs)

    if title is not None:
        put.set_title(ax=ax, title=title, fontsize=fontsize, **kwargs)

    put.set_spines(ax=ax, **kwargs)

    return fig

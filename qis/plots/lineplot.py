"""
line plot
"""
# packages
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Union, Tuple, Optional, Dict
import qis.plots.utils as put


def plot_line(df: Union[pd.Series, pd.DataFrame],
              x: str = None,
              y: str = None,
              hue: str = None,
              linestyles: List[str] = None,
              linestyle: str = '-',
              linewidth: float = 1.0,
              legend_title: str = None,
              legend_loc: Optional[Union[str, bool]] = 'upper left',
              legend_stats: put.LegendStats = put.LegendStats.NONE,
              legend_labels: List[str] = None,
              x_labels: List[str] = None,
              xlabel: str = None,
              ylabel: str = None,
              title: str = None,
              xvar_format: Optional[str] = None,  # '{:,.2f}', # unless data is numerical
              yvar_format: Optional[str] = '{:,.2f}',
              markers: Union[str, List[str]] = False,
              fontsize: int = 10,
              colors: List[str] = None,
              x_limits: Tuple[Optional[float], Optional[float]] = None,
              y_limits: Tuple[Optional[float], Optional[float]] = None,
              is_log: bool = False,
              ax: plt.Subplot = None,
              **kwargs
              ) -> Optional[plt.Figure]:
    """
    line plot wrapper for sns, lineplot
    plot df using index
    x: str = None: x column
    y: str = None: y column
    hue: str = None: hue
    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    if df.empty:
        warnings.warn('df is empty: no data to plot')
        return fig

    if isinstance(df, pd.DataFrame):
        pass
    elif isinstance(df, pd.Series):
        df = df.to_frame()
    else:
        raise TypeError(f"unsuported data type {type(df)}")

    if x is not None:
        if y is None:
            raise ValueError(f"y column must be given with x column")
        else:
            if hue is not None:
                df = df[[x, y, hue]]
            else:
                df = df[[x, y]]

    if colors is None:
        if hue is None:
            colors = put.get_n_colors(n=len(df.columns), **kwargs)
        else:
            colors = put.get_n_colors(n=len(df[hue].unique()), **kwargs)

    sns.lineplot(data=df, x=x, y=y, hue=hue,
                 palette=colors, dashes=False, markers=markers, linestyle=linestyle, linewidth=linewidth,
                 style=hue,
                 ax=ax)

    if title is not None:
        put.set_title(ax=ax, title=title, fontsize=fontsize, **kwargs)

    if legend_loc is not None:
        if legend_labels is None:
            if hue is None:
                legend_labels = put.get_legend_lines(data=df, legend_stats=legend_stats, var_format=yvar_format)
            else:
                h, legend_labels = ax.get_legend_handles_labels()
        put.set_legend(ax=ax,
                       labels=legend_labels,
                       colors=colors,
                       legend_loc=legend_loc,
                       legend_title=legend_title,
                       markers=markers,
                       fontsize=fontsize,
                       **kwargs)

    else:
        ax.legend().set_visible(False)

    if linestyles is not None:
        put.set_linestyles(ax=ax, linestyles=linestyles)

    if y_limits is not None:
        put.set_y_limits(ax=ax, y_limits=y_limits)
    if x_limits is not None:
        put.set_x_limits(ax=ax, x_limits=x_limits)

    put.set_ax_ticks_format(ax=ax, fontsize=fontsize, xvar_format=xvar_format, yvar_format=yvar_format)

    if x_labels is None:
        put.set_ax_ticks_format(ax=ax, fontsize=fontsize, **kwargs)

    put.set_ax_xy_labels(ax=ax, xlabel=xlabel, ylabel=ylabel, fontsize=fontsize, **kwargs)
    put.set_spines(ax=ax, **kwargs)

    if is_log:
        ax.set_yscale('log')

    return fig


def plot_lines_list(xy_datas: Dict[str, pd.DataFrame],
                    data_labels: List[List[str]],
                    colors: List[str] = None,
                    markers: List[str] = None,
                    is_fill_annotations: bool = False,
                    xvar_format: str = '{:.1f}',
                    yvar_format: str = '{:.1f}',
                    xlabel: str = None,
                    ylabel: str = None,
                    title: str = None,
                    fontsize: int = 10,
                    x_limits: Tuple[Union[float, None], Union[float, None]] = None,
                    y_limits: Tuple[Union[float, None], Union[float, None]] = None,
                    legend_loc: str = 'upper left',
                    ax: plt.Subplot = None,
                    **kwargs
                    ) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    if colors is None:
        colors = put.get_n_colors(n=len(xy_datas.keys()))

    for idx, (key, xy) in enumerate(xy_datas.items()):
        color = colors[idx]
        labels = data_labels[idx]

        if markers is not None:
            marker = markers[idx % len(markers)] if markers else None
        else:
            marker = 'None'
        sns.lineplot(x=xy.columns[0], y=xy.columns[1], data=xy, marker=marker, color=color, ax=ax)

        for label, x, y in zip(labels, xy.iloc[:, 0].to_numpy(), xy.iloc[:, 1].to_numpy()):
            if label != '':
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

    if legend_loc is not None:
        legend_labels = list(xy_datas.keys())
        put.set_legend(ax=ax,
                       labels=legend_labels,
                       colors=colors,
                       legend_loc=legend_loc,
                       markers=markers,
                       fontsize=fontsize,
                       **kwargs)
    else:
        ax.legend().set_visible(False)

    put.set_spines(ax=ax, **kwargs)

    return fig

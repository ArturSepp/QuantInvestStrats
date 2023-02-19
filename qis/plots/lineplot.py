"""
line plot
"""

# packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Union, Tuple, Optional, Any
from enum import Enum

# qis
import qis.plots.utils as put


def plot_line(df: Union[pd.Series, pd.DataFrame],
              linestyles: List[str] = None,
              linestyle: Any = '-',
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
              ax: plt.Subplot = None,
              **kwargs
              ) -> Optional[plt.Figure]:

    if isinstance(df, pd.DataFrame):
        pass
    elif isinstance(df, pd.Series):
        df = df.to_frame()
    else:
        raise TypeError(f"unsuported data type {type(df)}")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    if colors is None:
        colors = put.get_n_colors(n=len(df.columns), **kwargs)

    sns.lineplot(data=df, palette=colors, dashes=False, markers=markers, linestyle=linestyle, linewidth=linewidth, ax=ax)

    if title is not None:
        put.set_title(ax=ax, title=title, fontsize=fontsize, **kwargs)

    if legend_loc is not None:
        if legend_labels is None:
            legend_labels = put.get_legend_lines(data=df, legend_stats=legend_stats, var_format=yvar_format)
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

    return fig


class UnitTests(Enum):
    LINEPLOT = 1
    MOVE_DATA = 2


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.LINEPLOT:
        x = np.linspace(0, 14, 100)
        y = np.sin(x)
        data = pd.Series(y, index=x, name='data')
        fig, axs = plt.subplots(2, 1, figsize=(8, 6))
        global_kwargs = dict(fontsize=12, linewidth=2.0, weight='normal', markersize=2)
        plot_line(df=data, legend_stats=put.LegendStats.AVG_LAST, ax=axs[0], **global_kwargs)
        plot_line(df=data, legend_stats=put.LegendStats.AVG_LAST,
                  linestyle='dotted',
                  ax=axs[1], **global_kwargs)

    elif unit_test == UnitTests.MOVE_DATA:
        with sns.axes_style("darkgrid"):
            fig, axs = plt.subplots(2, 1, figsize=(12, 6))
            global_kwargs = dict(fontsize=12, linewidth=2.0, weight='normal', markersize=2)
            index = ['06-07JUN22', '10-17JUN22', '17-24JUN22', '24JUN-01JUL22', '26JUN-30SEP22', '30SEP-30DEC22']
            data = [0.7644, 0.7602, 0.7306, 0.7524, 0.8192, 0.8204]
            df = pd.DataFrame(data, index=index, columns=['Expected Move % annualized'])
            print(df)
            markers = put.get_n_markers(n=1)
            plot_line(df=df,
                      title='ATM forward volatilities Implied from BTC MOVE contracts on 06-Jun-2022',
                      legend_stats=put.LegendStats.NONE,
                      yvar_format='{:.0%}',
                      xvar_format=None,
                      markers=markers,
                      ax=axs[0],
                      **global_kwargs)

            plot_line(df=df,
                      title='ATM forward volatilities Implied from BTC MOVE contracts on 06-Jun-2022',
                      legend_stats=put.LegendStats.NONE,
                      yvar_format='{:.0%}',
                      xvar_format=None,
                      linewidth=0.,
                      markers=['s'] * len(df.columns),
                      ax=axs[1])

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.MOVE_DATA

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

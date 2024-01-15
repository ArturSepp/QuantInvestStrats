"""
errorbar plot
"""

# packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Union, Tuple, Optional
from enum import Enum

# qis
import qis.plots.utils as put


def plot_errorbar(df: Union[pd.Series, pd.DataFrame],
                  y_std_errors: Union[float, pd.Series, pd.DataFrame] = 0.5,
                  exact: pd.Series = None,  # can add exact solution
                  legend_title: str = None,
                  legend_loc: Optional[Union[str, bool]] = 'upper left',
                  xlabel: str = None,
                  ylabel: str = None,
                  var_format: Optional[str] = '{:.0f}',
                  title: Union[str, bool] = None,
                  fontsize: int = 10,
                  capsize: int = 10,
                  colors: List[str] = None,
                  exact_color: str = 'green',
                  exact_marker: str = "v",
                  y_limits: Tuple[Optional[float], Optional[float]] = None,
                  ax: plt.Subplot = None,
                  **kwargs
                  ) -> Optional[plt.Figure]:

    if isinstance(df, pd.DataFrame):
        pass
    elif isinstance(df, pd.Series):
        df = df.to_frame()
    else:
        raise TypeError(f"unsupported data type {type(df)}")

    columns = df.columns

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    if colors is None:
        colors = put.get_n_colors(n=len(columns), **kwargs)

    for idx, column in enumerate(columns):
        if isinstance(y_std_errors, pd.DataFrame):
            yerr = y_std_errors[column].to_numpy()  # columnwese
        elif isinstance(y_std_errors, pd.Series):
            yerr = y_std_errors.to_numpy()
        else:
            yerr = y_std_errors

        ax.errorbar(x=df.index, y=df[column].to_numpy(), yerr=yerr, color=colors[idx], fmt='o', capsize=capsize)

    if exact is not None:
        for idx, index in enumerate(df.index):
            put.add_scatter_points(ax=ax, label_x_y=[(index, exact[index])], color=exact_color,
                                   marker=exact_marker, **kwargs)
        labels = columns.to_list() + [exact.name]
        colors = colors + [exact_color]
        markers = ['o']*len(columns) + [exact_marker]
    else:
        labels = columns
        markers = ['o']*len(columns)

    if title is not None:
        put.set_title(ax=ax, title=title, fontsize=fontsize)

    if legend_loc is not None:
        put.set_legend(ax=ax,
                       markers=markers,
                       labels=labels,
                       colors=colors,
                       legend_loc=legend_loc,
                       legend_title=legend_title,
                       handlelength=0,
                       fontsize=fontsize,
                       **kwargs)

    else:
        ax.legend().set_visible(False)

    if y_limits is not None:
        put.set_y_limits(ax=ax, y_limits=y_limits)

    if var_format is not None:
        put.set_ax_ticks_format(ax=ax, fontsize=fontsize, xvar_format=None, yvar_format=var_format, **kwargs)
    else:
        put.set_ax_ticks_format(ax=ax, fontsize=fontsize, **kwargs)
    put.set_ax_xy_labels(ax=ax, xlabel=xlabel, ylabel=ylabel, fontsize=fontsize, **kwargs)
    put.set_spines(ax=ax, **kwargs)

    return fig


class UnitTests(Enum):
    ERROR_BAR = 1


def run_unit_test(unit_test: UnitTests):

    n = 10
    x = np.linspace(0, 10, n)
    dy = 0.8
    y1 = pd.Series(np.sin(x) + dy * np.random.randn(n), index=x, name='y1')
    y2 = pd.Series(np.cos(x) + dy * np.random.randn(n), index=x, name='y2')
    data = pd.concat([y1, y2], axis=1)

    if unit_test == UnitTests.ERROR_BAR:

        global_kwargs = {'fontsize': 8,
                         'linewidth': 0.5,
                         'weight': 'normal',
                         'markersize': 1}

        with sns.axes_style('darkgrid'):
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            plot_errorbar(df=data,
                          ax=ax,
                          **global_kwargs)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.ERROR_BAR

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

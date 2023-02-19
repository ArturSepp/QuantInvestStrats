"""
quantile-quantile plot
"""
# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as stats
from statsmodels import api as sm
from typing import List, Union, Tuple, Optional
from enum import Enum

# qis
import qis.perfstats.returns as ret
import qis.plots.utils as put
import qis.perfstats.desc_table as dsc


def plot_qq(df: Union[pd.DataFrame, pd.Series],
            colors: List[str] = None,
            markers: List[str] = None,
            legend_loc: str = 'upper left',
            var_format: str = '{:.2f}',
            is_drop_na: bool = True,
            fontsize: int = 10,
            markersize: int = 2,
            title: str = None,
            xlabel: str = 'Theoretical quantiles',
            ylabel: str = 'Empirical quantiles',
            desc_table_type: dsc.DescTableType = dsc.DescTableType.SHORT,
            legend_stats: put.LegendStats = put.LegendStats.NONE,
            x_limits: Tuple[Optional[float], Optional[float]] = None,
            y_limits: Tuple[Optional[float], Optional[float]] = None,
            ax: plt.Subplot = None,
            **kwargs
            ) -> plt.Figure:

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    if isinstance(df, pd.Series):
        df = df.to_frame()
        line = 'q'
    else:
        line= None

    if colors is None:
         colors = put.get_n_colors(n=len(df.columns), **kwargs)

    if markers is None:
        markers = len(df.columns) * ['o']

    for idx, column in enumerate(df.columns):
        data0 = df[column]
        if is_drop_na:
            data0 = data0.dropna()
        sm.qqplot(data0, stats.norm, fit=True, line=line, ax=ax,  fmt=colors[idx],
                  markerfacecolor=colors[idx], markeredgecolor=colors[idx], marker=markers[idx],
                  markersize=markersize)
    if line is None:
        sm.qqline(ax, line='45', fmt='-', color='red')

    if desc_table_type != dsc.DescTableType.NONE:
        stats_table = dsc.compute_desc_table(df=df,
                                         desc_table_type=desc_table_type,
                                         var_format=var_format)
        put.set_legend_with_stats_table(stats_table=stats_table,
                                        ax=ax,
                                        colors=colors,
                                        legend_loc=legend_loc,
                                        fontsize=fontsize,
                                        **kwargs)
    else:
        legend_labels = put.get_legend_lines(data=df,
                                             legend_stats=legend_stats,
                                             var_format=var_format)
        put.set_legend(ax=ax,
                       labels=legend_labels,
                       colors=colors,
                       legend_loc=legend_loc,
                       fontsize=fontsize,
                       **kwargs)

    put.set_ax_xy_labels(ax=ax, xlabel=xlabel, ylabel=ylabel, fontsize=fontsize, **kwargs)
    put.set_ax_ticks_format(ax=ax, xvar_format=var_format, yvar_format=var_format, fontsize=fontsize, **kwargs)
    put.set_spines(ax=ax, **kwargs)

    if y_limits is not None:
        put.set_y_limits(ax=ax, y_limits=y_limits)
    if x_limits is not None:
        put.set_x_limits(ax=ax, x_limits=x_limits)

    if title is not None:
        put.set_title(ax=ax, title=title, fontsize=fontsize)

    return fig


def plot_xy_qq(x: pd.Series,
               y: pd.Series,
               colors: List[str] = None,
               markers: List[str] = None,
               labels: List[str] = None,
               legend_loc: str = 'upper left',
               is_drop_na: bool = True,
               ax: plt.Subplot = None,
               **kwargs
               ) -> plt.Figure:

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    if colors is None:
         colors = put.get_n_colors(n=1)

    if is_drop_na:
        x = x.dropna()
        y = y.dropna()
    x = x.to_numpy()
    y = y.to_numpy()

    qs = np.linspace(0, 1, min(len(x), len(y)))

    x_qs = np.quantile(x, qs)
    y_qs = np.quantile(y, qs)
    ax.scatter(x_qs, y_qs, c=colors[0])

    sm.qqline(ax, line='45', fmt='k--')

    put.set_legend(ax=ax,
                   labels=labels,
                   colors=colors,
                   legend_loc=legend_loc,
                   **kwargs)

    put.set_ax_xy_labels(ax=ax, **kwargs)
    put.set_ax_ticks_format(ax=ax, **kwargs)

    return fig


class UnitTests(Enum):
    RETURNS = 1
    XY_PLOT = 2


def run_unit_test(unit_test: UnitTests):

    from qis.test_data import load_etf_data
    prices = load_etf_data().dropna()

    df = ret.to_returns(prices=prices, drop_first=True)

    if unit_test == UnitTests.RETURNS:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        global_kwargs = dict(fontsize=8, linewidth=0.5, weight='normal', markersize=1)

        plot_qq(df=df,
                desc_table_type=dsc.DescTableType.SKEW_KURTOSIS,
                ax=ax,
                **global_kwargs)

    elif unit_test == UnitTests.XY_PLOT:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        global_kwargs = dict(fontsize=8, linewidth=0.5, weight='normal', markersize=1)
        plot_xy_qq(x=df.iloc[:, 1],
                   y=df.iloc[:, 0],
                   ax=ax,
                   **global_kwargs)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.RETURNS

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
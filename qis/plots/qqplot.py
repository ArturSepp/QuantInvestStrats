"""
quantile-quantile plot
"""
# packages
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as stats
from statsmodels import api as sm
from typing import List, Union, Tuple, Optional

# qis
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
    """
    quantile-quantile plot of each column against the normal distribution.

    The diagnostic for whether a return series is normal, and where it is not: points bending
    away from the reference line in the lower left are the left tail being fatter than normal,
    which is the usual finding and the reason the package reports skew and kurtosis alongside
    volatility.

    Arguments shared with every ``plot_*`` function are documented in
    ``qis/docs/plotting_kwargs.md``.

    Args:
        df: returns, one column per series. A Series is plotted alone
        markers: matplotlib marker style per column
        is_drop_na: drop missing observations before ranking. False leaves them in and shifts
            the empirical quantiles, so leave this True unless the gaps are meaningful
        desc_table_type: descriptive statistics table drawn beside the plot; see
            :class:`DescTableType`
        legend_stats: summary statistics appended to each legend entry

    Returns:
        the figure drawn on
    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    if df.empty:
        warnings.warn('df is empty: no data to plot')
        return fig

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

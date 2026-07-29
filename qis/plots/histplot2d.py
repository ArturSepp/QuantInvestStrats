"""
a two-dimensional histogram of the two columns of a frame, binned by ``sns.histplot`` and shaded
by probability. ``plot_histplot2d`` is the only entry point: ``add_corr_legend`` puts the
Spearman rank correlation and its p-value in the legend, and ``a_min``/``a_max`` clip first.
"""
# packages
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from typing import Optional

# qis
import qis.plots.utils as put


def plot_histplot2d(df: pd.DataFrame,
                    title: str = None,
                    a_min: float = None,
                    a_max: float = None,
                    xvar_format: str = '{:.1f}',
                    yvar_format: str = '{:.1f}',
                    add_corr_legend: bool = True,
                    legend_loc: Optional[str] = 'upper left',
                    color: str = 'navy',
                    fontsize: int = 10,
                    ax: plt.Subplot = None,
                    **kwargs
                    ) -> plt.Figure:

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = None

    if df.empty:
        warnings.warn('df is empty: no data to plot')
        return fig

    if len(df.columns) != 2:
        raise ValueError(f"should be 2 columns")

    if a_min is not None or a_max is not None:
        df = np.clip(df, a_min=a_min, a_max=a_max)

    sns.histplot(data=df,
                 x=df.columns[0],
                 y=df.columns[1],
                 bins=100,
                 cbar=False,
                 stat='probability',
                 cbar_kws=dict(shrink=.75),
                 ax=ax)

    put.set_ax_ticks_format(ax=ax, xvar_format=xvar_format, yvar_format=yvar_format, **kwargs)

    if add_corr_legend:
        rho, pval = stats.spearmanr(df.to_numpy(), nan_policy='omit', axis=0)  # column is variable
        label = f"Rank corr={rho:0.2f}, p-val={pval:0.2f}"
        lines = [(label, {'color': color})]

        put.set_legend(ax=ax,
                       legend_loc=legend_loc,
                       fontsize=fontsize,
                       lines=lines,
                       **kwargs)

    put.align_xy_limits(ax=ax)

    if title is not None:
        ax.set_title(title, fontsize=fontsize, **kwargs)

    return fig

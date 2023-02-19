"""
plot histogram 2d
"""
# packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from typing import Optional
from enum import Enum

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

    if len(df.columns) != 2:
        raise ValueError(f"should be 2 columns")

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = None

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


class UnitTests(Enum):
    TEST = 1


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.TEST:
        np.random.seed(1)
        n_instruments = 1000
        exposures_nm = np.random.normal(0.0, 1.0, size=(n_instruments, 2))
        data = pd.DataFrame(data=exposures_nm, columns=[f"id{n+1}" for n in range(2)])

        fig, ax = plt.subplots(1, 1, figsize=(3.9, 3.4), tight_layout=True)
        global_kwargs = dict(fontsize=6, linewidth=0.5, weight='normal', first_color_fixed=True)
        plot_histplot2d(df=data, ax=ax, **global_kwargs)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.TEST

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

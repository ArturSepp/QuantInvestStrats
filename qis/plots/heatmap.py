"""
heatmap plots
"""

# packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional
from enum import Enum

# qis
import qis.plots.utils as put


def plot_heatmap(df: pd.DataFrame,
                 transpose: bool = False,
                 inverse: bool = False,
                 date_format: Optional[str] = '%Y',
                 cmap: str = 'RdYlGn',
                 var_format: Optional[str] = '{:.1%}',
                 alpha: float = 1.0,
                 fontsize: int = 10,
                 title: Optional[str] = None,
                 top_x_label: bool = True,
                 square: bool = False,
                 vline_columns: List[int] = None,
                 hline_rows: List[int] = None,
                 vmin: float = None,
                 vmax: float = None,
                 labelpad: int = 50,
                 ax: plt.Subplot = None,
                 **kwargs
                 ) -> Optional[plt.Figure]:

    if ax is None:
        fig, ax = plt.subplots()
    else:  # add table to existing axis
        fig = None

    df = df.copy()

    if date_format is not None:  # index may include 'Total'
        df.index = [date.strftime(date_format) if isinstance(date, pd.Timestamp) else date for date in df.index]

    if transpose:
        df = df.T
        inverse = False

    if inverse:
        df = df.reindex(index=df.index[::-1])

    if var_format is not None:
        var_format = var_format.replace('{:', '').replace('}', '')  # no {}

    sns.heatmap(data=df,
                center=0,
                annot=True,
                fmt=var_format,
                cmap=cmap,
                alpha=alpha,
                cbar_kws={'size': fontsize},
                cbar=False,
                annot_kws={'size': fontsize},
                xticklabels=True,  # important for full display of labels
                yticklabels=True,  # important for full display of labels
                square=square,
                vmin=vmin,
                vmax=vmax,
                ax=ax)  # ,"ha": 'right' #cbar_kws={'format': '%0.2f%%'}

    if top_x_label:
        ax.xaxis.tick_top()

    if not transpose:
        pass
        # bottom, top = ax.get_ylim()
        # ax.set_ylim(bottom + 0.5, top - 0.5)
    else:
        ax.xaxis.labelpad = labelpad

    put.set_ax_tick_params(ax=ax, fontsize=fontsize, labelbottom=not top_x_label, labeltop=top_x_label, **kwargs)
    put.set_ax_tick_labels(ax=ax, fontsize=fontsize, **kwargs)

    if vline_columns is not None:
        for vline_column in vline_columns:
            ax.vlines([vline_column], *ax.get_ylim(), lw=1)

    if hline_rows is not None:
        for hline_row in hline_rows:
            ax.hlines([hline_row], *ax.get_xlim(), lw=1)

    if title is not None:
        put.set_title(ax=ax, title=title, fontsize=fontsize, **kwargs)

    ax.set_ylabel('')
    ax.set_xlabel('')

    return fig


class UnitTests(Enum):
    HEATMAP = 1


def run_unit_test(unit_test: UnitTests):

    from qis.test_data import load_etf_data
    prices = load_etf_data().dropna()

    if unit_test == UnitTests.HEATMAP:
        corrs = prices.pct_change().corr()
        plot_heatmap(corrs, inverse=False, x_rotation=90)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.HEATMAP

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

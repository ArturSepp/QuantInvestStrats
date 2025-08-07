"""
pieplot
"""
# packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, List
from enum import Enum

# qis
import qis.plots.utils as put


def plot_pie(df: [pd.Series, pd.DataFrame],
             y_column: str = None,
             ylabel: str = '',
             title: str = None,
             colors: List[str] = None,
             legend_loc: Optional[str] = None,
             autopct: Optional[str] = '%.0f%%',
             ax: plt.Subplot = None,
             **kwargs
             ) -> Optional[plt.Figure]:

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    if y_column is None and isinstance(df, pd.DataFrame):
        y_column = df.columns[0]
    if colors is None:
        colors = put.get_cmap_colors(n=len(df.index), **kwargs)

    df.plot.pie(y=y_column, autopct=autopct, colors=colors, ax=ax)

    if legend_loc is None:
        ax.legend().set_visible(False)

    if title is not None:
        put.set_title(ax=ax, title=title, **kwargs)

    ax.set_ylabel(ylabel)

    return fig


class LocalTests(Enum):
    PORTFOLIO = 1


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    if local_test == LocalTests.PORTFOLIO:

        df = pd.DataFrame({'Conservative': [0.5, 0.25, 0.25],
                           'Balanced': [0.30, 0.30, 0.40],
                           'Growth': [0.10, 0.40, 0.50]},
                          index=['Stables', 'Market-neutral', 'Crypto-Beta'])
        print(df)
        kwargs = dict(fontsize=8, linewidth=0.5, weight='normal', markersize=1)

        with sns.axes_style("darkgrid"):
            fig, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)
            plot_pie(df=df,
                     ax=ax,
                     **kwargs)

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.PORTFOLIO)

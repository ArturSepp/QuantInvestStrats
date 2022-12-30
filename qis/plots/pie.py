"""
pieplot
"""
# built in
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional
from enum import Enum

# qis
import qis.plots.utils as put
from qis.plots.utils import LegendLineType


def plot_pie(df: [pd.Series, pd.DataFrame],
             y_column: str = None,
             ax: plt.Subplot = None,
             **kwargs
             ) -> Optional[plt.Figure]:

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    if y_column is None and isinstance(df, pd.DataFrame):
        y_column = df.columns[0]

    df.plot.pie(y=y_column, ax=ax)

    return fig


class UnitTests(Enum):
    PORTFOLIO = 1


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.PORTFOLIO:

        df = pd.DataFrame({'Conservative': [0.5, 0.25, 0.25],
                           'Balanced': [0.30, 0.30, 0.40],
                           'Growth': [0.10, 0.40, 0.50]},
                          index=['Stables', 'Market-neutral', 'Crypto-Beta'])
        print(df)
        kwargs = dict(fontsize=8, linewidth=0.5, weight='normal', markersize=1)

        with sns.axes_style("darkgrid"):
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

            plot_pie(df=df,
                     ax=ax,
                     **kwargs)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.PORTFOLIO

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)


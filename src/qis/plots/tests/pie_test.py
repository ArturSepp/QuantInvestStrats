import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from enum import Enum
from qis.plots.pie import plot_pie


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

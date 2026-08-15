
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from enum import Enum
from qis.plots.errorbar import plot_errorbar


class LocalTests(Enum):
    ERROR_BAR = 1


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    n = 10
    x = np.linspace(0, 10, n)
    dy = 0.8
    y1 = pd.Series(np.sin(x) + dy * np.random.randn(n), index=x, name='y1')
    y2 = pd.Series(np.cos(x) + dy * np.random.randn(n), index=x, name='y2')
    data = pd.concat([y1, y2], axis=1)

    if local_test == LocalTests.ERROR_BAR:

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

    run_local_test(local_test=LocalTests.ERROR_BAR)

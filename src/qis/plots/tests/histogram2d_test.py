
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
from qis.plots.histplot2d import plot_histplot2d


class LocalTests(Enum):
    TEST = 1


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    if local_test == LocalTests.TEST:
        np.random.seed(1)
        n_instruments = 1000
        exposures_nm = np.random.normal(0.0, 1.0, size=(n_instruments, 2))
        data = pd.DataFrame(data=exposures_nm, columns=[f"id{n+1}" for n in range(2)])

        fig, ax = plt.subplots(1, 1, figsize=(3.9, 3.4), tight_layout=True)
        global_kwargs = dict(fontsize=6, linewidth=0.5, weight='normal', first_color_fixed=True)
        plot_histplot2d(df=data, ax=ax, **global_kwargs)

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.TEST)

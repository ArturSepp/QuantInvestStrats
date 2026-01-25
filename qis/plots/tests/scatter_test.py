
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

from qis.plots.scatter import (plot_scatter,
                               plot_classification_scatter,
                               estimate_classification_scatter)


def get_random_data(is_random_beta: bool = True,
                    n: int = 10000
                    ) -> pd.DataFrame:

    x = np.random.normal(0.0, 1.0, n)
    eps = np.random.normal(0.0, 1.0, n)
    if is_random_beta:
        beta = np.random.normal(1.0, 1.0, n)*np.square(x)
    else:
        beta = np.ones(n)
    y = beta*x + eps
    df = pd.concat([pd.Series(x, name='x'), pd.Series(y, name='y')], axis=1)
    df = df.sort_values(by='x', axis=0)

    return df


class LocalTests(Enum):
    SCATTER = 1
    CLASSIFICATION_SCATTER = 2
    CLASSIFICATION_REGRESSION = 3


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    np.random.seed(2)
    df = 1000.0+get_random_data(n=100000)
    print(df)

    if local_test == LocalTests.SCATTER:
        plot_scatter(df=df, fit_intercept=False)

    elif local_test == LocalTests.CLASSIFICATION_SCATTER:
        plot_classification_scatter(df=df, x='x', y='y')

    elif local_test == LocalTests.CLASSIFICATION_REGRESSION:
        y_rpeds = estimate_classification_scatter(df=df, x='x', y='y')
        plot_classification_scatter(df=df, x='x', y='y')
        print(y_rpeds)
        y_rpeds.plot()

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.SCATTER)

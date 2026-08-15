import numpy as np
import pandas as pd
from enum import Enum
from qis.utils.df_cut import x_bins_cut, add_classification


class LocalTests(Enum):
    CUT = 1
    CLASS = 2


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    np.random.seed(2)  # freeze seed

    if local_test == LocalTests.CUT:

        n = 1000000
        x = np.random.normal(0.0, 1.0, n)
        bins = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])

        pd1 = pd.cut(x, bins)
        print('pd.cut')
        print(pd1)

        pd2, labels = x_bins_cut(x, bins)
        print('x_bins_cut')
        print(pd2)

    elif local_test == LocalTests.CLASS:

        n = 10000
        x = np.random.normal(0.0, 1.0, n)
        eps = np.random.normal(0.0, 1.0, n)
        y = x + eps
        bins = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
        df = pd.concat([pd.Series(x, name='x'), pd.Series(y, name='y')], axis=1)

        df1, labels = add_classification(df=df, class_var_col='x', bins=bins)
        print(df1)


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.CUT)

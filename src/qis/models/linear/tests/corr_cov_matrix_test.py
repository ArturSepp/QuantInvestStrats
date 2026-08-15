import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum

# qis
import qis.plots.time_series as pts
import qis.models.linear.ewm as ewm
from qis.models.linear.corr_cov_matrix import (
    compute_ewm_corr_df,
    compute_masked_covar_corr,
    corr_to_pivot_row,
    matrix_regularization,
)


class LocalTests(Enum):
    CORR = 1
    EWMA_CORR_MATRIX = 2
    PLOT_CORR_MATRIX = 3
    MATRIX_REGULARIZATION = 4


def test_compute_masked_corr_uses_pairwise_complete_observations():
    """Keep ragged-history correlations bounded and equal to pandas pairwise correlation."""
    data = pd.DataFrame({
        'long': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
        'short': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.0, 1.0, 2.0, 3.0],
    })

    actual = compute_masked_covar_corr(data=data, is_covar=False)

    pd.testing.assert_frame_equal(actual, data.corr())
    assert np.nanmax(np.abs(actual.to_numpy())) <= 1.0


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    if local_test == LocalTests.CORR:
        t = 100
        n = 4
        data = np.random.normal(0, 1.0, (t, n))
        pivot = data[:, 0]
        corrs = corr_to_pivot_row(pivot=pivot, data=data, is_normalized=True)
        print(corrs)

    elif local_test == LocalTests.EWMA_CORR_MATRIX:
        dates = pd.date_range(start='12/31/2018', end='12/31/2019', freq='B')
        n = 3
        mean = [-2.0, -1.0, 0.0]
        returns = pd.DataFrame(data=np.random.normal(mean, 1.0, (len(dates), n)),
                               index=dates,
                               columns=['x' + str(m + 1) for m in range(n)])

        corr = ewm.compute_ewm_covar_tensor(a=returns.to_numpy(), is_corr=True)
        print('corr')
        print(corr)
        print('corr_00')
        print(corr[0, 0, :])
        print('corr_01')
        print(corr[0, 1, :])
        print('corr_02')
        print(corr[0, 2, :])
        print('corr_last')
        print(corr[:, :, -1])

    elif local_test == LocalTests.PLOT_CORR_MATRIX:

        dates = pd.date_range(start='31Dec2020', end='31Dec2021', freq='B')
        n = 3
        mean = [-2.0, -1.0, 0.0]
        returns = pd.DataFrame(data=np.random.normal(mean, 1.0, (len(dates), n)),
                               index=dates,
                               columns=['x' + str(m + 1) for m in range(n)])
        corrs = compute_ewm_corr_df(df=returns)
        print(corrs)

        pts.plot_time_series(df=corrs,
                             legend_stats=pts.LegendStats.AVG_LAST,
                             trend_line=pts.TrendLine.AVERAGE)

    elif local_test == LocalTests.MATRIX_REGULARIZATION:
        covar = np.array([[1.0, -0.01, 0.01],
                         [-0.01, 0.5, 0.005],
                         [0.01, 0.005, 0.0001]])
        covar_a = matrix_regularization(covar=covar)
        print(covar_a)

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.MATRIX_REGULARIZATION)

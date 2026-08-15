import matplotlib.pyplot as plt
from enum import Enum
import qis.perfstats.returns as ret
import qis.models.linear.corr_cov_matrix as ccm
from qis.models.linear.plot_correlations import (plot_corr_matrix_from_covar,
                                                 plot_returns_corr_matrix_time_series,
                                                 plot_returns_ewm_corr_table,
                                                 plot_returns_corr_table)

class LocalTests(Enum):
    CORR_TABLE = 1
    CORR_MATRIX = 2
    EWMA_CORR = 3
    PLOT_CORR_FROM_COVAR = 4


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """
    from qis.tests.price_data_test import load_etf_data
    prices = load_etf_data().dropna()

    if local_test == LocalTests.CORR_TABLE:
        plot_returns_corr_table(prices=prices)

    elif local_test == LocalTests.CORR_MATRIX:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), constrained_layout=True)
        plot_returns_corr_matrix_time_series(prices=prices, regime_benchmark='SPY', ax=ax)

    elif local_test == LocalTests.EWMA_CORR:
        plot_returns_ewm_corr_table(prices=prices.iloc[:, :5])

    elif local_test == LocalTests.PLOT_CORR_FROM_COVAR:
        returns = ret.to_returns(prices=prices, freq='ME')
        covar = 12.0 * ccm.compute_masked_covar_corr(data=returns, is_covar=True)
        plot_corr_matrix_from_covar(covar)

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.PLOT_CORR_FROM_COVAR)


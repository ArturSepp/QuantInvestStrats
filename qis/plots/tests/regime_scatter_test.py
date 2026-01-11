
# packages
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum

from qis.perfstats.regime_classifier import (RegimeClassifier,
                                             BenchmarkReturnsQuantilesRegime,
                                             BenchmarkReturnsQuantilesRegime,
                                             BenchmarkReturnsPositiveNegativeRegime)
from qis.plots.derived.regime_scatter import plot_scatter_regression


class LocalTests(Enum):
    """Enumeration of available local test cases."""
    SCATTER_ALL = 1
    SCATTER_1 = 2


def run_local_test(local_test: LocalTests):
    """
    Run local tests for development and debugging purposes.

    Integration tests that download real data and generate regression plots.
    Use for quick verification during development.

    Args:
        local_test: Which test case to run
    """
    from qis.test_data import load_etf_data

    # Load sample ETF price data
    prices = load_etf_data().dropna()
    # regime_classifier = BenchmarkReturnsQuantilesRegime(freq='QE')
    regime_classifier = BenchmarkReturnsPositiveNegativeRegime()

    if local_test == LocalTests.SCATTER_ALL:
        # Test with all assets
        with sns.axes_style('darkgrid'):
            plot_scatter_regression(
                prices=prices,
                regime_benchmark='SPY',
                regime_classifier=regime_classifier,
                add_last_date=False,
                display_estimated_betas=True
            )

    elif local_test == LocalTests.SCATTER_1:
        # Test with single asset pair
        with sns.axes_style('darkgrid'):
            plot_scatter_regression(
                prices=prices[['SPY', 'TLT']],
                regime_benchmark='SPY',
                regime_classifier=regime_classifier,
                add_last_date=True,
                display_estimated_betas=True
            )

    plt.show()


if __name__ == '__main__':
    run_local_test(local_test=LocalTests.SCATTER_ALL)

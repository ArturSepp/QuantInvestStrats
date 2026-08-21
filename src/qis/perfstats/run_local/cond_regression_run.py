
from enum import Enum
from qis import BenchmarkReturnsQuantilesRegime
from qis.plots.derived.regime_scatter import get_regime_regression_params


class Locals(Enum):
    """Enumeration of available local test cases."""
    REGRESSION = 1


def run_local(local: Locals):
    """
    Run local tests for development and debugging purposes.

    Integration tests that download real data and generate reports.
    Use for quick verification during development.

    Args:
        local: Which test case to run
    """
    from qis.run_local.price_data_run import load_etf_data

    # Load sample ETF price data
    prices = load_etf_data().dropna()
    print(prices)

    if local == Locals.REGRESSION:
        # Test regime-conditional regression with SPY as benchmark
        estimated_params = get_regime_regression_params(
            prices=prices,
            regime_classifier=BenchmarkReturnsQuantilesRegime(),
            benchmark='SPY',
            drop_benchmark=True,
            is_print_summary=True,
            is_add_alpha=True
        )
        print("\nEstimated Parameters:")
        print(estimated_params)


if __name__ == '__main__':
    run_local(local=Locals.REGRESSION)

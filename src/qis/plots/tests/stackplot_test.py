import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
from qis.plots.stackplot import plot_stack



def create_sample_portfolio_data() -> pd.DataFrame:
    """Create sample portfolio data for demonstration"""
    dates = pd.date_range('2010-01-01', periods=120, freq='ME')

    # Sample portfolio weights (can be positive or negative)
    np.random.seed(42)
    assets = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'META']

    # Generate random weights that sum to approximately 100% (allowing for some leverage)
    weights_data = []
    for i in range(len(dates)):
        weights = np.random.normal(20, 15, len(assets))  # Can be negative (short positions)
        weights = weights / weights.sum() * 100  # Normalize to sum to 100%
        weights_data.append(weights)

    weights_df = pd.DataFrame(weights_data, columns=assets, index=dates)
    return weights_df


class LocalTests(Enum):
    WEIGHTS = 1


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """
    if local_test == LocalTests.WEIGHTS:

        weights = create_sample_portfolio_data()

        fig, axs = plt.subplots(2, 1, figsize=(8, 10), tight_layout=True)
        plot_stack(df=weights,
                   stacked=False,
                   use_bar_plot=True,
                   x_rotation=90,
                   yvar_format='{:,.0%}',
                   date_format='%b-%y',
                   fontsize=6,
                   ax=axs[0])

        plot_stack(df=weights,
                   stacked=False,
                   use_bar_plot=False,
                   x_rotation=90,
                   yvar_format='{:,.0%}',
                   date_format='%b-%y',
                   fontsize=6,
                   ax=axs[1])

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.WEIGHTS)

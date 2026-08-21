
import matplotlib.pyplot as plt
from enum import Enum
from qis.plots.heatmap import plot_heatmap


class Locals(Enum):
    HEATMAP = 1


def run_local(local: Locals):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    from qis.run_local.price_data_run import load_etf_data
    prices = load_etf_data().dropna()

    if local == Locals.HEATMAP:
        corrs = prices.pct_change().corr()
        plot_heatmap(corrs, inverse=False, x_rotation=90)

    plt.show()


if __name__ == '__main__':

    run_local(local=Locals.HEATMAP)

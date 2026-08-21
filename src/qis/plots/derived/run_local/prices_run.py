"""Development runner extracted from ``qis.plots.derived.prices``."""

import matplotlib.pyplot as plt
from enum import Enum
from qis.perfstats.config import PerfStat, PerfParams

from qis.plots.derived.prices import (
    get_performance_labels_for_stats,
    plot_prices,
    plot_prices_with_dd,
)

class Locals(Enum):
    PERFORMANCE_LABELS = 1
    PRICE = 2
    PRICE_WITH_DD = 3

def run_local(local: Locals):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    from qis.run_local.price_data_run import load_etf_data
    prices = load_etf_data().dropna()

    if local == Locals.PERFORMANCE_LABELS:
        this = get_performance_labels_for_stats(prices=prices, perf_stats_labels=[PerfStat.PA_RETURN,
                                                                                  PerfStat.VOL,
                                                                                  PerfStat.SHARPE_RF0,
                                                                                  PerfStat.MAX_DD])
        print(this)

    elif local == Locals.PRICE:
        perf_params = PerfParams(freq='B')
        plot_prices(prices=prices, perf_params=perf_params)

    elif local == Locals.PRICE_WITH_DD:
        perf_params = PerfParams(freq='ME')
        plot_prices_with_dd(prices=prices,
                            regime_benchmark=prices.columns[0],
                            perf_params=perf_params)

    plt.show()

if __name__ == "__main__":
    run_local(local=Locals.PERFORMANCE_LABELS)

"""Development runner extracted from ``qis.plots.derived.drawdowns``."""

import matplotlib.pyplot as plt
from enum import Enum

from qis.plots.derived.drawdowns import (
    plot_rolling_drawdowns,
    plot_rolling_time_under_water,
    plot_top_drawdowns_paths,
)

class Locals(Enum):
    DRAWDOWN_TS = 1
    ROLLING_TIME = 2
    PLOT_TOP_DRAWDOWNS = 3

def run_local(local: Locals):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    from qis.run_local.price_data_run import load_etf_data
    prices = load_etf_data() # .dropna()

    if local == Locals.DRAWDOWN_TS:
        plot_rolling_drawdowns(prices=prices)

    elif local == Locals.ROLLING_TIME:
        plot_rolling_time_under_water(prices=prices)

    elif local == Locals.PLOT_TOP_DRAWDOWNS:
        # plot_top_drawdowns_ts(price=prices['TLT'], freq='D')
        plot_top_drawdowns_paths(price=prices['TLT'], highlight_ongoing=True, freq='D')

    plt.show()

if __name__ == "__main__":
    run_local(local=Locals.DRAWDOWN_TS)

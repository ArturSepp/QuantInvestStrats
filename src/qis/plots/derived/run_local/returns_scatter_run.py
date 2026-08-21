"""Development runner extracted from ``qis.plots.derived.returns_scatter``."""

import matplotlib.pyplot as plt
from enum import Enum

from qis.plots.derived.returns_scatter import (
    plot_returns_scatter,
)

class Locals(Enum):
    RETURNS = 1
    RETURNS2 = 2

def run_local(local: Locals):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    from qis.run_local.price_data_run import load_etf_data
    prices = load_etf_data().dropna()

    if local == Locals.RETURNS:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        global_kwargs = dict(fontsize=8, linewidth=0.5, weight='normal', markersize=1)
        plot_returns_scatter(prices=prices,
                             benchmark='SPY',
                             var_format='{:.2%}',
                             ax=ax,
                             **global_kwargs)

    elif local == Locals.RETURNS2:

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        global_kwargs = dict(fontsize=8, linewidth=0.5, weight='normal', markersize=1)

        plot_returns_scatter(prices=prices[['SPY', 'TLT']],
                             benchmark='TLT',
                             y_column='benchmarks',
                             ylabel='SPY',
                             var_format='{:.2%}',
                             ax=ax,
                             **global_kwargs)

    plt.show()

if __name__ == "__main__":
    run_local(local=Locals.RETURNS)

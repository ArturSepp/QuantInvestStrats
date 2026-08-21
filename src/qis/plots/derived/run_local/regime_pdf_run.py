"""Development runner extracted from ``qis.plots.derived.regime_pdf``."""

import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum

from qis.plots.derived.regime_pdf import (
    plot_regime_pdf,
)

class Locals(Enum):
    REGIME_PDF = 1

def run_local(local: Locals):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    from qis.run_local.price_data_run import load_etf_data
    prices = load_etf_data()[['SPY', 'TLT']].dropna()

    if local == Locals.REGIME_PDF:
        with sns.axes_style("darkgrid"):
            fig, axs = plt.subplots(1, 2, figsize=(15, 8), tight_layout=True)
            plot_regime_pdf(prices=prices, benchmark='SPY', is_histogram=False, ax=axs[0])
            plot_regime_pdf(prices=prices, benchmark='SPY', is_histogram=True, ax=axs[1])

    plt.show()

if __name__ == "__main__":
    run_local(local=Locals.REGIME_PDF)

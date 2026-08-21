"""Development runner extracted from ``qis.plots.derived.regime_class_table``."""

import matplotlib.pyplot as plt
from enum import Enum

from qis.plots.derived.regime_class_table import (
    plot_quantile_class_table,
)

class Locals(Enum):
    QUANTILE_CLASS_TABLE = 1

def run_local(local: Locals):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    from qis.run_local.price_data_run import load_etf_data
    prices = load_etf_data().dropna()
    returns = prices.asfreq('QE', method='ffill').pct_change().dropna()

    if local == Locals.QUANTILE_CLASS_TABLE:
        plot_quantile_class_table(data=returns, x_column='SPY', num_buckets=4, hue_name='quantile regime')

    plt.show()

if __name__ == "__main__":
    run_local(local=Locals.QUANTILE_CLASS_TABLE)

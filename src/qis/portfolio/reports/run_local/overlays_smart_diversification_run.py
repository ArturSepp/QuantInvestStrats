"""Development runner extracted from ``qis.portfolio.reports.overlays_smart_diversification``."""

import matplotlib.pyplot as plt
from enum import Enum
from qis import PerfStat

from qis.portfolio.reports.overlays_smart_diversification import (
    SmartDiversificationReport,
)

class Locals(Enum):
    CURVE = 1

def run_local(local: Locals):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    from qis.run_local.price_data_run import load_etf_data
    prices = load_etf_data()
    print(prices)
    overlays = ['TLT', 'GLD']
    prices = prices[['SPY']+overlays].dropna()

    if local == Locals.CURVE:

        sd_report = SmartDiversificationReport(principal_nav=prices.iloc[:, 0], overlay_navs=prices[overlays])

        # strategies_report.plot_nav()
        sd_report.plot_smart_diversification_curve(x_var=PerfStat.BEAR_SHARPE,
                                                   y_var=PerfStat.SHARPE_RF0,
                                                   title='Total Sharpe vs Bear Sharpe')
        # strategies_report.plot_smart_diversification_curve(x_var=PerfStat.VOL, y_var=PerfStat.PA_RETURN, title='Total P.A vs Vol')

        plt.show()

if __name__ == "__main__":
    run_local(local=Locals.CURVE)

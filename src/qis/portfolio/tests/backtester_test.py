import numpy as np
import pandas as pd
from enum import Enum
from qis.portfolio.backtester import backtest_model_portfolio



class LocalTests(Enum):
    BLENDED = 1
    COSTS = 2


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    import matplotlib.pyplot as plt
    import qis.plots.derived.prices as ppd

    from qis.tests.price_data_test import load_etf_data
    prices = load_etf_data().dropna()

    prices = prices[['SPY', 'TLT']]
    # prices.iloc[:200, :] = np.nan
    print(prices)

    if local_test == LocalTests.BLENDED:

        portfolio_nav_1_0 = backtest_model_portfolio(prices=prices,
                                                     weights=np.array([1.0, 0.0]),
                                                     rebalancing_freq='QE').get_portfolio_nav()

        portfolio_nav_5_5 = backtest_model_portfolio(prices=prices,
                                                     weights=np.array([1.0, 0.5]),
                                                     rebalancing_freq='QE').get_portfolio_nav()

        portfolio_nav_0_1 = backtest_model_portfolio(prices=prices,
                                                     weights=np.array([1.0, 1.0]),
                                                     rebalancing_freq='QE').get_portfolio_nav()

        portfolio_nav = pd.concat([portfolio_nav_1_0, portfolio_nav_5_5, portfolio_nav_0_1], axis=1)
        portfolio_nav.columns = ['x1=100, x2=0', 'x1=100, x2=50', 'x1=100, x2=100']
        print(portfolio_nav)
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ppd.plot_prices(prices=portfolio_nav, ax=ax)

    elif local_test == LocalTests.COSTS:
        portfolio_nav = backtest_model_portfolio(prices=prices,
                                                 weights=np.array([1.0, 1.0]),
                                                 rebalancing_freq='QE')

        portfolio_nav.plot_pnl()

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.BLENDED)
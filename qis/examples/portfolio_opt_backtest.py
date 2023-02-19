# packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from typing import Tuple, List, Dict
from enum import Enum

# qis
import qis
from qis.portfolio.optimization.qp_solvers import PortfolioObjective, max_portfolio_sharpe_qp
import qis.portfolio.backtester as bp
import qis.portfolio.optimization.rolling_portfolios as rlp


class UnitTests(Enum):
    MAX_DIVERSIFICATION = 1


def run_unit_test(unit_test: UnitTests):

    # data
    tickers = ['SPY', 'QQQ', 'EEM', 'TLT', 'IEF', 'LQD', 'HYG', 'SHY', 'GLD']
    prices = yf.download(tickers, start=None, end=None)['Adj Close'].dropna()

    kwargs = dict(add_mean_levels=True,
                  is_yaxis_limit_01=True,
                  baseline='zero',
                  bbox_to_anchor=(0.4, 1.1),
                  legend_stats=qis.LegendStats.AVG_STD_LAST,
                  ncol=len(prices.columns)//3,
                  var_format='{:.0%}')

    if unit_test == UnitTests.MAX_DIVERSIFICATION:
        weights = rlp.solve_optimal_weights_ewm_covar(prices=prices,
                                                      portfolio_objective=PortfolioObjective.MAX_DIVERSIFICATION,
                                                      weight_mins=np.zeros(len(prices.columns)),
                                                      weight_maxs=np.ones(len(prices.columns)),
                                                      rebalancing_freq='Q',
                                                      ewm_lambda=0.97)

        portfolio_out = bp.backtest_model_portfolio(prices=prices,
                                                    weights=weights,
                                                    is_rebalanced_at_first_date=True,
                                                    ticker='MaxDiversification',
                                                    is_output_portfolio_data=True)


        portfolio_out.plot_nav()
    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.MAX_DIVERSIFICATION

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

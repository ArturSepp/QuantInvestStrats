"""Draw a first qis portfolio-backtest chart from core-only synthetic data."""

import matplotlib.pyplot as plt

import qis
from qis.datasets import generate_synthetic_universe


universe = generate_synthetic_universe(start='2018-01-02', end='2025-12-31')
prices = universe.prices[['SEQ_US', 'SBD_TSY']]

print('Running first backtest (Numba compiles the portfolio kernel on first use)...')
portfolio = qis.backtest_model_portfolio(
    prices=prices,
    weights={'SEQ_US': 0.60, 'SBD_TSY': 0.40},
    rebalancing_freq='QE',
    rebalancing_costs=0.0010,
    ticker='Quarterly 60/40',
)
figure = qis.plot_prices(
    prices=portfolio.get_portfolio_nav(),
    perf_stats_labels=None,
    title='Quarterly rebalanced 60/40 portfolio',
)
plt.show()
plt.close(figure)

"""``PortfolioData.plot_ra_perf_table`` draws a benchmark table with default ``perf_params``.

The title quotes the volatility frequency. When ``perf_params`` is None the method must use the
parameters the table itself defaults to, rather than dereference None.
"""

# packages
import matplotlib
matplotlib.use('Agg')  # noqa: E402  - a headless backend, set before pyplot is imported
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# qis
import qis


def test_benchmark_table_without_perf_params() -> None:
    """No AttributeError, and the title names the frequency inferred from the NAV index."""
    dates = pd.bdate_range('2021-01-01', periods=300)
    steps = np.arange(len(dates))
    prices = pd.DataFrame({'Asset A': 100.0 * np.exp(0.0004 * steps + 0.02 * np.sin(steps / 7.0)),
                           'Asset B': 100.0 * np.exp(0.0001 * steps + 0.01 * np.cos(steps / 5.0))},
                          index=dates)
    portfolio = qis.backtest_model_portfolio(prices=prices, weights=np.array([0.6, 0.4]),
                                             rebalancing_freq='ME', ticker='ra table')
    fig, ax = plt.subplots(figsize=(8.0, 3.0))
    try:
        portfolio.plot_ra_perf_table(benchmark_price=prices['Asset A'], is_grouped=False,
                                     perf_params=None, ax=ax)
        texts = [ax.get_title()] + [text.get_text() for text in fig.texts]
    finally:
        plt.close(fig)
    assert any('B-freq returns with beta to Asset A' in text for text in texts), texts

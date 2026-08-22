"""Walk-forward calendar-month seasonality with point-in-time annual refits.

The example uses the seeded synthetic universe and a ten-year trailing estimation sample. A
position recorded at month end *t* is applied to the simple return over *[t, t+1]*. No return
from the investment year enters that year's signal estimate.

Run with ``python -m examples.portfolios.seasonality_backtest``.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import qis
from qis.datasets.synthetic import generate_synthetic_prices

from examples.portfolios.strats.seasonality_strat import compute_rolling_seasonal_signals


TICKERS = ['SEQ_US', 'SBD_TSY', 'SBD_IG', 'SCM_GLD']
ESTIMATION_YEARS = 10


def compute_seasonality_backtest(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the walk-forward strategy and its monthly decision schedule.

    Args:
        prices: Daily prices for the strategy universe.

    Returns:
        Tuple of ``(navs, decisions)``. ``navs`` contains the seasonality strategy and an
        equal-weight long-only benchmark; ``decisions`` is dated before the return it earns.
    """
    decisions = compute_rolling_seasonal_signals(
        prices=prices,
        num_sample_years=ESTIMATION_YEARS,
    )
    monthly_prices = prices.resample('ME').last()
    monthly_returns = monthly_prices.pct_change(fill_method=None)
    implemented_positions = decisions.reindex(monthly_returns.index).shift(1)
    strategy_returns = monthly_returns.multiply(implemented_positions).mean(axis=1).dropna()
    strategy_returns = strategy_returns.rename('Walk-forward seasonality')
    benchmark_returns = monthly_returns.mean(axis=1).reindex(strategy_returns.index)
    benchmark_returns = benchmark_returns.rename('Equal-weight long only')
    navs = qis.returns_to_nav(pd.concat([strategy_returns, benchmark_returns], axis=1))
    return navs, decisions


def run_example(show: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the offline seasonality example and optionally display its chart."""
    prices = generate_synthetic_prices(
        start='1995-01-02',
        end='2025-12-31',
        apply_quirks=False,
    )[TICKERS]
    navs, decisions = compute_seasonality_backtest(prices=prices)
    print('Return convention: simple monthly returns; positions are decided one month earlier.')
    print(navs.tail().round(3).to_string())

    with sns.axes_style('darkgrid'):
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
        qis.plot_prices_with_dd(
            prices=navs,
            perf_params=qis.PerfParams(freq='ME'),
            title='Point-in-time calendar-month seasonality',
            axs=axs,
        )
    if show:
        plt.show()
    else:
        plt.close(fig)
    return navs, decisions


if __name__ == '__main__':
    run_example()

"""Demonstrate tracking-error and active-risk analytics on an offline synthetic universe.

The example builds a point-in-time EWMA covariance model, reports ex-ante tracking error,
benchmark beta, and Euler marginal TE contributions, then compares them with realised EWMA
tracking error from portfolio and benchmark backtests.
"""
import pandas as pd

import qis
from qis.datasets.synthetic import generate_synthetic_universe


ASSETS = ['SEQ_US', 'SEQ_EU', 'SBD_TSY', 'SBD_IG', 'SCM_GLD', 'SAL_HF']
PORTFOLIO_WEIGHTS = pd.Series(
    [0.30, 0.20, 0.25, 0.10, 0.10, 0.05], index=ASSETS, name='portfolio'
)
BENCHMARK_WEIGHTS = pd.Series(
    [0.35, 0.15, 0.30, 0.10, 0.05, 0.05], index=ASSETS, name='benchmark'
)
RETURNS_FREQ = 'ME'
REBALANCING_FREQ = 'QE'
EWMA_SPAN = 36


def run_example() -> None:
    """Run the offline tracking-error and active-risk example."""
    universe = generate_synthetic_universe(
        start='2014-01-02', end='2025-12-31', apply_quirks=False
    )
    prices = universe.prices[ASSETS]
    group_data = universe.group_data.reindex(ASSETS)

    covar_dict = qis.estimate_rolling_ewma_covar(
        prices=prices,
        returns_freq=RETURNS_FREQ,
        rebalancing_freq=REBALANCING_FREQ,
        span=EWMA_SPAN,
    )
    risk_model = qis.RiskModel(covar=covar_dict)
    risk_date = risk_model.dates[-1]

    ex_ante_tre = risk_model.compute_tre_at_date(
        benchmark_weights=BENCHMARK_WEIGHTS,
        portfolio_weights=PORTFOLIO_WEIGHTS,
        date=risk_date,
        group_data=group_data,
    )
    benchmark_beta = risk_model.compute_benchmark_beta_at_date(
        benchmark_weights=BENCHMARK_WEIGHTS,
        portfolio_weights=PORTFOLIO_WEIGHTS,
        date=risk_date,
    )
    marginal_tre = risk_model.compute_marginal_tre_at_date(
        benchmark_weights=BENCHMARK_WEIGHTS,
        portfolio_weights=PORTFOLIO_WEIGHTS,
        date=risk_date,
        group_data=group_data,
    )

    portfolio_nav = qis.backtest_model_portfolio(
        prices=prices,
        weights=PORTFOLIO_WEIGHTS.to_dict(),
        rebalancing_freq=REBALANCING_FREQ,
        ticker='Active portfolio',
    ).get_portfolio_nav()
    benchmark_nav = qis.backtest_model_portfolio(
        prices=prices,
        weights=BENCHMARK_WEIGHTS.to_dict(),
        rebalancing_freq=REBALANCING_FREQ,
        ticker='Benchmark',
    ).get_portfolio_nav()
    realised_tre = qis.compute_ewma_realised_tracking_error(
        portfolio_nav=portfolio_nav,
        benchmark_nav=benchmark_nav,
        ewma_span=EWMA_SPAN,
        freq=RETURNS_FREQ,
        is_log_returns=False,
    )

    print(f'Risk date: {risk_date:%Y-%m-%d}')
    print('\nEx-ante tracking error by standalone group, annualised percent:')
    print((100.0 * ex_ante_tre).round(2).to_string())
    print(f'\nBenchmark beta: {benchmark_beta:.3f}')
    print('\nEuler marginal tracking-error contributions, annualised percent:')
    print((100.0 * marginal_tre).round(2).to_string())
    print(f'\nLatest realised EWMA tracking error: {100.0 * realised_tre.dropna().iloc[-1]:.2f}%')


if __name__ == '__main__':
    run_example()

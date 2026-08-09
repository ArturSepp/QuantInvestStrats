"""Demonstrate ex-post tracking-error and benchmark-risk analytics offline.

The example uses simple monthly returns from synthetic strategy and benchmark backtests. It
reports 36-month EWMA realised tracking error, whole-sample tracking error and information
ratio, and point-in-time 36-month EWMA benchmark beta and annualised alpha.
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
    """Run the offline ex-post tracking-error and benchmark-risk example."""
    universe = generate_synthetic_universe(
        start='2014-01-02', end='2025-12-31', apply_quirks=False
    )
    prices = universe.prices[ASSETS]

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
    monthly_returns = qis.to_returns(
        prices=pd.concat([portfolio_nav, benchmark_nav], axis=1),
        freq=RETURNS_FREQ,
        is_log_returns=False,
        drop_first=True,
    )

    return_diffs = monthly_returns[['Active portfolio']].sub(
        monthly_returns['Benchmark'], axis=0
    )
    return_diffs = return_diffs.rename(
        columns={'Active portfolio': 'Active portfolio vs benchmark'}
    )
    whole_sample_te, whole_sample_ir = qis.compute_te_ir_errors(return_diffs=return_diffs)
    realised_tre = qis.compute_ewma_realised_tracking_error(
        portfolio_nav=portfolio_nav,
        benchmark_nav=benchmark_nav,
        ewma_span=EWMA_SPAN,
        freq=RETURNS_FREQ,
        is_log_returns=False,
    )
    beta, alpha, _, _, _, _ = qis.compute_ewm_beta_alpha_forecast(
        x_data=monthly_returns['Benchmark'],
        y_data=monthly_returns[['Active portfolio']],
        span=EWMA_SPAN,
        init_type=qis.InitType.X0,
        beta_init_value=1.0,
    )
    annualised_alpha = alpha * qis.get_annualization_factor(freq=RETURNS_FREQ)

    print('Return convention: simple monthly returns; TE and alpha are annualised.')
    print(f'Latest realised EWMA tracking error: {100.0 * realised_tre.dropna().iloc[-1]:.2f}%')
    print(f'Whole-sample tracking error: {100.0 * whole_sample_te.iloc[0]:.2f}%')
    print(f'Whole-sample information ratio: {whole_sample_ir.iloc[0]:.3f}')
    print(f'Latest EWMA benchmark beta: {beta.iloc[-1, 0]:.3f}')
    print(f'Latest EWMA annualised alpha: {100.0 * annualised_alpha.iloc[-1, 0]:.2f}%')


if __name__ == '__main__':
    run_example()

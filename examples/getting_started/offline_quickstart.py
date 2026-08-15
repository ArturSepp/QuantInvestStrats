"""First portfolio-analytics result from a core qis install, entirely offline.

Run from a repository checkout with
``python examples/getting_started/offline_quickstart.py``. The same complete script is included
in the documentation quickstart so it can be copied after ``pip install qis``.
"""

import pandas as pd

import qis
from qis.datasets import generate_synthetic_universe


ASSETS = ['SEQ_US', 'SBD_TSY', 'SEQ_EM']
STATIC_WEIGHTS = {'SEQ_US': 0.50, 'SBD_TSY': 0.30, 'SEQ_EM': 0.20}
REBALANCING_FREQ = 'QE'
TRANSACTION_COST = 0.0010  # fractional units: 10 bp of traded notional


# The source panel is business-daily. SEQ_EM starts late, so the schedule must allocate over the
# instruments that are live at each rebalance without looking beyond that date.
universe = generate_synthetic_universe(start='2018-01-02', end='2025-12-31')
prices = universe.prices[ASSETS]
benchmark_nav = universe.benchmark_prices.iloc[:, 0].rename('SBM_6040')
weight_schedule = qis.generate_static_weights_schedule(
    prices=prices,
    weights=STATIC_WEIGHTS,
    rebalancing_freq=REBALANCING_FREQ,
)

# A target observed at t sets units at that close; those units earn the return over [t, t+1].
# The DataFrame owns its quarterly rebalance dates, so rebalancing_freq is None in the backtest.
portfolio_data = qis.backtest_model_portfolio(
    prices=prices,
    weights=weight_schedule,
    rebalancing_freq=None,
    rebalancing_costs=TRANSACTION_COST,
    weight_implementation_lag=0,
    ticker='Live-universe portfolio',
)
portfolio_nav = portfolio_data.get_portfolio_nav()

# Monthly simple returns and the arithmetic Sharpe convention are explicit. The rf=0 arithmetic
# Sharpe is selected below; an excess Sharpe would additionally require rates_data.
perf_params = qis.PerfParams(
    freq='ME',
    return_type=qis.ReturnTypes.RELATIVE,
    sharpe_convention=qis.SharpeConvention.ARITHMETIC,
)
performance_columns = [
    qis.PerfStat.PA_RETURN.to_str(),
    qis.PerfStat.VOL.to_str(),
    qis.PerfStat.SHARPE_ARITH.to_str(),
    qis.PerfStat.MAX_DD.to_str(),
]
performance_table = qis.compute_ra_perf_table(
    prices=portfolio_nav,
    perf_params=perf_params,
)[performance_columns]

# Tracking error and information ratio use the same monthly simple-return difference.
monthly_returns = qis.to_returns(
    prices=pd.concat([portfolio_nav, benchmark_nav], axis=1),
    freq='ME',
    is_log_returns=False,
    drop_first=True,
)
return_difference = (
    monthly_returns['Live-universe portfolio'] - monthly_returns['SBM_6040']
).to_frame('Portfolio vs SBM_6040')
tracking_error, information_ratio = qis.compute_te_ir_errors(return_difference)

final_target_weights = weight_schedule.iloc[-1]
final_realised_weights = portfolio_data.weights.iloc[-1]
final_nav = float(portfolio_nav.iloc[-1])

print(f'Date range: {prices.index[0]:%Y-%m-%d} to {prices.index[-1]:%Y-%m-%d}')
print(f'Prices: business-day frequency, shape={prices.shape}')
print(
    f'Target schedule: {weight_schedule.shape}, {REBALANCING_FREQ}, '
    'live-universe-aware'
)
print(f'Final target weights: {final_target_weights.round(4).to_dict()}')
print(f'Final realised weights: {final_realised_weights.round(4).to_dict()}')
print('\nMonthly simple-return performance (arithmetic Sharpe, rf=0):')
print(performance_table.round(3).to_string())
print(f'\nFinal NAV: {final_nav:.4f}')
print(
    'Benchmark-relative monthly simple returns: '
    f'TE={tracking_error.iloc[0]:.4f}, IR={information_ratio.iloc[0]:.4f}'
)
print(
    'Next reporting step (no file is written here): '
    'https://quantinveststrats.readthedocs.io/en/latest/factsheets_and_reporting.html'
)

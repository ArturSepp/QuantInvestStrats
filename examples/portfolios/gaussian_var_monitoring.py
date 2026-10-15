"""Compare supplied-covariance Gaussian VaR for two allocations, entirely offline.

Run ``python -m examples.portfolios.gaussian_var_monitoring`` from the repository root.
The frozen synthetic universe supplies monthly log returns. Annualised 36-month
EWMA covariances are sampled monthly after a three-span burn-in. The allocations
are fixed illustrations, not optimised portfolios or realised strategy histories.
"""

import pandas as pd

import qis
from qis.datasets import generate_synthetic_universe


ASSETS = ['SEQ_US', 'SEQ_EU', 'SBD_TSY', 'SBD_IG', 'SCM_GLD']
BENCHMARK = pd.Series([0.40, 0.20, 0.25, 0.10, 0.05], index=ASSETS)
ALLOCATIONS = {
    'Strategic': pd.Series([0.45, 0.10, 0.30, 0.10, 0.05], index=ASSETS),
    'Active': pd.Series([0.50, 0.10, 0.20, 0.10, 0.10], index=ASSETS),
}


def compute_monitoring_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute current and historical risk using one reusable simulation engine.

    Returns:
        Latest allocation comparison, indexed by allocation name, and the monthly
        risk history, indexed by date and allocation. VaR is one-year, 99%, and
        loss-positive; volatility and tracking error are annualised fractions.
    """
    universe = generate_synthetic_universe(
        start='2014-01-02', end='2025-12-31', seed=20260725, apply_quirks=False
    )
    covariances = qis.estimate_rolling_ewma_covar(
        prices=universe.prices[ASSETS],
        returns_freq='ME',
        rebalancing_freq='ME',
        span=36,
        demean=True,
        apply_an_factor=True,
        warmup_period=36,
        time_period=qis.TimePeriod('31Dec2022', '31Dec2025'),
    )
    simulation = qis.GaussianMonteCarlo(qis.GaussianVarSpec())
    rows = []
    for date, covariance in sorted(covariances.items()):
        for name, weights in ALLOCATIONS.items():
            risk = qis.compute_gaussian_portfolio_risk(
                covariance=covariance,
                portfolio_weights=weights,
                benchmark_weights=BENCHMARK,
                date=date,
                simulation=simulation,
            )
            rows.append({'date': date, 'allocation': name, **risk})
    history = pd.DataFrame(rows).set_index(['date', 'allocation']).sort_index()
    latest_date = history.index.get_level_values('date').max()
    return history.xs(latest_date, level='date').copy(), history


def run_example() -> None:
    """Print a latest comparison and the last three monthly risk snapshots."""
    latest, history = compute_monitoring_tables()
    percent_columns = [
        'portfolio_var', 'benchmark_var', 'tracking_error', 'portfolio_volatility',
        'benchmark_volatility', 'analytical_var', 'mc_relative_error',
    ]
    display = latest.copy()
    display[percent_columns] *= 100.0
    date = history.index.get_level_values('date').max()
    print(f'Risk date: {date:%Y-%m-%d}; one-year 99% Gaussian VaR.')
    print('VaR, volatility, TE and MC relative error below are percentages; beta/ratio are units.')
    print(display.round(4).to_string())
    print('\nRecent history, fractional units:')
    print(history[['portfolio_var', 'var_ratio', 'ex_ante_beta', 'tracking_error']].tail(6))


if __name__ == '__main__':
    run_example()

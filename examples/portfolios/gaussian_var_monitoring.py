"""Compare supplied-covariance Gaussian VaR for two allocations, entirely offline.

Run ``python -m examples.portfolios.gaussian_var_monitoring`` from the repository root.
The frozen synthetic universe supplies monthly log returns. Annualised 36-month
EWMA covariances are sampled monthly after a three-span burn-in. The allocations
are fixed illustrations, not optimised portfolios or realised strategy histories.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import qis
from qis.datasets import generate_synthetic_universe
from qis.portfolio.risk.one_year_tail_risk import compute_one_year_tail_risk


ASSETS = ['SEQ_US', 'SEQ_EU', 'SBD_TSY', 'SBD_IG', 'SCM_GLD']
BENCHMARK = pd.Series([0.40, 0.20, 0.25, 0.10, 0.05], index=ASSETS)
ALLOCATIONS = {
    'Strategic': pd.Series([0.45, 0.10, 0.30, 0.10, 0.05], index=ASSETS),
    'Active': pd.Series([0.50, 0.10, 0.20, 0.10, 0.10], index=ASSETS),
}


def _inputs():
    """Return the same seeded asset prices and annualised covariances for both exhibits."""
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
    return universe.prices[ASSETS], covariances


def compute_monitoring_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute current and historical risk using one reusable simulation engine.

    Returns:
        Latest allocation comparison, indexed by allocation name, and the monthly
        risk history, indexed by date and allocation. VaR is one-year, 99%, and
        loss-positive; volatility and tracking error are annualised fractions.
    """
    _, covariances = _inputs()
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


def compute_tail_comparison(block_months: int = 3) -> pd.DataFrame:
    """Return the USD one-year comparison for the Active targets and the benchmark."""
    prices, covariances = _inputs()
    date = max(covariances)
    monthly = prices.loc[:date].resample('ME').last().pct_change(fill_method=None)
    return compute_one_year_tail_risk(
        covariances[date], ALLOCATIONS['Active'], BENCHMARK, date, 'USD', monthly,
        block_months=block_months)


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
    columns = ['strategy_var', 'strategy_es', 'benchmark_var', 'benchmark_es']
    tails = compute_tail_comparison()
    print('\nOne-year 99% loss (% of NAV), Active and benchmark in USD:')
    print((100 * tails[columns]).round(2).to_string(na_rep='—'))
    print('\nCircular block sensitivity (% of NAV), 100,000 paths per row, seed 0:')
    sensitivity = {n: (tails if n == 3 else compute_tail_comparison(n)).loc[
        'Historical bootstrap', columns].astype(float) for n in (1, 3, 6)}
    print(pd.DataFrame(sensitivity).T.mul(100).round(2).rename_axis('block months'))


def save_tail_appendix(directory: Path) -> None:
    """Build the public weights report and export its Appendix figure and numeric frame."""
    prices, covariances = _inputs()
    allocations = [('Active', ALLOCATIONS['Active']), ('Benchmark', BENCHMARK)]
    portfolios = [qis.backtest_model_portfolio(prices, weights, rebalancing_freq='ME', ticker=name)
                  for name, weights in allocations]
    groups = pd.Series(['Equity', 'Equity', 'Bonds', 'Bonds', 'Gold'], index=ASSETS)
    for portfolio in portfolios:
        portfolio.set_group_data(groups)
    multi = qis.MultiPortfolioData(portfolio_datas=portfolios, covar_dict=covariances,
                                   benchmark_prices=prices[['SEQ_US']])
    monthly = prices.resample('ME').last().pct_change(fill_method=None)
    with plt.rc_context({'figure.max_open_warning': 0}):
        figs, dfs = qis.weights_tracking_error_report_by_ac_subac(
            multi, ac_group_data=groups, sub_ac_group_data=groups, turnover_groups=groups,
            add_one_year_tail_risk=True, tail_risk_currency='USD',
            tail_risk_monthly_returns=monthly)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        figs['one_year_tail_risk'].savefig(directory / 'one_year_tail_risk.pdf')
        dfs['one_year_tail_risk'].to_csv(directory / 'one_year_tail_risk.csv')
    finally:
        for fig in figs.values():
            plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--appendix-dir', type=Path, help='Save the report Appendix PDF and CSV.')
    args = parser.parse_args()
    run_example()
    if args.appendix_dir is not None:
        save_tail_appendix(args.appendix_dir)

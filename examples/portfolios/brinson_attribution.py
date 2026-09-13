"""Offline BHB attribution, native-date linking and net-trading-cost reconciliation."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import qis
from qis.datasets.synthetic import generate_synthetic_universe


def run_example(output_dir: Path = None):
    """Backtest synthetic portfolios and reconcile linked attribution at every endpoint.

    Args:
        output_dir: Optional destination for the QIS attribution PDF, PNG and CSV tables.
            None prints results without writing files.

    Returns:
        MultiPortfolioData and the canonical five-table monthly Brinson result.
    """
    universe = generate_synthetic_universe(
        start='2018-12-31', end='2025-12-31', apply_quirks=False)
    assets = ['SEQ_US', 'SBD_TSY', 'SBD_IG', 'SCM_GLD']
    prices = universe.prices[assets]
    schedule = prices.index[::10]
    equity = .45 + .15 * np.sin(np.arange(len(schedule)) / 3.0)
    weights = pd.DataFrame({
        'SEQ_US': equity, 'SBD_TSY': .15, 'SBD_IG': .75 - equity, 'SCM_GLD': .10,
    }, index=schedule)
    strategy = qis.backtest_model_portfolio(
        prices, weights, management_fee=0.0, rebalancing_costs=.001, ticker='Strategy')
    benchmark = qis.backtest_model_portfolio(
        prices, [.40, .40, .20, .0], rebalancing_freq='QE',
        management_fee=0.0, rebalancing_costs=.001, ticker='Benchmark')
    for portfolio in (strategy, benchmark):
        portfolio.set_group_data(
            universe.group_data.reindex(assets), ['Equities', 'Bonds', 'Commodities'])
    multi = qis.MultiPortfolioData([strategy, benchmark])
    period = qis.TimePeriod('2020-12-31', '2025-12-31')
    monthly = multi.compute_brinson_attribution(
        time_period=period, freq='ME', is_net=True)
    native = multi.compute_brinson_attribution(
        time_period=period, is_net=True)
    quarterly = multi.compute_brinson_attribution(
        time_period=period, freq='QE', is_net=True)

    navs = pd.concat([strategy.nav, benchmark.nav], axis=1).loc[period.start:period.end]
    cumulative = navs.div(navs.iloc[0]) - 1.0
    expected = cumulative['Strategy'] - cumulative['Benchmark']
    np.testing.assert_allclose(native[1].sum(axis=1).cumsum(), expected.iloc[1:], atol=1e-12)
    np.testing.assert_allclose(
        monthly[1].sum(axis=1).cumsum(), expected.resample('ME').last().iloc[1:], atol=1e-12)
    pd.testing.assert_frame_equal(monthly[0], quarterly[0])
    for column, label in enumerate(['Strategy', 'Benchmark']):
        np.testing.assert_allclose(monthly[0].loc['Total Sum', f'{label}\nReturn Total'],
                                   cumulative.iloc[-1, column], atol=1e-12)
    print('Arithmetic native-date returns; linked effects; net of realised trading costs.')
    print('Management fees and funding are zero. The benchmark has no commodity allocation.')
    print(monthly[0].to_string(float_format=lambda value: f'{value:.2%}'))
    print(f'Compounded active return: {expected.iloc[-1]:.4%}')
    print('Every native/monthly endpoint reconciles; monthly and quarterly totals agree.')

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        figure, axes = plt.subplots(5, 1, figsize=(12, 18))
        figure.subplots_adjust(top=.94, bottom=.04, hspace=.55)
        figure.suptitle('Brinson attribution | native daily effects, monthly display', fontsize=15)
        qis.plot_brinson_attribution_table(
            *monthly, axs=list(axes), var_format='{:.1%}', fontsize=8,
            x_date_freq='YE', date_format='%Y', x_rotation=0)
        for axis in axes[1:]:
            axis.set_axisbelow(True)
            axis.grid(True, alpha=.3)
        figure.savefig(output_dir / 'brinson_attribution.pdf')
        figure.savefig(output_dir / 'brinson_attribution.png', dpi=100)
        plt.close(figure)
        for name, table in zip(
                ['summary', 'aggregate', 'allocation', 'selection', 'interaction'], monthly):
            table.to_csv(output_dir / f'{name}.csv')
    return multi, monthly


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, help='Optional figure/table output directory')
    run_example(parser.parse_args().output_dir)

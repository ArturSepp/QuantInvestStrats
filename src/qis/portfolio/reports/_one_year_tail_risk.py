"""Render the one-year strategy and benchmark tail-risk Appendix."""

from textwrap import fill

import matplotlib.pyplot as plt
import pandas as pd

from qis.portfolio.risk.one_year_tail_risk import compute_one_year_tail_risk
from qis.utils.df_to_weights import align_weights_to_columns


def _target_at(portfolio, as_of):
    """Read original dated or static targets, including QIS zero-position missing entries."""
    targets = portfolio.input_weights
    if targets is None:
        raise ValueError(f'{portfolio.ticker}: one-year tail risk requires target weights')
    if isinstance(targets, pd.DataFrame):
        if not isinstance(targets.index, pd.DatetimeIndex) or not targets.index.is_unique:
            raise ValueError('Target schedules require unique observation dates')
        known = targets.loc[targets.index <= as_of].sort_index()
        if known.empty:
            raise ValueError(f'{portfolio.ticker}: no target weights at or before {as_of.date()}')
        return known.iloc[-1].fillna(0.0), known.index[-1]
    return align_weights_to_columns(targets, portfolio.prices.columns).fillna(0.0), pd.NaT


def plot_one_year_tail_risk(table, figsize=(11.7, 8.3)):
    """Format the same full-precision frame that the report returns for export."""
    analytic, mc, history = (row for _, row in table.iterrows())
    scale = min(figsize[0] / 11.7, figsize[1] / 8.3, 1.15)
    fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    ax.set_axis_off()
    ax.text(.02, .94, 'Appendix | One-year tail risk', size=16 * scale, weight='bold',
            transform=ax.transAxes)
    ax.text(.02, .86, f'{analytic.as_of:%d %b %Y}   |   {analytic.currency}   |   '
            f'{analytic.confidence:.0%} confidence   |   Loss as % of NAV',
            fontsize=11 * scale, transform=ax.transAxes)
    ax.text(.02, .78, f'Strategy: {analytic.strategy}     Benchmark: {analytic.benchmark}',
            fontsize=11 * scale, transform=ax.transAxes)
    metrics = ['strategy_var', 'strategy_es', 'benchmark_var', 'benchmark_es']
    formatted = [[method] + ['—' if pd.isna(row[k]) else f'{row[k]:.2%}' for k in metrics]
                 for method, row in table.iterrows()]
    grid = ax.table(cellText=formatted, colLabels=['Method', 'Strategy VaR', 'Strategy ES',
                    'Benchmark VaR', 'Benchmark ES'], cellLoc='center',
                    colWidths=[.26, .18, .18, .19, .19], bbox=[.02, .47, .96, .25])
    grid.auto_set_font_size(False)
    grid.set_fontsize(9 * scale)
    for (i, _), cell in grid.get_celld().items():
        cell.set_edgecolor('white')
        cell.set_facecolor('#dbe8ef' if i == 0 else ('#f1f5f7' if i % 2 else 'white'))
    vintage = (f'{analytic.covariance_date:%d %b %Y}' if pd.notna(analytic.covariance_date)
               else 'unavailable')
    sample = (f'{history.sample_start:%b %Y} to {history.sample_end:%b %Y}'
              if history.matched_months else 'unavailable')
    notes = [
        f'Gaussian covariance: {vintage}. {analytic.coverage}',
        (f'Gaussian Monte Carlo: {mc.paths:,} antithetic paths, seed {mc.seed}. '
         'The engine reports VaR only.' if mc.paths else 'Gaussian Monte Carlo unavailable.'),
        f'Bootstrap history: {sample} ({history.matched_months} months). {history.coverage}',
        (f'{history.paths:,} paths, seed {history.seed}, circular {history.block_months}-month '
         'blocks. Targets rebalance monthly. Annual compounded returns are centred across paths.'
         if history.paths else 'No historical paths were simulated.'),
    ]
    ax.text(.02, .39, '\n\n'.join(fill(line, 105) for line in notes), va='top',
            fontsize=9 * scale, transform=ax.transAxes)
    return fig


def get_one_year_tail_risk_table(strategy, benchmark, covariances, as_of,
                                 monthly_returns, currency):
    """Select both targets and the latest eligible risk matrix at the same report cut-off."""
    if as_of is None:
        as_of = min(p.nav.dropna().index.max() for p in (strategy, benchmark))
    as_of = pd.Timestamp(as_of)
    p, p_date = _target_at(strategy, as_of)
    b, b_date = _target_at(benchmark, as_of)
    vintage = max((d for d in (covariances or {}) if d <= as_of), default=None)
    covariance = covariances[vintage] if vintage is not None else None
    table = compute_one_year_tail_risk(covariance, p, b, as_of, currency,
                                       monthly_returns, covariance_date=vintage)
    table['strategy'] = strategy.ticker
    table['benchmark'] = benchmark.ticker
    table['strategy_target_date'] = p_date
    table['benchmark_target_date'] = b_date
    return table

"""Compare low-beta and momentum rank ICs on real ETFs using adjusted Yahoo Finance closes.

From the QIS checkout run python -m examples.perfstats.yahoo_etf_signal_ic.
Pass prices_csv to run_example to replay saved daily adjusted closes (date by ticker).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from examples.perfstats.ewma_benchmark_residuals import benchmark_adjustment
from qis.perfstats.forward_return_targets import compute_forward_return_targets
from qis.perfstats.signal_ic import compute_matched_signal_ic, compute_signal_ic_summary


BENCHMARKS = ['SPY', 'TLT']
ASSETS = ['IWM', 'EFA', 'EEM', 'LQD', 'HYG', 'GLD', 'DBC', 'VNQ']
HORIZONS = (1, 3, 6, 12)
FIRST_MONTH = '2007-12-31'
FIRST_DECISION = '2010-12-31'
LAST_MONTH = '2025-12-31'


def download_prices() -> pd.DataFrame:
    """Download split- and distribution-adjusted daily ETF closes using the QIS data extra."""
    import yfinance as yf

    data = yf.download(BENCHMARKS + ASSETS, start='2007-12-01', end='2026-01-01',
                       auto_adjust=True, ignore_tz=True, progress=False, threads=False)
    if data.empty or 'Close' not in data.columns:
        raise ValueError('Yahoo returned no adjusted closes for the ETF comparison')
    return data['Close']


def load_prices(prices_csv: str | Path | None = None) -> pd.DataFrame:
    """Download adjusted closes or read them from an explicit CSV path."""
    if prices_csv is None:
        return download_prices()
    return pd.read_csv(prices_csv, index_col=0, parse_dates=True)


def month_end_closes(daily: pd.DataFrame) -> pd.DataFrame:
    """Take the final observed session of each calendar month on one shared ETF grid."""
    tickers = BENCHMARKS + ASSETS
    days = daily.sort_index().loc['2007-12-01':LAST_MONTH].reindex(columns=tickers)
    missing = days.columns[~days.notna().any()]
    if len(missing):
        raise ValueError(f'Missing Yahoo price histories: {", ".join(missing)}')
    if not days.index.is_unique:
        raise ValueError('Expected one adjusted-close row per trading day')
    monthly = days.groupby(days.index.to_period('M')).tail(1).copy()
    monthly.index = monthly.index.to_period('M').to_timestamp('M')
    monthly = monthly.reindex(pd.date_range(FIRST_MONTH, LAST_MONTH, freq='ME'))
    invalid = (~np.isfinite(monthly) | monthly.le(0)).stack()
    if invalid.any():
        date, ticker = invalid[invalid].index[0]
        raise ValueError(f'Missing or invalid month-end price for {ticker} at {date.date()}')
    return monthly


def formation_panel(daily: pd.DataFrame, *, ewma_span: int = 36
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the forward targets and formation scores, retaining monthly log prices."""
    log_prices = np.log(month_end_closes(daily))
    returns = log_prices.diff().iloc[1:]
    decisions = returns.loc[FIRST_DECISION:].index
    adjustment = benchmark_adjustment(returns[ASSETS], returns[BENCHMARKS], decisions[:-1],
                                      ewma_span=ewma_span)
    targets = compute_forward_return_targets(
        decision_dates=decisions, asset_increments_by_freq={'ME': returns[ASSETS]},
        horizons=HORIZONS, adjustments={'residual': adjustment},
    )
    low_beta = pd.DataFrame({t: -b['SPY'] for t, b in adjustment.loadings_by_date.items()}).T
    # At t, 12-1 momentum uses prices at t-1 and t-12 (eleven monthly increments).
    momentum = log_prices[ASSETS].shift(1) - log_prices[ASSETS].shift(12)
    scores = pd.DataFrame({'low_beta': low_beta.stack(), 'momentum': momentum.stack()})
    scores.index.names = ['formation_date', 'asset']
    panel = targets.join(scores, on=['formation_date', 'asset'], validate='many_to_one')
    panel['eligible'] = panel.asset.isin(ASSETS)
    return panel, log_prices


def matched_membership(panel: pd.DataFrame) -> pd.DataFrame:
    """Count common, finite ETF windows before testing the ranks for dispersion."""
    matched = panel.loc[panel.eligible & np.isfinite(
        panel[['low_beta', 'momentum', 'raw', 'residual']]).all(axis=1)]
    return pd.crosstab(matched.asset, matched.horizon).reindex(
        index=ASSETS, columns=HORIZONS, fill_value=0)


def dated_ics(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute matched ICs and require paired realisations for the common sample."""
    dated = compute_matched_signal_ic(
        panel, signals=['low_beta', 'momentum'], targets=['raw', 'residual'],
        reference_target='raw', min_obs_per_date=5, method='spearman',
    )
    common = dated.loc[dated['sample'].eq('common')].copy()
    # Both means use the same realisations even if one cross-section has a constant rank.
    paired = common.groupby(['signal', 'horizon', 'formation_date']).ic.transform('count').eq(2)
    dated.loc[common.index[~paired], 'ic'] = np.nan
    return dated


def compare_signals(daily: pd.DataFrame, *, ewma_span: int = 36
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return paired mean rank ICs and matched window counts by ETF and horizon."""
    panel, _ = formation_panel(daily, ewma_span=ewma_span)
    dated = dated_ics(panel)
    common = dated.loc[dated['sample'].eq('common')]
    summary = compute_signal_ic_summary(
        common, as_of=pd.Timestamp(LAST_MONTH), ewma_span=ewma_span, rolling_span=None,
    ).summary.set_index(['signal', 'horizon', 'target'])
    means = summary.mean_ic.unstack('target')[['raw', 'residual']].rename(
        columns={'raw': 'raw_ic', 'residual': 'residual_ic'})
    coverage = summary.xs('residual', level='target')[['n_valid', 'mean_n_assets']]
    return means.join(coverage).sort_index(), matched_membership(panel)


def run_example(prices_csv: str | Path | None = None) -> None:
    """Print the signal comparison and the ETFs that supplied each matched sample."""
    daily = load_prices(prices_csv)
    ewma_span = 36
    comparison, membership = compare_signals(daily, ewma_span=ewma_span)
    print(f'Raw and SPY/TLT-residual Spearman IC through {LAST_MONTH}. '
          f'EWMA span {ewma_span} monthly returns for joint betas and '
          f'{ewma_span} valid IC dates per horizon for the means.')
    print(comparison.to_string(float_format=lambda x: f'{x:.4f}'))
    print('\nMatched windows per ETF (columns in months)')
    print(membership.to_string())


if __name__ == '__main__':
    run_example()

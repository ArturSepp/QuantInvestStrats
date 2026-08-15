"""
Global run: reproduce the strategy factsheet across every reporting-frequency config
on one and the same data panel.

A single vol-parity multi-asset portfolio is backtested once on daily (business-day) data,
then ``qis.generate_strategy_factsheet`` is rendered for the full DAILY / WEEKLY / MONTHLY /
QUARTERLY x {long, short} grid driven by ``fetch_default_report_kwargs``:

    - long  period  -> full history,           long  config (regime 'QE', annual heatmap)
    - short period  -> last 3 years of data,    short config (regime 'ME', monthly heatmap)

fetch_default_report_kwargs() auto-selects the long vs short preset from the reported span
(its ``long_threshold_years`` default is 5y), so the 3y window lands on the short configs.

Each (frequency, span) is written to its own PDF and all pages are also collected into a
single combined PDF, so the same panel can be compared across reporting frequencies.

Set ``use_synthetic_data=True`` (or run with no network) to generate a reproducible GBM panel
instead of downloading from yfinance; the real-data path mirrors examples/factsheets/strategy.py.
"""
import matplotlib
matplotlib.use('Agg')  # batch PDF generation, no interactive display needed

import numpy as np
import pandas as pd
from typing import Tuple

import qis
from qis import TimePeriod, PortfolioData, ReportingFrequency
from qis.portfolio.reports.config import fetch_default_report_kwargs

# custom universe with asset-class grouping (same panel reused for every config)
UNIVERSE_DATA = dict(SPY='Equities',
                     QQQ='Equities',
                     EEM='Equities',
                     TLT='Bonds',
                     IEF='Bonds',
                     LQD='Credit',
                     HYG='HighYield',
                     GLD='Gold')

# the four reporting-frequency configs to reproduce on the same panel
REPORTING_FREQUENCIES = (ReportingFrequency.DAILY,
                         ReportingFrequency.WEEKLY,
                         ReportingFrequency.MONTHLY,
                         ReportingFrequency.QUARTERLY)


def fetch_universe_data(start: str = "2003-12-31") -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """real-data path: daily prices from yfinance, made business-day frequency."""
    import yfinance as yf
    tickers = list(UNIVERSE_DATA.keys())
    prices = yf.download(tickers=tickers, start=start, end=None, ignore_tz=True, auto_adjust=True)['Close'][tickers]
    prices = prices.asfreq('B', method='ffill')
    benchmark_prices = prices[['SPY', 'TLT']]
    return prices, benchmark_prices, pd.Series(UNIVERSE_DATA)


def generate_synthetic_universe(start: str = '2003-12-31',
                                end: str = '2025-12-31',
                                seed: int = 1
                                ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """offline fallback: reproducible correlated-GBM business-day panel with the same tickers."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, end=end, freq='B')
    tickers = list(UNIVERSE_DATA.keys())
    # per-asset annualised drift / vol, loosely by asset class
    ann_mu = dict(SPY=0.08, QQQ=0.10, EEM=0.06, TLT=0.03, IEF=0.025, LQD=0.035, HYG=0.05, GLD=0.05)
    ann_vol = dict(SPY=0.16, QQQ=0.22, EEM=0.20, TLT=0.12, IEF=0.06, LQD=0.07, HYG=0.10, GLD=0.15)
    dt = 1.0 / 260.0
    n = len(dates)
    market = rng.standard_normal(n)  # common factor to induce cross-asset correlation
    cols = {}
    for ticker in tickers:
        beta = 0.6 if UNIVERSE_DATA[ticker] in ('Equities', 'HighYield') else (-0.1 if UNIVERSE_DATA[ticker] == 'Bonds' else 0.2)
        idio = rng.standard_normal(n)
        shock = beta * market + np.sqrt(max(1.0 - beta ** 2, 0.0)) * idio
        rets = ann_mu[ticker] * dt + ann_vol[ticker] * np.sqrt(dt) * shock
        cols[ticker] = 100.0 * np.exp(np.cumsum(rets))
    prices = pd.DataFrame(cols, index=dates, columns=tickers)
    benchmark_prices = prices[['SPY', 'TLT']]
    return prices, benchmark_prices, pd.Series(UNIVERSE_DATA)


def load_universe(use_synthetic_data: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """try yfinance, fall back to the synthetic panel when offline or download is empty."""
    if not use_synthetic_data:
        try:
            prices, benchmark_prices, group_data = fetch_universe_data()
            if prices is not None and not prices.dropna(how='all').empty:
                return prices, benchmark_prices, group_data
        except Exception as e:  # offline / yfinance failure -> synthetic
            print(f"yfinance download failed ({e}); using synthetic panel")
    return generate_synthetic_universe()


def generate_volparity_portfolio(prices: pd.DataFrame,
                                 group_data: pd.Series,
                                 span: int = 30,
                                 vol_target: float = 0.15,
                                 rebalancing_costs: float = 0.0010
                                 ) -> PortfolioData:
    """equal-vol (vol-parity) weights, backtested once on the daily panel."""
    ra_returns, weights, ewm_vol = qis.compute_ra_returns(returns=qis.to_returns(prices=prices, is_log_returns=True),
                                                          span=span,
                                                          vol_target=vol_target)
    weights = weights.divide(weights.sum(axis=1), axis=0)
    portfolio = qis.backtest_model_portfolio(prices=prices,
                                             weights=weights,
                                             rebalancing_costs=rebalancing_costs,
                                             weight_implementation_lag=1,
                                             ticker='VolParity')
    portfolio.set_group_data(group_data=group_data, group_order=list(group_data.unique()))
    return portfolio


def run_global_report(use_synthetic_data: bool = False,
                      short_period_years: int = 3,
                      add_rates_data: bool = False,
                      output_path: str = None
                      ) -> str:
    """render the strategy factsheet for the full {frequency} x {long, short} grid on one panel."""
    prices, benchmark_prices, group_data = load_universe(use_synthetic_data=use_synthetic_data)

    end = prices.index[-1]
    time_period_long = TimePeriod(prices.index[0], end)                          # full history
    time_period_short = TimePeriod(end - pd.DateOffset(years=short_period_years), end)  # last N years

    portfolio_data = generate_volparity_portfolio(prices=prices, group_data=group_data)
    name = portfolio_data.nav.name

    spans = (('long', time_period_long), ('short', time_period_short))
    all_figs = []
    if output_path is None:
        output_path = qis.local_path.get_output_path()

    for reporting_frequency in REPORTING_FREQUENCIES:
        for span_label, time_period in spans:
            report_kwargs = fetch_default_report_kwargs(time_period=time_period,
                                                        reporting_frequency=reporting_frequency,
                                                        add_rates_data=add_rates_data)
            factsheet_name = f"{name} factsheet \u2013 {reporting_frequency.name} reporting, {span_label} period"
            figs = qis.generate_strategy_factsheet(portfolio_data=portfolio_data,
                                                   benchmark_prices=benchmark_prices,
                                                   time_period=time_period,
                                                   factsheet_name=factsheet_name,
                                                   **report_kwargs)
            qis.save_figs_to_pdf(figs=figs,
                                 file_name=f"{name}_factsheet_{reporting_frequency.name.lower()}_{span_label}",
                                 local_path=output_path)
            all_figs.extend(figs)
            print(f"rendered {factsheet_name}  ({len(figs)} page(s))")

    combined = qis.save_figs_to_pdf(figs=all_figs,
                                    file_name=f"{name}_factsheet_all_reporting_frequencies",
                                    local_path=output_path)
    print(f"\ncombined report ({len(all_figs)} pages): {combined}")
    return combined


if __name__ == '__main__':
    # real-data run (yfinance). Pass use_synthetic_data=True to run fully offline.
    run_global_report(use_synthetic_data=False)

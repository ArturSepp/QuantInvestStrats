"""
Global run: reproduce the multi-strategy factsheet across every reporting-frequency config
on one and the same data panel.

A vol-parity strategy is backtested once per EWMA ``span`` on daily (business-day) data - a
parameter sweep - and the resulting strategies are wrapped in one MultiPortfolioData, then
``generate_multi_portfolio_factsheet`` is rendered for the full
DAILY / WEEKLY / MONTHLY / QUARTERLY x {long, short} grid driven by ``fetch_default_report_kwargs``:

    - long  period -> full history,            long  config (regime 'QE', annual heatmap)
    - short period -> last 3 years of data,     short config (regime 'ME', monthly heatmap)

This is the multi-strategy (cross-sectional comparison) analogue of
examples/factsheets/strategy_reporting_frequencies.py and
examples/factsheets/strategy_benchmark_reporting_frequencies.py, and reuses their universe and
frequency grid. The span-sweep MultiPortfolioData construction mirrors
examples/factsheets/multi_strategy.py.

The strategies are backtested once on the full daily panel; the reporting window only slices the
panel for each (frequency, span) report. Each report is written to its own PDF and all pages are
also collected into a single combined PDF, so the same comparison can be inspected across
reporting frequencies. Set ``use_synthetic_data=True`` (or run with no network) for a reproducible
GBM panel.
"""
import matplotlib
matplotlib.use('Agg')  # batch PDF generation, no interactive display needed

import pandas as pd
from typing import List, Tuple

import qis
from qis import TimePeriod, MultiPortfolioData, ReportingFrequency
from qis.portfolio.reports.config import fetch_default_report_kwargs
from qis.portfolio.reports.multi_strategy_factsheet import generate_multi_portfolio_factsheet
# reuse the exact universe + reporting-frequency grid from the single-strategy runner
from qis.examples.factsheets.strategy_reporting_frequencies import (UNIVERSE_DATA,
                                                                    REPORTING_FREQUENCIES,
                                                                    load_universe)

# EWMA spans for the vol-parity parameter sweep (one strategy per span), as in multi_strategy.py
SWEEP_SPANS: Tuple[int, ...] = (5, 10, 20, 40, 60, 120)


def generate_volparity_multi_strategy(prices: pd.DataFrame,
                                      benchmark_prices: pd.DataFrame,
                                      group_data: pd.Series,
                                      spans: Tuple[int, ...] = SWEEP_SPANS,
                                      vol_target: float = 0.15,
                                      rebalancing_costs: float = 0.0010
                                      ) -> MultiPortfolioData:
    """vol-parity span sweep: one strategy per span, each backtested once on the full daily panel."""
    returns = qis.to_returns(prices=prices, is_log_returns=True)
    portfolio_datas = []
    for span in spans:
        ra_returns, weights, ewm_vol = qis.compute_ra_returns(returns=returns, span=span, vol_target=vol_target)
        weights = weights.divide(weights.sum(axis=1), axis=0)
        portfolio_data = qis.backtest_model_portfolio(prices=prices,
                                                      weights=weights,
                                                      rebalancing_costs=rebalancing_costs,
                                                      weight_implementation_lag=1,
                                                      ticker=f"VP span={span}")
        portfolio_data.set_group_data(group_data=group_data, group_order=list(group_data.unique()))
        portfolio_datas.append(portfolio_data)
    return MultiPortfolioData(portfolio_datas=portfolio_datas, benchmark_prices=benchmark_prices)


def run_global_multi_strategy_report(use_synthetic_data: bool = False,
                                     short_period_years: int = 3,
                                     add_rates_data: bool = False,
                                     spans: Tuple[int, ...] = SWEEP_SPANS,
                                     add_group_exposures_and_pnl: bool = False,
                                     add_strategy_factsheets: bool = False,
                                     output_path: str = None
                                     ) -> str:
    """render the multi-strategy span-sweep factsheet for the full {frequency} x {long, short} grid on one panel."""
    prices, benchmark_prices, group_data = load_universe(use_synthetic_data=use_synthetic_data)

    end = prices.index[-1]
    time_period_long = TimePeriod(prices.index[0], end)                                 # full history
    time_period_short = TimePeriod(end - pd.DateOffset(years=short_period_years), end)   # last N years

    multi_portfolio_data = generate_volparity_multi_strategy(prices=prices,
                                                             benchmark_prices=benchmark_prices,
                                                             group_data=group_data,
                                                             spans=spans)
    report_name = 'VolParity_span_sweep'   # file-safe base name
    report_title = 'VolParity span sweep'  # human-readable suptitle

    report_windows = (('long', time_period_long), ('short', time_period_short))
    all_figs: List = []
    if output_path is None:
        output_path = qis.local_path.get_output_path()

    for reporting_frequency in REPORTING_FREQUENCIES:
        for span_label, time_period in report_windows:
            report_kwargs = fetch_default_report_kwargs(time_period=time_period,
                                                        reporting_frequency=reporting_frequency,
                                                        add_rates_data=add_rates_data)
            backtest_name = (f"{report_title} \u2013 "
                             f"{reporting_frequency.name} reporting, {span_label} period")
            figs = generate_multi_portfolio_factsheet(multi_portfolio_data=multi_portfolio_data,
                                                      time_period=time_period,
                                                      backtest_name=backtest_name,
                                                      add_group_exposures_and_pnl=add_group_exposures_and_pnl,
                                                      add_strategy_factsheets=add_strategy_factsheets,
                                                      **report_kwargs)
            qis.save_figs_to_pdf(figs=figs,
                                 file_name=f"{report_name}_factsheet_{reporting_frequency.name.lower()}_{span_label}",
                                 local_path=output_path)
            all_figs.extend(figs)
            print(f"rendered {backtest_name}  ({len(figs)} page(s))")

    combined = qis.save_figs_to_pdf(figs=all_figs,
                                    file_name=f"{report_name}_factsheet_all_reporting_frequencies",
                                    local_path=output_path)
    print(f"\ncombined report ({len(all_figs)} pages): {combined}")
    return combined


if __name__ == '__main__':
    # real-data run (yfinance). Pass use_synthetic_data=True to run fully offline.
    run_global_multi_strategy_report(use_synthetic_data=False)

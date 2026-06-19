"""
Global run: reproduce the strategy-vs-benchmark factsheet across every reporting-frequency config
on one and the same data panel.

A single vol-parity strategy and a 1/N (equal-weight) benchmark are backtested once on daily
(business-day) data and wrapped in one MultiPortfolioData, then
``generate_strategy_benchmark_factsheet_plt`` is rendered for the full
DAILY / WEEKLY / MONTHLY / QUARTERLY x {long, short} grid driven by ``fetch_default_report_kwargs``:

    - long  period -> full history,            long  config (regime 'QE', annual heatmap)
    - short period -> last 3 years of data,     short config (regime 'ME', monthly heatmap)

This is the benchmark-factsheet analogue of examples/factsheets/strategy_reporting_frequencies.py
(same {frequency} x {span} sweep for the single-strategy ``generate_strategy_factsheet``) and reuses
its universe and frequency grid. The MultiPortfolioData construction mirrors
examples/factsheets/strategy_benchmark.py.

Each (frequency, span) is written to its own PDF and all pages are also collected into a single
combined PDF, so the same comparison can be inspected across reporting frequencies. Set
``use_synthetic_data=True`` (or run with no network) for a reproducible GBM panel.
"""
import matplotlib
matplotlib.use('Agg')  # batch PDF generation, no interactive display needed

import numpy as np
import pandas as pd

import qis
from qis import TimePeriod, MultiPortfolioData, ReportingFrequency
from qis.portfolio.reports.config import fetch_default_report_kwargs
from qis.portfolio.reports.strategy_benchmark_factsheet import generate_strategy_benchmark_factsheet_plt
# reuse the exact universe + reporting-frequency grid from the single-strategy runner
from qis.examples.factsheets.strategy_reporting_frequencies import (UNIVERSE_DATA,
                                                                    REPORTING_FREQUENCIES,
                                                                    load_universe)


def generate_volparity_multiportfolio(prices: pd.DataFrame,
                                      benchmark_prices: pd.DataFrame,
                                      group_data: pd.Series,
                                      span: int = 30,
                                      vol_target: float = 0.15,
                                      rebalancing_costs: float = 0.0010
                                      ) -> MultiPortfolioData:
    """vol-parity strategy vs equal-weight benchmark, both backtested once on the full daily panel."""
    ra_returns, weights, ewm_vol = qis.compute_ra_returns(returns=qis.to_returns(prices=prices, is_log_returns=True),
                                                          span=span,
                                                          vol_target=vol_target)
    weights = weights.divide(weights.sum(axis=1), axis=0)
    strategy = qis.backtest_model_portfolio(prices=prices,
                                            weights=weights,
                                            rebalancing_costs=rebalancing_costs,
                                            weight_implementation_lag=1,
                                            ticker='VolParity')
    strategy.set_group_data(group_data=group_data, group_order=list(group_data.unique()))

    benchmark = qis.backtest_model_portfolio(prices=prices,
                                             weights=np.ones(len(prices.columns)) / len(prices.columns),
                                             rebalancing_costs=rebalancing_costs,
                                             weight_implementation_lag=1,
                                             ticker='EqualWeight')
    benchmark.set_group_data(group_data=group_data, group_order=list(group_data.unique()))

    return MultiPortfolioData(portfolio_datas=[strategy, benchmark], benchmark_prices=benchmark_prices)


def run_global_benchmark_report(use_synthetic_data: bool = False,
                                short_period_years: int = 3,
                                add_rates_data: bool = False,
                                add_strategy_factsheet: bool = False,
                                output_path: str = None
                                ) -> str:
    """render the strategy-vs-benchmark factsheet for the full {frequency} x {long, short} grid on one panel."""
    prices, benchmark_prices, group_data = load_universe(use_synthetic_data=use_synthetic_data)

    end = prices.index[-1]
    time_period_long = TimePeriod(prices.index[0], end)                                 # full history
    time_period_short = TimePeriod(end - pd.DateOffset(years=short_period_years), end)   # last N years

    multi_portfolio_data = generate_volparity_multiportfolio(prices=prices,
                                                             benchmark_prices=benchmark_prices,
                                                             group_data=group_data)
    strategy_name = multi_portfolio_data.portfolio_datas[0].nav.name
    benchmark_name = multi_portfolio_data.portfolio_datas[1].nav.name

    spans = (('long', time_period_long), ('short', time_period_short))
    all_figs = []
    if output_path is None:
        output_path = qis.local_path.get_output_path()

    for reporting_frequency in REPORTING_FREQUENCIES:
        for span_label, time_period in spans:
            report_kwargs = fetch_default_report_kwargs(time_period=time_period,
                                                        reporting_frequency=reporting_frequency,
                                                        add_rates_data=add_rates_data)
            backtest_name = (f"{strategy_name} vs {benchmark_name} \u2013 "
                             f"{reporting_frequency.name} reporting, {span_label} period")
            figs = generate_strategy_benchmark_factsheet_plt(multi_portfolio_data=multi_portfolio_data,
                                                             time_period=time_period,
                                                             backtest_name=backtest_name,
                                                             add_strategy_factsheet=add_strategy_factsheet,
                                                             add_brinson_attribution=False,
                                                             add_exposures_pnl_attribution=False,
                                                             add_exposures_comp=False,
                                                             add_grouped_exposures=False,
                                                             add_grouped_cum_pnl=False,
                                                             is_grouped=False,
                                                             add_joint_instrument_history_report=False,
                                                             **report_kwargs)
            qis.save_figs_to_pdf(figs=figs,
                                 file_name=(f"{strategy_name}_vs_{benchmark_name}_benchmark_factsheet_"
                                            f"{reporting_frequency.name.lower()}_{span_label}"),
                                 orientation='landscape',
                                 local_path=output_path)
            all_figs.extend(figs)
            print(f"rendered {backtest_name}  ({len(figs)} page(s))")

    combined = qis.save_figs_to_pdf(figs=all_figs,
                                    file_name=(f"{strategy_name}_vs_{benchmark_name}"
                                               f"_benchmark_factsheet_all_reporting_frequencies"),
                                    orientation='landscape',
                                    local_path=output_path)
    print(f"\ncombined report ({len(all_figs)} pages): {combined}")
    return combined


if __name__ == '__main__':
    # real-data run (yfinance). Pass use_synthetic_data=True to run fully offline.
    run_global_benchmark_report(use_synthetic_data=False)

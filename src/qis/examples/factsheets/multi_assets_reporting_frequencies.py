"""
Global run: reproduce the multi-asset factsheet across every reporting-frequency config
on one and the same data panel.

The asset universe (equities / bonds / credit / commodities) is reported once per frequency via
``generate_multi_asset_factsheet`` with SPY as the regime/beta benchmark, for the full
DAILY / WEEKLY / MONTHLY / QUARTERLY x {long, short} grid driven by ``fetch_default_report_kwargs``:

    - long  period -> full history,            long  config (regime 'QE', annual heatmap)
    - short period -> last 3 years of data,     short config (regime 'ME', monthly heatmap)

This is the multi-asset (universe comparison) analogue of
examples/factsheets/strategy_reporting_frequencies.py,
examples/factsheets/strategy_benchmark_reporting_frequencies.py and
examples/factsheets/multi_strategy_reporting_frequencies.py, and reuses their universe and
frequency grid. Unlike those, there is no backtest - the raw asset prices are compared directly -
so the reporting window only slices the panel for each (frequency, span) report. The asset
factsheet mirrors examples/factsheets/multi_assets.py.

Each report is written to its own PDF and all pages are also collected into a single combined PDF,
so the same universe can be inspected across reporting frequencies. Set ``use_synthetic_data=True``
(or run with no network) for a reproducible GBM panel.
"""
import matplotlib
matplotlib.use('Agg')  # batch PDF generation, no interactive display needed

import pandas as pd
from typing import List

import qis
from qis import TimePeriod, ReportingFrequency
from qis.portfolio.reports.config import fetch_default_report_kwargs
from qis.portfolio.reports.multi_assets_factsheet import generate_multi_asset_factsheet
# reuse the exact universe + reporting-frequency grid from the single-strategy runner
from qis.examples.factsheets.strategy_reporting_frequencies import (UNIVERSE_DATA,
                                                                    REPORTING_FREQUENCIES,
                                                                    load_universe)


def run_global_multi_asset_report(use_synthetic_data: bool = False,
                                  short_period_years: int = 3,
                                  add_rates_data: bool = False,
                                  output_path: str = None
                                  ) -> str:
    """render the multi-asset factsheet for the full {frequency} x {long, short} grid on one panel."""
    prices, benchmark_prices, group_data = load_universe(use_synthetic_data=use_synthetic_data)
    benchmark = benchmark_prices.columns[0]  # SPY as the primary regime / beta / scatter benchmark

    end = prices.index[-1]
    time_period_long = TimePeriod(prices.index[0], end)                                 # full history
    time_period_short = TimePeriod(end - pd.DateOffset(years=short_period_years), end)   # last N years

    report_name = 'MultiAsset_universe'   # file-safe base name
    report_title = 'Multi-asset universe'  # human-readable suptitle

    report_windows = (('long', time_period_long), ('short', time_period_short))
    all_figs: List = []
    if output_path is None:
        output_path = qis.local_path.get_output_path()

    for reporting_frequency in REPORTING_FREQUENCIES:
        for span_label, time_period in report_windows:
            report_kwargs = fetch_default_report_kwargs(time_period=time_period,
                                                        reporting_frequency=reporting_frequency,
                                                        add_rates_data=add_rates_data)
            factsheet_name = (f"{report_title} \u2013 "
                              f"{reporting_frequency.name} reporting, {span_label} period")
            fig = generate_multi_asset_factsheet(prices=prices,
                                                 benchmark_prices=benchmark_prices,
                                                 benchmark=benchmark,
                                                 time_period=time_period,
                                                 factsheet_name=factsheet_name,
                                                 **report_kwargs)
            qis.save_figs_to_pdf(figs=[fig],
                                 file_name=f"{report_name}_factsheet_{reporting_frequency.name.lower()}_{span_label}",
                                 local_path=output_path)
            all_figs.append(fig)
            print(f"rendered {factsheet_name}  (1 page)")

    combined = qis.save_figs_to_pdf(figs=all_figs,
                                    file_name=f"{report_name}_factsheet_all_reporting_frequencies",
                                    local_path=output_path)
    print(f"\ncombined report ({len(all_figs)} pages): {combined}")
    return combined


if __name__ == '__main__':
    # real-data run (yfinance). Pass use_synthetic_data=True to run fully offline.
    run_global_multi_asset_report(use_synthetic_data=False)

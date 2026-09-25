---
myst:
  html_meta:
    description: >-
      A practical reference for qis.factsheet: report dispatch, benchmark inputs,
      figure and PDF output, frequency configuration, and offline examples.
---

# Factsheets & reporting

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-06-19](https://github.com/ArturSepp/QuantInvestStrats/commit/a39da9f97a11a25848e1d6ff8bc644f7501d53a8)*

Implemented in [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

Use this reference to select a report and configure its output. The
[methodology overview](factsheets_and_reporting.md) explains the calculation conventions;
the [gallery](gallery.md) shows the four report forms. These are site pages. The text-only
[reporting-frequency note](_included/reporting_frequencies.md) also ships inside the package
as `qis/docs/reporting_frequencies.md`.

## Quick start

This complete example produces figures in memory from a fixed synthetic price panel.
No data download or output directory is needed.

~~~python
import matplotlib.pyplot as plt

import qis
from qis.datasets import generate_synthetic_universe

universe = generate_synthetic_universe(
    start='2018-01-02', end='2025-12-31', seed=20260725, apply_quirks=False
)
prices = universe.prices[['SEQ_US', 'SBD_TSY', 'SCM_GLD']]
figures = qis.factsheet(
    prices,
    benchmark_prices=universe.benchmark_prices,
    reporting_frequency='quarterly',
    add_rates_data=False,
    factsheet_name='Synthetic universe | quarterly reporting',
)
assert figures and all(isinstance(figure, plt.Figure) for figure in figures)
for figure in figures:
    figure.canvas.draw()
    plt.close(figure)
~~~

The pandas input selects a **multi-asset** report even when a separate benchmark is supplied.
A single-strategy report requires a `PortfolioData` object, typically returned by
`qis.backtest_model_portfolio`.

## Report selection

| Input | Report | Wrapped generator |
|---|---|---|
| `Series` or `DataFrame` | Multi-asset universe | `qis.generate_multi_asset_factsheet` |
| `PortfolioData` | Single strategy | `qis.generate_strategy_factsheet` |
| `MultiPortfolioData` | Multiple strategies | `qis.generate_multi_portfolio_factsheet` |
| `MultiPortfolioData`, `kind='strategy_benchmark'` | Strategy versus benchmark portfolio | `qis.generate_strategy_benchmark_factsheet_plt` |

For a pair, portfolio positions 0 and 1 are the default strategy and comparison portfolio.
The generator's `strategy_idx` and `benchmark_idx` keywords select other positions.
Store the reference price panel on `MultiPortfolioData` for market regimes and beta
calculations. This reference series is distinct from the portfolio used for active comparison.

## Facade arguments and output

| Argument | Meaning |
|---|---|
| `benchmark` | Reference column on the multi-asset path. If neither reference argument is supplied, the first input column is used. |
| `benchmark_prices` | Separate reference levels for pandas input or the required reference for a single-strategy report. Multi-portfolio reports read the reference stored on their object. |
| `data_is_returns` | On the multi-asset path, compound the input and any separate benchmark as simple fractional returns. Default false. |
| `reporting_frequency` | `'daily'`, `'weekly'`, `'monthly'`, `'quarterly'`, or a `qis.ReportingFrequency` member. Default monthly. |
| `time_period` | Restrict the displayed span. Omitted, the facade infers the span from the input history. |
| `long_threshold_years` | Use the long preset when the span exceeds this threshold; default 5.0. |
| `add_rates_data` | Attempt to download a rate series for excess-return statistics. Default false; inspect missing/empty rate data if enabled. |
| `factsheet_name` | Report title; mapped to the underlying multi-portfolio report title when appropriate. |
| `file_name`, `local_path` | Save a PDF and return its path when a filename is supplied. Use an existing output directory explicitly. |
| Additional keywords | Forwarded to the selected generator and override preset values; use that generator's documented parameters. |

With no filename, the result is always a **list** of figures, including for a one-page report.
Direct generators can return a figure or a list; the facade normalises their outputs.

To save through the facade, add `file_name='book'` and `local_path=str(output_directory)` to
the call, where `output_directory` is an existing directory you have chosen. The result becomes
a PDF path string. The default PDF helper appends the current date, so do not hardcode the
returned filename. Alternatively, save an existing figure list with `qis.save_figs_to_pdf`;
it accepts `add_current_date=False` when a stable filename is needed.

Saving does not close the figures. Close retained figures after use, or use `plt.close('all')`
in a dedicated batch process after facade PDF output. Page count may grow when appendices,
strategy pages or a long-history heatmap are included.

## Full control

Build frequency-calibrated keywords before calling a generator directly. This independent
example uses the quarterly long preset with an explicit reporting period.

~~~python
import matplotlib.pyplot as plt

import qis
from qis.datasets import generate_synthetic_universe

universe = generate_synthetic_universe(
    start='2018-01-02', end='2025-12-31', seed=20260725, apply_quirks=False
)
prices = universe.prices[['SEQ_US', 'SBD_TSY', 'SCM_GLD']]
period = qis.get_time_period(df=prices)
report_kwargs = qis.fetch_default_report_kwargs(
    time_period=period,
    reporting_frequency=qis.ReportingFrequency.QUARTERLY,
    add_rates_data=False,
)
assert report_kwargs['vol_rolling_window'] == 12
assert report_kwargs['turnover_rolling_period'] == 4
assert report_kwargs['perf_params'].freq == 'QE'
figure = qis.generate_multi_asset_factsheet(
    prices=prices,
    benchmark='SEQ_US',
    time_period=period,
    **report_kwargs,
)
assert isinstance(figure, plt.Figure)
figure.canvas.draw()
plt.close(figure)
~~~

Use the enum for configuration helpers; the facade also parses the human-readable strings.
For manual configuration, `qis.FactsheetConfig` stores the fields and
`qis.fetch_factsheet_config_kwargs` expands them into generator keywords, including
`PerfParams` and a regime classifier. That lower-level helper defaults to
`add_rates_data=True`; pass false for offline use. It also accepts an explicit `rates_data`
series, allowing rates to be supplied without a download.

`FactsheetConfig()` defaults to the daily long preset. Changing a single frequency field
does not recalibrate the remaining fields. Prefer the complete presets and review related
frequencies, windows, annualisation and labels together when overriding them.

## Reporting frequency

The base grids are `B`, `W-WED`, `ME` and `QE`. Window and span lengths count 260, 52, 12 and 4
periods per year; return statistics annualise with 252, 52, 12 and 4 periods per year.
Native-path drawdowns, sampled risk tables, regime bins and heatmaps have distinct roles;
a monthly reporting choice does not imply monthly sampling for every visible panel.

See the [methodology overview](factsheets_and_reporting.md#methodology) for the distinction
and the [full preset table](_included/reporting_frequencies.md) for exact windows and spans.
The frequency guard rejects a base grid finer than the inferred input cadence. It does not
validate stale prices or information content.

## Examples

The [gallery workflow](gallery.md#build-the-four-report-types) constructs all four input
forms offline. The [documentation analytics folder](https://github.com/ArturSepp/QuantInvestStrats/tree/main/tools/docs_analytics)
contains the common generation command, manifest and producers for the documentation images.

The [factsheet example directory](https://github.com/ArturSepp/QuantInvestStrats/tree/main/examples/factsheets)
also contains `strategy.py`, `strategy_benchmark.py`, `multi_strategy.py` and `multi_assets.py`.
Those market-data examples use the optional `data` extra. The four
`*_reporting_frequencies.py` runners demonstrate daily, weekly, monthly and quarterly
reports with long and short spans. Their default entry points attempt market-data downloads;
call their runner functions with `use_synthetic_data=True` for the explicit offline mode.

## Tests

The [reporting convention tests](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/tests/test_reporting_conventions.py)
check presets, guards, panel labels, generator parameters and facade figure/PDF output.
The [report geometry tests](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/tests/test_reporting_goldens.py)
check panel count and axes geometry across reporting frequencies. They do not compare baseline
images. Tables, labels and legends still need visual inspection at the intended page width.

## References

- qis contributors. [Factsheet facade](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/reports/factsheet_facade.py)
  and [configuration implementation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/reports/config.py).
- Sepp, A. qis: Performance analytics, portfolio backtesting, risk analysis, and factsheet
  reporting in Python. [Software citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

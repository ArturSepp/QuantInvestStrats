# Factsheets & reporting

`qis` produces multi-page, print-ready (A4 PDF) performance factsheets for portfolios and asset
universes. There are four report archetypes, a single configuration layer that calibrates every
statistic to a chosen reporting frequency, and a one-call entry point (`qis.factsheet`) for the
common case. The verbose `generate_*_factsheet` functions remain the full-control API.

For rendered examples of all four report types, see the [gallery](gallery.md).

## Quick start

```python
import qis

# one strategy vs a benchmark, monthly reporting, full history -> writes a PDF, returns its path
qis.factsheet(prices, benchmark_prices=spy, reporting_frequency='monthly', file_name='book')

# the same universe at quarterly cadence -> returns the list of figures (no file)
figs = qis.factsheet(prices, benchmark_prices=spy, reporting_frequency='quarterly')
```

`qis.factsheet` selects the report from the input type and either returns the figures or, when
`file_name` is given, writes a PDF and returns its path:

| input | report | generator wrapped |
|---|---|---|
| prices / returns (`Series` or `DataFrame`) | multi-asset universe | `generate_multi_asset_factsheet` |
| `PortfolioData` | single strategy | `generate_strategy_factsheet` |
| `MultiPortfolioData` | multi-strategy | `generate_multi_portfolio_factsheet` |
| `MultiPortfolioData` + `kind='strategy_benchmark'` | strategy vs benchmark | `generate_strategy_benchmark_factsheet_plt` |

Pass `data_is_returns=True` if `data` is returns rather than prices; pass `benchmark` (a column
name) or `benchmark_prices` to set the regime / beta reference; pass `time_period` to override the
default full-history span.

## Full control

The facade only removes boilerplate. To drive the generators directly, build the
frequency-calibrated keyword arguments with `fetch_default_report_kwargs` and spread them in — it
sets every rolling window, regression frequency, regime-classification frequency and annualisation
for the chosen reporting frequency, auto-selecting the long vs short horizon from the reporting
span:

```python
from qis import ReportingFrequency
from qis.portfolio.reports.config import fetch_default_report_kwargs

report_kwargs = fetch_default_report_kwargs(time_period=tp,
                                            reporting_frequency=ReportingFrequency.QUARTERLY)
fig = qis.generate_multi_asset_factsheet(prices=prices, benchmark='SPY',
                                         time_period=tp, **report_kwargs)
```

`make_factsheet_config(reporting_frequency, is_long_period, **overrides)` returns the underlying
`FactsheetConfig` when you want to pin the horizon or override individual fields.

## Reporting frequency

The same book can be reported daily / weekly / monthly / quarterly with every statistic kept
internally consistent, every panel labelled with the frequency it was computed at, and an
up-sampling guard that refuses to report finer than the data supports. See
[reporting_frequencies.md](reporting_frequencies.md) for the two axes, the window / grid / regime
presets, the guard and the per-panel labelling discipline.

## Examples

`examples/factsheets/` has runnable scripts for each report type — `strategy.py`,
`strategy_benchmark.py`, `multi_strategy.py`, `multi_assets.py` — and, for the frequency
convention, the four `*_reporting_frequencies.py` runners that render each report across the full
`DAILY / WEEKLY / MONTHLY / QUARTERLY × {long, short}` grid on one data panel.

## Tests

`qis/tests/test_reporting_conventions.py` locks the calibration presets, the up-sampling guard, the
per-panel frequency labels (all four reports), the frequency-invariant vs frequency-dependent
statistics, and the `FactsheetConfig`-to-generator parameter contract.
`qis/tests/test_reporting_goldens.py` is the optional `pytest-mpl` visual-regression tier (run with
`--mpl` after generating baselines locally).

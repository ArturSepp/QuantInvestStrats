---
myst:
  html_meta:
    description: >-
      Build multi-asset, strategy, strategy-versus-benchmark, and multi-strategy factsheets with
      frequency-calibrated qis reporting.
---

# Factsheets and reporting

Use `qis.factsheet` when a practitioner needs a consistent multi-panel or multi-page review of an
asset universe, a backtested strategy, or several strategies. The facade chooses one of four
report archetypes and applies one reporting-frequency configuration across performance, rolling
risk, regression, regime, and drawdown panels. Use the underlying `generate_*_factsheet`
functions when page composition or individual settings need full control.

## Inputs and the four report archetypes

| Input | Selected report | Additional requirement |
|---|---|---|
| price/return `Series` or `DataFrame` | multi-asset universe | reference defaults to the first column; override with `benchmark` or `benchmark_prices` |
| `PortfolioData` | single strategy | supply `benchmark_prices` |
| `MultiPortfolioData` | multiple strategies | none beyond valid contained portfolios |
| `MultiPortfolioData` with `kind='strategy_benchmark'` | strategy versus benchmark strategy | the object must contain the intended pair |

By default, pandas inputs are positive price or NAV levels. With `data_is_returns=True`, inputs
are simple fractional periodic returns (`0.01` means 1%); the facade geometrically compounds them
to NAVs before reporting. Columns and benchmark identifiers must be unique and aligned to the
intended economics.

## Frequency, annualisation, and missing values

`reporting_frequency` accepts daily, weekly, monthly, or quarterly. The corresponding base grids
and annualisation factors are `B`/260, `W-WED`/52, `ME`/12, and `QE`/4. Window counts, EWM spans,
regressions, and panel labels are calibrated together. Long and short reporting spans use
different window lengths; the [full reporting-frequency convention](
_included/reporting_frequencies.md) records the exact presets.

Default performance tables compute volatility from log returns, report compound p.a. return as
the headline return, and use a zero risk-free rate. `add_rates_data=True` supplies the downloaded
cash series for excess-return statistics; it does not change the input price/return convention.

Factsheets reject a reporting frequency finer than the native data: monthly observations can be
reported monthly or quarterly, but not as daily or weekly information. Running drawdown panels
use the native price path, while frequency-dependent risk tables use the reporting grid; their
labels make that distinction explicit.

NaNs are handled by the underlying analytics rather than by a universal imputation policy.
Ragged starts, internal gaps, stale values, delisted tails, and genuinely low-frequency sleeves
have different meanings. Check the input and the resulting observation windows; the facade does
not certify that a forward-filled value is economically tradable.

## Minimal offline example

The clean synthetic subset keeps the example focused on reporting rather than data repair.

```python
import matplotlib.pyplot as plt

import qis
from qis.datasets import generate_synthetic_universe

universe = generate_synthetic_universe(
    start='2018-01-02', end='2025-12-31', apply_quirks=False
)
prices = universe.prices[['SEQ_US', 'SBD_TSY', 'SCM_GLD']]
figures = qis.factsheet(
    prices,
    benchmark='SEQ_US',
    reporting_frequency='monthly',
)

assert all(isinstance(figure, plt.Figure) for figure in figures)
for figure in figures:
    plt.close(figure)
```

Without `file_name`, the result is a list of Matplotlib `Figure` objects for inspection,
embedding, or custom saving. The call renders figures but does not write a report. With
`file_name='book'`, `qis.factsheet` writes an A4 PDF and returns its path as a string; pass an
explicit `local_path` when the destination matters. A saved PDF is an output artefact, not a
different calculation.

## How to interpret and choose a report

- **Multi-asset:** compare instruments against a reference series, including cumulative return,
  risk-adjusted statistics, rolling risk, drawdowns, correlation, and regimes.
- **Single strategy:** add weights, turnover, costs, and holding-level attribution from a
  `PortfolioData` backtest.
- **Strategy versus benchmark:** compare two portfolio books and their active difference.
- **Multi-strategy:** compare several portfolio variants on shared tables and axes.

The [factsheet gallery](gallery.md) shows the four rendered forms. The wheel-shipped
[factsheet convention note](factsheets.md) maps each facade input to its lower-level
generator.

## Constraints and common failure modes

- A `PortfolioData` single-strategy report without `benchmark_prices` raises `ValueError`.
- A `kind` override must match the input type; it cannot turn raw prices into a portfolio object.
- Reporting finer than the input frequency raises rather than inventing observations.
- Wide universes can exceed a page's legend capacity. The renderer warns; reduce the number of
  series or adjust the documented figure/font settings rather than ignoring a collapsed layout.
- `add_rates_data=True` downloads risk-free-rate data and therefore needs the `data` extra and
  network access. Core/offline reporting leaves it false.
- `file_name` authorises a filesystem write. Omit it for a pure figure-list result and close
  figures in batch or test processes.

## See also

- {doc}`Generated qis.factsheet API <api/generated/qis.factsheet>`
- [Factsheet gallery](gallery.md)
- [Factsheet convention](factsheets.md)
- [Reporting-frequency convention](_included/reporting_frequencies.md)
- [Canonical multi-asset example (requires the `data` extra)](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/factsheets/multi_assets.py)

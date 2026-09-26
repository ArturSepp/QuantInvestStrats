---
myst:
  html_meta:
    description: >-
      Factsheet inputs, report selection, return conventions, reporting grids, and
      reproducible offline examples for qis portfolio and multi-asset analytics.
---

# Factsheets and reporting

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-08-15](https://github.com/ArturSepp/QuantInvestStrats/commit/b04f87a11327bf814cc38ea46934c5b993ad4ef8)*

Implemented in [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

A factsheet combines performance, risk and portfolio diagnostics into a set of related panels.
Its interpretation depends on the input history, benchmark, return convention and observation
grid. A common page layout alone does not make two analyses comparable.

## Overview

`qis.factsheet` selects a report from its input and applies a reporting-frequency preset.
The four forms answer different questions:

- **Multi-asset:** how do instruments compare with a reference series?
- **Single strategy:** how do a portfolio's returns relate to its holdings, turnover and costs?
- **Strategy versus benchmark:** how do two portfolio books differ?
- **Multi-strategy:** how do several portfolio variants compare on shared tables and axes?

The [reference](factsheets.md) lists the API choices. The [gallery](gallery.md) illustrates
the page forms and links to scripts for regenerating their analytics.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | Simple returns build levels; table volatility uses log returns |
| Sampling grid | The reporting preset grid; running drawdowns use the native path |
| Annualisation | $\mathrm{AN}$ of the base grid: 252, 52, 12 or 4 |
| Mean adjustment | Sample moments, demeaned, as in the performance statistics |
| Timing | Units are held between rebalancings; weights at $t$ apply over $(t,t+1]$ |
| Output units | Decimal returns, dimensionless ratios and two-sided turnover |
| qis default | `factsheet(reporting_frequency='monthly')`; the long preset for spans over five years |

### Inputs and the four report archetypes

| Input | Selected report | Benchmark information |
|---|---|---|
| Price or return `Series` or `DataFrame` | Multi-asset universe | A reference column via `benchmark`, or a separate series/panel via `benchmark_prices`. With neither supplied, the first input column is used. |
| `PortfolioData` | Single strategy | Supply `benchmark_prices`. |
| `MultiPortfolioData` | Multiple strategies | Store the reference price panel on the object for regime and beta panels. |
| `MultiPortfolioData` with `kind='strategy_benchmark'` | Strategy versus benchmark portfolio | The first two portfolios are the default pair; the object's reference price panel supplies market regimes/betas. |

`qis.factsheet` dispatches each archetype to a generator that can also be called directly:
`qis.generate_multi_asset_factsheet` for a price universe, `qis.generate_strategy_factsheet`
for one `PortfolioData`, `qis.generate_multi_portfolio_factsheet` for several strategies, and
`qis.generate_strategy_benchmark_factsheet_plt` for a strategy against a benchmark portfolio,
whose exposure panel uses `qis.plot_exposures_strategy_vs_benchmark_stack`. Direct calls take
their windows and grids from a preset such as `qis.FACTSHEET_CONFIG_DAILY_DATA_SHORT_PERIOD`
or from `qis.fetch_default_report_kwargs`.

Use a sorted `DatetimeIndex` and meaningful, unique string column names. Prices and NAVs
are positive levels by default. On the **multi-asset path only**,
`data_is_returns=True` treats both the pandas input and any separately supplied
`benchmark_prices` as simple fractional returns: 0.01 means 1%. Do not combine return
input with an unconverted benchmark price panel under that switch. Portfolio objects already
contain their calculated NAVs and accounting records; the switch does not rebuild them.

Let $P_t$ denote a price/NAV level at observation $t$, and $r_t$ its simple return from the
previous observation. State whether the supplied history includes trading costs, fees and
cash flows. Raw price inputs do not imply any portfolio rebalancing or cost model.

## Methodology

### From returns to levels

For a complete positive history with an explicit starting level,

$$
r_t = \frac{P_t}{P_{t-1}} - 1,
\qquad
P_t = P_0 \prod_{j=1}^{t}(1+r_j).
$$

The multi-asset return switch uses `qis.returns_to_nav` with its simple-return defaults.
Supply an explicit baseline row, and inspect missing observations before conversion.
The helper has its own initialisation and forward-fill behaviour; the switch is not a
general missing-data policy. See [incomplete and mixed-frequency data](
incomplete_and_mixed_frequency_data.md).

### Frequency, annualisation, and missing values

Three time scales matter: the native observation path, the base grid for sampled statistics,
and the regime/heatmap bins. The reporting preset coordinates their settings; it does not
resample every visible curve to one common grid.

| Reporting choice | Base grid | Window periods per year | Annualisation factor |
|---|---|---:|---:|
| Daily | Business day (`B`) | 260 | 252 |
| Weekly | Wednesday week-end (`W-WED`) | 52 | 52 |
| Monthly | Month-end (`ME`) | 12 | 12 |
| Quarterly | Quarter-end (`QE`) | 4 | 4 |

Window lengths and EWM spans are sized from the window column. Volatility and Sharpe ratios
are annualised with `qis.get_annualization_factor`, which gives 252 for business days.

The facade infers the full available reporting span when `time_period` is omitted.
A span **greater than** `long_threshold_years` (default 5.0) selects the long preset;
otherwise it selects the short preset. In a monthly long report, volatility/variance spans,
rolling Sharpe windows and beta spans are 36 observations. Turnover and cost totals roll over
12 observations, regimes use quarterly bins, and the annual-return heatmap uses yearly bins.
In a monthly short report, those risk windows/spans are 12 observations and regimes use
monthly bins. EWM spans are decay parameters; they are not fixed-length rolling windows.

The [reporting-frequency convention](_included/reporting_frequencies.md) records every preset.
Running drawdown and time-under-water panels use the native price path; risk-table drawdowns
use the configured sampling grid, extended by each asset's final observation so that a loss in
the current, unfinished period is included. Earlier intramonth losses can still appear in the
native drawdown panel without appearing in month-end risk statistics. A reporting grid finer than
the inferred input frequency is rejected. This guard cannot detect information hidden by
forward-filled monthly observations in a daily index.

### Performance, costs and turnover

Default performance tables use log returns for volatility, compound annual return as the
headline return, and zero-rate Sharpe statistics. These are separate conventions, described
in [performance analytics and Sharpe ratios](performance_analytics_and_sharpe.md).
`add_rates_data=False` is the offline default. Enabling rate downloads does not change
the input price/return convention; inspect the resulting rate data before interpreting an
excess-return statistic.

Portfolio reports inherit their NAV, costs and turnover from `PortfolioData`. A reporting
frequency changes how those records are summarised, not the underlying rebalancing schedule.
Between rebalancings, qis holds units and weights drift with prices.

For investor-capital turnover, use two-sided executed notional divided by NAV. Managed
futures require full contract notionals through `turnover_unit_notional`. Gross-normalised
book churn and volatility-normalised signal turnover answer different questions; neither
should silently replace realised investor turnover. The [turnover methodology](
turnover_conventions.md) specifies these conventions and their inputs.

## Worked example

### Compounding is not addition

A 10% gain followed by a 10% loss leaves 99% of the initial capital:

$$
\frac{P_2}{P_0} = 1.10 \times 0.90 = 0.99.
$$

This complete monthly example includes a zero-return baseline and performs no filling.

```python
import numpy as np
import pandas as pd

import qis

returns = pd.Series(
    [0.0, 0.10, -0.10],
    index=pd.to_datetime(['2025-01-31', '2025-02-28', '2025-03-31']),
    name='Illustration',
)
nav = qis.returns_to_nav(
    returns, is_log_returns=False, ffill_between_nans=False
)
np.testing.assert_allclose(nav.to_numpy(), [1.0, 1.10, 0.99])
assert np.isclose(nav.iloc[-1] / nav.iloc[0] - 1.0, -0.01)
```

### Minimal offline example

This report uses a clean subset of the frozen synthetic fixture. It is a demonstration of
report construction, not historical market performance. The 2018–2025 sample and seed are fixed.

```python
import matplotlib.pyplot as plt

import qis
from qis.datasets import generate_synthetic_universe

universe = generate_synthetic_universe(
    start='2018-01-02', end='2025-12-31', seed=20260725, apply_quirks=False
)
prices = universe.prices[['SEQ_US', 'SBD_TSY', 'SCM_GLD']]
figures = qis.factsheet(
    prices,
    benchmark='SEQ_US',
    reporting_frequency='monthly',
    add_rates_data=False,
    factsheet_name='Synthetic assets | monthly reporting',
)
assert figures and all(isinstance(figure, plt.Figure) for figure in figures)
for figure in figures:
    figure.canvas.draw()
    plt.close(figure)
```

The result is a multi-asset report. Selecting an input column as a reference does not create
a strategy backtest. The [gallery workflow](gallery.md#build-the-four-report-types) constructs
portfolio objects for the other three forms.

## Implementation in qis

Without `file_name`, the facade returns a list of Matplotlib figures and writes no report
file. With a filename, it saves the generated pages to PDF and returns the path string.
Pass an explicit, existing `local_path`; the default otherwise uses the package's configured
output directory. The PDF helper appends the current date by default. A PDF generation date
does not change the analytics sample cutoff.

Page count depends on the report type, selected appendices and available history.
Do not assume every call produces one page. Close figures after saving or inspection in
batch processes.

For direct generator calls, use `qis.fetch_default_report_kwargs` to build the preset.
Caller overrides take precedence, so overrides to one frequency/window should be reviewed
alongside the related fields. See the [full-control example](factsheets.md#full-control).

The [facade source](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/reports/factsheet_facade.py)
defines dispatch and output handling; the
[configuration source](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/reports/config.py)
defines presets. The [batch analytics scripts](https://github.com/ArturSepp/QuantInvestStrats/tree/main/tools/docs_analytics)
generate figures, supporting tables and provenance together.

## Interpretation and limitations

<a id="how-to-interpret-and-choose-a-report"></a>
<a id="constraints-and-common-failure-modes"></a>

- A single-strategy `PortfolioData` report without `benchmark_prices` raises
  `ValueError`. A `kind` override must match the supplied object; it does not construct
  portfolios from raw prices.
- In multi-portfolio reports, the comparison portfolio and the market reference series are
  separate inputs. Supply the reference panel on `MultiPortfolioData`; the facade's
  standalone `benchmark_prices` argument is used on the raw-pandas and single-strategy paths.
- Missing observations, stale values and ragged histories can give panels different effective
  samples. Neither the facade nor its frequency guard certifies tradability or data quality.
- `add_rates_data=True` attempts a download through the optional `data` extra.
  Missing yfinance or an empty result can leave `rates_data=None`. Verify coverage and
  conventions; do not infer successful excess-return calculation from that flag alone.
- Dense tables and legends require visual review at the intended display size. Layout tests
  check geometry, not readability. Reduce the series count or adjust documented display settings.
- Synthetic reports demonstrate behaviour under specified inputs. Their returns, Sharpe ratios
  and drawdowns are not estimates of an investable strategy's expected performance.

## See also

- [Factsheet API and configuration reference](factsheets.md)
- [Factsheet gallery and regeneration](gallery.md)
- [Frequency and annualisation](frequency_convention_note.md)
- [Portfolio backtesting](portfolio_backtesting.md)
- [Brinson attribution](brinson_attribution.md)
- [Documentation and figure standard](documentation_standard.md)

## References

1. Bacon, C. R. (2008). *Practical Portfolio Performance Measurement and Attribution*, 2nd edition. Wiley. The performance-measurement conventions behind factsheet statistics.
2. Sepp, A. qis: Performance analytics, portfolio backtesting, risk analysis, and factsheet reporting in Python. [Software citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

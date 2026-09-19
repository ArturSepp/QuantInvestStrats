---
myst:
  html_meta:
    description: >-
      Compare the documented inputs and workflows of qis, bt, QuantStats,
      pyfolio-reloaded and open-source vectorbt, with links to primary sources.
---

# Choosing between qis, bt, QuantStats, pyfolio-reloaded, and vectorbt

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-08-15](https://github.com/ArturSepp/QuantInvestStrats/commit/378db166e53ca50c0e5adcdaf5b7c5749ca0e9f5)*

This comparison is part of [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

These libraries overlap, but they start from different analytical objects. The useful comparison
is the input each workflow expects and the output it provides. Official documentation and
repository material were reviewed on **13 September 2026**. This is a documented-workflow
comparison, without a speed benchmark, numerical equivalence test or quality ranking.

## Workflow decision guide

The following starting points are an interpretation of the documented workflows linked below.

| Starting point | Package to examine | Documented workflow |
|---|---|---|
| Prices plus an external target-weight schedule | [qis](portfolio_backtesting.md) | Held-unit portfolio history, analytics, risk and reports. |
| Selection, weighting and rebalance rules to compose inside a framework | [bt](https://pmorissette.github.io/bt/) | Algorithm stacks and nested strategies. |
| A periodic return series | [QuantStats](https://github.com/ranaroussi/quantstats) | Statistics, plots and HTML reports. |
| An existing strategy's returns, positions and transactions | [pyfolio-reloaded](https://pyfolio.ml4trading.io/api-reference.html#pyfolio.tears.create_full_tear_sheet) | Performance and trading diagnostics. |
| Signal or order arrays, including parameter combinations | [vectorbt](https://vectorbt.dev/api/portfolio/base/) | Portfolio simulation and analysis of its records. |

A workflow can use more than one library. Before comparing their numerical outputs, align the
return convention, data grid, execution timing, costs, annualisation, risk-free rate and benchmark.

<a id="qis-and-bt-adjacent-not-interchangeable"></a>

## qis and bt: strategy construction and portfolio analysis

In bt, an ordered stack of `Algo` objects expresses scheduling, selection, weights and
rebalancing. Strategy/security trees support nested portfolios; the framework also records
positions, transactions, costs and performance. Its documentation demonstrates these steps
in a complete synthetic backtest. [bt overview](https://pmorissette.github.io/bt/).

In qis, `backtest_model_portfolio` accepts prices and externally specified weights or a dated
target schedule. It converts targets into held units, applies costs and produces a
`PortfolioData` object for subsequent analytics. The
[backtesting guide](portfolio_backtesting.md) defines that accounting contract.

Both workflows can produce portfolio NAVs. The distinction is where the research rules are
expressed and what analysis follows. A claim that one engine is more accurate or faster would
require a separate controlled comparison with matching assumptions.

<a id="analytics-and-reporting-capability-matrix"></a>

## Documented analytics and reporting

### qis

The core combines [performance and Sharpe statistics](performance_analytics_and_sharpe.md),
[drawdowns and rolling analytics](performance_analytics_and_sharpe.md#rolling-statistics-and-drawdowns),
[covariance-based risk and realised tracking error](tracking_error_and_risk.md),
and [four factsheet forms](factsheets_and_reporting.md). Matplotlib figures can be inspected,
customised or saved to PDF without an optional reporting backend.

Additional focused workflows cover [incomplete histories](incomplete_and_mixed_frequency_data.md),
[private-asset unsmoothing](private_asset_unsmoothing.md), [FX translation and hedging](
fx_hedging_and_market_data.md), [regime/event analysis](stress_testing.md) and
[portfolio attribution](model_layer_attribution.md). These links describe the implementation
contracts and limitations; the presence of a topic does not establish equivalence to another
library's similarly named method.

### QuantStats

The official repository separates return-series statistics, plots and report generation.
Its documented reports accept a benchmark and can produce HTML tear sheets. The method list
includes drawdowns, rolling risk, information ratio and Monte Carlo return simulations.
These operate on periodic returns; period win rates are not automatically trade win rates.
[QuantStats README](https://github.com/ranaroussi/quantstats).

A target-weight execution engine, forecast-covariance portfolio-risk workflow, private-asset
unsmoothing and FX hedge construction were not assessed in that material. This is a limit
of the review, not a claim that no related function or extension exists.

### pyfolio-reloaded

The hosted API documents return, position, transaction, capacity and full tear sheets.
Inputs include noncumulative daily returns and, when available, position values and executed
trades. Documented diagnostics include drawdowns, rolling beta/Sharpe, turnover, slippage sweeps,
round trips, event windows and factor performance attribution.
[pyfolio API](https://pyfolio.ml4trading.io/api-reference.html).

This is analysis of supplied strategy records. An execution simulator or a forecast-covariance
portfolio-risk interface was not assessed. The hosted API identifies an older documentation
build; check the [maintained repository](https://github.com/stefan-jansen/pyfolio-reloaded)
and installed release before relying on an exact signature.

### vectorbt

The open-source portfolio API documents construction from orders, signals and custom order
functions, with fees, fixed fees, slippage and broadcast parameter combinations.
[Portfolio API](https://vectorbt.dev/api/portfolio/base/).
Its return accessors include benchmark-relative statistics and rolling information ratios;
its configurable plot builder supports interactive exploration.
[Returns API](https://vectorbt.dev/api/returns/accessors/),
[plot builder](https://vectorbt.dev/api/generic/plots_builder/).

Its data layer documents policies for aligning differing indexes, including keeping missing
values, dropping dates or raising an error.
[Data alignment](https://vectorbt.dev/api/data/base/#vectorbt.data.base.Data.align_index).
That is distinct from an economic policy for appraisal smoothing or mixed reporting frequencies.
This review does not assess a printable factsheet artifact, private-asset unsmoothing or FX
hedge construction. It covers the open-source documentation, not VectorBT PRO.

## Comparing numerical results

Use a small common input before comparing a full strategy report. Record:

- Prices versus simple/log returns, currency, date coverage and missing-data treatment.
- Target weights versus held weights or orders, rebalance dates, execution lag and cash treatment.
- Fees, turnover denominator, borrowing/funding assumptions and whether results are gross or net.
- Sampling grid, annualisation factor, standard-deviation convention and risk-free-rate series.
- Benchmark definition and common observation window.

A matching metric label is insufficient: compounded annual return divided by volatility differs
from an arithmetic-return Sharpe, and a month-end drawdown can miss intramonth losses.
The qis [frequency](frequency_convention_note.md) and
[Sharpe](performance_analytics_and_sharpe.md) guides make those conventions explicit.

## Where qis fits in its maintainer's stack

qis supplies shared analytics to packages including `optimalportfolios`, `trendfollowing` and
`privateassets`. Portfolio optimisation belongs to the construction package; reusable performance
statistics, risk and reporting belong in qis. The
[software design guide](software_design.md#package-stack-boundary) and
[ecosystem map](https://github.com/ArturSepp/QuantInvestStrats/blob/main/AGENTS.md)
describe the current boundaries.

## How this comparison was made

The review used the linked official documentation and repositories, plus the current qis
source. It did not install or benchmark the other libraries. Positive descriptions identify
documented capabilities; “not assessed” identifies topics outside the verified material.

Documentation on a default branch or a `latest` site can be ahead of a published release.
Consult each project's release metadata when selecting an environment:
[qis](https://pypi.org/project/qis/), [bt](https://pypi.org/project/bt/),
[QuantStats](https://pypi.org/project/quantstats/),
[pyfolio-reloaded](https://pypi.org/project/pyfolio-reloaded/) and
[vectorbt](https://pypi.org/project/vectorbt/).
This page does not maintain a rolling “latest version” table or infer quality from release counts.

## References

- bt contributors. [Official overview and workflow](https://pmorissette.github.io/bt/).
- QuantStats contributors. [Official repository and feature documentation](https://github.com/ranaroussi/quantstats).
- pyfolio-reloaded contributors. [Hosted API](https://pyfolio.ml4trading.io/api-reference.html)
  and [repository](https://github.com/stefan-jansen/pyfolio-reloaded).
- vectorbt contributors. [Portfolio API](https://vectorbt.dev/api/portfolio/base/),
  [returns API](https://vectorbt.dev/api/returns/accessors/) and
  [plot builder](https://vectorbt.dev/api/generic/plots_builder/).
- Sepp, A. qis: Performance analytics, portfolio backtesting, risk analysis, and factsheet
  reporting in Python. [Software citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

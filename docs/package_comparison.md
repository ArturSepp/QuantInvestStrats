---
myst:
  html_meta:
    description: >-
      A neutral, evidence-linked comparison of qis, QuantStats, pyfolio-reloaded, and vectorbt
      for performance analytics, portfolio simulation, risk, and reporting workflows.
---

# Choosing between qis, QuantStats, pyfolio-reloaded, and vectorbt

These libraries overlap, but their primary workflows differ. `qis` combines portfolio analytics,
target-weight backtesting, risk analysis, and factsheet reporting. QuantStats turns return series
into metrics, plots, and HTML reports. pyfolio-reloaded diagnoses an existing backtest from
returns, positions, and transactions. vectorbt builds and explores portfolio simulations from
holdings, signals, or orders at scale.

This comparison was reviewed on **15 August 2026**. It is a guide to documented workflows, not a
performance benchmark or a claim that an unlisted function cannot exist. “Not assessed” means a
dedicated workflow was not identified in the official material reviewed for this page.

## Capability matrix

Every positive cell links to the relevant official documentation or source. A qualified cell
states what is documented without extending it to a broader claim.

| Capability | qis | QuantStats | pyfolio-reloaded | vectorbt |
|---|---|---|---|---|
| Return and performance statistics | [Price/NAV analytics with explicit return and Sharpe conventions](performance_analytics_and_sharpe.md) | [Return-series statistics module](https://github.com/ranaroussi/quantstats#quantstats-portfolio-analytics-for-quants) | [Daily-return statistics in returns tear sheets](https://pyfolio.ml4trading.io/api-reference.html#pyfolio.tears.create_returns_tear_sheet) | [Portfolio and returns-accessor statistics](https://vectorbt.dev/api/portfolio/base/#vectorbt.portfolio.base.Portfolio.returns_stats) |
| Drawdowns and rolling analytics | [Rolling statistics and drawdown episodes](performance_analytics_and_sharpe.md#rolling-statistics-and-drawdowns) | [Drawdown and rolling plot functions](https://github.com/ranaroussi/quantstats#available-methods) | [Rolling beta/Sharpe, drawdowns, and underwater plots](https://pyfolio.ml4trading.io/api-reference.html#pyfolio.tears.create_returns_tear_sheet) | [Drawdown records](https://vectorbt.dev/api/generic/drawdowns/) and [rolling return metrics](https://vectorbt.dev/api/returns/accessors/#vectorbt.returns.accessors.ReturnsAccessor.rolling_information_ratio) |
| Tear sheets or factsheets | [Four figure/report archetypes](factsheets_and_reporting.md) | [Metrics, plots, and HTML tear sheets](https://github.com/ranaroussi/quantstats#creating-a-report) | [Thematic and full tear sheets](https://pyfolio.ml4trading.io/api-reference.html#tear-sheets) | [Configurable portfolio statistics and plots](https://vectorbt.dev/api/portfolio/base/#returns-stats); a document artifact was not assessed |
| Portfolio simulation from weights or orders | [Targets become units that drift between rebalances](portfolio_backtesting.md) | Not assessed; the documented scope is [period return-series analytics](https://github.com/ranaroussi/quantstats#important-period-based-vs-trade-based-metrics) | Post-analysis rather than execution simulation: the full tear sheet [consumes returns, positions, and transactions](https://pyfolio.ml4trading.io/api-reference.html#pyfolio.tears.create_full_tear_sheet) | [Order and target-percentage simulation](https://vectorbt.dev/api/portfolio/base/#vectorbt.portfolio.base.Portfolio.from_orders) |
| Transaction costs | [Fractions of traded notional at rebalance](portfolio_backtesting.md#data-and-calculation-contract) | Not assessed in the documented [return/report modules](https://github.com/ranaroussi/quantstats#quantstats-portfolio-analytics-for-quants) | [Slippage adjustment/sweeps and transaction tear sheets](https://pyfolio.ml4trading.io/api-reference.html#pyfolio.tears.create_txn_tear_sheet); this analyzes supplied trades | [Percentage fees, fixed fees, and slippage in order simulation](https://vectorbt.dev/api/portfolio/base/#vectorbt.portfolio.base.Portfolio.from_orders) |
| Benchmark-relative analytics | [Benchmark tables, alpha/beta, and active risk](tracking_error_and_risk.md) | [Reports accept a benchmark Series or ticker](https://github.com/ranaroussi/quantstats#creating-a-report) | [Benchmark returns, rolling beta, and comparative tear sheets](https://pyfolio.ml4trading.io/api-reference.html#pyfolio.tears.create_returns_tear_sheet) | [Benchmark returns and information ratio](https://vectorbt.dev/api/returns/accessors/#vectorbt.returns.accessors.ReturnsAccessor.information_ratio) |
| Ex-ante risk | [Covariance-based volatility, tracking error, and decomposition](tracking_error_and_risk.md#ex-ante-covariance-risk) | Not assessed in the reviewed [official feature surface](https://github.com/ranaroussi/quantstats#available-methods) | [Factor exposures and realised performance attribution](https://pyfolio.ml4trading.io/api-reference.html#performance-attribution) are documented; forecast-covariance portfolio risk was not assessed | Not assessed in the reviewed [portfolio API](https://vectorbt.dev/api/portfolio/) |
| Ex-post tracking error and information ratio | [Realised EWMA TE plus whole-sample TE/IR](tracking_error_and_risk.md#ex-post-realised-tracking-error) | [`information_ratio` is listed](https://github.com/ranaroussi/quantstats#available-methods); a dedicated tracking-error path was not assessed | Not assessed; named tracking-error or information-ratio APIs were not identified in the reviewed [hosted API](https://pyfolio.ml4trading.io/api-reference.html) | [Information ratio and rolling information ratio](https://vectorbt.dev/api/returns/accessors/#vectorbt.returns.accessors.ReturnsAccessor.rolling_information_ratio); a separately named tracking-error API was not assessed |
| Regime or event analysis | [Regime classification and conditional analytics examples](https://github.com/ArturSepp/QuantInvestStrats/tree/main/examples/regimes) | Not assessed in the reviewed [statistics and plots list](https://github.com/ranaroussi/quantstats#available-methods) | [Predefined “interesting times” event tear sheet](https://pyfolio.ml4trading.io/api-reference.html#pyfolio.tears.create_interesting_times_tear_sheet); this is not a general regime classifier | Not assessed in the reviewed [generic analytics modules](https://vectorbt.dev/api/generic/) |
| Mixed or incomplete histories | [Separate policies for starts, gaps, staleness, delisting, and reporting frequency](incomplete_and_mixed_frequency_data.md) | Not assessed in the reviewed [return-series documentation](https://github.com/ranaroussi/quantstats#important-period-based-vs-trade-based-metrics) | Inputs are documented as [daily returns and positions](https://pyfolio.ml4trading.io/api-reference.html#pyfolio.tears.create_full_tear_sheet); a heterogeneous-frequency policy was not assessed | [Index alignment can keep NaN, drop missing dates, or raise](https://vectorbt.dev/api/data/base/#vectorbt.data.base.Data.align_index) |
| Private-asset unsmoothing | [Rolling AR and static GLM unsmoothing, separate from de-levering](private_asset_unsmoothing.md) | Not assessed in the reviewed [official feature surface](https://github.com/ranaroussi/quantstats#available-methods) | Not assessed in the reviewed [hosted API](https://pyfolio.ml4trading.io/api-reference.html) | Not assessed in the reviewed [returns API](https://vectorbt.dev/api/returns/) |
| FX translation and hedging | [CIP-based reference-currency translation and hedge analytics](fx_hedging_and_market_data.md) | Not assessed in the reviewed [stable feature surface](https://github.com/ranaroussi/quantstats#available-methods) | Not assessed in the reviewed [hosted API](https://pyfolio.ml4trading.io/api-reference.html) | Not assessed in the reviewed [portfolio API](https://vectorbt.dev/api/portfolio/) |
| Plot and report customization | [Matplotlib/seaborn primitives, configs, and figure-list reports](factsheets_and_reporting.md#minimal-offline-example) | [Plot module and optional Plotly conversion](https://github.com/ranaroussi/quantstats#quantstats-portfolio-analytics-for-quants) | [Matplotlib axes and keyword customization](https://pyfolio.ml4trading.io/api-reference.html#plotting-functions) | [Plotly-based configurable subplot builder](https://vectorbt.dev/api/generic/plots_builder/) |
| Primary documented audience | Investment analytics and reporting from [prices, weights, benchmarks, and risk inputs](index.md) | [Quants and portfolio managers profiling return series](https://github.com/ranaroussi/quantstats#quantstats-portfolio-analytics-for-quants) | [Performance and risk analysis of an existing trading algorithm](https://pyfolio.ml4trading.io/) | [Large-scale strategy research and backtesting](https://vectorbt.dev/#why-vectorbt) |

## Workflow decision guide

Choose based on the object you already have and the next decision you need to make:

- **Choose [QuantStats](https://github.com/ranaroussi/quantstats#creating-a-report)** when you
  already have a clean periodic return Series and want a concise
  collection of statistics, plots, or an HTML report. Its documented period-based scope is also
  a useful boundary: trade-entry and fill analysis requires a different input model.
- **Choose [pyfolio-reloaded](https://pyfolio.ml4trading.io/api-reference.html#pyfolio.tears.create_full_tear_sheet)**
  when a Zipline-like backtest has already produced daily returns,
  positions, and transactions and you want tear sheets, turnover/slippage diagnostics, round-trip
  analysis, event windows, or factor performance attribution.
- **Choose [vectorbt](https://vectorbt.dev/#why-vectorbt)** when the central task is generating
  and comparing many holdings, signal, or order simulations with fees and slippage, followed by
  interactive exploration of portfolio, trade, and drawdown records.
- **Choose [`qis`](index.md)** when the workflow starts from price/NAV panels or target weights
  and needs explicitly calibrated performance tables, held-unit portfolio histories, factsheets,
  ex-ante and ex-post risk, incomplete-history handling, private-asset unsmoothing, or FX
  translation.

A dedicated trading simulator is the better fit when order sequencing, fills, partial execution,
or broad parameter sweeps are the main research object. A lighter return-report generator is the
better fit when portfolio construction and risk decomposition are already complete. These choices
can coexist in a research stack, but statistics should be compared across packages after aligning
return convention, frequency, annualisation, risk-free rate, and benchmark definition.

## Where qis fits in its maintainer's stack

`qis` is the analytics and reporting base layer in the maintainer's open-source stack.
`optimalportfolios` consumes it for portfolio construction, while `trendfollowing` consumes it
for strategy analytics. That boundary is intentional: optimization belongs in the construction
package; reusable performance statistics, drawdowns, risk reporting, and factsheets belong here.
See the [repository ecosystem map](https://github.com/ArturSepp/QuantInvestStrats#ecosystem) for
the current package relationships.

Use a dedicated simulator such as [vectorbt](https://vectorbt.dev/api/portfolio/base/) when order
mechanics are the problem. Use a compact reporting layer such as
[QuantStats](https://github.com/ranaroussi/quantstats#creating-a-report) when a return Series is
the complete input. Use
[pyfolio-reloaded](https://pyfolio.ml4trading.io/api-reference.html#pyfolio.tears.create_full_tear_sheet)
when the diagnostic object is an existing strategy's returns, positions, and transactions. `qis`
does not attempt to erase those distinctions.

## How this comparison was made

Stable package versions were read from PyPI on 15 August 2026; technical claims were then checked
against the linked official documentation or repositories. No competitor package was installed,
and no timing or numerical benchmark was run.

| Package | Stable version reviewed | Official technical surface | Qualification |
|---|---:|---|---|
| [`qis`](https://pypi.org/project/qis/) | 5.9.4 (11 Aug 2026) | [Current documentation](https://quantinveststrats.readthedocs.io/en/latest/) | The `latest` site follows the repository's documentation branch and can contain unreleased documentation changes. |
| [QuantStats](https://pypi.org/project/quantstats/) | 0.0.81 (13 Jan 2026) | [Official repository README and source](https://github.com/ranaroussi/quantstats) | The README states that fuller documentation is forthcoming, so unlisted workflows are marked not assessed. |
| [pyfolio-reloaded](https://pypi.org/project/pyfolio-reloaded/) | 0.9.9 (2 Jun 2025) | [Hosted API](https://pyfolio.ml4trading.io/api-reference.html) and [current repository](https://github.com/stefan-jansen/pyfolio-reloaded) | The hosted API header identifies an older documentation build; version status comes from PyPI, and claims are limited to documented APIs. |
| [vectorbt](https://pypi.org/project/vectorbt/) | 1.1.0 (5 Jul 2026) | [Official documentation](https://vectorbt.dev/) | This page compares the open-source package, not VectorBT PRO. |

The matrix does not score packages, count stars or downloads, or infer quality from release
frequency. Defaults and formulas can differ even when two rows use the same label. Verify the
relevant convention in the selected package before comparing numerical output.

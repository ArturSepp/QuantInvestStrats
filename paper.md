---
title: 'qis: reproducible performance analytics and portfolio backtesting in Python'
tags:
  - Python
  - quantitative finance
  - portfolio analytics
  - backtesting
  - performance attribution
  - risk analysis
authors:
  - name: Artur Sepp
    orcid: 0000-0002-7038-1748
    affiliation: 1
affiliations:
  - name: LGT Bank (Schweiz) AG, Zürich, Switzerland
    index: 1
date: 11 October 2026
bibliography: paper.bib
---

# Summary

`qis` computes the quantities a quantitative portfolio manager or investor reports. It turns a
series of portfolio weights, generated externally, and a panel of underlying asset prices into a
simulated portfolio, measures it against a benchmark, and renders the result as a factsheet. In
addition to the backtesting engine, `qis` provides strategy and reporting building blocks:
exponentially weighted covariance, block bootstrap resampling, risk measures,
regime-conditional statistics, currency hedging, and return unsmoothing.

We designed `qis` around a separation of responsibilities: the researcher supplies target
weights, while the library reconstructs holdings and applies explicit accounting and reporting
conventions. Signal generation and portfolio construction remain with the caller.

`qis` is the base layer of a stack of quantitative finance packages — `optimalportfolios`,
`trendfollowing` and `privateassets`, all three public. Each capability is written once and reused
across that stack. It requires only the scientific Python stack [@numpy; @scipy; @pandas;
@statsmodels; @matplotlib; @seaborn], runs on Python 3.10 to 3.14, and is
released under the MIT licence.

# Statement of need

Reported portfolio performance numbers depend on conventions that are rarely stated. Volatility depends on
the return frequency, a Sharpe ratio on whether a risk-free rate is subtracted and on which rate,
and a backtested return on whether holdings drift between rebalancings or are reset every period.
Two implementations of the same strategy can therefore report different numbers without either
containing an error.

We answer each of these questions once, inside the library, and report the answer beside the
number. The return convention is an argument rather than an assumption, three Sharpe conventions are named and selected explicitly, and the
reporting frequency appears on every rendered panel. The simulation holds units between
rebalancings, so the realised weights drift with prices, which is what a portfolio does.

The stationary bootstrap illustrates why conventions need verification. Before version 5.1.0,
`qis` truncated blocks at the sample boundary. Since 26 July 2026 it uses circular wrapping,
as in @politis1994. In the package's fixed synthetic example, truncation draws the first
observation of a 250-period sample at 0.11 times its uniform weight and the first decile at
0.53 times. With drift rising through the sample, the resampled mean exceeds the source mean
by 2.15% per year; circular wrapping reproduces the source mean within 0.32%. These are sample-specific
results, reproduced by the example and pinned by regression tests. Results generated under
the earlier convention therefore require the earlier software version to reproduce.

A backtest needs two inputs: a strategy generator produces weights, and a price panel aligned to
those weights supplies everything else. When each consumer of a result rebuilds the reported
quantities for itself, the Sharpe ratio of one strategy, read from a chart, a summary table and a
factsheet, comes out as three numbers. We build all three from one
`PortfolioData` object constructed from weights and prices, and from the same object come the
comparisons a pipeline actually runs. In addition, we implement two other base pipelines for
strategy profiling: a bumped strategy against its base, and several building blocks of the same
strategy side by side. All three reports are rendered by one reporting layer rather than a figure
written per comparison.

# State of the field

The following comparison covers selected documented workflows, rather than an exhaustive
feature or performance benchmark. The references identify the software and its documentation.

| package | its documented interface takes | and returns |
|---|---|---|
| `pyfolio-reloaded` [@pyfolio] | returns, optionally positions and transactions | tear sheets and round-trip analysis |
| `bt` [@bt] | target weights through `WeighTarget` in an Algo stack | statistics, weights, positions and turnover |
| `vectorbt` [@vectorbt] | orders, signals or an order-generating callback | order and trade records, statistics and plots |
| `Riskfolio-Lib` [@riskfolio] | returns and optimisation constraints | weights, reports and plots |
| `skfolio` [@skfolio] | returns fitted through a scikit-learn interface | weights, cross-validation, risk measures and plots |
| `quantstats` [@quantstats] | returns | metrics, plots and HTML tear sheets |
| `empyrical-reloaded` [@empyrical] | returns | return and risk metrics |
| `ffn` [@ffn] | prices | analytics; its documentation refers simulation to `bt` |
| `PyPortfolioOpt` [@Martin2021] | prices, or expected returns and a risk model | weights, discrete allocation, performance and plots |

`arch` provides a related resampling primitive: `StationaryBootstrap` takes several aligned
arrays, Series or DataFrames in one object, so one index draw propagates across them [@arch].

`qis` differs in where the boundary sits. Only weights and prices cross it, so any generator
producing weights can be measured. The simulation returns a `PortfolioData` object carrying the net asset value, the realised weights, the held units, the
instrument-level profit and loss and the realised costs together, so attribution is a property of
that object rather than a later calculation on a series that has already lost what it needs.

Building a separate library, rather than extending `bt` or `ffn`, served this shared result-object
boundary across the author's research packages. `bt` already accepts target weights, and `ffn`
already supplies analytics; `qis` combines holdings, costs, attribution and comparative reports
without requiring callers to adopt an Algo stack or reconstruct those fields from returns.
This choice trades the flexibility of a general strategy framework for a common accounting
contract; it does not imply that the alternatives cannot be extended.

The integrated analytics include return unsmoothing for appraisal-based valuations
[@getmansky2004], regime-conditional reporting by benchmark return quantile [@Sepp2019], and
paired bootstrap resampling of aligned factor and residual panels. These methods are not
claimed as unique: R's `PerformanceAnalytics` implements Geltner unsmoothing
[@performanceanalytics], and `arch` supplies the paired resampling primitive above.

# Software design

We organise the package into 14 capability groups, which `src/qis/api.py` names.

The backtester is 441 lines. It converts target weights to units at each rebalancing, holds those
units until the next one, applies costs, and returns the result. Holding units preserves natural
weight drift; multiplying every period's returns by unchanged target weights would instead
assume repeated rebalancing. This path dependence motivates a date-sequential `numba` kernel
[@numba], while array-based analytics operate on its outputs. The weights-and-prices boundary suits
portfolio research; order-book events and signal generation remain outside this engine.

The public interface is `qis.__all__`, which holds 474 names; `src/qis/api.py` records that list as a
literal together with a documented core of 166 symbols grouped by capability, and the suite fails
when either record disagrees with the namespace. The broad namespace preserves convenient
imports and compatibility for consumers, at the cost of discoverability; the curated core
provides a smaller entry point without removing existing exports.

Tests check exported plots on synthetic panels, documented core arguments, local documentation
links and the measurements quoted here. Selected offline examples execute in the suite;
static checks cover example imports and calls, including network-dependent examples.
Plot smoke tests establish that figures draw, while layout and readability require visual review.

The suite runs without network access on a core installation, from a frozen seeded simulator
reproducing the defects of real panels: ragged starts, missing observations, stale prices, appraisal
smoothing.

# Research impact statement

Two public projects demonstrate use outside the author's own package stack.
[`tradingstrategy-ai/optimalportfolios-defi`](https://github.com/tradingstrategy-ai/optimalportfolios-defi)
declares `qis` as a dependency and includes an
[example](https://github.com/tradingstrategy-ai/optimalportfolios-defi/blob/ea8970bcd7aff61a22078b7baa55600bbb0bfd31/scripts/example.py)
that uses its backtester, strategy factsheets and report export functions. The
[`luna-007/extra`](https://github.com/luna-007/extra) repository contains a
[backtesting example](https://github.com/luna-007/extra/blob/c71e30753aa47beb3e622a6123a3d9f949228cca/backtest.py)
that uses `qis` to compare leveraged ETF allocations, account for funding costs and generate
a performance factsheet. These repositories provide publicly inspectable examples of external reuse.

The public consumers below are maintained by the author, so they demonstrate research reuse
within this stack rather than independent adoption. `optimalportfolios` declares `qis` a
mandatory dependency and references 137 of its symbols
at 1105 sites. `trendfollowing`, which carries the trend-following work cited below, references 87 symbols
at 428 sites. Both counts are taken at the commits recorded in
`docs/audit/consumers.json` and reproduced by `tools/audit_consumers.py --pinned`.

A third public package, `privateassets`, applies the unsmoothing layer to private-asset
returns; we cite it for the range of asset classes served, not as adoption evidence.

Within the author's research stack, `qis` supports published work on cryptocurrency allocation
[@sepp2023crypto] and robust strategic and tactical asset allocation [@sepp2026robust]. Its
regime-conditional analytics support the working paper on trend-following systems [@sepp2026trend],
while its resampling functionality supports research on capital market assumptions using multi-asset
tradable factors, forthcoming in The Journal of Portfolio Management in 2027 [@sepp2026matf]. The
package also supplies regime-conditional analytics and reporting for *The Convexity Premium of
Portfolio Overlays*, forthcoming in the Journal of Investment Management [@sepp2026convexity]. The
paper's [public companion example](https://github.com/ArturSepp/OptimalPortfolios/tree/main/papers/smart_diversification_joim_2026)
demonstrates these calculations and portfolio-overlay reports using synthetic data.

Commits run from December 2022 to October 2026, with activity in 40 of the 47 calendar months.
External contributions include the numerical, performance and reporting work acknowledged below
and recorded in the repository's public history. We do not infer wider research adoption from
these contributions.

# AI usage disclosure

During July–October 2026, Claude Opus 4.8 and 5.5 and OpenAI Codex assisted with software
development, tests, documentation, and paper preparation. The author made the methodological
and architectural decisions, reviewed and validated the assisted outputs, and remains
responsible for the work.

# Acknowledgements

The author thanks John D. Reynolds for contributions to numerical correctness, performance,
reporting, and regression test coverage in QIS.

# References

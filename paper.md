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

Quantitative portfolio managers and investors evaluate a strategy by simulating a portfolio that
follows it and reporting its risk and return. From the target weights of a strategy and the
prices of its assets, `qis` simulates the portfolio, measures it against a benchmark and renders
the result as a factsheet. Beyond the backtesting engine, the package provides building blocks
for strategies and reports: exponentially weighted covariance, bootstrap resampling, risk
measures, regime-conditional statistics, currency hedging and return unsmoothing.

We designed `qis` around a weights-and-prices boundary. The library reconstructs holdings from
the weights and applies explicit accounting and reporting conventions, while signal generation and
portfolio construction remain with the researcher.

The `qis` package is the base layer of a public stack of quantitative finance packages, whose
other members are `optimalportfolios`, `trendfollowing` and `privateassets`. The package depends
on NumPy, SciPy, pandas, statsmodels, Matplotlib, seaborn, Numba, openpyxl and PyYAML [@numpy;
@scipy; @pandas; @statsmodels; @matplotlib; @seaborn; @numba], runs on Python 3.10 to 3.14 and is
released under the MIT licence.

# Statement of need

The reported performance of a portfolio depends on conventions that are rarely stated. Volatility
depends on the return frequency, and a Sharpe ratio depends on whether a risk-free rate is
subtracted and on which rate. A backtested return depends on whether holdings drift between
rebalancings or are reset every period. Two implementations of the same strategy can therefore
report different numbers without either containing an error.

The target users of `qis` are researchers and portfolio managers who generate weights with their
own models. We settle each convention once, inside the library, and report the choice beside the
number. The caller states the return convention as an argument, and the reporting frequency
appears on every rendered panel. Three Sharpe conventions, with compounded, log and arithmetic
numerators, are named and selected explicitly.

A convention inside one named method can move a result by more than two percentage points a year.
The stationary bootstrap of @politis1994 wraps blocks circularly at the sample boundary, and `qis`
truncated them there before version 5.1.0. In the fixed synthetic example, truncation samples the
first observation of a 250-period sample at 0.11 times its uniform probability and the first
decile at 0.53 times. With drift rising through the sample, the truncated resampled mean exceeds
the sample mean by 2.15% per year, while circular wrapping reproduces it within 0.32% per year.
Both figures are specific to this sample, and regression tests pin them to the example. Results
generated before version 5.1.0 therefore require that version to reproduce.

Conventions also diverge inside one research pipeline, because each consumer of a result may
rebuild the reported quantities. One strategy can then show three Sharpe ratios, one each in a
chart, a summary table and a factsheet.

# State of the field

| Package | Documented input | Documented output |
|---|---|---|
| `pyfolio-reloaded` [@pyfolio] | returns, optionally positions and transactions | tear sheets and round-trip analysis |
| `bt` [@bt] | target weights through `WeighTarget` in an Algo stack | statistics, weights, positions and turnover |
| `cvxportfolio` [@cvxportfolio] | a trading policy, such as `FixedWeights` for supplied target weights, and market data | value, weights, holdings, turnover and statistics |
| `cvxsimulator` [@cvxsimulator] | prices, with units, weights or cash positions set in a loop over time | a portfolio object tracking units, cash and profit, with a snapshot plot |
| `vectorbt` [@vectorbt] | orders, signals or an order-generating callback | order and trade records, statistics and plots |
| `Riskfolio-Lib` [@riskfolio] | returns and optimisation constraints | weights, reports and plots |
| `skfolio` [@skfolio] | returns fitted through a scikit-learn interface | weights, cross-validation, risk measures and plots |
| `quantstats` [@quantstats] | returns | metrics, plots and HTML tear sheets |
| `empyrical-reloaded` [@empyrical] | returns | return and risk metrics |
| `ffn` [@ffn] | prices | analytics, with simulation referred to `bt` |
| `PyPortfolioOpt` [@Martin2021] | prices, or expected returns and a risk model | weights, discrete allocation, performance and plots |

: Documented interfaces of related Python packages, accessed on 11 October 2026. Three of the
eleven accept supplied weights. []{label="tab:field"}

The table omits event-driven engines such as `zipline-reloaded` [@zipline] and `backtrader`
[@backtrader]. Their strategies submit orders from a callback on each bar, including orders that
target a percentage of portfolio value.

The `qis` package differs in where the weights-and-prices boundary sits. Only weights and prices
cross it, so any generator that produces weights can be measured. Among the packages in
\autoref{tab:field}, the closest is `cvxportfolio`, whose `FixedWeights` policy trades to supplied
target weights and holds the positions between them. Its documented backtest result reports value,
weights, holdings, turnover and total profit, within a framework built around optimisation
policies and cost models. The `qis` simulation returns a
`PortfolioData` object that also carries the held units, the profit and loss of each instrument
and the realised costs. Attribution is therefore a property of the object, whereas returns-based
analytics start from a series that no longer carries holdings or costs.

We built a separate library, instead of extending `bt`, `ffn` or `cvxportfolio`, because our
research packages exchange one result object. The `bt` package already accepts target weights,
and `ffn` already supplies analytics. The `qis` package combines holdings, costs, attribution and
comparative reports, and its callers adopt neither an Algo stack nor a policy framework. This
choice gives up the flexibility of a general strategy framework. We accept that cost because one
shared result object keeps the numbers of our packages comparable.

The integrated analytics include return unsmoothing for appraisal-based valuations
[@geltner1993; @getmansky2004], regime-conditional reporting by benchmark return quantile [@Sepp2019] and
paired bootstrap resampling of aligned factor and residual panels. We do not claim these methods
as unique. R's `PerformanceAnalytics` implements the first-order unsmoothing of @geltner1993
[@performanceanalytics], and the `StationaryBootstrap` class of `arch` resamples several aligned
arrays with one index draw [@arch]. The `qis` unsmoothing extends the first-order case to any lag
order, estimated once for the sample or over an exponentially weighted rolling window. Its
bootstrap resamples returns and recompounds them into price panels.

# Software design

The backtester is 441 lines, so its accounting can be audited line by line. The backtester
converts target weights to units at each rebalancing, holds those units until the next one,
applies costs and returns the result. Holding units lets the realised weights drift with prices,
whereas multiplying each period's returns by fixed target weights would assume rebalancing in
every period. The path dependence of held units motivates a date-sequential `numba` kernel
[@numba], and array-based analytics operate on its outputs. The weights-and-prices boundary suits
portfolio research, while order-book events and signal generation remain outside the engine.

We build the chart, the summary table and the factsheet of a strategy from one `PortfolioData`
object, constructed from weights and prices. The same object feeds the three comparisons of a
research pipeline: a strategy against its benchmark, a parameter perturbation against its base
case and the sub-strategies of one strategy side by side. One reporting layer renders all three
comparisons, so no comparison needs plotting code of its own.

The public interface is `qis.__all__`, which holds 474 names. The module `src/qis/api.py` records
that list as a literal, together with a documented core of 166 symbols in 14 capability groups.
The test suite fails when either record disagrees with the namespace. The broad namespace
preserves convenient imports and compatibility for existing consumers, at the cost of
discoverability. The curated core gives new users a smaller entry point without removing any
export.

Tests check exported plots on synthetic panels, the documented arguments of the core, local
documentation links and every measurement this paper quotes. Selected offline examples execute in
the suite, and static checks cover the imports and calls of all examples. Plot smoke tests
establish that figures draw, while layout and readability require visual review. The suite runs
without network access on a core installation, using a frozen, seeded simulator that reproduces
the defects of real panels: ragged starts, missing observations, stale prices and appraisal
smoothing.

# Research impact statement

Two public projects use `qis` outside our own package stack. The repository
[`tradingstrategy-ai/optimalportfolios-defi`](https://github.com/tradingstrategy-ai/optimalportfolios-defi),
described by its owners as an `optimalportfolios` example for DeFi data, declares `qis` as a
dependency. Its
[example](https://github.com/tradingstrategy-ai/optimalportfolios-defi/blob/ea8970bcd7aff61a22078b7baa55600bbb0bfd31/scripts/example.py)
runs the `qis` backtester, strategy factsheet and report export on an optimised ETF portfolio. The
[`luna-007/extra`](https://github.com/luna-007/extra) repository contains a
[backtesting example](https://github.com/luna-007/extra/blob/c71e30753aa47beb3e622a6123a3d9f949228cca/backtest.py)
that uses `qis` to compare leveraged ETF allocations, account for funding costs and generate a
performance factsheet.

We maintain the public consumers below, so they show reuse within this stack and not independent
adoption. The `optimalportfolios` package declares `qis` a mandatory dependency and
references 137 of its symbols at 1105 sites. The `trendfollowing` package, the companion code of
@sepp2026trend, references 87 symbols at 428 sites. Both counts are taken at the commits recorded
in `docs/audit/consumers.json` and reproduced by `tools/audit_consumers.py --pinned`. We also list
`privateassets`, which applies the unsmoothing layer to private-asset returns, for the range of
asset classes served and not as evidence of adoption.

Within our research stack, `qis` supports published work on cryptocurrency allocation
[@sepp2023crypto] and robust strategic and tactical asset allocation [@sepp2026robust]. Its
regime-conditional analytics support the working paper on trend-following systems
[@sepp2026trend]. Its bootstrap resampling supports research on capital market assumptions using
multi-asset tradable factors, forthcoming in The Journal of Portfolio Management [@sepp2026matf].
The package also supplies regime-conditional analytics and reports for research on portfolio
overlays, forthcoming in the Journal of Investment Management [@sepp2026convexity]. The
[public companion example](https://github.com/ArturSepp/OptimalPortfolios/tree/main/papers/smart_diversification_joim_2026)
of that paper demonstrates these calculations on synthetic data.

On 11 October 2026, the repository had 642 GitHub stars and 72 forks, and pypistats.org reported
9342 PyPI downloads over the last month, excluding known mirrors. The commit history runs from
December 2022 to October 2026, with commits in 40 of the 47 calendar months.

# AI usage disclosure

From July to October 2026, Anthropic's Claude (Opus 5, Opus 5.5, Fable 5 and Fable 5.1) and
OpenAI Codex (Sol 5.6) assisted with software development, tests, documentation and paper preparation. The
author made the methodological and architectural decisions, reviewed and validated the assisted
outputs and remains responsible for the work. Assisted changes pass through the same test suite
and continuous integration as all other changes, and the suite checks each measurement this paper
quotes against a recomputed or recorded value.

# Acknowledgements

We thank John D. Reynolds for contributions to numerical correctness, performance, reporting and
regression test coverage in `qis`. This work received no specific funding.

# References

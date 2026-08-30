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
  - name: LGT Bank
    index: 1
date: 28 July 2026
bibliography: paper.bib
---

# Summary

`qis` computes the quantities a quantitative portfolio manager or investor reports. It turns a
series of portfolio weights, generated externally, and a panel of underlying asset prices into a
simulated portfolio, measures it against a benchmark, and renders the result as a factsheet. In
addition to the backtesting engine, `qis` provides strategy and reporting building blocks:
exponentially weighted covariance, block bootstrap resampling, risk measures,
regime-conditional statistics, currency hedging, and return unsmoothing.

We designed `qis` around one assumption: a strategy is a rule for producing weights, and, for the
simulation and reporting workflows `qis` targets, everything downstream of those weights is
reconstruction rather than research. The researcher develops the signal and the portfolio
construction — the things that actually matter for a successful investment process;
the rest belongs to the library.

`qis` is the base layer of a stack of quantitative finance packages — `optimalportfolios`,
`trendfollowing` and `privateassets`, all three public. Each capability is written once and reused
across that stack. It requires only the scientific Python stack, runs on Python 3.10 to 3.14, and is
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

The cost of leaving a convention unstated is measurable rather than notional. The stationary
bootstrap resamples a series in blocks, and a block drawn near the end of the sample either wraps
around to the start or stops there. Both forms run, and neither reports which it used. Under the
truncating form the first observation of a 250-period sample is drawn at 0.11 times its uniform
weight and the first decile at 0.53 times, because a block can only run forwards. On a series
whose drift rises through the sample, that uneven draw reports a mean return 2.15% per year above
the source, while the wrapping form reports it within 0.32%. Two researchers running a stationary
bootstrap on the same data would publish annual returns more than two percentage points apart,
and nothing in either output would explain the gap. The example producing these numbers ships
with the package, and its values are pinned by the test suite.

A backtest needs two inputs: a strategy generator produces weights, and a price panel aligned to
those weights supplies everything else. When each consumer of a result rebuilds the reported
quantities for itself, the Sharpe ratio of one strategy, read from a chart, a summary table and a
factsheet, comes out as three numbers. We build all three from one
`PortfolioData` object constructed from weights and prices, and from the same object come the
comparisons a pipeline actually runs. In addition, we implement two other base pipelines for
strategy profiling: a bumped strategy against its base, and several building blocks of the same
strategy side by side. All three reports are rendered by one reporting layer rather than a figure
written per comparison.

The same discipline answers a second need. A research group accumulates analytics faster than it
consolidates them, and the same method is written again in each project. In our own repositories
we found four independent block bootstrap implementations and two return unsmoothers, all
reimplementing code `qis` already exported. They had diverged, so the same nominal method
produced different numbers in different papers.

The argument extends to agentic AI tools: a researcher who directs one at `qis` for simulation and
reporting, rather than letting it write those steps, holds the conventions fixed across sessions
instead of having each session reimplement them and reproduce the divergence above.

# State of the field

Packages in this area are usually described as splitting into reporting libraries and backtesting
frameworks; their documented interfaces do not, so the comparison below is capability by capability,
at the versions current on 27 July 2026.

| package | its documented interface takes | and returns |
|---|---|---|
| `pyfolio-reloaded` 0.9.9 | returns, and optionally positions, transactions, market data, factor returns and loadings | tear sheets, including round-trip analysis rebuilt from positions and transactions |
| `bt` 1.2.0 | `WeighTarget` takes a target-weight `DataFrame` from the caller, as part of an Algo-stack strategy | a result exposing statistics, weights, positions and turnover |
| `vectorbt` 1.1.0 | `Portfolio.from_orders`, `from_signals` and `from_order_func` take orders, signals, or an order-generating callback produced elsewhere | order and trade records, statistics and plots |
| `Riskfolio-Lib` 7.3.0 | returns and optimisation constraints | weights, with documented `Reports` and `PlotFunctions` modules |
| `skfolio` 0.20.1 | a scikit-learn estimator fitted to returns | weights, walk-forward and combinatorial purged cross-validation, risk measures and plots |
| `quantstats` 0.0.81 | a return series, not trade data | metrics, plots and an HTML tear sheet |
| `empyrical-reloaded` 0.5.12 | a return series | return and risk metrics as functions, with no reporting layer |
| `ffn` 1.1.5 | prices | analytics; its documentation refers simulation to `bt` |
| `PyPortfolioOpt` 1.6.0 | prices, or expected returns and a risk model | weights and a discrete allocation, `portfolio_performance()`, a `plotting` module |

`arch` has no row, being a resampling and econometrics library rather than a portfolio stack, but
it documents the closest primitive: `StationaryBootstrap` takes several aligned arrays, Series or
DataFrames in one object, so one index draw propagates across them.

`qis` differs in where the boundary sits. Only weights and prices cross it, so any generator
producing weights can be measured. The simulation returns a `PortfolioData` object carrying the net asset value, the realised weights, the held units, the
instrument-level profit and loss and the realised costs together, so attribution is a property of
that object rather than a later calculation on a series that has already lost what it needs.

Two capabilities have no counterpart in the documented interfaces above. Return unsmoothing
corrects the serial correlation induced by appraisal-based valuation, which matters for private
assets and hedge funds [@getmansky2004]. Regime-conditional reporting partitions every statistic
by benchmark return quantile [@Sepp2019]. A third, paired block bootstrap resampling — one index
draw across several aligned panels, so a factor and a residual panel resample together
[@politis1994] — has the primitive noted above.

# Software design

We organise the package into 12 capability groups, which `src/qis/api.py` names.

The backtester is 376 lines, and that follows from the design assumption rather than from
compression. Because a strategy is a rule for producing weights, the backtester holds no strategy
logic: it converts target weights to units at each rebalancing, holds those units until the next
one, applies costs, and returns the result. The recursion over dates is compiled with `numba`,
one of the few loops here that cannot be vectorised.

The public interface is `qis.__all__`, which holds 413 names; `src/qis/api.py` records that list as a
literal together with a documented core of 120 symbols grouped by capability, and the suite fails
when either record disagrees with the namespace.

Properties are enforced by tests rather than by convention: every exported plotting function draws a
figure on a synthetic panel; every example references symbols and keyword arguments that exist;
every core symbol documents its arguments and documents none it does not take; documentation links
resolve; and the measurements this paper quotes match a generated record. Every example that needs no data
vendor runs in the suite; the rest are checked statically.

The suite runs without network access on a core installation, from a frozen seeded simulator
reproducing the defects of real panels: ragged starts, missing observations, stale prices, appraisal
smoothing.

# Research impact statement

We state plainly that the stack is the author's. Two public consumers carry this section. `optimalportfolios` declares `qis` a mandatory dependency and references 96 of its symbols
at 765 sites. `trendfollowing`, which carries the trend-following work cited below, references 87 symbols
at 441 sites. Both counts are taken at the commits recorded in
`docs/audit/consumers.json` and reproduced by `tools/audit_consumers.py --pinned`.

A third public package, `privateassets`, applies the unsmoothing layer to private-asset
returns; we cite it for the range of asset classes served, not as adoption evidence.

The capabilities carry named results. The portfolio and optimisation layers support work on robust
strategic and tactical allocation [@sepp2026robust], the backtester and performance layer work on
cryptocurrency allocation [@sepp2023crypto], the regime-conditional layer work on trend-following
systems [@sepp2026trend], and the resampling layer work on capital market assumptions built from
multi-asset tradable factors [@sepp2026matf]. The last two are working papers.

Commits run from December 2022 to August 2026, with activity in 38 of the 45 calendar months.

# AI usage disclosure

The analytical methods, the conventions they implement, and the architecture of the package are
the author's, and no method in `qis` originated from a language model.

In July 2026 the author used Anthropic's Claude (2026 models), through an agentic coding
interface, for a documentation and test infrastructure effort preceding this submission. That work
produced the test suite described above, the docstrings on the documented core, the documentation
configuration, the audit scripts under `tools/`, and repairs of defects the new tests exposed. It also implemented two changes the author specified:
circular block wrapping in the stationary bootstrap, and a fixed-coefficient option in the static
unsmoothing estimator. The co-authored commits can be listed with
`git log --grep='Co-authored-by: Claude'`; their number is not quoted here because it moves.

We drafted this paper with the same assistance, working from measurements the assistant recorded
under `docs/audit/`. Every row of the comparison above, and every bibliographic claim, was
verified against primary documentation on 29 July 2026. The author reviewed, edited and validated all
assisted output, wrote the economic argument in the statement of need, made the design
decisions, and is responsible for the content.

# References

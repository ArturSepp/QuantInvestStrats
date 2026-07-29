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

`qis` computes the quantities a quantitative investor reports. It turns a series of portfolio
weights and a panel of prices into a simulated portfolio, measures that portfolio against a
benchmark, and renders the result as a factsheet. Around those three steps it provides the
supporting analytics reporting depends on: exponentially weighted covariance estimation, block
bootstrap resampling, risk measures, regime-conditional statistics, currency hedging, and return
unsmoothing for assets that report infrequently.

We designed it around one assumption: a strategy is a rule for producing weights, and everything
downstream of those weights is reconstruction rather than research. The researcher keeps the
signal and the portfolio construction; the rest belongs to the library.

`qis` is the base layer of a stack of quantitative finance packages — `optimalportfolios`,
`trendfollowing` and `privateassets`, all three public — and the computational layer beneath
published research in portfolio construction and systematic strategies. Each capability is
written once and reused across that stack. It requires only the scientific Python stack, runs on
Python 3.10 to 3.14, and is released under the MIT licence.

# Statement of need

Reported performance numbers depend on conventions that are rarely stated. Annualisation depends
on the return frequency, and a Sharpe ratio depends on whether a risk-free rate is subtracted and
on which rate. A backtested return depends on whether holdings drift between rebalancings or are
reset every period. Two implementations of the same strategy can therefore report different
numbers without either containing an error, and the difference is invisible in the output.

We answer each of these questions once, inside the library, and report the answer beside the
number. The return convention is an argument rather than an assumption, annualisation follows
from the stated frequency, three Sharpe conventions are named and selected explicitly, and the
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
those weights supplies everything else. The common alternative makes the profit and loss depend
on state held inside the generator, so every consumer of the result rebuilds those quantities
from whatever the generator emitted. The Sharpe ratio of one strategy, read from a chart, a
summary table and a factsheet, then comes out as three numbers. We build all three from one
`PortfolioData` object constructed from weights and prices, and from the same object the
comparisons a pipeline actually runs — a variant against its base, several strategies side by
side — are rendered by one reporting layer rather than a figure written per comparison.

The same discipline answers a second need. A research group accumulates analytics faster than it
consolidates them, and the same method is written again in each project. In our own repositories
we found four independent block bootstrap implementations and two return unsmoothers, all
reimplementing code `qis` already exported. They had diverged, so the same nominal method
produced different numbers in different papers.

The argument now extends to agentic AI tools. A researcher who directs such a tool at `qis` for
simulation and reporting, rather than letting it write those steps, holds the conventions fixed
across sessions; otherwise each session reimplements them and the divergence above reappears one
conversation at a time.

# State of the field

Python packages in this area are often described as splitting into reporting libraries and
backtesting frameworks. Their documented interfaces do not split that way, so the comparison
below is capability by capability, at the versions current on 27 July 2026.

| package | its documented interface takes | and returns |
|---|---|---|
| `pyfolio-reloaded` 0.9.9 | returns, and optionally positions, transactions, market data, factor returns and loadings | tear sheets, including round-trip analysis rebuilt from positions and transactions |
| `bt` 1.2.0 | `WeighTarget` takes a target-weight `DataFrame` from the caller | a backtest result; the strategy need not be written as Algo blocks |
| `vectorbt` 1.1.0 | `Portfolio.from_orders` takes arrays of size, price, fees and direction produced elsewhere | order and trade records, statistics and plots |
| `Riskfolio-Lib` 7.3.0 | returns and optimisation constraints | weights, with documented `Reports` and `PlotFunctions` modules |
| `skfolio` 0.20.1 | a scikit-learn estimator fitted to returns | weights, walk-forward and combinatorial purged cross-validation, risk measures and plots |
| `quantstats` 0.0.81 | a return series, not trade data | metrics, plots and an HTML tear sheet |
| `empyrical-reloaded` 0.5.12 | a return series | return and risk metrics as functions, with no reporting layer |
| `ffn` 1.1.5 | prices | analytics; its documentation refers simulation to `bt` |
| `PyPortfolioOpt` 1.6.0 | prices, or expected returns and a risk model | weights and a discrete allocation, `portfolio_performance()`, a `plotting` module |

`qis` differs in where the boundary sits. Only weights and prices cross it, so any generator
producing weights can be measured. The simulation returns a `PortfolioData` object carrying the net asset value, the realised weights, the held units, the
instrument-level profit and loss and the realised costs together, so attribution is a property of
that object rather than a later calculation on a series that has already lost what it needs.

We found no counterpart in the documented interfaces above for three capabilities. Return
unsmoothing corrects the serial correlation induced by appraisal-based valuation, which matters
for private assets and hedge funds [@getmansky2004]. Regime-conditional reporting partitions every
statistic by benchmark return quantile [@Sepp2019]. Paired block bootstrap resampling applies one
index draw to several aligned panels, so a factor and a residual panel resample together
[@politis1994].

# Software design

We organise the package into 12 capability groups: among them performance statistics, portfolio
and backtesting, factsheets, exponentially weighted estimation, currency hedging, regime
reporting, resampling and unsmoothing.

The backtester is 323 lines, and that follows from the design assumption rather than from
compression. Because a strategy is a rule for producing weights, the backtester holds no strategy
logic: it converts target weights to units at each rebalancing, holds those units until the next
one, applies costs, and returns the result. The recursion over dates is compiled with `numba`,
one of the few paths here where a loop cannot be vectorised.

The public interface is `qis.__all__`, fixed when the package is imported so that it does not
depend on which submodules a process has loaded. It holds 403 names, nine of them subpackage
bindings. `qis/api.py` records that list as a literal, so a change to the surface appears in a
diff rather than as a count that moves, and records a documented core of 116 symbols grouped by
capability. The suite fails when either record disagrees with the namespace.

Properties are enforced by tests rather than by convention: every exported plotting function
draws a figure on a synthetic panel; every example references symbols and keyword arguments that
exist; every core symbol documents its arguments and documents none it does not take; the
recorded export list matches the namespace; documentation links resolve; and the measurements
this paper quotes match a generated record. The convention measurement above is pinned to its
published values rather than only executed. Nine of the 58 examples need no data vendor and are
run; the other 49 are checked statically.

The suite runs without network access on a core installation: its data comes from a frozen
seeded simulator reproducing the defects of real panels: ragged starts, missing observations,
stale prices, appraisal smoothing.

# Research impact statement

We state plainly that the stack is the author's. Two consumers carry this section, both public
and both using the package deeply. `optimalportfolios`, which implements portfolio optimisation
solvers and rolling backtests, declares `qis` a mandatory dependency and references 94 of its
symbols at 683 sites. `trendfollowing`, which carries the code behind the trend-following work
cited below, references 87 symbols at 443 sites. Both counts are taken at the commits recorded in
`docs/audit/consumers.json` and reproduced by `tools/audit_consumers.py --pinned`, whose docstring
defines a symbol and a site.

A third public package, `privateassets`, applies the unsmoothing layer to private-asset returns;
it is recent and its dependency small, so we cite it for the range of asset classes served rather
than as evidence of adoption. Private repositories account for further use, among them a
production allocation system; we give no figure for them, because a count a reader cannot
reproduce is not evidence.

The capabilities carry named results. The portfolio and optimisation layers support work on
robust strategic and tactical allocation [@sepp2026robust], the backtester and performance layer
work on cryptocurrency allocation [@sepp2023crypto], the regime-conditional layer work on
trend-following systems [@sepp2026trend], and the resampling layer work on capital market
assumptions built from multi-asset tradable factors [@sepp2026matf]. The last two are working
papers, under submission at the SIAM Journal on Financial Mathematics and at The Journal of
Portfolio Management.

Development has been continuous rather than concentrated: commits run from December 2022 to
July 2026, with activity in 37 of the 44 calendar months in that span.

# AI usage disclosure

Generative AI assisted this project, and we state where.

The analytical methods, the conventions they implement, and the architecture of the package are
the author's, and no method in `qis` originated from a language model.

Between 25 and 28 July 2026 the author used Anthropic's Claude, through an agentic coding
interface, for a documentation and test infrastructure effort preceding this submission. That work
produced the test suite described above, the docstrings on the documented core, the documentation
configuration, the audit scripts under `tools/`, and the repair of four defects the new tests
exposed in exported plotting functions. It also implemented two changes the author specified:
circular block wrapping in the stationary bootstrap, and a fixed-coefficient option in the static
unsmoothing estimator. The co-authored commits can be listed with
`git log --grep='Co-authored-by: Claude'`; their number is not quoted here because it moves.

We drafted this paper with the same assistance, working from measurements the assistant recorded
under `docs/audit/`. Every row of the comparison above, and every bibliographic claim, was
verified against primary documentation on 27 July 2026. The author wrote the economic argument in
the statement of need and is responsible for the content.

# References

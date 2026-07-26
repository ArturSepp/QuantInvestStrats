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
date: 26 July 2026
bibliography: paper.bib
---

# Summary

`qis` computes the quantities a quantitative investor reports. The package turns a series of
portfolio weights and a panel of prices into a simulated portfolio, measures that portfolio
against a benchmark, and renders the result as a factsheet. Around those three steps it provides
the supporting analytics that reporting depends on, including exponentially weighted covariance
estimation, block bootstrap resampling, regime-conditional statistics, currency hedging, and
return unsmoothing for assets that report infrequently.

We designed the package around one assumption: a strategy is a rule for producing weights, and
everything downstream of those weights is reconstruction rather than research. `qis` performs
that reconstruction, so a researcher writes the rule and the package supplies the simulation, the
statistics, the conventions, and the exhibits.

`qis` serves as the base layer of a stack of quantitative finance packages, three of them public,
and as the computational layer beneath published research in portfolio construction and
systematic strategies. Each capability is written once and reused across that stack. It requires only the scientific Python stack, runs on Python 3.10 to 3.14,
and is released under the MIT licence.

# Statement of need

Reported performance numbers depend on conventions that are rarely stated. Annualisation depends
on the return frequency, and a Sharpe ratio depends on whether a risk-free rate is subtracted and
on which rate. A backtested return depends on whether holdings drift between rebalancings or are
reset every period. Two implementations of the same strategy can therefore report different
numbers without either containing an error, and the difference is invisible in the output.

We answer each of these questions once, inside the library, and report the answer beside the
number. The return convention is an argument rather than an assumption, and annualisation follows
from the stated frequency. Three Sharpe conventions are named and selected explicitly, and the
reporting frequency appears on every rendered panel. The simulation holds units between
rebalancings, so the realised weights drift with prices, which is what a portfolio does.

The cost of leaving a convention unstated is measurable rather than notional. The stationary
bootstrap resamples a series in blocks, and a block drawn near the end of the sample either wraps
around to the start or stops there. Both forms run, and neither reports which it used. Under the
truncating form the first observation of a 250-period sample is drawn at 0.11 times its uniform
weight and the first decile at 0.53 times, because a block can only run forwards. Applied to a
series whose drift rises through the sample, that uneven draw reports a mean return 2.15% per
year above the source series, while the wrapping form reports it within 0.32%. Two researchers
running a stationary bootstrap on the same data would publish annual returns differing by more
than two percentage points, and nothing in either output would explain the gap. The example that
produces these numbers ships with the package and runs in the test suite.

A backtest needs two inputs. A strategy generator produces weights, and a price panel aligned to
those weights supplies everything else. Returns, turnover, realised trading costs and attribution
are reconstruction rather than research, and we reconstruct them once inside the library. The
common alternative makes the profit and loss depend on state held inside the strategy generator.
That design obliges every consumer of the result to rebuild the same quantities from whatever the
generator chose to emit, and the rebuilt versions stop agreeing with one another. The consequence
appears in a place every practitioner recognises. The Sharpe ratio of one strategy, read from a
chart, from a summary table and from a factsheet, comes out as three numbers. We build all three
from one `PortfolioData` object constructed from weights and prices, so they cannot disagree.
Every other method in the package is a building block on the same terms.

The building-block discipline is the second need, because a research group accumulates analytics
faster than it consolidates them, and the same method is then written again in each project. In
our own repositories we found four independent block bootstrap implementations and two return
unsmoothers, all of them reimplementing code that `qis` already exported, and one carrying a
comment that recorded the intention to move it into `qis`. Those implementations had diverged, so
the same nominal method produced different numbers in different papers. A general-purpose
analytics layer is worth maintaining because the alternative is not an absence of code, but
several copies of it that no longer agree.

# State of the field

Python packages in this area take one of two designs, and `qis` sits between them.

Reporting libraries take a return series and compute statistics from it. `quantstats` and
`pyfolio-reloaded` render tear sheets, and `empyrical-reloaded` supplies the statistics as
functions. The simulation that produced the series happens outside the library, so its
conventions are the caller's responsibility and the library cannot report them. `ffn` (1.1.5) works from prices, which moves the return convention inside the library but leaves
the simulation outside it.

Backtesting frameworks perform the simulation, and express the strategy inside the framework.
`bt` (1.2.0) composes strategies from Algo blocks, and `vectorbt` (1.0.0) builds portfolios from
signals or orders. `PyPortfolioOpt` and `Riskfolio-Lib` construct portfolios without a general
reporting layer.

`qis` performs the simulation but keeps the strategy outside it. Only weights and prices cross
the boundary, so any generator that produces weights can be measured, and the package needs no
opinion about how the weights were formed. The simulation returns a `PortfolioData` object
carrying the net asset value, the realised weights, the held units, the instrument-level profit
and loss, and the realised costs together. Attribution is then a property of that object rather
than a later calculation on a series that has already lost the information attribution requires.

[TODO: versions above were checked on 2026-07-26 against the current PyPI release of each
package. Recheck at submission and update the numbers. `quantstats`, `pyfolio-reloaded`,
`empyrical-reloaded`, `PyPortfolioOpt` and `Riskfolio-Lib` are described from their documented
entry points rather than from a version-pinned check.]

We are not aware of a counterpart in the packages listed above for three capabilities. Return
unsmoothing corrects the serial correlation induced by appraisal-based valuation, which matters
for private assets and hedge funds [@getmansky2004]. Regime-conditional reporting partitions
every statistic by benchmark return quantile. Paired block bootstrap resampling applies one index
draw to several aligned panels, so a factor panel and a residual panel resample together
[@politis1994].

# Software design

We organise the package into eleven capability groups, among them performance statistics,
portfolio and backtesting, factsheets, exponentially weighted estimation, currency hedging,
regime reporting, resampling and unsmoothing.

The backtester is 273 lines, and that number follows from the design assumption rather than from
compression. Because a strategy is a rule for producing weights, the backtester holds no strategy
logic: it converts target weights to units at each rebalancing, holds those units until the next
one, applies costs, and returns the result. We compile the recursion over dates with `numba`,
since it is one of the few paths in the package where a loop cannot be replaced by a vectorised
operation.

The public interface is defined as the symbols exported from the package namespace, currently
403, and a smaller documented core of 116 symbols is recorded in machine-readable form and
grouped by capability. Both the documentation build and the test suite read that record, so the
documented surface cannot drift away from the published one.

We enforce four properties through the test suite rather than through convention. Every exported
plotting function must run on a synthetic panel and produce a figure, and every example must
reference symbols and keyword arguments that exist. Every symbol in the documented core must
document its arguments, and every documented argument must exist in the signature. The suite also executes the
examples that need no data vendor, including the one that produces the convention measurement
quoted above, so those numbers cannot drift away from the code that generates them. The suite
contains 929 tests and runs without network access on a core installation, because its data comes
from a frozen seeded simulator that reproduces the defects of real panels, including ragged start
dates, missing observations, stale prices, and appraisal smoothing.

# Research impact statement

`qis` is the base layer of a stack of packages, and we state plainly that the stack is the
author's. Two consumers carry this section, because both are public and both use the package
deeply.
`optimalportfolios`, which implements portfolio optimisation solvers and rolling backtests,
declares `qis` as a mandatory dependency and calls 73 of its symbols at 541 call sites.
`TrendFollowingSystems`, which carries the code behind the trend-following work cited below,
calls 96 symbols at 538 sites. A reader can reproduce both counts from a clone with a short
script.

A third public package, `privateassets`, applies the unsmoothing layer to private-asset returns.
It is recent and its dependency is small, so we cite it for the range of asset classes served
rather than as evidence of adoption. Three private repositories account for the remainder,
among them a production asset allocation system. Across six consumers we measure 2,738 call
sites covering 240 distinct symbols.

The capabilities carry named results. The portfolio and optimisation layers support work on
robust strategic and tactical allocation [@sepp2026robust], the backtester and performance layer
support work on cryptocurrency allocation [@sepp2023crypto], and the regime-conditional layer
supports work on trend-following systems [@sepp2025trend]. The trend-following paper is under
submission at the SIAM Journal on Financial Mathematics.

Development has been continuous rather than concentrated. The repository holds 278 commits made
between December 2022 and July 2026, with activity in 37 of the 44 calendar months in that span.

# AI usage disclosure

Generative AI assisted this project, and we state where.

The analytical methods, the conventions they implement, and the architecture of the package are
the author's, and no method in `qis` originated from a language model.

On 25 and 26 July 2026 the author used Anthropic's Claude, through an agentic coding interface,
for a documentation and test infrastructure effort preceding this submission. That work produced the test suite described above, the
docstrings on the documented core, the documentation configuration, and the repair of four
defects the new tests exposed in exported plotting functions. It also implemented two changes the author specified: circular block
wrapping in the stationary bootstrap, and a fixed-coefficient option in the static unsmoothing
estimator. Eight commits in the repository carry Claude as a co-author, and a reader can identify
them with `git shortlog`.

We drafted this paper with the same assistance, working from measurements the assistant produced
and recorded. The author wrote the passage giving the economic argument in the statement of need,
verified every comparative and bibliographic claim, and is responsible for the content.

# References

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
  - name: "[TODO: affiliation as it should appear in print]"
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

`qis` serves as the base layer of a stack of quantitative finance packages, two of which are
public, and as the computational layer beneath published research in portfolio construction and
systematic strategies. It requires only the scientific Python stack, runs on Python 3.10 to 3.14,
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

[TODO: Artur to write the economic-mechanism paragraph here, per the house rule that this passage
is drafted without AI assistance. The argument to make is why convention drift matters in
practice rather than in principle, anchored in a concrete case where two correct implementations
disagree.]

The second need is compositional, because a research group accumulates analytics faster than it
consolidates them, and the same method is then written repeatedly across projects. We found four
independent block bootstrap implementations and two return unsmoothers across four of our own
repositories, all of them reimplementing code that `qis` already exported, and one carrying a
comment that recorded the intention to move it into `qis`. Those implementations had diverged, so
the same nominal method produced different numbers in different papers. A general-purpose
analytics layer is worth maintaining because the alternative is not an absence of code, but
several copies of it that no longer agree.

# State of the field

Several Python packages report portfolio performance. `quantstats` and `pyfolio-reloaded` produce
tear sheets from a return series, while `empyrical-reloaded` supplies performance statistics as
functions. `ffn` and `bt` combine statistics with a strategy framework, and `vectorbt` and
`Riskfolio-Lib` include reporting layers beneath a broader backtesting or optimisation library.
`PyPortfolioOpt` covers portfolio construction without a reporting layer.

`qis` differs in what it treats as the unit of work. The packages above take a return series as
their input, which places the simulation that produced the series outside the library and makes
its conventions the caller's responsibility. `qis` takes weights and prices, performs the
simulation, and returns a `PortfolioData` object carrying the net asset value, the realised
weights, the held units, the instrument-level profit and loss, and the realised costs together.
Attribution is then a property of that object rather than a later calculation on a series that
has already lost the information attribution requires.

We are not aware of a counterpart in the packages listed above for three capabilities. Return
unsmoothing corrects the serial correlation induced by appraisal-based valuation, which matters
for private assets and hedge funds [@getmansky2004]. Regime-conditional reporting partitions
every statistic by benchmark return quantile. Paired block bootstrap resampling applies one index
draw to several aligned panels, so a factor panel and a residual panel resample together
[@politis1994].

[TODO: verify each comparative claim in this section against the current release of every named
package before submission, and record the version checked and the date.]

# Software design

We organise the package into eleven capability groups: performance statistics, portfolio and
backtesting, factsheets, exponentially weighted estimation, market data and currency hedging,
regime reporting, bootstrap resampling, unsmoothing, plotting, date handling, and data frame
utilities.

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
document its arguments, and every documented argument must exist in the signature. The suite
contains 928 tests and runs without network access on a core installation, because its data comes
from a frozen seeded simulator that reproduces the defects of real panels, including ragged start
dates, missing observations, stale prices, and appraisal smoothing.

# Research impact statement

`qis` is the base layer of a stack of packages. `optimalportfolios`, which implements portfolio
optimisation solvers and rolling backtests, declares `qis` as a mandatory dependency and calls 73
of its symbols at 541 call sites. `TrendFollowingSystems`, which carries the code for a paper in
submission on diversification of systematic strategies, calls 96 symbols at 538 sites. Both
packages are public, so a reader can reproduce both counts from a clone.

Four further repositories depend on the package, among them a production asset allocation system
and four paper-specific repositories. Across six consumers we measure 2,738 call sites covering
240 distinct symbols. The capabilities carry named results: regime-conditional reporting supports
published work on diversification, the bootstrap layer supports work on achievable Sharpe ratios,
and the portfolio layer supports work on mandate architecture.

[TODO: list the four to six papers with venue, year, volume and pages, each verified. Do not
include any citation that has not been checked.]

Development has been continuous rather than concentrated. The repository holds 278 commits made
between December 2022 and July 2026, with activity in 37 of the 44 calendar months in that span.

# AI usage disclosure

Generative AI assisted this project, and we state where.

The analytical methods, the conventions they implement, and the architecture of the package are
the author's, and no method in `qis` originated from a language model.

On 25 and 26 July 2026 the author used Anthropic's Claude, through an agentic coding interface,
for a documentation and test infrastructure effort preceding this submission. That work produced
the test suite described above, the docstrings on the documented core, the generated
documentation configuration, and the repair of four defects that the new tests exposed in
exported plotting functions. It also implemented two changes the author specified: circular block
wrapping in the stationary bootstrap, and a fixed-coefficient option in the static unsmoothing
estimator. Eight commits in the repository carry Claude as a co-author, and a reader can identify
them with `git shortlog`.

We drafted this paper with the same assistance, working from measurements the assistant produced
and recorded. The author wrote the passage giving the economic argument in the statement of need,
verified every comparative and bibliographic claim, and is responsible for the content.

# Acknowledgements

[TODO: acknowledgements, or delete this section.]

# References

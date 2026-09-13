---
myst:
  html_meta:
    description: >-
      Build instrument portfolios with funds, stocks, calls, puts and futures; evaluate factor
      scenarios and custom payoffs; generate a standard portfolio stress report with qis.
---

# Instrument portfolios and standard stress reports

*[author / affiliation / date — placeholder]*

A workflow guide for [qis](https://github.com/ArturSepp/QuantInvestStrats);
see the [software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

`qis.portfolio.stress` evaluates an absolute-position snapshot under an assigned factor model.
Funded assets, signed intrinsic calls/puts, futures and consumer-owned composite payoffs share
one portfolio interface. The numerical result feeds the same PDF, Excel and CSV report.

Use this interface when original position IDs, option strikes, futures multipliers, currencies
or payoff nonlinearities must survive the scenario calculation. For a funded exposure matrix
or a standalone conditional factor calculation, the existing [factor stress helpers](stress_testing.md)
remain available. Both paths use the canonical [RiskModel](tracking_error_and_risk.md) analytics.

These interfaces and report enhancements are included from 5.30.1. Earlier local stress
branch version labels did not identify published distributions containing this framework.

## Start with the runnable examples

Both examples run on the core installation and the repository's synthetic data generator.
They require no credentials, market-data service, estimator package or private consumer code.
Their seven-factor names are illustrative; the inputs are not a production MATF calibration.

| Example | What it demonstrates |
|---|---|
| `examples/portfolios/instrument_portfolio_stress.py` | Funded and mixed portfolios; all four primitive types; continuing accumulator/decumulator legs; EUR local quotes with USD fitted responses; Credit and Carry families; monthly replay; four conditional grids; standard reporting. |
| `examples/portfolios/composite_payoff_stress.py` | A terminal-knockout wrapper implementing `HoldingPayoff`, retaining source marks, vanilla valuation and shared-response risk; independent terminal-P&L and finite-difference checks. |

From the source checkout, using its configured Python interpreter:

~~~console
python -m examples.portfolios.instrument_portfolio_stress
python -m examples.portfolios.composite_payoff_stress
~~~

By default the examples compute and verify results and write no files. To generate reports,
supply fresh output directories outside the source checkout:

~~~console
python -m examples.portfolios.instrument_portfolio_stress --case all --output-dir /path/to/new/instrument_reports
python -m examples.portfolios.composite_payoff_stress --output-dir /path/to/new/composite_report
~~~

`--case all` creates `funded/` and `mixed/`; `--case funded` and `--case mixed` select one.
The custom-payoff example writes directly to its supplied directory. On Windows, replace the
example paths with quoted absolute paths on the local C drive and use the repository's external
interpreter. An existing target is rejected before report output is written.

## Inputs, objects and outputs

| Step | Public object or function | Caller supplies | QIS produces |
|---|---|---|---|
| 1 | `RiskModel` | Dated response covariance, factor loadings, factor covariance, residual variances and optional `FactorGroupSpec` definitions | One assigned model shared by holdings and scenario calculations |
| 2 | `Underlying` | Actual quote ID, positive local spot, currency, fitted response ID and `ResponseBasis` | A distinct valuation quote connected to a possibly shared risk response |
| 3 | `PortfolioHolding` | Original ID/name, observed mark, signed `InstrumentLeg` terms or `HoldingPayoff` | One attributed holding, regardless of its number of synthetic legs |
| 4 | `InstrumentPortfolio` | Holdings, model/position dates, quote and FX registries, positive reporting denominator | `get_mtm`, `get_pnl`, batch `evaluate` and current `response_jacobian` |
| 5 | `StressScenarios` | Factor/family anchors, simple/log convention and completion policy | Complete factor log-shock vectors; independent or jointly conditional |
| 6 | `run_portfolio_stress_test` | Portfolio, requests, optional monthly history and named grids | Detached `PortfolioStressResult`: full valuations, exposures, local risk, attribution and audit tables |
| 7 | `generate_portfolio_stress_report` | Completed result and `StressReportConfig` | Eleven analysis PDF pages, optional coverage, final notation guide, all numerical tables and artifact hashes |

The application owns quote acquisition, factor estimation, unsmoothing, contract interpretation
and settlement/credit decisions. A consumer can construct `RiskModel` directly or use its own
estimation adapter. No estimator is called by scenario evaluation or report rendering.

The [shipped instrument-portfolio guide](_included/portfolio_stress.md) gives the valuation
formulas, FX conventions, complete custom-payoff contract and factor-family rules. It is the
single source of those conventions and is also available inside an installed QIS package.

## Reading the resulting report

- **Current betas and dollar exposures** use today's payoff Jacobian, aggregated by shared
  underlying response. A zero-mark future can carry substantial exposure. Options crossing
  strikes or barriers can have large losses despite a small current beta.
- **Requested and conditional scenarios** value the full payoff. A missing anchor is a free
  factor; an explicit zero pins that factor. Conditional completion keeps all expanded family
  members fixed and solves for the other factors jointly.
- **Credit family grids** split a total simple bump before converting to log returns. A -10%
  total bump gives Credit and Credit EM -5% each. Summed family exposures and summed Euler
  contributions answer aggregation questions; they do not use those split weights.
- **Historical months** replay each complete monthly factor vector on today's holdings. They
  are ranked by exact portfolio P&L and are not the portfolio's realized investment history.
- **Euler volatility contributions** are signed allocations of total model volatility.
  Factor terms plus the residual Euler term add to total volatility. The page-five panels
  select factors by absolute Euler contribution, then rank their holding contributions.
- **Sensitivity grids** fit a through-zero quadratic for every portfolio, including
  derivatives. Legends show the equation and uncentered R-squared on the displayed grid.
  Exact payoff points remain authoritative, including at strikes and knockout jumps.
  Blue shading shows conditional +/-1sigma and +/-2sigma volatility ranges using
  scenario-local exposures for funded and derivative holdings. Factor Euler plus
  residual Euler contributions reconcile to each band's horizon volatility. OLS
  mean-fit confidence intervals remain exported diagnostics, not chart shading.
  Local bands omit curvature and boundary-crossing risk; they are not exact nonlinear
  confidence intervals. Scenario-local risk and Euler exports are included from 5.30.1.
- **Loadings and fit** distinguish unit underlying response risk from portfolio exposure. Fitted
  R-squared and original clustering trees must come from the caller; absent diagnostics are
  labelled unavailable. No fitted R-squared or original trees are invented by these examples.

Percentages use the application's explicit reporting denominator. Derivative report captions
label it as such; it need not be debt-net NAV. Observed marks remain anchors even when the
intrinsic proxy has a different baseline. The result is not automatically a liquidation or
margin calculation.

Each example supplies a four-row appendix through `StressReportConfig`, demonstrating where
application-specific explanations belong. Display names use at most 20 characters in these
examples; full source IDs and payoff terms remain in the exports. The linked workbook contents
page points to all result and audit sheets, with readable widths, numerical formats and frozen
panes. It contains stored numerical results, not an interactive option-pricing spreadsheet.

## Full source: funded and mixed portfolios

~~~{literalinclude} ../examples/portfolios/instrument_portfolio_stress.py
:language: python
~~~

## Full source: a custom payoff

The example replaces one remaining-quantity accumulator with a terminal knockout at EUR 115
and moves its strike to EUR 95. Current spot is EUR 100. Away from the strike and barrier, a
central finite difference of public portfolio P&L agrees with the analytic factor sensitivity.
At or beyond the barrier the example declares zero remaining payoff and zero local sensitivity.
This boundary convention does not make a discontinuous barrier differentiable and is not a
path simulation. Its limitations are carried into the position audit and report appendix.

~~~{literalinclude} ../examples/portfolios/composite_payoff_stress.py
:language: python
~~~

## Verification

The existing offline-example harness discovers and executes both examples, so these commands
exercise the same source displayed above:

~~~console
python -m pytest src/qis/tests/test_examples.py -k "instrument_portfolio_stress or composite_payoff_stress"
python -m pytest --pyargs qis.portfolio.stress.tests
~~~

The checks cover zero-shock mark identity, full attribution reconciliation, the Credit split,
Euler additivity, nonzero exposure on a zero-mark future, exact local FX conversion, the custom
terminal payoff and finite-difference factor sensitivities. The shared stress test suite covers
label validation, ambiguous factor instructions, shared residuals and kink policies.

## Related guides

- [Factor stress helpers, conditioning and funded prediction bands](stress_testing.md)
- [RiskModel and tracking error](tracking_error_and_risk.md)
- [Factsheets and reporting](factsheets_and_reporting.md)
- [Installed instrument-portfolio conventions](_included/portfolio_stress.md)

~~~{toctree}
:hidden:

_included/portfolio_stress
~~~

## Cluster contributions

The exhibit following the fitted dendrogram groups the existing holding results by
the caller's fitted response memberships. Cadence prefixes distinguish, for example,
ME-1 from QE-1. No clustering or payoff valuation is repeated.

The top heatmap displays requested conditional-scenario P&L summed over holdings in
each cluster, divided by the full reporting notional. The bottom left shows current
holding response dollar sensitivities times factor betas, summed by cluster and
divided by that same notional. Values are exposure ratios, displayed to two decimals.
Both tables append an additive portfolio row.

The bottom right stacks signed systematic and idiosyncratic Euler contributions to
annual **factor-model portfolio volatility**. These are not standalone cluster
volatilities. Systematic contributions aggregate the existing holding-factor Euler
allocation. Shared-response residual Euler terms are scaled to the model-volatility
denominator and allocated by signed current response sensitivities before grouping.
Offsets within the same response retain their shared residual identity. Every
component and cluster together reconciles to model portfolio volatility; negative
diversifying contributions remain signed.

All three panels use gross-MTM ordering. The display keeps at most eight groups,
reserving explicit unassigned and multi-cluster buckets and combining smaller
regular clusters as Other clusters. The labels show net notional weight and holding
count. Every cluster remains individually available in CSV/workbook tables for
membership, display mapping, MTM, dollar/weighted factor exposure, Euler risk, and
all independent, conditional, requested and historical scenario contributions.

A holding with missing fitted response membership is unassigned. A holding spanning
multiple known clusters has a separate multi-cluster bucket, avoiding arbitrary
allocation of nonlinear P&L. Positions excluded from the supplied model remain
outside these diagnostics, with their count and gross MTM stated explicitly;
their missing analytics are never replaced with zero and the full notional remains
the denominator. Without fitted memberships the page displays an unassigned group.

### Family panels and cluster descriptions

The largest-contributor and default sensitivity panels use the same six groups,
ranked by absolute summed Euler risk. Declared disjoint families combine all member
factor exposures and holding Euler contributions by summation. Overlapping scenario
groups retain atomic factor reporting, avoiding double-counting. Individual grids
remain available in the numerical audit. A family sensitivity curve is an exact
revaluation, not an average of individual curves: the default equal total simple bump x anchors each
of n members at log(1+x/n), then jointly conditions every free factor.
Custom QIS groups may supply other weights; their axes state the actual allocation.

StressReportConfig.cluster_labels accepts a mapping from cadence-prefixed IDs
(such as ME-1) to plain descriptive labels. The fitted dendrogram membership table
shows these alongside raw IDs. The QIS plotting helper plot_clusters accepts the
same optional mapping. Providers own label generation; QIS never imports an estimator
or rebuilds a tree. ROSAA calls FactorLasso's factor_labels/labels_at workflow on the
single fitted snapshot with equal cluster-member weighting. These are descriptive
factor/volatility labels, not claims about cluster persistence.

The cluster page puts scenario and factor names above both heatmaps. Its adjacent
Top contributor column identifies, for each displayed row, the holding with the
largest absolute P&L under that row's worst requested conditional scenario. It shows
the signed contribution divided by full notional, plus the scenario. The portfolio
row uses its own worst scenario, and an Other clusters row is evaluated after
combining its memberships. Full raw-cluster and displayed-row results are exported.

The factor-risk bars use the portfolio's five largest absolute **atomic** factor
Euler contributions with fixed colours across all cluster rows. Their annotations
are signed subtotals over those five factors. They exclude other systematic factors
and residual risk; the adjacent total-risk bars retain the full systematic and
idiosyncratic decomposition. Full cluster-by-factor Euler tables are exported.

## Final notation and analysis guide

The report ends with a two-column guide to the eleven analysis exhibits. It defines the
reporting denominator, factor and response sensitivities, covariance, residual risk and
Euler contributions, then explains each chart and table, including scenario attribution,
conditional bands, loading aggregates and cluster allocations. The guide uses the configured
model name and reports the selected band horizon. Its 10-point body text remains above the
shared 9-point footnote minimum.

The optional parser-owned coverage table precedes this guide: reports have thirteen pages
with coverage and twelve without it. Page 10 shows the dated correlation/volatility matrix
and target-to-factor mappings. Page 11 illustrates conditional mean shocks and contains the
conditional covariance and local-band formulas. Portfolio valuation and risk are unchanged.

## Conditional-shock illustration (5.36)

Two colour-coded tables independently anchor every atomic factor at -10% and +10% simple
return. Each column is one anchored factor; each row is an affected factor. The diagonal
retains the anchor and is outlined. No family splitting is used on this page. Both tables
share a symmetric colour scale, display signed percentages and retain original factor order.

For anchor a and simple bump s, QIS fixes z_a = log(1+s), completes z_i = Sigma_ia / Sigma_aa
x z_a and displays exp(z_i)-1. Thus a negative correlation can produce an opposite-sign
co-move, and the two simple-return tables need not be exact negatives. The tables are
conditional mean returns, not covariance matrices. Zero-variance anchors are unavailable.

The accompanying Schur-complement formula explains remaining covariance. It is the same
for both signs when the anchored set is unchanged; scenario-local sensitivities can still
change the conditional risk-band widths. No separate correlation-matrix shock is imposed.

`PortfolioStressResult.report_diagnostics` contains `Conditional factor shocks -10%` and
`Conditional factor shocks +10%`. Both are exported without PDF rounding. The renderer only
consumes these detached calculations and does not reprice or refit a portfolio.

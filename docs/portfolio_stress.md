---
myst:
  html_meta:
    description: >-
      Build instrument portfolios with funds, stocks, calls, puts and futures; evaluate factor
      scenarios and custom payoffs; generate a standard portfolio stress report with qis.
---

# Instrument portfolios and standard stress reports

`qis.portfolio.stress` evaluates an absolute-position snapshot under an assigned factor model.
Funded assets, signed intrinsic calls/puts, futures and consumer-owned composite payoffs share
one portfolio interface. The numerical result feeds the same PDF, Excel and CSV report.

Use this interface when original position IDs, option strikes, futures multipliers, currencies
or payoff nonlinearities must survive the scenario calculation. For a funded exposure matrix
or a standalone conditional factor calculation, the existing [factor stress helpers](stress_testing.md)
remain available. Both paths use the canonical [RiskModel](tracking_error_and_risk.md) analytics.

These APIs were added in 5.28.0; the factor/family Euler exhibits, optional appendix and restored
report layout are included in 5.29.0. A documentation build describes its source version; it does
not imply that development changes have already been published to the package index.

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
| 7 | `generate_portfolio_stress_report` | Completed result and `StressReportConfig` | Nine core PDF pages, optional tenth page, all numerical tables and artifact hashes |

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
  confidence intervals. Scenario-local risk and Euler exports are included in 5.30.0.
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

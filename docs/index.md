---
myst:
  html_meta:
    description: >-
      Documentation for qis: performance analytics, portfolio backtesting, risk and attribution
      methods, factsheet reports, runnable examples, and API reference.
---

# qis documentation

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-07-25](https://github.com/ArturSepp/QuantInvestStrats/commit/33fb329654dd1ab6064a009c785f11f08910953e)*

<a id="qis"></a>
<a id="qis-performance-analytics-portfolio-backtesting-risk-analysis-and-factsheet-reporting"></a>

[qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats)
is a Python library for performance analytics, portfolio backtesting, risk analysis and factsheet
reporting. It works with price and NAV time series and externally constructed portfolio weights.
These guides explain the methods, calculation conventions and runnable examples behind its reports.

Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

## Start here

1. [Install qis](install.md). The core installation command is `python -m pip install qis`.
2. Follow the [offline quickstart](quickstart.md) for a first chart, portfolio backtest,
   performance table and benchmark-relative results.
3. Browse the [factsheet gallery](gallery.md) to choose a report, and read
   [Notation and conventions](notation_and_conventions.md) before interpreting or comparing
   results.

After installation, the quickstart calculations use fixed synthetic data without network access
or optional extras. The complete portfolio workflow lives in
[`examples/getting_started/offline_quickstart.py`](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/getting_started/offline_quickstart.py).
The quickstart page includes that source and provides links for Markdown viewers.

## The qis analytics handbook

The methodology chapters form one book. Each chapter defines its method with formulas and
concise proofs, states its conventions in a seven-row convention card, works a small example
whose numbers the test suite checks, and links to the functions that implement it. Symbols
keep one meaning throughout; see [Notation and conventions](notation_and_conventions.md) and
the [bibliography](bibliography.md).

### Part I: Foundations

- [Notation and conventions](notation_and_conventions.md): reserved symbols, simple and log
  returns, per-annum returns, annualisation, excess returns and timing.
- [Returns, NAVs, excess returns, fees and leverage](returns_and_navs.md): from prices to
  returns and back, cash-rate deduction, fee crystallisation and levered returns.
- [Reporting frequency and annualisation](frequency_convention_note.md):
  sampling grids, the variance ratio and interpretation of reported statistics.
- [Incomplete and mixed-frequency data](incomplete_and_mixed_frequency_data.md):
  missing observations, instrument lifetimes and differing reporting schedules.

### Part II: Performance measurement

- [The performance-statistic catalogue](performance_statistics.md): one formula for every
  column of the performance table.
- [Sharpe ratios: conventions and inference](performance_analytics_and_sharpe.md):
  the Sharpe conventions of qis, the volatility drag between them and their sampling error.
- [Drawdowns and time under water](drawdowns.md): running and maximum drawdowns, episodes,
  Calmar ratios and their dependence on the grid and the horizon.
- [Alpha, beta and benchmark-relative performance](benchmark_relative_performance.md):
  the single-index regression, alpha inference, EWMA betas and beta attribution.
- [Regime-conditional performance](regime_conditional_performance.md): benchmark regimes and
  the additive decomposition of the Sharpe ratio.

### Part III: Estimation

- [Exponentially weighted estimators](ewm_estimators.md): spans, half-lives, effective sample
  sizes, initialisation and EWM volatility and covariance.
- [Covariance, correlation and principal components](covariance_correlation_pca.md):
  estimation, masking, eigen-decomposition and the noise floor.
- [Serial dependence and autocorrelation](serial_dependence.md): autocorrelation functions,
  their standard errors and lagged betas.
- [Regression and HAC inference](regression_and_hac.md): OLS and EWMA regressions with
  heteroskedasticity- and autocorrelation-consistent standard errors.
- [Resampling and the bootstrap](reproducibility.md): IID, block and stationary bootstraps, and
  what an unstated sampling convention costs.
- [Private-asset unsmoothing](private_asset_unsmoothing.md):
  serial correlation, return reconstruction and de-levering.

### Part IV: Portfolios

- [Portfolio backtesting](portfolio_backtesting.md):
  decisions, execution timing, held units and portfolio histories.
- [Turnover conventions](turnover_conventions.md):
  traded notional, transaction costs and turnover reporting.
- [Risk-adjusted returns and volatility targeting](risk_adjusted_returns.md): scaling returns
  by lagged volatility estimates and what targeting does to realised risk.
- [Signal diagnostics: information coefficient and information ratio](signal_diagnostics.md):
  rank correlations of signals with forward returns and their aggregation.

### Part V: Risk

- [Portfolio risk and Euler contributions](risk_contributions.md): marginal, total and relative
  risk contributions and their grouping.
- [Factor risk models](factor_risk_models.md): EWMA factor models, betas, residual risk and the
  model covariance.
- [Tracking error and benchmark-relative risk](tracking_error_and_risk.md):
  ex-ante risk estimates and realised tracking error and information ratio.
- [Portfolio breadth](portfolio_breadth.md): effective instrument counts and concentration.
- [Factor stress testing](stress_testing.md):
  specified factor shocks, valuation changes and prediction bands.
- [Instrument portfolios and stress reports](portfolio_stress.md):
  funded assets, options and futures in one stress interface.
- [Stress testing with options](stress_testing_with_options.md):
  five stocks, ten short VOP-priced options and a four-ETF EWMA risk model.
- [FX hedging and market data](fx_hedging_and_market_data.md):
  currency conversion, hedging assumptions and data contracts.

### Part VI: Attribution

- [Brinson attribution](brinson_attribution.md):
  allocation, selection and interaction effects against a benchmark.
- [Model-layer attribution](model_layer_attribution.md):
  risk, signal and integration layers, with factor and feature contribution methods.

## Reporting guides

- [Factsheets and reporting](factsheets_and_reporting.md):
  inputs, report types and the calculation conventions used by the reporting workflow.
- [Factsheet reference](factsheets.md):
  call patterns, configuration, output objects and PDF saving.
- [Factsheet gallery](gallery.md): the four report types on fixed synthetic data.

## Reference

- [API reference](api/index.rst): function and class documentation.
  [Public API source catalog](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/api.py).
- [Bibliography](bibliography.md): every work cited by the handbook, in one style.
- [Software design](software_design.md): module ownership, public API and dependency boundaries.
- [Package comparison](package_comparison.md): documented workflows in qis and related libraries.
- [API migration history](REMOVED_5_0.md): renamed and removed symbols, with current module imports.
- [Documentation standard](documentation_standard.md): article, equation, citation and figure rules.

The following compact notes also ship with the Python package. Site builds render them alongside
the articles; the source links remain usable when reading this page in a checkout.

| Convention | Site note | Packaged source |
|---|---|---|
| Sharpe ratios | [Convention summary](_included/sharpe_conventions.md) | [sharpe_conventions.md](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/docs/sharpe_conventions.md) |
| Reporting grids | [Frequency reference](_included/reporting_frequencies.md) | [reporting_frequencies.md](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/docs/reporting_frequencies.md) |
| Plot styling | [Shared plotting arguments](_included/plotting_kwargs.md) | [plotting_kwargs.md](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/docs/plotting_kwargs.md) |

## Project resources

- [PyPI package](https://pypi.org/project/qis/) and
  [source repository](https://github.com/ArturSepp/QuantInvestStrats).
- [Issue tracker](https://github.com/ArturSepp/QuantInvestStrats/issues) and
  [contributor guide](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CONTRIBUTING.md).
- [Governance, maintenance and support](https://github.com/ArturSepp/QuantInvestStrats/blob/main/GOVERNANCE.md).
- [Changelog](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CHANGELOG.md).
- [Software paper source](https://github.com/ArturSepp/QuantInvestStrats/blob/main/paper.md).

```{toctree}
:hidden:
:maxdepth: 1
:caption: Start here

install
quickstart
gallery
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Part I - Foundations

notation_and_conventions
returns_and_navs
Reporting frequency and annualisation <frequency_convention_note>
incomplete_and_mixed_frequency_data
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Part II - Performance measurement

performance_statistics
performance_analytics_and_sharpe
drawdowns
benchmark_relative_performance
regime_conditional_performance
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Part III - Estimation

ewm_estimators
covariance_correlation_pca
serial_dependence
regression_and_hac
Resampling and the bootstrap <reproducibility>
Private-asset unsmoothing <private_asset_unsmoothing>
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Part IV - Portfolios

Portfolio backtesting <portfolio_backtesting>
Turnover conventions <turnover_conventions>
risk_adjusted_returns
signal_diagnostics
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Part V - Risk

risk_contributions
factor_risk_models
Tracking error and risk <tracking_error_and_risk>
Portfolio breadth <portfolio_breadth>
Factor stress testing <stress_testing>
Instrument portfolios and stress reports <portfolio_stress>
Stress testing with options <stress_testing_with_options>
FX hedging and market data <fx_hedging_and_market_data>
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Part VI - Attribution

Brinson attribution <brinson_attribution>
Model-layer attribution <model_layer_attribution>
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Reporting guides

factsheets_and_reporting
Factsheet reference <factsheets>
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Reference

api/index
bibliography
software_design
Package comparison <package_comparison>
REMOVED_5_0
documentation_standard
Sharpe convention summary <_included/sharpe_conventions>
Reporting-frequency reference <_included/reporting_frequencies>
Shared plotting arguments <_included/plotting_kwargs>
```

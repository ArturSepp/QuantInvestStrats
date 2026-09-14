---
myst:
  html_meta:
    description: >-
      Documentation for qis: performance analytics, portfolio backtesting, risk and attribution
      methods, factsheet reports, runnable examples, and API reference.
---

# qis documentation

*[author / affiliation / date — placeholder]*

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
3. Browse the [factsheet gallery](gallery.md) to choose a report, and read the
   [reproducibility guide](reproducibility.md) before interpreting or comparing results.

After installation, the quickstart calculations use fixed synthetic data without network access
or optional extras. The complete portfolio workflow lives in
[`examples/getting_started/offline_quickstart.py`](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/getting_started/offline_quickstart.py).
The quickstart page includes that source and provides links for Markdown viewers.

## Performance and reporting

- [Performance analytics and Sharpe conventions](performance_analytics_and_sharpe.md):
  return, volatility, drawdown and risk-adjusted performance measures.
- [Reporting-frequency methodology](frequency_convention_note.md):
  sampling grids, annualisation and interpretation of reported statistics.
- [Factsheets and reporting](factsheets_and_reporting.md):
  inputs, report types and the calculation conventions used by the reporting workflow.
- [Factsheet reference](factsheets.md):
  call patterns, configuration, output objects and PDF saving.

## Portfolio accounting and risk

- [Portfolio backtesting](portfolio_backtesting.md):
  decisions, execution timing, held units and portfolio histories.
- [Turnover conventions](turnover_conventions.md):
  traded notional, transaction costs and turnover reporting.
- [Portfolio breadth](portfolio_breadth.md):
  effective instrument counts and concentration.
- [Tracking error and benchmark-relative risk](tracking_error_and_risk.md):
  ex-ante risk estimates and realised tracking error and information ratio.
- [Brinson attribution](brinson_attribution.md):
  allocation, selection and interaction effects against a benchmark.

## Estimation and market data

- [Model-layer attribution](model_layer_attribution.md):
  risk, signal and integration layers, with factor and feature contribution methods.
- [Factor stress testing](stress_testing.md):
  specified factor shocks, valuation changes and prediction bands.
- [Stress testing with options](stress_testing_with_options.md):
  five stocks, ten short VOP-priced options and a four-ETF EWMA risk model.
- [Incomplete and mixed-frequency data](incomplete_and_mixed_frequency_data.md):
  missing observations, instrument lifetimes and differing reporting schedules.
- [Private-asset unsmoothing](private_asset_unsmoothing.md):
  serial correlation, return reconstruction and estimation limitations.
- [FX hedging and market data](fx_hedging_and_market_data.md):
  currency conversion, hedging assumptions and data contracts.

## Implementation and reference

- [Software design](software_design.md): module ownership, public API and dependency boundaries.
- [Package comparison](package_comparison.md): documented workflows in qis and related libraries.
- [API reference](api/index.rst): function and class documentation.
  [Public API source catalog](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/api.py).
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
Reproducibility <reproducibility>
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Performance and reporting

performance_analytics_and_sharpe
Reporting-frequency methodology <frequency_convention_note>
factsheets_and_reporting
Factsheet reference <factsheets>
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Portfolio accounting and risk

Portfolio backtesting <portfolio_backtesting>
Turnover conventions <turnover_conventions>
Portfolio breadth <portfolio_breadth>
Tracking error and risk <tracking_error_and_risk>
Brinson attribution <brinson_attribution>
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Estimation and market data

Model-layer attribution <model_layer_attribution>
Factor stress testing <stress_testing>
Instrument portfolios and stress reports <portfolio_stress>
Stress testing with options <stress_testing_with_options>
incomplete_and_mixed_frequency_data
Private-asset unsmoothing <private_asset_unsmoothing>
FX hedging and market data <fx_hedging_and_market_data>
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Implementation and reference

software_design
Package comparison <package_comparison>
api/index
REMOVED_5_0
documentation_standard
Sharpe convention summary <_included/sharpe_conventions>
Reporting-frequency reference <_included/reporting_frequencies>
Shared plotting arguments <_included/plotting_kwargs>
```

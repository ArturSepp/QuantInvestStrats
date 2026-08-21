---
myst:
  html_meta:
    description: >-
      qis provides performance analytics, portfolio backtesting, risk analysis, and factsheet
      reporting for quantitative investment strategies in Python.
---

# qis: performance analytics, portfolio backtesting, risk analysis, and factsheet reporting

qis - performance analytics, portfolio backtesting, risk analysis, and factsheet reporting in
Python.

Quantitative Investment Strategies covers time-series and cross-sectional performance,
drift-aware portfolio histories, ex-ante and ex-post risk, and reproducible reports.

Install with `pip install qis`, then follow the [offline quickstart](quickstart.md) for a
deterministic portfolio backtest, performance table, and benchmark-relative result. Its single
source is
[`examples/getting_started/offline_quickstart.py`](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/getting_started/offline_quickstart.py):
the documentation includes that complete runnable file rather than maintaining another code copy.
It needs no network, data vendor, credentials, optional extra, or output directory.

## Project resources

- [PyPI package](https://pypi.org/project/qis/)
- [Source repository](https://github.com/ArturSepp/QuantInvestStrats)
- [Issue tracker](https://github.com/ArturSepp/QuantInvestStrats/issues)
- [Governance, maintenance, and support](https://github.com/ArturSepp/QuantInvestStrats/blob/main/GOVERNANCE.md)
- [Changelog](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CHANGELOG.md)
- [Citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff)
- [JOSS submission paper](https://github.com/ArturSepp/QuantInvestStrats/blob/main/paper.md)
  (under review; not accepted or published)

```{toctree}
:maxdepth: 2
:caption: Getting started

install
quickstart
reproducibility
gallery
```

```{toctree}
:maxdepth: 1
:caption: Focused guides

performance_analytics_and_sharpe
software_design
factsheets_and_reporting
tracking_error_and_risk
portfolio_backtesting
incomplete_and_mixed_frequency_data
private_asset_unsmoothing
fx_hedging_and_market_data
package_comparison
```

```{toctree}
:maxdepth: 1
:caption: Conventions

_included/sharpe_conventions
_included/reporting_frequencies
_included/frequency_convention_note
_included/factsheets
_included/plotting_kwargs
_included/REMOVED_5_0
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
```

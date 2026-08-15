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

Install with `pip install qis`, then build a factsheet from a price panel:

```python
import qis
from qis.datasets import generate_synthetic_universe

universe = generate_synthetic_universe()
qis.factsheet(universe.prices,
              benchmark_prices=universe.benchmark_prices,
              reporting_frequency='monthly')
```

That snippet needs no network and no data vendor: the panel is generated in-process from a fixed
seed and carries the defects real panels carry — ragged starts, missing observations, stale
prices, a delisted tail and a monthly-reported illiquid sleeve.

## Project resources

- [PyPI package](https://pypi.org/project/qis/)
- [Source repository](https://github.com/ArturSepp/QuantInvestStrats)
- [Issue tracker](https://github.com/ArturSepp/QuantInvestStrats/issues)
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

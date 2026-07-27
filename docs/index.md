# qis

Python analytics for visualisation of financial data, performance reporting, factsheets and
analysis of quantitative strategies.

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

```{toctree}
:maxdepth: 2
:caption: Getting started

install
quickstart
reproducibility
_included/gallery
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

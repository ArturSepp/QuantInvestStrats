---
myst:
  html_meta:
    description: >-
      Install qis, choose optional extras, verify the imported package, and run
      the offline quickstart or a development checkout.
---

# Installation

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-07-25](https://github.com/ArturSepp/QuantInvestStrats/commit/33fb329654dd1ab6064a009c785f11f08910953e)*

[qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats)
provides portfolio analytics and reporting in Python.
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

## Core installation

Use Python 3.10 or later and install into your chosen Python environment:

~~~console
python -m pip install qis
~~~

The distribution and import names are both `qis`. The core dependencies are NumPy, pandas,
SciPy, statsmodels, Numba, Matplotlib, seaborn, openpyxl and PyYAML. No data-vendor client is
installed by default. The core supports the offline analytics, Matplotlib figures and PDF
factsheets shown in the [quickstart](quickstart.md) and [reporting guide](factsheets.md).

The [project metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/pyproject.toml)
defines Python/dependency constraints for the current source. A source checkout can be ahead
of the [published distribution](https://pypi.org/project/qis/); use the metadata for the release
you install when reproducing an earlier environment.

## Verify the installation

Run this in the same interpreter or notebook kernel that will run your analysis:

~~~python
import sys
from importlib.metadata import version

import qis

print('Python:', sys.version)
print('Interpreter:', sys.executable)
print('Installed qis metadata:', version('qis'))
print('Imported qis source:', qis.__file__)
assert 'factsheet' in qis.__all__
assert 'backtest_model_portfolio' in qis.__all__
~~~

The version identifies installed distribution metadata; the import path identifies the source
actually loaded. An editable install or a custom `PYTHONPATH` can make those differ.
For a calculation and chart check, continue with the [offline quickstart](quickstart.md).

## Optional extras

Install only the integrations your workflow uses.

| Extra | Packages added | Purpose |
|---|---|---|
| `data` | yfinance, pandas-datareader | Download prices for market-data examples. |
| `io` | pyarrow, fsspec | Parquet/Feather and associated filesystem support. |
| `reports` | pybloqs, jinja2 | The optional PyBloqs HTML report backend. |
| `visualization` | plotly | Interactive plots. |
| `database` | psycopg2, SQLAlchemy | Database integrations. |
| `jupyter` | jupyter, notebook, jupyterlab, ipykernel, ipywidgets | Notebook tools and widgets. |
| `docs` | sphinx, myst-parser, furo | Build the documentation site. |
| `all` | All extras above | Install all optional integrations together. |

For example:

~~~console
python -m pip install "qis[data,io]"
~~~

The `reports` extra is not required for `qis.factsheet` or `qis.save_figs_to_pdf`.
A backend's external programs and data access requirements remain separate from installing
its Python packages. The [software design guide](software_design.md#optional-backends)
describes the import boundary.

Optional functions usually import their backend when called. Failure handling belongs to the
specific function: it may raise a missing-dependency error or use a documented fallback.
For example, a factsheet's attempted rate download may leave rate data unavailable; installing
`data` alone does not establish that a download succeeded.

## From source

For a development checkout, install the core package and pytest explicitly:

~~~console
git clone https://github.com/ArturSepp/QuantInvestStrats.git
cd QuantInvestStrats
python -m pip install -e . pytest
python -m pytest
~~~

An editable install imports changes directly from the checkout. Tests live under `src/qis`,
and the repository configures pytest to discover them. Extra-dependent tests may skip on a
core installation; repository-only documentation checks are unavailable in an installed wheel.

Contributors should follow [CONTRIBUTING.md](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CONTRIBUTING.md)
for the maintained development and CI commands. On the maintainer's OneDrive-hosted Windows
checkout, follow [AGENTS.md](https://github.com/ArturSepp/QuantInvestStrats/blob/main/AGENTS.md)
before running those commands: use the external `C:\Python\QuantInvestStrats312` environment
and keep generated state and verification runs on local C storage.

## References

- qis contributors. [Packaging metadata and optional dependencies](https://github.com/ArturSepp/QuantInvestStrats/blob/main/pyproject.toml).
- Sepp, A. qis: Performance analytics, portfolio backtesting, risk analysis, and factsheet
  reporting in Python. [Software citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

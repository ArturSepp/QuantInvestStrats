# Installation

```bash
pip install qis
```

qis requires Python 3.10 or later. The core install pulls only the scientific stack: `numpy`,
`pandas`, `scipy`, `statsmodels`, `numba`, `matplotlib`, `seaborn`, `openpyxl` and `PyYAML`.
No data-vendor client is installed by default.

## Optional extras

| extra | adds | for |
|---|---|---|
| `data` | `yfinance`, `pandas-datareader` | downloading prices in examples |
| `io` | `pyarrow`, `fsspec` | parquet and feather |
| `reports` | `pybloqs`, `jinja2` | HTML and PDF factsheet rendering |
| `visualization` | `plotly` | interactive figures |
| `database` | `psycopg2`, `SQLAlchemy` | database-backed storage |
| `jupyter` | `jupyter`, `notebook`, `jupyterlab` | notebook use |
| `all` | every extra above | |

```bash
pip install "qis[data,io]"
```

The analytics core never imports an optional dependency at module level. Where one is needed, the
import is function-local and raises an `ImportError` naming the extra.

## From source

```bash
git clone https://github.com/ArturSepp/QuantInvestStrats.git
cd QuantInvestStrats
pip install -e ".[data,io]"
pytest
```

The test suite passes on a core install; tests needing an extra are skipped, not failed.

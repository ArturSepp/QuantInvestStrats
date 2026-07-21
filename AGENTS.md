# AGENTS.md

Guidance for AI coding agents working in the **QuantInvestStrats** repository.

## Project overview

`qis` (Quantitative Investment Strategies) is the analytics and reporting engine of the
stack: time-series and cross-sectional performance analytics, risk-adjusted performance
tables, factsheet generation, and a matplotlib/seaborn visualisation layer for financial
data. It is a dependency of `optimalportfolios` and `trendfollowing`, so changes to its
public API have downstream consequences outside this repository.

Distribution name `qis`; import name `qis`. Licensed MIT (`LICENSE.txt`).

## Ecosystem position

This package is one of eight open-source Python libraries maintained at
[github.com/ArturSepp](https://github.com/ArturSepp). Before implementing anything
non-trivial, check whether it already exists in one of these:

| Package | Repository | Purpose |
|---|---|---|
| `qis` | QuantInvestStrats | Performance analytics, factsheets, visualisation |
| `optimalportfolios` | OptimalPortfolios | Portfolio construction and backtesting |
| `factorlasso` | factorlasso | Sparse factor models and factor covariance estimation |
| `bbg-fetch` | BloombergFetch | Bloomberg data fetching |
| `trendfollowing` | TrendFollowingSystems | Trend-following systems: closed-form theory and replication |
| `goal-based-allocation` | GoalBasedAllocation | Dynamic MV allocation under regime-switching jump-diffusions |
| `stochvolmodels` | StochVolModels | Stochastic volatility pricing analytics |
| `vanilla-option-pricers` | VanillaOptionPricers | Vanilla option pricers and implied volatility fitters |

Actual package dependencies within the stack: `optimalportfolios` depends on `qis`
and `factorlasso`; `trendfollowing` depends on `qis`; `stochvolmodels` has an
optional `research` extra that pulls in `qis`. The others are independent.

Do not vendor or copy code between these packages. If functionality belongs in a
sibling package, say so rather than reimplementing it here.

## Repository layout

```
qis/
  perfstats/      performance and risk statistics
  plots/          visualisation layer (matplotlib/seaborn)
  portfolio/      portfolio objects, factsheets, multi-strategy reports
  models/         estimators and models
  market_data/    data access helpers
  examples/       runnable examples
  docs/           notebooks and documentation sources
  file_utils.py, local_path.py, sql_engine.py, settings.yaml
notebooks/        Jupyter notebooks
```

Tests live **inside** the package next to the code they cover, as
`qis/<subpackage>/tests/*_test.py` — there is no top-level `tests/` directory.

## Commands

```bash
pip install -e ".[data]"          # editable install with yfinance/pandas-datareader
pytest qis/                       # run the test suite (see Known issues)
pytest qis/perfstats/tests/ -v    # run one subpackage
ruff check qis/                   # lint
```

Optional extras: `data`, `reports`, `visualization`, `io`, `database`, `jupyter`.
Supported Python is >= 3.10; CI runs the matrix 3.10 – 3.14.

## Conventions

- Test files are named `*_test.py` (suffix, not prefix) and live in a `tests/`
  directory inside the subpackage under test. Follow the local convention.
- Line length 100 (`ruff`, rules `E`, `F`, `W`, `I`). Run `ruff check qis/` before
  finishing.
- Enums are used heavily (100+ modules) for options and switches; prefer an enum
  member over a string literal when one already exists.
- Dataclasses are used for configuration and result containers.
- The public surface is re-exported from `qis/__init__.py`; anything added there is
  part of the public API and must not be removed casually.
- Data structures are pandas `DataFrame`/`Series` with a `DatetimeIndex`. Functions
  return pandas objects rather than numpy arrays unless there is a reason not to.

## Constraints — do not do these

- Do not change the signature or behaviour of anything exported from `qis/__init__.py`
  without flagging it: `optimalportfolios` and `trendfollowing` import from it.
- Do not add hard runtime dependencies. Optional functionality belongs behind an
  extra in `[project.optional-dependencies]` with a guarded import.
- Do not commit generated factsheets, PDFs, or figure output.
- Do not modify `settings.yaml` or `local_path.py` to hardcode a machine-specific path.
- Examples must run on free data (yfinance) — do not make an example require Bloomberg.

## Release checklist

A release touches three version locations. All three must agree:

1. `version` in `pyproject.toml`
2. `version` and `date-released` in `CITATION.cff`
3. the software BibTeX entry in `README.md` (if it pins a version)

Then: commit, tag `v<version>`, build and publish to PyPI, and cut a GitHub Release
with the same tag. Do not bump versions as part of an unrelated change, and do not
publish without the maintainer explicitly asking for a release.

## Known issues

`[tool.pytest.ini_options] testpaths = ["tests"]` in `pyproject.toml` points at a
directory that does not exist in this repository; a bare `pytest` invocation therefore
collects nothing. Always pass an explicit path (`pytest qis/`). Fixing `testpaths` to
`["qis"]` is a welcome change if the maintainer asks for it.

---
myst:
  html_meta:
    description: >-
      The qis data flow, public API, module ownership, numerical contracts,
      optional backends, documentation trees, and verification boundaries.
---

# Software design

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-08-21](https://github.com/ArturSepp/QuantInvestStrats/commit/10cc0ab622d064908a782dddfce7205e44dea7fc)*

[qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats)
is an analytics and reporting library built around labelled data and explicit conventions.
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

This page explains how inputs become portfolio histories and reports, where functionality
belongs, and which contracts an extension must preserve. The [offline quickstart](quickstart.md)
demonstrates the public workflow.

## Layers and data flow

A typical portfolio workflow is:

~~~text
prices + an externally specified target-weight schedule
                         |
                         v
             backtest: trades and held units
                         |
                         v
        PortfolioData: NAV, weights, costs, holdings
                         |
                         v
       performance, risk and attribution calculations
                         |
                         v
            tables, plots and factsheet pages
~~~

Raw prices or existing NAVs can also enter performance analytics and reports directly;
they do not need a backtest first. Return conversion and estimators serve both paths.

Time-series interfaces generally use pandas `Series` or `DataFrame` objects with a
`DatetimeIndex`. Covariance matrices and exposure vectors instead align on instrument or
factor labels. Numerical kernels may use NumPy arrays; callers must preserve the ordering
that connects their results back to those labels.

| Module area | Responsibility |
|---|---|
| `qis.perfstats` | Return conversion, annualisation, performance statistics, drawdowns and regimes. |
| `qis.models` | Reusable estimators and models, including EWMA, bootstrap and unsmoothing methods. |
| `qis.utils` | Shared data operations, calendar handling, regression utilities and numerical helpers. |
| `qis.portfolio` | Portfolio accounting, risk, attribution and reporting objects. |
| `qis.plots` | Matplotlib/seaborn plots and derived presentation functions. |
| `qis.market_data` | Market-data transformations and optional integration helpers. |
| `qis.datasets` | The fixed synthetic fixture used by offline examples and tests. |

Model-layer attribution, feature attribution, portfolio breadth and Brinson attribution live
in `qis.portfolio.attribution`. Their orchestration uses portfolio NAV/allocation semantics;
regression and covariance estimation remain in reusable lower-level modules. Presentation
functions live in the plotting/reporting layers rather than defining a second calculation.

The source is under
[`src/qis`](https://github.com/ArturSepp/QuantInvestStrats/tree/main/src/qis).
For calculation contracts, see [backtesting](portfolio_backtesting.md),
[risk](tracking_error_and_risk.md), [model-layer attribution](model_layer_attribution.md)
and [Brinson attribution](brinson_attribution.md).

## Public API boundary

**`qis.__all__` defines the top-level public surface.** The exports are assembled in
[`src/qis/__init__.py`](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/__init__.py).
[`qis.api.PUBLIC_API`](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/api.py)
records the same set as a literal. `CORE_API` identifies a curated, documented subset;
it is not a separate implementation or an additional export mechanism.

The generated [API reference](api/index.rst) follows the public exports, so an internal module
move need not change a supported top-level import. `dir(qis)` also contains attributes bound by
submodule imports; use `__all__` when checking whether a name is public:

~~~python
import qis

matches = sorted(name for name in qis.__all__ if 'tracking_error' in name.lower())
print(matches)
assert 'compute_ewma_realised_tracking_error' in matches
~~~

Before adding an export, decide whether it is a stable operation needed by callers or an
internal helper. Public changes affect consumers such as `optimalportfolios` and
`trendfollowing`. Synchronise the literal record with `python tools/sync_public_api.py`
after an authorised export change; `--check` detects drift without writing.
[The API tests](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/tests/test_core_api.py)
check export agreement and the documented core.

A module-level import in the [migration reference](REMOVED_5_0.md) may locate an internal helper
without making it a top-level public API. Refer to the release changelog when upgrading such code.

## Numerical contracts

- A decision at time $t$ sets the units held over $[t,t+1]$. Estimation on a backtest path must
  use information available at that decision.
- Between rebalances, held units remain fixed and weights drift with prices. Explicit costs
  and the weight implementation lag are part of the backtest input.
- Return type, observation grid and annualisation are stated. For `qis.to_returns`, pass
  `is_log_returns` explicitly.
- `SharpeConvention.PA`, `ARITHMETIC` and `LOG` choose different numerators/return conventions.
  The risk-free-rate choice is a separate dimension; excess variants require appropriate
  `PerfParams.rates_data`. See [Sharpe conventions](performance_analytics_and_sharpe.md).
- Missing starts, internal gaps, stale observations and delisted tails have different meanings.
  The [incomplete-history guide](incomplete_and_mixed_frequency_data.md) explains their policies.

A numerical verification should use a reference computed independently of the implementation.
Estimation windows, weight normalisation, annualisation, unsmoothing and resampling also need
checks for look-ahead and convention drift. A successful plot or plausible terminal NAV alone
does not establish those properties.

## Optional backends

The core imports without data-vendor clients, database drivers, Plotly, PyBloqs or PyArrow.
Optional dependencies are declared as [extras](install.md#optional-extras).
Reachable optional imports are function-local; a dedicated backend module may import its
dependency at module level when the top-level `qis` import cannot reach it.

Ruff's `TID253` rule checks this boundary with explicit exceptions in
[`pyproject.toml`](https://github.com/ArturSepp/QuantInvestStrats/blob/main/pyproject.toml).
Core and wheel test jobs exercise imports without optional extras. A function should document
whether a missing backend raises an error or permits a fallback. Tests requiring a backend
skip when it is absent.

## Package-stack boundary

The [ecosystem map](https://github.com/ArturSepp/QuantInvestStrats/blob/main/AGENTS.md)
records the package relationships. `qis` supplies reusable analytics to its consumers:

| Owning package | Responsibility outside qis |
|---|---|
| [optimalportfolios](https://github.com/ArturSepp/OptimalPortfolios) | Portfolio construction and optimisation. |
| [factorlasso](https://github.com/ArturSepp/FactorLasso) | Sparse factor-model estimation. |
| [bbg-fetch](https://github.com/ArturSepp/BloombergFetch) | Bloomberg data acquisition. |
| [trendfollowing](https://github.com/ArturSepp/TrendFollowingSystems) | Dedicated trend-following strategy analytics. |

Changes should follow these ownership boundaries and avoid dependency cycles. A second
performance-statistics implementation in a consumer would make shared conventions harder
to maintain.

## Verification and packaging

The [CI workflow](https://github.com/ArturSepp/QuantInvestStrats/blob/main/.github/workflows/ci.yml)
defines the tested environments. Its core matrix covers Python 3.10–3.14 on Linux, plus Windows
and macOS on Python 3.12. A separate locked lane covers the `data` and `io` extras. The wheel
lane builds from a source distribution, checks the archive contents, installs the core wheel
into a clean environment, and runs the shipped tests and offline quickstart outside the checkout.
These are configured checks, not a claim that a particular CI run has passed.

Automated tests live in source-adjacent `tests/` packages. Component diagnostics live beside
their implementation in `run_local/<subject>_run.py` and are excluded from wheels.
Repository examples live under `examples/` and are also excluded; documented offline scripts
can still run against the installed package.

There are two documentation trees:

| Source | Distribution and purpose |
|---|---|
| `src/qis/docs/*.md` | Text-only package notes shipped in the wheel and referenced by docstrings. |
| `docs/` | Site pages and generated audit records; not shipped in the wheel. |

The [Sphinx configuration](https://github.com/ArturSepp/QuantInvestStrats/blob/main/docs/conf.py)
mirrors packaged notes into `docs/_included/` at build time and generates API pages.
The top-level [Brinson article](brinson_attribution.md) is authoritative; its packaged note is
a pointer. Repository-integrity tests skip when their source files are absent from a wheel.

Contributor commands and authoring requirements are in
[CONTRIBUTING.md](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CONTRIBUTING.md)
and the [documentation standard](documentation_standard.md).

## References

- qis contributors. [Public API record](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/api.py),
  [packaging and lint configuration](https://github.com/ArturSepp/QuantInvestStrats/blob/main/pyproject.toml),
  and [CI workflow](https://github.com/ArturSepp/QuantInvestStrats/blob/main/.github/workflows/ci.yml).
- Sepp, A. qis: Performance analytics, portfolio backtesting, risk analysis, and factsheet
  reporting in Python. [Software citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

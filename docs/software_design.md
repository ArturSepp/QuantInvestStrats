---
myst:
  html_meta:
    description: >-
      Software architecture, public API boundaries, numerical contracts, optional dependencies,
      and extension points in qis.
---

# Software design

`qis` is the base analytics and reporting layer in a wider quantitative-finance stack. Its design
centers on labelled pandas objects, explicit statistical conventions, and a stable top-level API.
This page explains where functionality belongs and which contracts an extension must preserve.

## Layers and data flow

The typical workflow moves through five layers:

```text
prices / NAVs / synthetic fixtures
              |
              v
returns, estimators, and performance statistics
              |
              v
target schedules -> held units -> realised portfolio history
              |
              v
ex-ante and ex-post risk, attribution, and benchmark analysis
              |
              v
tables, plots, and factsheet reports
```

The layers share `DataFrame` and `Series` objects with a `DatetimeIndex`. Labels are retained
through calculations so a covariance matrix, weight vector, contribution table, and report can be
aligned and audited without reconstructing instrument order from an array.

- `qis.perfstats` owns return conversion, annualisation, performance statistics, drawdowns, and
  regime analysis.
- `qis.models` owns reusable estimators such as EWMA, regressions, bootstrap methods, and
  unsmoothing models.
- `qis.portfolio` turns target weights into a held-unit portfolio history and exposes attribution,
  risk, and reporting objects.
- `qis.plots` supplies the matplotlib/seaborn visual layer used directly and by factsheets.
- `qis.market_data` contains vendor-neutral transformations. Optional fetchers are integrations,
  not a prerequisite for the analytics core.

The focused guides describe the user-facing contracts in more detail: [performance and Sharpe
conventions](performance_analytics_and_sharpe.md), [portfolio
backtesting](portfolio_backtesting.md), [tracking error and risk](tracking_error_and_risk.md), and
[factsheets and reporting](factsheets_and_reporting.md).

## Public API boundary

The supported import surface is re-exported from `qis.__init__` and recorded as a literal in
`qis.api.PUBLIC_API`. `qis.api.CORE_API` identifies the documented capability-oriented subset used
by downstream packages, examples, and guides. The generated [API reference](api/index.rst) follows
those records rather than the physical module tree, so an internal module move does not change the
documented import path.

Before adding a new export, decide whether the operation is a stable cross-package primitive or an
internal helper. A new public name creates a compatibility obligation for `optimalportfolios`,
`trendfollowing`, and other consumers. Synchronize the literal export record with
`python tools/sync_public_api.py` whenever the top-level surface changes.

## Numerical contracts

Several conventions are structural rather than implementation details:

- A decision made at time *t* is applied over *[t, t+1]*; estimation inside a backtest is
  point-in-time and must not use later observations.
- Between rebalances, QIS holds instrument units. Realised weights drift with asset prices; a
  weighted average of returns is not a replacement for the backtest.
- Return type, sampling frequency, and annualisation are explicit. Use `qis.to_returns` with an
  explicit `is_log_returns` choice instead of inferring a convention from context.
- Sharpe statistics have separately labelled arithmetic, geometric, and excess-return
  conventions. Excess variants require the corresponding rates data in `PerfParams`.
- Missing starts, interior gaps, stale observations, and delisted tails are distinct states. The
  [incomplete-history guide](incomplete_and_mixed_frequency_data.md) documents their policies.

A test for a numerical change should compare against a reference computed through a different
path. Tests for estimation windows, weight normalisation, annualisation, unsmoothing, or resampling
also need an explicit check for look-ahead or convention drift.

## Optional backends

The core installation is offline and importable without data vendors, database drivers, Plotly,
PyBloqs, or PyArrow. Optional dependencies are declared as extras and imported inside the function
that needs them. A module dedicated to one optional backend may import it at module level only when
the top-level `qis` import cannot reach that module.

CI enforces this boundary statically with Ruff's `TID253` rule and dynamically by running the full
suite from a clean core wheel. New optional functionality should include a clear missing-dependency
error and a test that skips, rather than fails, when the extra is absent.

## Package-stack boundary

Dependencies flow toward QIS, not back from it. Portfolio construction and optimisation belong in
[`optimalportfolios`](https://github.com/ArturSepp/OptimalPortfolios), generic sparse factor models
in [`factorlasso`](https://github.com/ArturSepp/factorlasso), and Bloomberg data acquisition in
[`bbg-fetch`](https://github.com/ArturSepp/BloombergFetch). Reimplementing those layers here would
create two conventions for one concept and a dependency cycle.

## Verification and packaging

The default pytest suite runs against a core installation on every supported Python version and on
Linux, Windows, and macOS at the primary interpreter. A separate locked lane exercises the data and
I/O extras with a line-coverage ratchet. Static checks defend import direction, optional-dependency
boundaries, naming conventions, and production docstring coverage.

Automated tests stay in source-adjacent `tests/` packages. Interactive or local-data diagnostics
stay equally close to their implementation, but under `run_local/` as `<subject>_run.py`. Those
runners use `Locals` plus `run_local(local=...)`; production modules never import them and wheels
do not distribute them.

The wheel job builds the artifact users receive, verifies its tests and package documentation,
installs it into a clean environment, runs the shipped suite outside the checkout, and executes the
[offline quickstart](quickstart.md) against the installed artifact. Contributor commands matching
these lanes are in the repository's
[CONTRIBUTING.md](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CONTRIBUTING.md).

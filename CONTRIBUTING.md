---
myst:
  html_meta:
    description: >-
      Contribution scope, reproducible starter tasks, development checks and review for QuantInvestStrats.
---

# Contributing to QuantInvestStrats

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-07-21](https://github.com/ArturSepp/QuantInvestStrats/commit/a5e30ad58fac37bab018f03885ce5548c6308acd)*

Implemented in [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

Thanks for your interest in `qis`. `qis` is the analytics and reporting engine of a wider stack, so contributions here can affect downstream packages.

## Scope

In scope:

- Bug fixes in performance statistics, factsheets, or plotting
- New performance or risk statistics with a reference for the definition used
- Plotting improvements that follow the existing matplotlib/seaborn layer
- Documentation, examples, and tests

Out of scope — these will be declined, so please open an issue to discuss before
writing code:

- New hard runtime dependencies. Optional functionality belongs behind an extra in
  `[project.optional-dependencies]` with a guarded import
- Portfolio optimisation, which belongs in
  [`optimalportfolios`](https://github.com/ArturSepp/OptimalPortfolios)
- Factor model estimation, which belongs in
  [`factorlasso`](https://github.com/ArturSepp/factorlasso)
- Data vendor integrations. Bloomberg access belongs in
  [`bbg-fetch`](https://github.com/ArturSepp/BloombergFetch)
- Examples that require a paid data subscription to run

## Starter tasks

These are proposed, unassigned tasks, not reports of confirmed numerical defects. Check for
existing or completed work and agree a bounded scope before starting. The
[shared contributor guide](https://github.com/ArturSepp/ArturSepp/blob/main/docs/contributing_to_stack.md)
explains setup, review, attribution and validation of automation-assisted work. Reuse this
repository's [feature request form](.github/ISSUE_TEMPLATE/feature_request.yml), or its
[bug report form](.github/ISSUE_TEMPLATE/bug_report.yml) for an observed defect.
No starter label implies that numerical redesign is included.

Commands below use PowerShell's `& $py`, where `$py` is the selected development interpreter.
Run them from the repository source root (a C-local source export on the maintainer host).
Follow [development setup](#development-setup) first; for this Windows/OneDrive checkout,
use the external environment and generated-state helper described in `AGENTS.md` before
running Python or uv. The shared guide gives the exact interpreter table. Each command is
a focused check, not a substitute for the package's documentation and PR gates.

### QIS-1: Map the quickstart output to its Python objects

**Owner and code home:** QuantInvestStrats; [docs/quickstart.md](docs/quickstart.md).
**Read first:** [examples/getting_started/offline_quickstart.py](examples/getting_started/offline_quickstart.py), [src/qis/tests/test_quickstart.py](src/qis/tests/test_quickstart.py).

**Current gap:** The script prints targets, realised weights, performance and benchmark-relative results; the page has no compact output-to-object lookup table.

**Reproduce:** Read the printed labels in the canonical script, then run the existing quickstart tests below.

**Acceptance and scope:** Add one small table mapping those four output groups to weight_schedule, portfolio_data.weights, performance_table and the TE/IR objects. State that targets and drifted holdings can differ; link existing timing and Sharpe guidance. Keep every computed value, seed and executable snippet unchanged.

**Useful verification:**

```powershell
& $py -m pytest src/qis/tests/test_quickstart.py -q
```

### QIS-2: Make the quickstart test enforce its offline promise

**Owner and code home:** QuantInvestStrats; [src/qis/tests/test_quickstart.py](src/qis/tests/test_quickstart.py).
**Read first:** [examples/getting_started/offline_quickstart.py](examples/getting_started/offline_quickstart.py).

**Current gap:** test_quickstart_executes_and_reports_sane_results runs the example but does not block socket connections.

**Reproduce:** Inspect that test: it changes the working directory and checks results, without patching socket.socket.connect or socket.create_connection. Run it unchanged to establish the baseline.

**Acceptance and scope:** Use monkeypatch to reject both connection entry points during the existing run, with a useful error. Check that a deliberate connection attempt is rejected by the guard in an isolated test before trusting the green example run. Preserve current NAV assertions and the canonical script. Do not add a network service, dependency or new analytics implementation.

**Useful verification:**

```powershell
& $py -m pytest src/qis/tests/test_quickstart.py -q
```

### QIS-3: Explain how to diagnose a source quickstart mismatch

**Owner and code home:** QuantInvestStrats; [docs/quickstart.md](docs/quickstart.md).
**Read first:** [src/qis/tests/test_quickstart.py](src/qis/tests/test_quickstart.py), [tools/check_docs.py](tools/check_docs.py).

**Current gap:** The page links its checks but has no contributor decision table separating source-copy drift, a wrong interpreter and a numerical assertion failure.

**Reproduce:** Follow the quickstart source link and inspect the test assertions and the minimum supported version on the page.

**Acceptance and scope:** Add a concise table: check the loaded qis file/version for interpreter mismatch; compare the canonical script and clean notebook for copy drift; report the exact failing assertion and environment for a numerical discrepancy. Link existing installation and contribution guidance. Do not recommend updating expected numbers or changing the frozen fixture to make a check pass.

**Useful verification:**

```powershell
& $py tools/check_docs.py --files docs/quickstart.md
```

## Reporting a bug

Open an issue using the bug report template. A report needs the `qis` version, your
Python version, a minimal self-contained reproducer, and the full traceback or the
incorrect numbers. Reproducers that depend on proprietary or licensed data cannot be
run, so please use generated or public data.

## Asking a question

Open an issue and describe what you are trying to do. Questions about methodology are
welcome; where a question is really about the published papers, please say which paper
and section you are reading.

Support is best-effort and has no guaranteed response time. The decision model, maintenance
expectations, release policy, and private route for sensitive reports are documented in
[GOVERNANCE.md](GOVERNANCE.md).

## Development setup

For the maintainer's Windows/OneDrive checkout, first run the setup helper named in
`AGENTS.md` and select the registered C:\Python interpreter. Set
`UV_PROJECT_ENVIRONMENT` to that external environment before any uv project operation;
never use the unqualified uv examples below against a OneDrive-local environment.
Keep generated output and pure verification source exports on C: as described in the
[shared contributor guide](https://github.com/ArturSepp/ArturSepp/blob/main/docs/contributing_to_stack.md#setup-and-a-first-check).

```bash
git clone https://github.com/ArturSepp/QuantInvestStrats.git
cd QuantInvestStrats
uv sync --python 3.12 --group test --locked
uv run --no-sync pytest
```

The sync command selects Python 3.12 and installs the core package plus the PEP 735 `test`
dependency group exactly as recorded in `uv.lock`; it fails instead of re-resolving when the
lock and `pyproject.toml` differ. This is sufficient for the core test suite and offline
quickstart. `--all-extras` also installs the optional PostgreSQL driver, whose source build needs
native tools such as `pg_config`. See the [installation guide](docs/install.md#postgresql-and-all-extras-installation-on-macos)
for macOS prerequisites and an interpreter check. Other supported Python versions can also be
selected explicitly.

Tests live inside `src/qis/`, and the project configuration supplies that path automatically.
Files matching `*_test.py` or `test_*.py` contain automated pytest tests only. A diagnostic that
needs local data, interactive plots, or visual inspection lives beside the component it develops
as `run_local/<subject>_run.py`, using `Locals` and `run_local(local=...)`. These development-only
runners are excluded from wheels.

The CI extras and coverage lane is:

```bash
uv sync --python 3.12 --group test --extra data --extra io --locked
uv run --no-sync pytest --cov=qis --cov-report=term:skip-covered
```

The static gates use the locked `lint` group without installing the scientific stack:

```bash
uv run --locked --only-group lint ruff check --select TID251,TID253,ICN,F src/qis/
uv run --locked --only-group lint interrogate src/qis
```

The bare `ruff check src/qis/` command also selects the legacy `E` and `W` backlog and is not the
CI gate. CI additionally checks newly added lines with `.github/lint_changed_lines.py`. Running an
`--only-group lint` command replaces the active project environment; run the first `uv sync`
command again before returning to tests.

Build the documentation with warnings treated as errors:

```bash
uv sync --extra docs --locked
uv run --no-sync sphinx-build -W -b html docs tmp/docs-build
```

Build and inspect the wheel before a packaging change:

```bash
uv build --sdist --clear --out-dir dist
uv build --wheel dist/*.tar.gz --out-dir dist
```

Building the wheel from the source distribution reproduces CI and prevents files left in a local
build tree from leaking into the release artifact.

After installing that wheel and `pytest` into a clean environment, the supported post-install
check is `python -m pytest --pyargs qis`. The CI wheel job also runs the offline quickstart from
outside the checkout and verifies its deterministic NAV and benchmark-relative output.

`AGENTS.md` documents the layout, numerical conventions, verification loop, and scope constraints
in more detail. It is written for coding agents but is equally useful to human contributors.

## Pull requests

- One topic per pull request. Unrelated changes in the same PR make review slower and
  are likely to be asked to split.
- Add or update tests for behaviour you change. A bug fix should come with a test that
  fails before the fix.
- Run the documented CI-equivalent command set before submitting.
- Do not bump the version in `pyproject.toml` or `CITATION.cff`; releases are cut
  separately.
- Do not commit generated output: figures, factsheets, backtest results, or data files.
- Keep the public API stable. If a change alters a public signature or default, say so
  explicitly in the PR description.

## Conduct

Be civil and assume good faith. Technical disagreement is welcome; personal remarks are
not.

## Licence

This project is MIT licensed. By contributing, you agree that your contributions are licensed under
the MIT licence of this project.

# Contributing to QuantInvestStrats

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

```bash
git clone https://github.com/ArturSepp/QuantInvestStrats.git
cd QuantInvestStrats
uv sync --group test --locked
uv run --no-sync pytest
```

The first command installs the core package plus the PEP 735 `test` dependency group exactly as
recorded in `uv.lock`; it fails instead of re-resolving when the lock and `pyproject.toml` differ.
Tests live inside `src/qis/`, and the project configuration supplies that path automatically.
Files matching `*_test.py` or `test_*.py` contain automated pytest tests only. A diagnostic that
needs local data, interactive plots, or visual inspection lives beside the component it develops
as `run_local/<subject>_run.py`, using `Locals` and `run_local(local=...)`. These development-only
runners are excluded from wheels.

The CI extras and coverage lane is:

```bash
uv sync --group test --extra data --extra io --locked
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

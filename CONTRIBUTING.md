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

## Development setup

```bash
git clone https://github.com/ArturSepp/QuantInvestStrats.git
cd QuantInvestStrats
pip install -e ".[data]"
pytest qis/            # tests live inside the package, not in a top-level tests/
ruff check qis/
```

`AGENTS.md` in this repository documents the layout, commands, conventions, and
constraints in more detail — it is written for AI coding agents but is equally useful
to human contributors.

## Pull requests

- One topic per pull request. Unrelated changes in the same PR make review slower and
  are likely to be asked to split.
- Add or update tests for behaviour you change. A bug fix should come with a test that
  fails before the fix.
- Run the test suite and `ruff` before submitting.
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

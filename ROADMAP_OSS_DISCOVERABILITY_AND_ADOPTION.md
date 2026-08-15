# ROADMAP_OSS_DISCOVERABILITY_AND_ADOPTION

Version 1.0, 2026-08-15. Executor: a coding agent working on this repository checkout.
`AGENTS.md` is the source of truth for repository conventions, verification commands, numerical
invariants, generated records, and release policy. Execute one stage per request. Do not combine
stages merely to reduce the number of commits.

This roadmap adapts the maintainer-approved discoverability and adoption programme from
`OptimalPortfolios` to the actual role and current repository structure of `qis`. The roadmap
itself lives at the repository root, as required for feature roadmaps. Stage reports and dated
measurement records live under `agents/`; public user documentation remains under `docs/`, and
wheel-shipped package notes remain under `src/qis/docs/`.

## Objective

Improve qualified discovery and first-use adoption of `qis` through four coordinated
initiatives:

1. Strengthen the public documentation site with focused, indexable, practitioner-led pages.
2. Establish one consistent package identity around performance analytics, risk analysis,
   portfolio backtesting, and factsheet reporting.
3. Publish an accurate use-case comparison with adjacent Python performance-analysis and
   backtesting packages.
4. Reduce the time from `pip install qis` to a successful, meaningful offline result.

This is a documentation and adoption roadmap. It must not change performance-statistic
definitions, return conventions, annualisation, Sharpe conventions, portfolio mechanics,
tracking-error calculations, public signatures, random seeds, or computed results.

## Current baseline

As observed on 2026-08-15:

- PyPI serves `qis` 5.10.0. Its summary is the generic sentence "Quantitative Investment
  Strategies (QIS) package implements Python analytics for visualisation of financial data,
  performance reporting, analysis of quantitative strategies".
- The GitHub repository About description repeats that generic wording. The repository reports
  601 stars and 68 forks; these are dated adoption signals, not quality measures.
- The README opening is already stronger: it names financial-data visualisation, performance and
  risk analysis, portfolio backtesting, and factsheet reporting.
- The public Sphinx/Furo site is built from `docs/` at
  `https://quantinveststrats.readthedocs.io/en/latest/`. Its landing page returns HTTP 200 and its
  current HTML title is only `qis 5.10.0`.
- The documentation already has installation, an offline synthetic-data quickstart,
  reproducibility notes, a factsheet gallery, convention notes, and a generated public-API
  reference. Focused capability pages are still absent.
- `docs/quickstart.md` is executable and tested, but its complete workflow is maintained as
  Markdown code blocks rather than included from one canonical repository script.
- Repository examples live under `examples/`, intentionally outside the wheel. Static checks
  cover every example, and examples without data-vendor markers execute in the core test suite.
- `src/qis/datasets/synthetic.py` provides deterministic, vendor-free data with ragged starts,
  missing observations, stale prices, a delisted tail, fat tails, appraisal smoothing, and a
  monthly-reported sleeve. It is the correct source for first-use documentation and must remain
  frozen.
- Existing offline examples already cover ex-ante risk and tracking error, ex-post tracking
  error and information ratio, lagged weight implementation, and late-starting instruments.
- The three-letter token `qis` is intrinsically ambiguous. Discovery work must pair it naturally
  with "Quantitative Investment Strategies", Python, and a concrete use case. Owning the bare
  query `qis` is not a realistic success criterion.
- `qis` is under JOSS review. Adoption work must not interfere with the paper, its numerical
  claims, or the review process.

Credentialed Search Console data, complete index coverage, documentation referrals, and a
like-for-like download baseline are not represented in the repository. D0 records them or marks
them explicitly unknown.

## Global execution rules

1. **One stage per execution request.** Every stage ends with its own verification evidence and a
   status-log entry in this file.
2. **Small diffs.** A stage changes at most five implementation or product files. The one-line
   status update in this roadmap is bookkeeping; if the total scope would materially exceed
   roughly five paths, stop and propose a split before editing.
3. **No numerical changes.** Documentation examples exercise existing behavior. A numerical
   discrepancy is reported, not fixed under this roadmap.
4. **No look-ahead.** Backtest material states that a weight decided at time *t* applies over
   *[t, t+1]*. Estimation inputs are point-in-time, and `MeanAdjType.INSAMPLE` is never taught as
   a backtest default.
5. **State conventions.** Every numerical page states simple versus log returns, data and
   estimation frequency, annualisation, and the relevant Sharpe convention. Excess-return
   Sharpe examples provide `PerfParams.rates_data`.
6. **Respect portfolio mechanics.** Documentation says that `qis` holds units between
   rebalancings and distinguishes a backtest from a weighted average of returns.
7. **Use the public API.** User documentation imports names exported through `qis.__all__`.
   Internal module paths are reserved for contributor documentation.
8. **Runnable means verified.** Code advertised as runnable executes in the environment it
   claims to support. A successful Sphinx build is not proof of numerical correctness.
9. **Offline first.** Getting-started material uses `qis.datasets.synthetic` and requires no
   Yahoo Finance, Bloomberg, credentials, local data file, or optional extra.
10. **Preserve the two documentation trees.** User pages belong in `docs/`; package-relative
    notes needed after installation belong in `src/qis/docs/`. Do not duplicate a note between
    them.
11. **No generated artefacts in Git.** Do not commit PDFs, factsheets, figures, Sphinx output,
    notebook output cells, downloaded data, or execution caches.
12. **No keyword stuffing or unsupported superiority claims.** Write for investment and risk
    practitioners. Use technical terms naturally and substantiate every comparison.
13. **No competitor dependencies.** A comparison page may discuss adjacent packages, but this
    roadmap never imports or adds `quantstats`, `pyfolio`, `empyrical`, `ffn`, `bt`, `vectorbt`,
    or another analytics layer to `qis`.
14. **Concurrent-session safety.** Re-read every target file immediately before editing and
    preserve unrelated worktree changes.
15. **Plan -> patch -> verify.** Record the exact commands and observed outcomes in the stage
    report or status-log entry.
16. **Release-neutral.** Stages write user-visible changes under `[Unreleased]` when warranted.
    They do not bump versions, tag, upload, or release.

## Success measures

Record a baseline before D1 and compare at approximately 30, 60, and 90 days after D6, or D7 if
the optional notebook is approved. Search visibility may lag deployment; no stage fails merely
because a crawler has not revisited a page immediately.

Primary measures:

- Search Console impressions, clicks, click-through rate, and landing pages for qualified
  branded queries: `qis python`, `qis quantitative investment strategies`,
  `quantinveststrats`, `qis portfolio analytics`, and `qis factsheet`.
- The same measures for non-branded queries such as `portfolio performance analytics python`,
  `portfolio factsheet python`, `tracking error analytics python`, `Sharpe ratio python`,
  `portfolio backtesting python`, `private asset return unsmoothing python`, and
  `FX hedging analytics python`.
- Number of roadmap documentation pages indexed under the canonical Read the Docs property.
- Whether paired queries identify the package rather than unrelated meanings of `QIS`. Record
  the bare `qis` query only as an ambiguity diagnostic.
- Successful execution of the authoritative offline quickstart on Linux, Windows, and macOS.
- Time from a clean core installation to deterministic printed output from that quickstart.
- 30-day PyPI download trend, with the limitation that downloads are not unique users.
- GitHub stars, forks, dependent repositories, documentation referrals, issues attributable to
  the new pages, and scholarly citations as secondary indicators rather than quality gates.

Do not optimize a stage for a single vanity metric. The durable goal is qualified discovery that
leads a practitioner to a correct first result.

## Stage overview

| Stage | Deliverable | Precondition |
|---|---|---|
| D0 | Discovery, indexing, and conversion baseline report | none |
| D1 | Canonical `qis` identity in repository-owned metadata | D0 recorded |
| - | Maintainer gate A: GitHub About and Search Console | D1 deployed |
| D2 | Technical discoverability and canonical-index audit/remediation | gate A where credentials are needed |
| D3 | Focused documentation wave 1: performance, factsheets, tracking error | D2 |
| D4 | Focused documentation wave 2: backtesting, imperfect data, unsmoothing, FX | D3 |
| D5 | Neutral package comparison and choice guide | D4 |
| D6 | Single-source, offline first-success example | D5 |
| - | Maintainer gate B: hosted notebook policy decision | D6 |
| D7 | Optional thin Colab entry point | gate B approves a notebook |
| D8 | 30/60/90-day measurement report and next recommendation | D6, or D7 if approved |

---

## D0 - Establish the baseline

**Deliverable.** `agents/QIS_DISCOVERABILITY_BASELINE.md`; no product or public-documentation
change.

**Content.** Record:

- Current PyPI version, summary, keywords, classifiers, project links, and release cadence.
- Current GitHub About description, topics, homepage, documentation link, stars, forks, and
  dependent repositories, with source and timestamp.
- Current documentation URL, HTTP status, server-rendered page titles and descriptions,
  canonical tags, robots policy, and sitemap availability.
- Current Search Console property, submitted sitemap, index coverage, canonical-page status, and
  query data when available. Credentialed unknowns are labelled unknown rather than inferred.
- Manual text observations for the qualified branded and non-branded queries listed in Success
  measures. Do not save copyrighted search-result screenshots.
- Current 30-day PyPI downloads, documentation referrals, GitHub referrals, and scholarly
  citations, with definitions and limitations.
- The conversion path PyPI -> documentation -> quickstart -> API/gallery/examples, including
  broken, ambiguous, duplicated, or vendor-dependent steps.
- The current offline-example inventory and measured runtime of `docs/quickstart.md` in a core
  environment.

**Acceptance.** Every measurement has a date, source, definition, and limitation. Search Console
or analytics data that requires maintainer access is explicitly unknown until supplied.

**Verification.** Confirm the report contains the headings `Identity`, `Indexing`, `Queries`,
`Conversion path`, `Adoption signals`, and `Limitations`; confirm `git diff` contains only the
report.

**Out of scope.** Search Console configuration, metadata edits, documentation edits, SEO claims,
or changes to examples.

---

## D1 - Establish one canonical package identity

**Deliverable.** One commit touching at most five files, normally `pyproject.toml`, `README.md`,
`docs/index.md`, `docs/conf.py`, and `CHANGELOG.md`.

**Canonical position.** Use one concise factual formulation across package metadata and primary
page titles. Starting language for maintainer review:

> `qis` - performance analytics, portfolio backtesting, risk analysis, and factsheet reporting
> in Python.

The following sentence may explain scope without claiming exclusivity:

> Quantitative Investment Strategies covers time-series and cross-sectional performance,
> drift-aware portfolio histories, ex-ante and ex-post risk, and reproducible reports.

**Content.** Align:

- `[project].description` in `pyproject.toml`.
- The README title and first descriptive paragraph.
- The Sphinx landing-page title and opening paragraph.
- Sphinx `html_title` and any short-title metadata so result pages identify both the exact token
  `qis` and its use case rather than only a version number.
- The `[Unreleased]` changelog, describing the work as documentation and metadata positioning.

Preserve the distribution/import token `qis`, the repository name `QuantInvestStrats`, and the
expanded name `Quantitative Investment Strategies` where each is useful. Do not change package
keywords, dependencies, URLs, public API, or version in this stage.

**Acceptance.** The five surfaces do not contradict each other. The description fits PyPI core
metadata, remains readable, and does not become a keyword list.

**Verification.** Run:

```bash
pytest src/qis/tests/test_version_metadata.py src/qis/tests/test_documentation.py -q
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m build
python -m twine check dist/*
ruff check docs/conf.py
```

Inspect the wheel `METADATA` and confirm `Summary:` exactly matches the approved canonical
description. Remove local build output after verification if it is not already ignored.

**Out of scope.** Version bump, GitHub settings, custom domain, API changes, or documentation
content expansion.

---

## Maintainer gate A - External identity and Search Console

This gate contains credentialed settings and is not inferred from repository access.

The maintainer:

1. Updates the GitHub repository About description to the D1 canonical description and keeps
   the canonical documentation and PyPI links.
2. Verifies ownership of the active `quantinveststrats.readthedocs.io` property in Google Search
   Console. A custom domain is a separate policy decision and is not required by this roadmap.
3. Submits or confirms the canonical sitemap and records whether the landing page, quickstart,
   gallery, and API page are indexed.
4. Exports or summarizes the credentialed baseline needed by D8 without committing cookies,
   tokens, ownership secrets, or private analytics exports.

**Gate evidence.** A short maintainer confirmation or redacted Search Console summary.

---

## D2 - Audit and remediate technical discoverability

**Deliverable.** A report-only stage when the deployed Read the Docs configuration is already
correct. Otherwise, one minimal commit of at most five files plus a short report appended to the
D0 baseline.

**Checks.** Verify the deployed site, not only local HTML:

- HTTPS returns 200 for the root, install, quickstart, gallery, conventions, API reference,
  robots file, and sitemap.
- Pages contain no accidental `noindex`, `nosnippet`, robots exclusion, redirect loop, or
  canonical URL pointing to an unrelated host or version.
- One canonical HTTPS URL exists per page. `latest`, stable, and versioned Read the Docs URLs do
  not compete unnecessarily in the sitemap.
- Landing-page title, description, headings, and navigation are present in server-rendered HTML.
- Internal navigation reaches every public page without JavaScript.
- PyPI and GitHub link to the canonical documentation root, and documentation links back to
  PyPI, GitHub, issues, changelog, citation metadata, and the JOSS paper status truthfully.
- The sitemap is valid XML and reflects the current public pages. Prefer Read the Docs' native
  sitemap; add a Sphinx sitemap dependency only if a demonstrated defect requires it.
- The Sphinx build is warning-free. If so, change `.readthedocs.yaml` from
  `fail_on_warning: false` to `true`; otherwise record the blockers and leave it unchanged.

**Remediation rule.** Fix only demonstrated defects. Do not add `llms.txt`, speculative schema,
duplicate doorway pages, keyword landing pages, or special AI markup. If a dependency change is
genuinely required, update all locked or declared dependency records together and stay within
the five-file limit.

**Acceptance.** All listed pages are reachable and indexable, the sitemap is valid, and Search
Console can inspect the canonical landing page.

**Verification.** Run a warning-as-error Sphinx build, then scripted HTTP/title/canonical/robots/
sitemap checks after Read the Docs finishes. Record status codes, canonical URLs, deployment
build identifier, and any credentialed Search Console evidence.

**Out of scope.** Content expansion, custom-domain migration, paid SEO tooling, or crawler-only
content.

---

## D3 - Focused documentation wave 1

**Deliverable.** One commit containing `docs/index.md` and three focused pages:

1. `docs/performance_analytics_and_sharpe.md`
2. `docs/factsheets_and_reporting.md`
3. `docs/tracking_error_and_risk.md`

**Common page contract.** Each page contains:

- The practitioner problem and when the capability is appropriate.
- Exact inputs, units, return convention, frequency, annualisation, and NaN behavior.
- A minimal public-API example using the synthetic universe or a link to a verified canonical
  offline example.
- Expected output type and how to interpret it.
- Constraints, failure modes, and distinctions that prevent common misuse.
- A `See also` section linking the generated API reference, relevant convention note, and
  canonical repository example.

**Topic requirements.**

- Performance page: `PerfParams`, risk-adjusted performance tables, rolling statistics,
  drawdowns, simple versus log returns, and the three explicitly labelled Sharpe conventions.
- Factsheets page: `qis.factsheet`, the four input/report archetypes, reporting-frequency
  calibration, the difference between a figure list and a saved report, and links to the gallery
  and wheel-shipped factsheet convention note.
- Tracking-error page: distinguish ex-ante covariance-based `RiskModel` analysis from ex-post
  realised EWMA tracking error and whole-sample TE/IR. State units and annualisation. Reuse
  `examples/portfolios/ex_anti_tracking_error_and_risk.py` and
  `examples/portfolios/ex_post_tracking_error_and_risk.py`; do not create a second
  implementation of tracking error.

**Second pass.** Review every numerical statement against source and one independently computed
small reference. For tracking error, compare a whole-sample result with the standard deviation
of the explicitly formed return-difference series, using the same annualisation convention.

**Acceptance.** Every page is linked from the landing page, uses existing exported symbols, and
introduces no unverified numerical or stability claim.

**Verification.** Run:

```bash
python -m sphinx -W --keep-going -b html docs docs/_build/html
pytest src/qis/tests/test_examples.py src/qis/tests/test_documentation.py -q
pytest src/qis/tests/test_core_api.py -q
```

Execute every example or snippet advertised as core/offline and record the independent numerical
check. Expected: all pass.

**Out of scope.** New statistics, new factsheet behavior, API changes, competitor comparison,
or generated figures.

---

## D4 - Focused documentation wave 2

**Deliverable.** One commit containing `docs/index.md` and four focused pages:

1. `docs/portfolio_backtesting.md`
2. `docs/incomplete_and_mixed_frequency_data.md`
3. `docs/private_asset_unsmoothing.md`
4. `docs/fx_hedging_and_market_data.md`

**Content.** Apply the D3 page contract, with these mandatory distinctions:

- Backtesting: target weights versus held units, drift between rebalancings, implementation lag,
  transaction costs on traded notional, and the *t* to *t+1* no-look-ahead convention.
- Imperfect data: ragged starts, internal missing observations, stale prices, delisted tails, and
  genuinely low-frequency reporting are separate cases. Explain the explicit cash residual of
  untradeable target weights and reuse
  `examples/portfolios/static_weight_with_missing_prices.py`.
- Unsmoothing: de-levering is distinct from AR unsmoothing; descriptive full-sample estimators
  are distinct from point-in-time rolling use. State frequency and small-sample limitations and
  reuse `examples/perfstats/unsmoothing_and_delevering.py`. Label its `io`-extra requirement.
- FX and market data: `FxRatesData`, covered-interest-parity inputs, reference-currency return
  translation, hedged versus unhedged returns, and the boundary between core analytics and
  optional/vendor data acquisition. Do not make Bloomberg a prerequisite.

**Acceptance.** The four pages expose differentiated capabilities currently scattered across
the README, examples, and API without copying those sources wholesale or promising silent
recovery from arbitrary bad data.

**Verification.** Use the D3 commands plus an internal-link check. Execute each public snippet in
its declared environment. Run the late-starting-instrument example in the core environment and
the unsmoothing example with the declared `io` extra. Report external-link failures separately
because remote availability is not a deterministic repository gate.

**Out of scope.** Changes to backtest mechanics, missing-data policy, unsmoothing estimators,
FX formulas, data downloads, or optional dependencies.

---

## D5 - Publish a neutral comparison and choice guide

**Deliverable.** One commit touching `docs/package_comparison.md`, `docs/index.md`, and, if
needed, `README.md` and `CHANGELOG.md`.

**Comparison set.** `qis`, QuantStats, pyfolio-reloaded, and vectorbt. If one project is inactive
or no longer has authoritative current documentation at execution time, stop and propose a
maintainer-approved replacement rather than silently changing the set.

**Evidence standard.** Before writing, record the comparison date and current stable version of
each package. Every capability claim links to official documentation, source, release notes, or
paper. GitHub stars and download counts may appear only in a clearly labelled adoption section
with date, source, and limitations; they never determine a technical recommendation.

**Required structure.**

- A short statement that the libraries overlap but serve different workflows.
- A capability matrix covering at least: return/performance statistics, drawdowns and rolling
  analytics, tear sheets/factsheets, portfolio simulation from weights or orders, transaction
  costs, benchmark-relative analytics, ex-ante risk, ex-post tracking error/information ratio,
  regime analysis, mixed/incomplete histories, private-asset unsmoothing, FX translation and
  hedging, plotting/report customization, and primary design audience.
- A workflow decision guide with at least one use case favoring each package.
- A transparent `How this comparison was made` section with versions, official links, and known
  limits.
- A concise explanation of `qis`'s role as an analytics/reporting engine in the maintainer's
  stack, including when a dedicated trading simulator or a lighter report generator is the
  better choice.

**Wording rules.** Avoid `only`, `best`, `most comprehensive`, and `no other package` unless a
bounded reproducible test proves the exact statement. Do not install competitors, manufacture a
favorable benchmark, or add them as dependencies.

**Acceptance.** A reader can identify a credible use case favoring every compared package. Every
matrix cell is supported or explicitly marked unknown/not assessed.

**Verification.** Warning-free Sphinx build, internal-link check, and a manual citation audit in
which every nontrivial competitor claim resolves to an official source and the cited fragment or
section supports the cell.

**Out of scope.** Performance benchmarks, dependency additions, adversarial marketing, or pages
written solely for crawlers.

---

## D6 - Create one source of truth for first success

**Deliverable.** One commit, normally touching exactly these five files:

- `examples/getting_started/offline_quickstart.py`
- `docs/quickstart.md`
- `docs/index.md`
- `README.md`
- `CHANGELOG.md`

**Experience contract.** A user with a plain `pip install qis` can copy or run the workflow and
obtain a meaningful portfolio-analytics result without network access, credentials, optional
extras, or repository-local imports. Target elapsed time is under five minutes on a normal
laptop; record the measured time rather than promising it blindly.

The script is the source of truth. Documentation includes it with MyST/Sphinx `literalinclude`
or a mechanically equivalent method so the complete code cannot drift independently across the
README, docs, and examples. Prose may explain sections, but must not maintain a second full copy.

**Required flow.**

1. Load `generate_synthetic_universe` through its supported package path.
2. Select a small documented subset and period to keep execution fast.
3. State the price frequency and compute simple returns or a risk-adjusted table with explicit
   `PerfParams`.
4. Build a deterministic live-universe-aware weight schedule.
5. Backtest with an explicit rebalance frequency and transaction cost, stating that decisions at
   *t* apply to the following holding period.
6. Print compact deterministic evidence of success: date range, table or weight shape, final
   weights, final NAV, and one already-defined benchmark-relative output if it remains concise.
7. Point to the factsheet page as the next reporting step without writing an artefact by default.
8. Explain what to change first: statistic set, return/Sharpe convention, rebalance cadence,
   transaction cost, benchmark, and reporting frequency.

Do not commit generated NAVs, plots, factsheets, or output files. A short checked text excerpt may
appear in prose, but the executable script remains authoritative.

**Acceptance.** The script is automatically classified into the existing offline examples lane
and runs from a core wheel on Linux, Windows, and macOS. README and documentation point to the
same script.

**Verification.** Run:

```bash
pytest src/qis/tests/test_examples.py src/qis/tests/test_quickstart.py -q
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m build
```

Install the wheel into a clean environment and run the script from outside the checkout. Record
runtime, output shape, final NAV, absence of network attempts, and the imported `qis` path.

**Out of scope.** New analytics, plotting APIs, bundled output, optional data downloads, public
signature changes, or version/release work.

---

## Maintainer gate B - Decide whether to add a hosted notebook

`qis` already declares an optional `jupyter` extra, but a Colab notebook would still be a second
executable representation that can drift. The maintainer chooses one:

- **Approve a thin notebook.** Proceed to D7. It installs the released core package and mirrors
  D6 without relying on the repository checkout or the `jupyter` extra.
- **Keep first use script-and-docs only.** Skip D7. Link to the authoritative offline script and
  explain how to paste it into Colab. D6 remains the completed first-use implementation.

This decision does not block D0-D6.

---

## D7 - Optional thin Colab entry point

**Precondition.** Maintainer gate B explicitly approves a notebook.

**Deliverable.** A small repository-only `.ipynb`, a Colab badge/link, truthful documentation,
and a mechanical drift check. Split the stage before editing if it would exceed five files.

**Content.** The notebook:

- Installs the latest released `qis` from PyPI and prints its version and import path.
- Uses the synthetic universe and mirrors D6's workflow.
- Contains concise Markdown and no embedded generated plots, binary output, or execution state.
- Links back to the versioned documentation and canonical source example.
- Does not change core, optional, docs, dev, or all dependency groups.

Prefer mechanically generating or checking code cells from the D6 script. The notebook is a
hosted trial surface, not a new source of truth.

**Acceptance.** `Open in Colab` resolves to the notebook on `main`, a clean hosted runtime runs it
against the current PyPI release, and a repository test detects drift from D6.

**Verification.** Parse the notebook as JSON, run the drift check, execute D6 locally from the
installed wheel, and record one maintainer-confirmed clean Colab `Run all`. Never commit executed
output cells.

**Out of scope.** Notebook gallery, Binder infrastructure, new Jupyter dependencies, or
unreleased APIs.

---

## D8 - Measure impact and choose the next investment

**Deliverable.** `agents/QIS_DISCOVERABILITY_90_DAY_REPORT.md`, updated at approximately 30, 60,
and 90 days after deployment. No product change in this stage.

**Content.** Compare like-for-like periods against D0:

- Index coverage and canonical-page status.
- Qualified branded and non-branded impressions, clicks, CTR, and landing pages.
- Documentation entry paths, gallery/API referrals, and quickstart completion signals where
  available.
- PyPI downloads and GitHub adoption signals using the same definitions and caveats as D0.
- Search treatment of `qis` when paired with Python, the expanded package name, and concrete
  use cases.
- Issues, discussions, dependent repositories, citations, or external references attributable
  to the new material where evidence exists.
- Cross-platform quickstart results and observed time-to-first-result.

Recommend the next action from evidence: deepen the highest-performing topic, correct a
conversion failure, improve external distribution, or stop investing in a channel that produced
no qualified use. Do not infer causality from one metric alone.

**Acceptance.** Baseline and follow-up use the same definitions; missing data is explicit; the
recommendation follows from observed evidence.

---

## JOSS boundary

`qis` is already under JOSS review. This roadmap must remain independent of review decisions:

- Do not edit `paper.md`, `paper.bib`, numerical audit records, or manuscript claims merely to
  support discoverability wording.
- Do not describe an open review, reviewer recommendation, or passing check as formal
  acceptance.
- D0-D8 may proceed because they improve user discovery and first use regardless of the review
  outcome.
- After formal acceptance and publication details are available, create a separate
  maintainer-approved stage for DOI, citation, badge, `CITATION.cff`, and documentation updates.
  Do not append that work to a discoverability stage implicitly.

Conda-forge packaging, conference talks, paid promotion, custom domains, and broader ecosystem
distribution may be valuable, but they are outside the agreed documentation/discoverability/
comparison/first-success programme.

## Status log

Append one line per completed or skipped stage:

`YYYY-MM-DD · stage · branch/commit · PASS|PASS-LOCAL|SKIPPED|BLOCKED · concise verification result`

2026-08-15 · D0 · main@547d3f5 · PASS-LOCAL · required baseline headings present; public and
credentialed unknowns distinguished; quickstart test passed in 16.81 seconds on Windows.
2026-08-15 · D1 · main · PASS-LOCAL · canonical identity aligned; strict Sphinx, metadata tests,
package build, Twine validation, and wheel Summary inspection passed. Three docstring-only fixes
were included with maintainer approval to clear pre-existing strict-Sphinx warnings.

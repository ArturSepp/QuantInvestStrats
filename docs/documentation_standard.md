---
myst:
  html_meta:
    description: >-
      Authoring rules for qis methodology and analytics documentation: article structure,
      attribution, portable equations, reproducible examples, and figure provenance.
---

# Documentation standard

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-09-13](https://github.com/ArturSepp/QuantInvestStrats/commit/29fb6ce856e4480d9643508842a7ef5beff56cf9)*

This standard applies to human-authored documentation for
[qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Use the project's [citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff)
when citing the software.

This is the QIS supplement to the
[shared OSS documentation standard](https://github.com/ArturSepp/ArturSepp/blob/main/docs/documentation_standard.md).
The shared guide owns common authoring rules; this page retains QIS-specific examples,
source ownership, analytics tooling, and verification. General changes belong in the shared
guide. Existing section headings remain available for incoming links.

## Article structure

Use the shared [article structure](https://github.com/ArturSepp/ArturSepp/blob/main/docs/documentation_standard.md#user-content-article-structure),
with the H2 `Implementation in qis`. Utility pages use the shared shorter form.
The QIS checker enforces the eight methodology headings and their order.

Explain simple versus log returns, annualisation, estimation versus reporting grids, gross
versus net results, risk-free-rate assumptions, and prior versus current weights where relevant.
These are part of the calculation contract, including when a figure illustrates the method.

## Handbook conventions

The methodology articles form the *qis analytics handbook*. Beyond the shared structure, every
article follows these rules; `tools/check_docs.py` and the documentation tests enforce the
mechanical ones.

- **Convention card.** The first table under `Inputs, notation, and assumptions` has the header
  `| Convention | This article |` and exactly these rows, in order: Return basis, Sampling grid,
  Annualisation, Mean adjustment, Timing, Output units, qis default. The
  [notation chapter](notation_and_conventions.md) defines each row.
- **Reserved notation.** Symbols listed in the notation chapter keep one meaning in every
  article. The annualisation factor is $\mathrm{af}$, set upright as one symbol and
  named after the code's `af` argument. A local
  symbol is declared in the article's notation table and never reuses a reserved one. Write
  the transpose as `\top` and variance or covariance as `\operatorname{Var}` and
  `\operatorname{Cov}`; the checker rejects `\mathsf{T}`, `\intercal`, `\mathrm{Var}` and
  plain-text formulas such as `Sigma_` or `sqrt(`.
- **Results and concise proofs.** State a result in a paragraph opening with a bold
  **Definition.**, **Identity.** or **Proposition.** label, and follow each identity or
  proposition with a short **Proof.** paragraph ending in $\square$. Cite a longer derivation
  instead of reproducing it.
- **Insight and Pitfall callouts.** Write `> **Insight.** ...` or `> **Pitfall.** ...` as an
  ordinary blockquote. `docs/_ext/qis_callouts.py` renders these as admonitions in Sphinx;
  other viewers show a readable quotation.
- **Executed worked examples.** Every `python` block of a methodology article runs, offline,
  in `src/qis/tests/documentation_examples_test.py`. Put a comment line
  `<!-- docs-test: skip -->` immediately before a schematic block that cannot run.
- **One bibliography.** [The bibliography](bibliography.md) holds every cited work once, in one
  style. An article's References section is a numbered list whose items begin with a
  bibliography entry verbatim, optionally followed by a note, and it includes the software
  citation. `src/qis/tests/documentation_bibliography_test.py` enforces the match.
- **Coverage.** Every analytics symbol in the `CORE_API` groups of `src/qis/api.py` is named in a
  methodology article, and every `PerfStat` member has an entry in the
  [performance-statistic catalogue](performance_statistics.md);
  `src/qis/tests/documentation_coverage_test.py` enforces both. `docs/conf.py` links each core
  group on the API page to the articles listed for it in `CAPABILITY_CHAPTERS`.
- **Teaching figures.** A chapter figure comes from `tools/docs_analytics/handbook.py` on the
  frozen synthetic universe, carries an independent numerical check, and is registered in the
  analytics manifest like every other preview. Its caption states the question, the numbers the
  reader should take away, and the qis call that produced them.
- **Spelling.** Prose uses British spelling; Python names keep their published American spelling.

## Copyable methodology template

Copy the [shared methodology template](https://github.com/ArturSepp/ArturSepp/blob/main/docs/documentation_standard.md#user-content-copyable-methodology-template),
replace `PACKAGE` with `qis` and `REPOSITORY` with `QuantInvestStrats`, and supply the topic.
Keep the MyST description front matter. Follow the shared
[authorship and date rules](https://github.com/ArturSepp/ArturSepp/blob/main/docs/documentation_standard.md#user-content-authorship-and-dates);
a new uncommitted page uses the linked Artur Sepp byline without a date.

## Portable equations

Follow the shared [portable mathematics rules](https://github.com/ArturSepp/ArturSepp/blob/main/docs/documentation_standard.md#user-content-portable-mathematics).
The QIS targets are GitHub Markdown, MyST/Sphinx, and VS Code Markdown preview.

For example, let $p_i$ be an absolute capital share:

$$
N_{\mathrm{eff}} = \frac{1}{\sum_i p_i^2}.
$$

Four equal capital shares give an effective count of four. The source is:

````markdown
Let $p_i$ be an absolute capital share.

$$
N_{\mathrm{eff}} = \frac{1}{\sum_i p_i^2}.
$$
````

## References and source ownership

Apply the shared [reference and example rules](https://github.com/ArturSepp/ArturSepp/blob/main/docs/documentation_standard.md#user-content-references-and-executable-examples).
Use QIS's `CITATION.cff` for software metadata. A frozen result also records its actual
qis version and source commit or content hash.

Only names in `qis.__all__` are the top-level public API; label internal contributor references.
Keep canonical scripts under their existing example/tool ownership, with ordinary source links
beside Sphinx includes. Do not maintain copied full scripts.

Packaged notes under `src/qis/docs/` remain text-only. Their build-time copies are not another
authoring location. The [Brinson article](brinson_attribution.md) is the explicit exception whose
complete methodology lives in top-level `docs/`; its packaged note remains a pointer. The
packaged Sharpe note is a short convention summary that points to the Sharpe chapter.

## Figures and analytical results

Each analytics preview must have an identifiable producer, data sample, parameters, conventions,
and actual source version. Its caption explains the question and the result; alt text describes
the comparison rather than only naming a file. Keep figures readable at normal page width and
provide access to full-resolution factsheets when a small preview cannot show every panel.

The [batch producer registry](https://github.com/ArturSepp/QuantInvestStrats/tree/main/tools/docs_analytics)
covers all 71 current previews: 20 synthetic exhibits, eight empirical cash-rate previews,
21 empirical hedged-index comparisons and 22 empirical unhedged-index comparisons.
One command regenerates synthetic images and supporting CSVs, preserves reviewed empirical
image bytes and records provenance. Review the bundle, then use the publisher to validate the
complete set before updating the allowlisted previews and their shared provenance record.
`python -m tools.docs_analytics.publish --verify --repo <checkout>` checks published image hashes.

Generate and verify analytics on C-local storage. Only reviewed preview PNGs allowlisted by the
implemented manifest and their provenance record may be copied to `docs/images/` as intentional
deliverables. Full report bundles, PDFs, temporary plots, downloaded data, and caches are excluded.
This is the narrow exception recorded in AGENTS.md, not permission to commit arbitrary output.

Keep fixed synthetic sample periods fixed. Record generation time separately. Use the frozen
synthetic universe for new market-panel demonstrations; preserve established teaching simulations
with known model effects and their existing seeds. Live-data refresh must be an explicit operation.
Displayed tables and numerical captions should be checked against the same result used to plot.

The [cash-rate timing case study](cash_rate_timing_and_fx_adjustments.md) is a specifically
approved empirical exception. Its producer preserves reviewed historical PNGs and recorded
aggregate statistics without distributing raw vendor returns or private mandate results.
The complete offline batch validates frozen hashes and aggregate integrity; it does not claim
to refit undistributed observations. A separate, explicit private-input command can refit the
fixed sample and check all recorded statistics. Record input hashes, observation cutoff,
actual refit source, generation time and the limits of public reproducibility.

The [hedged-index replication study](hedged_index_replication.md) is a second specifically
approved empirical exception. It preserves 21 reviewed previews and 42 aggregate records
covering recent and full available histories. The explicit private-input helper rechecks
the recent 69-month derived panel only; it does not claim to refit undistributed full
histories or re-fetch vendor data. Original payoff/source verification and current bundle
generation have separate provenance. Raw return panels and private mandate data remain excluded.

The [unhedged-index replication study](unhedged_index_replication.md) is an approved empirical
exception with 22 reviewed three-panel previews and 44 aggregate records. Each preview compares
supplied generic spots and FX implied by a different index family. The latter is a held-out
consistency diagnostic, not an independent WMR replication. Offline generation validates
frozen records and image hashes; the explicit private helper rechecks both FX methods on
the recent 69-month derived panel. It neither acquires vendor data nor refits undistributed
full histories. Input hashes and original numerical review remain separate from bundle generation.

## Verification and migration

The standard's source checker is [tools/check_docs.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/tools/check_docs.py).
It checks descriptions, visible bylines and software links, heading structure, and display-math
source. It ignores fenced teaching examples and reports source line numbers for violations.
It does not validate all TeX syntax, methodological truth, bibliography accuracy, or rendered layout.

After the repository's mandatory environment setup, use its prescribed interpreter:

```console
python tools/check_docs.py
python tools/check_docs.py --files docs/portfolio_breadth.md
python tools/check_docs.py --all
```

The default validates explicitly adopted pages and names remaining pages as pending. `--files`
validates the requested revision batch, regardless of adoption status. `--all` is the final full
migration gate; the current inventory is fully adopted. Add newly authored pages to the explicit
inventory and adoption set; do not exempt an unknown page by leaving it unlisted.

Run relevant example and documentation tests, plus a strict Sphinx build, from a C-local source
export. Sphinx generates source files under `docs/`, so changing only its output directory is not
sufficient for a OneDrive checkout. Inspect rendered equations after the math engine completes
and inspect figures for clipping, table overlap, legibility, and light/dark contrast. Do not claim
a viewer was checked when only the source or another viewer was inspected.

## See also

- [Documentation home](index.md)
- [Performance and Sharpe conventions](performance_analytics_and_sharpe.md)
- [Factsheet gallery](gallery.md)
- [Contributor guidance](https://github.com/ArturSepp/QuantInvestStrats/blob/main/AGENTS.md)

## References

- Sepp, A. qis: Performance analytics, portfolio backtesting, risk analysis, and factsheet
  reporting in Python. [Citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
- [CommonMark specification](https://spec.commonmark.org/0.31.2/).
- [GitHub: writing mathematical expressions](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions).
- [MyST: math and equations](https://myst-parser.readthedocs.io/en/latest/syntax/math.html).
- [VS Code: Markdown](https://code.visualstudio.com/docs/languages/markdown).

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

Use the shared [article structure](https://github.com/ArturSepp/ArturSepp/blob/main/docs/documentation_standard.md#article-structure),
with the H2 `Implementation in qis`. Utility pages use the shared shorter form.
The QIS checker enforces the eight methodology headings and their order.

Explain simple versus log returns, annualisation, estimation versus reporting grids, gross
versus net results, risk-free-rate assumptions, and prior versus current weights where relevant.
These are part of the calculation contract, including when a figure illustrates the method.

## Copyable methodology template

Copy the [shared methodology template](https://github.com/ArturSepp/ArturSepp/blob/main/docs/documentation_standard.md#copyable-methodology-template),
replace `PACKAGE` with `qis` and `REPOSITORY` with `QuantInvestStrats`, and supply the topic.
Keep the MyST description front matter. Follow the shared
[authorship and date rules](https://github.com/ArturSepp/ArturSepp/blob/main/docs/documentation_standard.md#authorship-and-dates);
a new uncommitted page uses the linked Artur Sepp byline without a date.

## Portable equations

Follow the shared [portable mathematics rules](https://github.com/ArturSepp/ArturSepp/blob/main/docs/documentation_standard.md#portable-mathematics).
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

Apply the shared [reference and example rules](https://github.com/ArturSepp/ArturSepp/blob/main/docs/documentation_standard.md#references-and-executable-examples).
Use QIS's `CITATION.cff` for software metadata. A frozen result also records its actual
qis version and source commit or content hash.

Only names in `qis.__all__` are the top-level public API; label internal contributor references.
Keep canonical scripts under their existing example/tool ownership, with ordinary source links
beside Sphinx includes. Do not maintain copied full scripts.

Packaged notes under `src/qis/docs/` remain text-only. Their build-time copies are not another
authoring location. The [Brinson article](brinson_attribution.md) is the explicit exception whose
complete methodology lives in top-level `docs/`; its packaged note remains a pointer.

## Figures and analytical results

Each analytics preview must have an identifiable producer, data sample, parameters, conventions,
and actual source version. Its caption explains the question and the result; alt text describes
the comparison rather than only naming a file. Keep figures readable at normal page width and
provide access to full-resolution factsheets when a small preview cannot show every panel.

The [batch producer registry](https://github.com/ArturSepp/QuantInvestStrats/tree/main/tools/docs_analytics)
covers all seven current previews. One command regenerates their images, supporting CSVs and
provenance. Review the bundle, then use the publisher described in its README to validate the
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

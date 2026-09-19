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

Methodology articles explain an analytical concept before describing its implementation. Write
in a neutral, definition-led style: identify assumptions, explain the result, and cite the sources
of substantive methodological claims. Distinguish established methods from qis implementation
conventions. The author placeholder is deliberate and must remain until those details are supplied.

## Article structure

A methodology article has one H1, a visible byline and project/citation links, a short lead, and
the following H2 sections in order. Use descriptive H3 subsections for individual methods. Existing
section links can be preserved with named anchors when reorganizing an article.

| Section | Required content |
|---|---|
| Overview | The question the method answers and when it is useful. |
| Inputs, notation, and assumptions | Symbols, dimensions, units, observation frequency, timing, and data policies. |
| Methodology | Definitions and equations, followed by their interpretation. |
| Worked example | Fixed inputs, a small result, and what the result establishes. |
| Implementation in qis | Public entry points, input/output contract, runnable source, and verification context. |
| Interpretation and limitations | Assumptions, uncertainty, edge cases, and unsuitable uses. |
| See also | A small set of relevant methods or guides. |
| References | Verified method sources and the qis software citation. |

Explain simple versus log returns, annualisation, estimation versus reporting grids, gross versus
net results, risk-free-rate assumptions, and prior versus current weights wherever relevant.
A figure or number is not self-explanatory merely because a function name appears beside it.

Installation, navigation, quickstart, gallery, architecture, package comparison, and migration
history use a **utility form**. They retain one H1, the byline, project/citation links, a useful
description, and a logical heading hierarchy, but need no empty methodology or equation sections.
Generated API pages and build-time copies of packaged notes retain their own generating source.

## Copyable methodology template

Replace topic placeholders with actual content. Use the confirmed author and repository-date
format below. Record a tested version only after verifying the actual imported package;
local metadata, a PyPI release, and an uncommitted checkout can describe different source states.

````markdown
---
myst:
  html_meta:
    description: >-
      [A factual description of the concept and its qis implementation.]
---

# [Method or analytical concept]

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [YYYY-MM-DD](https://github.com/ArturSepp/QuantInvestStrats/commit/COMMIT_SHA)*

Implemented in [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

[Define the concept and its scope in a short lead.]

## Overview

[Purpose and appropriate uses.]

## Inputs, notation, and assumptions

| Symbol or input | Meaning | Units and convention |
|---|---|---|
| [symbol] | [definition] | [units, frequency, and timing] |

## Methodology

[Introduce the equations and explain their meaning.]

## Worked example

[Fixed inputs, result, and interpretation; label synthetic data explicitly.]

## Implementation in qis

[Public entry points, ordinary source links, and reproduction instructions.]
[Record the verified version/source and verification date when available.]

## Interpretation and limitations

[Assumptions, uncertainty, and edge cases.]

## See also

[Related methods and guides.]

## References

- [Verified author, year, title, venue, and DOI or primary-source link.]
- Sepp, A. qis: Performance analytics, portfolio backtesting, risk analysis, and
  factsheet reporting in Python.
  [Software citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
````

The confirmed author is [Artur Sepp](https://github.com/ArturSepp). Link that name in the
byline; omit affiliation unless supplied. **First recorded** is the earliest available
repository commit date for the article, following renames and earlier RST versions where
applicable. Link the date to that commit, using its full hash. Git history is evidence of
repository inclusion, not the exact time a page was pushed to GitHub or publicly posted.

Inspect `git log --follow --format="%H %cI" -- docs/<page>.md` and the previous source path.
Use the earliest entry's committer date, preserving its recorded timezone's calendar date.
For a Markdown conversion, also check its former RST source so reformatting does not reset
the article's date. Do not substitute the date of this edit, a release, a file modification,
or an analytics run. For a new page with no committed history, use
`*Author: [Artur Sepp](https://github.com/ArturSepp)*` until a repository date is available.

Author date, last substantive review date, data cutoff, and image generation time are separate
facts. A build must not silently claim that the methodology was reviewed again.

## Portable equations

The supported targets are GitHub Markdown, the MyST/Sphinx documentation site, and VS Code's
built-in Markdown preview. They share dollar-delimited math, but do not implement every TeX macro
identically. Basic CommonMark viewers may show TeX source; provide a rendered-site link when needed.
See [GitHub math](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions),
[MyST math](https://myst-parser.readthedocs.io/en/latest/syntax/math.html), and
[VS Code math preview](https://code.visualstudio.com/docs/languages/markdown#_math-formula-rendering).

Use `$...$` inline and `$$` on separate lines for display math, with blank lines around the block.
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

- Keep semantic mathematics out of code spans/fences. Fences are appropriate when showing source,
  as above. Do not use fenced `{math}` directives or `{math}`/`{eq}` roles in article prose.
- Place `aligned`, `cases`, or matrices inside a display block. Split long formulas at meaningful
  equalities. Use a small common vocabulary rather than custom TeX macros.
- Define symbols before use. Keep one meaning per symbol and use explicit subscripts for timing.
- Link to the ordinary heading of a derivation; avoid renderer-specific automatic equation labels.
- Keep complex formulas out of table cells. Use `\lvert`/`\rvert` and `\lVert`/`\rVert` for
  absolute values and norms where a raw pipe could be parsed as a table separator.
- Write currency amounts as `USD 100` or `CHF 100` near math. Do not globally unescape paths,
  URLs, code, or literal TeX examples when repairing a delimiter.

## References and source ownership

Cite methods at the relevant claim and give full bibliographic entries under References. Verify
author names, titles, dates, and DOIs against the publisher or another primary source. Use ordinary
Markdown links so citations work outside Sphinx. A software citation does not replace the source
of a mathematical method, and a paper citation does not identify the software that produced a chart.

Use `CITATION.cff` as the software metadata source. A frozen result additionally needs its actual
qis version and source commit or content hash; a moving `main` link alone cannot reproduce it.
Do not infer affiliation, publication acceptance, or a version-specific validation from a build.

Keep canonical runnable scripts under their existing example/tool ownership. Provide an ordinary
source link beside every Sphinx `literalinclude`. Link to a rendered API page or canonical source
when a generated API file is absent from a checkout. Do not maintain manually copied full scripts.
Only names in `qis.__all__` are the top-level public API; label internal contributor references.

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
migration gate; all 23 current pages have been adopted. Add newly authored pages to the explicit
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

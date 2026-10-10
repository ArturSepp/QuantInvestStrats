# Reproducible documentation analytics

This repository-only tool builds all 81 registered documentation previews, supporting aggregate
CSV tables and provenance. Thirty exhibits use offline synthetic inputs. Eight approved empirical
cash-rate previews preserve reviewed bytes and aggregate fits without distributing raw vendor
observations. Twenty-one hedged-index previews preserve the corresponding reviewed figures
and 42 aggregate records. Twenty-two unhedged-index previews compare supplied spots and
different-family index-implied FX, with 44 aggregate records. The package does not import this tooling.

## Run all analytics

On the maintainer's Windows host, apply the current session setup in [AGENTS.md](../../AGENTS.md)
and work from a verified C-local source export with its `src` directory on `PYTHONPATH`.
Use the external project interpreter:

~~~powershell
$docsPython = 'C:\Python\QuantInvestStrats312\Scripts\python.exe'
$bundle = Join-Path $agentRun 'analytics-bundle-01'

& $docsPython -m tools.docs_analytics.run --list
& $docsPython -m tools.docs_analytics.run --all --output-dir $bundle
& $docsPython -m tools.docs_analytics.run --validate-bundle $bundle
~~~

The second command regenerates the entire set. Each run needs a new output directory below
`AGENT_LOCAL_ROOT`, outside the source export and OneDrive. On other hosts, set that variable to
a local scratch root and use a core qis development installation. The runner uses Matplotlib's
Agg backend, bundled DejaVu Sans font, and existing core dependencies; no data extra is required.
It blocks Python socket connections and name resolution during generation.

Generation first writes into a new sibling staging directory. Missing producers, exceptions,
failed checks, source changes during the run, unreadable figures or incomplete outputs prevent
completion. Failed staging directories retain diagnostics and must not be published.
Existing output directories are never overwritten.

## Manifest and provenance

[manifest.json](manifest.json) is the coverage ledger. It lists each image's consumer, stable ID,
producer, fixed inputs, assumptions, expected tables and numerical checks. Adding a Markdown,
HTML or MyST image to the README or site without registering its producer fails the inventory check.
External status and Colab badges are explicitly classified as non-analytics. Code examples and generated Sphinx
mirrors are excluded.

A complete bundle contains:

- `images/*.png`: the 81 registered previews; synthetic images and empirical refits use 150 dpi.
- `tables/gallery/*.csv`, `tables/model_layer/*.csv` and `tables/handbook/*.csv`: inputs and
  supporting computed values.
- `analytics_manifest.json`: generation timestamp, fixed sample dates, parameters, conventions,
  Python and package versions, imported qis path, effective source hashes, output hashes and
  dimensions, numerical check results and summary values.

The record distinguishes `qis_source_version` (the exported project's version) from
`dependencies.qis` (installed distribution metadata); these may differ when running current
source through an older installed environment. The import path must belong to the source export.

The source fingerprint covers qis Python source, producer code, the model-layer and bootstrap
convention examples,
project metadata and available root lock/requirements files. This identifies a dirty source
export without pretending that HEAD or a distribution version identifies its contents.
Hashes of the input CSVs identify the data used. Supporting tables preserve full floating-point
precision; figure labels are presentation rounding.

Validation checks completeness, hashes, file readability, numerical-check status and agreement
with the current manifest and effective source. It does not certify visual readability or
independently recompute every factsheet statistic. A validator is an integrity check, not a
signature against deliberate rewriting of both files and their provenance.

## Producer contracts

[readme.py](readme.py) renders eight README previews from the current report APIs and the frozen
synthetic universe (seed 20260725, 2018–2025, no quirks). Eight instruments span equities, bonds
and commodities. Inverse-volatility targets use 21/63/126-observation estimators, monthly
rebalancing, one-business-day implementation lag and 10 bp costs; the 63-span portfolio is
compared with equal weights. Reports use the monthly preset and no cash-rate adjustment.
The standalone performance table uses a constant 3% annual cash rate, monthly risk statistics
and quarterly regression. Four main reports use the gallery's focused layout; the risk appendix
shows six existing panels, preserving their data. The positions page composes current
`PortfolioData` plots. Brinson table headings wrap and percentages use two decimal places.
The displayed weight totals include residual cash, independently marked as NAV less holdings
and lagged to each return period. The Cash row includes initial uninvested balances and cost
funding; no cash interest is supplied. Its gross attribution effects are zero. Native cash weights
and the displayed totals are saved alongside the unchanged instrument-only attribution tables.
Brinson uses native-date BHB effects and Frongello linking, gross of realised costs, with
interaction assigned to selection and benchmark-regime background colours. Independent
compounded-return and held-unit references check the linked attribution and backtest, and
plotted effects are compared with saved tables. These replace unregistered legacy screenshots;
they are new reproducible teaching exhibits, not a refresh of their historical Yahoo samples.

[gallery.py](gallery.py) uses the frozen `qis.datasets.generate_synthetic_universe` fixture,
seed 20260725 and the fixed 2018–2025 sample. Four continuously available instruments illustrate
three quarterly rebalanced allocations. Prices retain fixture quirks; no missing observation is
filled by the adapter. Costs are 10 bp per unit traded. All four reports call `qis.factsheet`.
The default output selects four panels per report and enlarges their labels for article width.
It preserves the selected line values and displayed table entries. The comparison table keeps
the two portfolio rows; the complete reports retain every original panel and row.
These replace the old A0–A5 gallery examples, whose exact original producer was not established.
They are new teaching exhibits, not numerical regressions against the old screenshots.

[model_layer.py](model_layer.py) composes the existing dedicated model-layer example: seeds
169/170, 2005-12-31 baseline, 240 monthly log returns through 2025-12-31, and Bartlett HAC(3)
intervals. It preserves that simulation's known layer and feature effects and calls its existing
independent identity/design checks. Tables and figures share the same computed attribution
objects. This named fixture is an exception to the market-panel fixture rule.

[handbook.py](handbook.py) draws fifteen teaching exhibits for thirteen methodology chapters from the
frozen synthetic universe (seed 20260725, 2005–2025, no quirks); the convexity-premium chapter has
two, both on quarterly returns from 2005-03-31 against the universe's `SBM_6040` benchmark. Three
explicit teaching constructions are applied on top of it: the volatility-targeting
figure scales the US equity returns by recorded volatility regimes, and the bootstrap figure
compares the qis draw with the truncating draw of `examples/models/bootstrap_convention.py`; the
two FX-hedging exhibits define a synthetic CHF-per-USD cross as normalised `SBD_TSY` raised to
1.5, with constant 3.5% USD and 3.0% CHF annual quotes. FX estimation uses the 2005–2025 monthly
history and a 36-month EWMA span; performance starts on 31 December 2010. Their public-QIS hedge
rules, lagged forward wealth, geometric returns and monthly-log-volatility Sharpe are checked
against independent formula/NumPy references. Each
exhibit has an independent check against a closed form or a direct numpy calculation, named after
its table.

A producer returns figures keyed by preview filename, tables keyed by the manifest table name,
Boolean checks, actual parameters and result summaries. Unexpected output sets or unsuccessful
checks fail the whole run. Producers must not access vendor data, local undistributed files,
test-private fixtures or `run_local` code.

### Approved empirical cash-rate case study

[cash_rate_case_study.py](cash_rate_case_study.py) preserves eight registered historical preview
PNGs and reconstructs the 24-row aggregate table from manifest parameters. This is the named
empirical exception: its default offline producer reads only allowlisted public PNGs and
aggregate records. It returns validated paths so the batch copies PNG bytes without re-rendering.
Hash and summary validation is not a claim to refit raw observations.

The explicit private-input refit command is separate from the unattended batch:

~~~console
python -m tools.docs_analytics.cash_rate_case_study --source-csv /private/cash_rate_monthly_comparison.csv --output-dir /local/new-refit
~~~

It verifies the input snapshot hash, reconstructs lag 0, lag 1 and midpoint from annual quotes,
checks all 24 recorded fits against SciPy linear regression and direct error moments, and uses
`qis.plot_scatter` with a linear fit and intercept. It requires authorised private input but no
data acquisition or network. Raw observations stay outside the repository. A refit with changed
inputs must be a new reviewed study rather than silently replacing this sample.

### Approved empirical hedged-index study

[hedged_index_case_study.py](hedged_index_case_study.py) preserves 21 registered comparison
PNGs and reconstructs the 42-row aggregate table. Its offline integrity checks cover
ticker/window completeness, the 69-month common sample, finite statistics, error bounds,
sample TE/RMSE consistency, geometric-return differences and frozen preview hashes.
Neither this producer nor the unattended batch reads private vendor observations.

~~~console
python -m tools.docs_analytics.hedged_index_case_study --source-csv /private/monthly_comparison.csv --output-dir /local/new-private-recheck
~~~

This explicit private command checks the frozen input hash, 21 recent-period regression
fits and error moments, the derived opening-principal hedge identity, canonical QIS sample
TE and geometric returns, then plots through QIS. It verifies the recent 69-month derived
panel, not undistributed full histories or raw FX acquisition. Full-history records retain
the original independently checked analysis. The registry records the original QIS
implementation hash separately from the complete bundle's current effective-source hash.
The original reviewed empirical PNG bytes are preserved by default; a new private refit
does not silently replace published previews. Raw returns and private mandates stay private.

### Approved empirical unhedged index study

[unhedged_index_case_study.py](unhedged_index_case_study.py) preserves 22 reviewed three-panel
previews, 44 aggregate rows, matched-pair identities and cross-family currency diagnostics.
Each figure has supplied-spot and held-out index-FX scatterplots plus cumulative relative NAV.
The diagnostic withholds the target family; it is not a temporally out-of-sample model or an
independently observed WMR series. The offline producer validates both sets of error bounds,
sample completeness, non-circular anchors, aggregate identities and frozen image hashes.

~~~console
python -m tools.docs_analytics.unhedged_index_case_study --source-csv /private/monthly_comparison.csv --output-dir /local/new-private-recheck
~~~

The explicit private command reconstructs both FX paths from the recent derived panel using
`FxRatesData`, rechecks regression, sample tracking error and geometric returns, and replots
through QIS. It does not re-fetch quotes or recheck full-history inputs. Publicly frozen image
bytes retain their original review and source identity. Raw histories and private mandates
are not part of the bundle.

## Complete gallery reports

For the complete reports, call the gallery producer with `full_reports=True` from the same
prepared source export. This writes four PNGs and four PDFs to a new C-local directory. These
larger reports are local deliverables; the documentation publisher accepts the focused previews.
Choose a new directory in the current task's scratch area before running the Python block:

~~~powershell
$env:QIS_DOCS_REPORT_DIR = Join-Path $agentRun 'full-reports-01'
~~~

~~~python
import os
from pathlib import Path

import matplotlib.pyplot as plt

from tools.docs_analytics.gallery import produce
from tools.docs_analytics.run import load_manifest, offline, output_boundary
from tools.docs_analytics.style import render_context, save_figure

output_dir = output_boundary(Path(os.environ['QIS_DOCS_REPORT_DIR']))
output_dir.mkdir()
with offline(), render_context():
    result = produce(load_manifest()['producers']['gallery'], full_reports=True)
    for name, figure in result['figures'].items():
        save_figure(figure, output_dir / name)
        figure.savefig(output_dir / Path(name).with_suffix('.pdf'), bbox_inches='tight')
        plt.close(figure)
~~~

## Review and publication

Generation does not update `docs/images/`. Compare two bundles from the same environment:
their `outputs` and `producers` records should match; generation time is intentionally different.
Review every PNG at full resolution and at its eventual page width. Check the captions against
the figures and supporting CSVs. Publish the reviewed bundle using an explicit target checkout:

~~~powershell
& $docsPython -m tools.docs_analytics.publish --publish-bundle $bundle --repo $targetCheckout
& $docsPython -m tools.docs_analytics.publish --verify --repo $targetCheckout
~~~

Set `$targetCheckout` to the intended repository path; use the C-local export first to review the
rendered site. The publisher validates every bundle output and requires the target's producer
source and manifest to match. It copies all 81 allowlisted PNGs together and writes
`docs/images/analytics_manifest.json` last. Supporting CSVs stay in the build bundle.

The publisher saves previous files in a new C-local backup directory beside the bundle and
reports that path. If a write or final verification fails, it restores files already replaced.
Per-file replacement and rollback reduce partial updates; they are not a filesystem transaction
and cannot guarantee recovery after process termination or power loss. Preserve the backup.

`--verify` checks the published PNG bytes against their recorded hashes and the current coverage
ledger. It treats the recorded source as a dated snapshot; it does not require a subsequent code
checkout to be byte-identical to the source that generated the images.

A fixed synthetic sample is deliberate. Regeneration time measures implementation freshness;
the observation cutoff measures sample coverage. Do not advance dates to make an exhibit look
recent. Cross-version or cross-platform byte-for-byte PNG reproducibility is not promised.

# Reproducible documentation analytics

This repository-only tool generates all 18 analytics images referenced by the documentation,
their supporting CSV tables, and a provenance record. It uses qis calculations and offline synthetic
inputs. The package does not import this tooling.

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
HTML or MyST image without registering its producer fails the inventory check. The external
Colab badge is explicitly classified as non-analytics. Code examples and generated Sphinx
mirrors are excluded.

A complete bundle contains:

- `images/*.png`: the 18 preview filenames, saved at 150 dpi.
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

[handbook.py](handbook.py) draws one teaching exhibit for each of eleven methodology chapters
from the frozen synthetic universe (seed 20260725, 2005–2025, no quirks). Two exhibits use an
explicitly stated teaching construction on top of it: the volatility-targeting figure scales the
US equity returns by recorded volatility regimes, and the bootstrap figure compares the qis draw
with the truncating draw of `examples/models/bootstrap_convention.py`. Each exhibit has an
independent check against a closed form or a direct numpy calculation, named after its table.

A producer returns figures keyed by preview filename, tables keyed by the manifest table name,
Boolean checks, actual parameters and result summaries. Unexpected output sets or unsuccessful
checks fail the whole run. Producers must not access vendor data, local undistributed files,
test-private fixtures or `run_local` code.

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
source and manifest to match. It copies all 18 allowlisted PNGs together and writes
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

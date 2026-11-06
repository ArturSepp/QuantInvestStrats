# Configuring qis resource and output directories

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-10-23](https://github.com/ArturSepp/QuantInvestStrats/commit/77fd15cc8acd19e527b6617730edc50f998f73d3)*

Implemented in [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

[qis](https://github.com/ArturSepp/QuantInvestStrats) uses [local_path.py](local_path.py)
and [settings.yaml](settings.yaml) for directory configuration. Software citation:
[CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
This guide describes the 5.44.7 configuration contract.

## Start without configuration

A fresh clone or installation reads `settings.yaml` beside `local_path.py`. No process
environment setting selects or overrides that YAML file. Its default directories are:

| YAML key | Default directory | Purpose |
|---|---|---|
| `RESOURCE_PATH` | `~/.qis/resources` | Resource files and deliberate download caches |
| `LOCAL_RESOURCE_PATH` | `~/.qis/resources` | Legacy local-resource setting |
| `UNIVERSE_PATH` | `~/.qis/resources/universes` | Saved universe files |
| `OUTPUT_PATH` | `~/.qis/outputs` | Figures, tables and reports |

`~` is the current user's home directory. On Windows these normally resolve below
`C:/Users/<username>/.qis`; they are not inside the cloned repository or installed package.
Public examples still need their documented dependencies and, for Yahoo downloads, network
access. Examples explicitly loading local files require those files to be supplied.

```python
from qis import local_path

paths = local_path.get_paths()  # Resolve settings; create no directories.
outputs = local_path.get_output_path()  # Create ~/.qis/outputs, including parents.
resources = local_path.get_resource_path(create=True)  # Prepare a download/cache directory.
```

`get_resource_path()` without `create=True` requires an existing resource directory and
raises if it is missing. Creating a directory does not download or fabricate its files.
`get_output_path(create=False)` resolves a destination without creating it, for report
writers that require a fresh path. No directories are created merely by importing qis.

## Explicit YAML overrides

The adjacent `settings.yaml` is the configuration source for ordinary QIS calls and
examples. For a particular call, pass another YAML file with `settings_path=`. Keep
private files outside the checkout and installed package; release checks reject
machine-specific distributed defaults and extra YAML files inside the package.

For example, a caller-owned `C:/Data/qis/private-settings.yaml` could contain:

```yaml
RESOURCE_PATH: 'C:/Data/qis/resources'
LOCAL_RESOURCE_PATH: 'C:/Data/qis/resources'
UNIVERSE_PATH: 'C:/Data/qis/resources/universes'
OUTPUT_PATH: 'C:/Data/qis/outputs'
```

Only include keys you want to override. Omitted keys retain their packaged defaults;
overriding `RESOURCE_PATH` does not automatically change `LOCAL_RESOURCE_PATH` or
`UNIVERSE_PATH`. Optional database users can also set `AWS_POSTGRES` privately; the
distributed value is an inert placeholder, not a configured service. Protect any
credentials and never commit the private file.

## Explicit overrides and resolution rules

Directory resolution uses this priority:

1. Explicit directory passed to a getter or example runner.
2. Explicit `settings_path` YAML passed to a getter.
3. `settings.yaml` beside `qis.local_path`.

```python
from qis import local_path

outputs = local_path.get_output_path('~/qis-one-run/outputs')
outputs = local_path.get_output_path(settings_path='~/private-qis.yaml')
# For an existing resource directory:
# resources = local_path.get_resource_path('~/my-data/resources')
```

Both `~` and environment variables are expanded. Relative paths inside YAML are relative
to that YAML file, not the current working directory. Explicit relative directory arguments
are relative to the working directory. Getters return absolute strings; join filenames
with `Path(directory) / filename`, not string concatenation.

An explicitly selected missing or malformed YAML file raises an error rather than silently
falling back. Invalid directory values, including old `...` placeholders, also raise.
Settings are cached by YAML file: after editing a file in a running Python session, call
`local_path.get_paths.cache_clear()`. Switching to another file is recognised by the next
getter call. Legacy `file_utils` constants are import-time snapshots; pass resolved paths
explicitly to readers and writers when changing configuration within a session.

## Keep the run_local convention

Repository examples retain `Locals`, `run_local(local=...)`, and a main guard containing
only the chosen runner call. There is no separate output-directory CLI parser. From a
checkout with qis installed, this offline synthetic example saves two PNGs and a CSV:

```python
from examples.plots.cluster_dendrograms import Locals, run_local

run_local(local=Locals.EXPORT)  # Default or private YAML OUTPUT_PATH.
# Alternatively, override the destination for one run:
# run_local(local=Locals.EXPORT, output_path='~/qis-one-run/clusters')
```

Examples are checkout workflows, not installed `qis` modules. Display-only and offline
check branches do not export reports; download/cache branches may still write resource
data. Stress reports require a fresh destination for each export. See the
[example catalogue](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/README.md)
for each runner's branches, dependencies and explicit overrides.

For maintainer tests and disposable runs, pass C-local temporary directories explicitly
and follow [AGENTS.md](https://github.com/ArturSepp/QuantInvestStrats/blob/main/AGENTS.md).

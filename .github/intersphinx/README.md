# Reviewed Sphinx reference inventories

*Author: [Artur Sepp](https://github.com/ArturSepp)*

These symbol indexes support offline strict HTML builds for
[qis](https://github.com/ArturSepp/QuantInvestStrats)
([software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff)).
They retain links to the official Python, NumPy, pandas and Matplotlib documentation.
`manifest.json` records the original URLs, inventory versions, retrieval timestamp and SHA-256.
The documentation configuration verifies the hashes before using the local files.

Run `python tools/refresh_intersphinx.py --output-dir <new-C-local-bundle>` to prepare an update.
Review the versions and hashes, then copy the five bundle files into this directory and build
strict HTML before committing. Refreshing these reference indexes does not change a scientific
result, package dependency, article authorship date or analytics sample.
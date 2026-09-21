"""Download a reviewable bundle of reference inventories for offline Sphinx builds.

Example: python tools/refresh_intersphinx.py --output-dir C:/path/to/new-bundle
Review the hashes/versions, then copy the intentional bundle to .github/intersphinx/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zlib
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

SOURCES = {
    "python": "https://docs.python.org/3/objects.inv",
    "numpy": "https://numpy.org/doc/stable/objects.inv",
    "pandas": "https://pandas.pydata.org/docs/objects.inv",
    "matplotlib": "https://matplotlib.org/stable/objects.inv",
}


def main():
    """Fetch inventories without modifying the checkout or suppressing build warnings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    root = Path(__file__).resolve().parents[1]
    if output == root or root in output.parents:
        raise ValueError("Generate the review bundle outside the checkout.")
    output.mkdir(parents=True, exist_ok=False)
    inventories = {}
    for name, url in SOURCES.items():
        request = Request(url, headers={"User-Agent": "qis-docs-inventory-refresh/1.0"})
        with urlopen(request, timeout=30) as response:
            data = response.read()
        header = data.split(b"\n", 4)
        if len(header) != 5 or header[0] != b"# Sphinx inventory version 2":
            raise ValueError(f"Not a version-2 Sphinx inventory: {url}")
        records = zlib.decompress(header[4]).decode("utf-8").splitlines()
        if not records:
            raise ValueError(f"Empty inventory: {url}")
        (output / f"{name}.inv").write_bytes(data)
        inventories[name] = {
            "source": url,
            "sha256": hashlib.sha256(data).hexdigest(),
            "project": header[1].decode(),
            "version": header[2].decode(),
            "entries": len(records),
        }
        print(f"{name}: {len(records)} entries, {inventories[name]['version']}")
    manifest = {
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "inventories": inventories,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

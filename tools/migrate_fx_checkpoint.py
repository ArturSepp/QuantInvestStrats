"""Explicitly migrate trusted pre-6.0 FX pickle paths into a new checkpoint.

This repository-only tool is not imported by QIS. Pickle can execute code: use it
only on trusted archives with their original dependency cohort available. Source
files are never overwritten. A receipt records hashes and mapped class names.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
from pathlib import Path
import pickle


_CLASS_PATHS = {
    ("qis.market_data.fx_rates_data", "FxRatesData"): "qis.fx.fx_rates_data",
    ("qis.market_data.factors_data", "FactorsData"): "qis.portfolio.risk.factors_data",
    **{
        ("qis.market_data.fx_returns", name): "qis.fx.fx_returns"
        for name in (
            "FxReturnType",
            "FxReturnCoordinate",
            "FxCashAccrual",
            "FxCashPolicy",
            "FxIndexNormalization",
            "FxAssetReturnsRequest",
            "FxCurrencyReturnsRequest",
            "FxReturnsData",
        )
    },
}


class FxCheckpointUnpickler(pickle.Unpickler):
    """Map only known renamed FX classes; all other dependencies load normally."""

    def __init__(self, stream):
        """Initialize the trusted archive stream and its actual remapping ledger."""
        super().__init__(stream)
        self.remapped: set[tuple[str, str, str]] = set()

    def find_class(self, module: str, name: str):
        """Resolve known old identities without installing namespace aliases."""
        target = _CLASS_PATHS.get((module, name))
        if target is not None:
            self.remapped.add((module, name, target))
            return getattr(importlib.import_module(target), name)
        if module == "qis.market_data" or module.startswith("qis.market_data."):
            raise pickle.UnpicklingError(f"Unmapped retired QIS identity: {module}.{name}")
        return super().find_class(module, name)


def migrate_checkpoint(source: Path, destination: Path) -> dict:
    """Write a new checkpoint and hash receipt without touching the source archive.

    Args:
        source: Trusted pickle created with the former FX namespace.
        destination: New file; existing files and in-place conversion are rejected.

    Returns:
        Source/output identities and the exact class paths remapped during loading.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    receipt_path = destination.with_suffix(destination.suffix + ".migration.json")
    if source == destination or destination.exists() or receipt_path.exists():
        raise FileExistsError("Use a new destination and retain the source checkpoint")
    raw = source.read_bytes()
    loader = FxCheckpointUnpickler(io.BytesIO(raw))
    value = loader.load()
    converted = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    # Check serialization before creating any output. Consumer-specific restoration
    # contexts, if required, must surround this explicitly invoked offline operation.
    pickle.loads(converted)
    receipt = {
        "source": str(source),
        "destination": str(destination),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "destination_sha256": hashlib.sha256(converted).hexdigest(),
        "remapped": [
            {"source_module": old, "class": name, "destination_module": new}
            for old, name, new in sorted(loader.remapped)
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation rejects a concurrent writer as well as existing archives.
    with destination.open("xb") as stream:
        stream.write(converted)
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return receipt


def main() -> None:
    """Migrate one explicitly selected trusted archive from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(migrate_checkpoint(args.source, args.destination), indent=2))


if __name__ == "__main__":
    main()

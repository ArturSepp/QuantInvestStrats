"""Initialize repository tooling before pytest 7 creates synthetic parent modules."""

import importlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
importlib.import_module("tools")

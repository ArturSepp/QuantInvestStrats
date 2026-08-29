"""Compatibility exports for the canonical descriptive-table implementation.

New code should import these symbols from ``qis`` or ``qis.perfstats.desc_table``. This historical
deep-import path remains available so existing callers receive the same enum and function objects.
"""

from collections.abc import Callable
from typing import Protocol, cast

import pandas as pd

# qis
import qis.perfstats.desc_table as canonical_desc_table
from qis.perfstats.desc_table import DescTableType


class _CanonicalDescTableProtocol(Protocol):
    """Typed view of the canonical function re-exported below."""

    compute_desc_table: Callable[..., pd.DataFrame]


_CANONICAL_DESC_TABLE = cast(_CanonicalDescTableProtocol, canonical_desc_table)

# Preserve object identity while giving the compatibility export a complete static type.
compute_desc_table = _CANONICAL_DESC_TABLE.compute_desc_table

__all__ = ['DescTableType', 'compute_desc_table']

"""Structural regressions for descriptive-table implementation identity.

The public descriptive-table implementation lives in ``qis.perfstats.desc_table``. The historical
``qis.plots.derived.desc_table`` import path remains available for compatibility, but it must expose
the same enum and function objects rather than maintaining a second implementation that can drift.
These identity assertions make every behavioral regression on the canonical function apply to both
import paths without duplicating the numerical test matrix.
"""

from typing import Protocol, cast

# qis
import qis
import qis.perfstats.desc_table as canonical_desc_table
import qis.plots.derived.desc_table as legacy_desc_table


class _ComputeDescTableExportProtocol(Protocol):
    """Typed identity-only view of the function exported by each module."""

    compute_desc_table: object


_CANONICAL_EXPORTS = cast(_ComputeDescTableExportProtocol, canonical_desc_table)
_LEGACY_EXPORTS = cast(_ComputeDescTableExportProtocol, legacy_desc_table)
_PUBLIC_EXPORTS = cast(_ComputeDescTableExportProtocol, qis)


# =============================================================================
# Canonical implementation identity
# =============================================================================


def test_legacy_desc_table_module_reexports_canonical_symbols() -> None:
    """Keep the legacy deep-import path as a narrow alias of the canonical implementation.

    Object identity is stronger than equivalent enum values or matching function behavior: it
    proves that callers cannot receive incompatible ``DescTableType`` classes and that fixes made
    to the canonical function cannot leave a second shipped implementation behind.
    """
    assert legacy_desc_table.DescTableType is canonical_desc_table.DescTableType
    assert _LEGACY_EXPORTS.compute_desc_table is _CANONICAL_EXPORTS.compute_desc_table
    assert legacy_desc_table.__all__ == ["DescTableType", "compute_desc_table"]


def test_public_desc_table_exports_resolve_to_canonical_symbols() -> None:
    """Preserve the existing top-level enum and function identities while retiring the duplicate."""
    assert qis.DescTableType is canonical_desc_table.DescTableType
    assert _PUBLIC_EXPORTS.compute_desc_table is _CANONICAL_EXPORTS.compute_desc_table

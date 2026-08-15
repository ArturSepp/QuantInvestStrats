"""
Shared helpers for qis examples.

These are not standalone examples themselves — they're imported by examples
to avoid duplicating boilerplate (default column sets, multi-figure performance
report layouts, common universe definitions).
"""
from qis.examples._helpers.reporting_helpers import (
    DEFAULT_RA_TABLE_COLUMNS,
    generate_performance_report,
)

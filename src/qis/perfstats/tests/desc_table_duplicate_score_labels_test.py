"""Regression coverage for duplicate labels in descriptive-score tables.

Pandas permits repeated DataFrame column labels, and the descriptive-table contract retains one
output row per physical input column. ``WITH_SCORE`` must therefore calculate each column's last
observed value and percentile rank by position rather than reselecting all columns sharing its
label. The deterministic panel below uses nonadjacent duplicates, missing observations, a tied
final score, and an all-missing duplicate to distinguish the physical histories and verify their
original order.
"""

import warnings
from typing import Protocol, cast

import numpy as np
import pandas as pd
import pytest

# qis
import qis.perfstats.desc_table as desc_table_module
from qis.perfstats.desc_table import DescTableType


class _DescTableModuleProtocol(Protocol):
    """Typed test-side interface for the public function exercised below."""

    def compute_desc_table(
            self,
            *,
            df: pd.DataFrame | pd.Series,
            desc_table_type: DescTableType) -> pd.DataFrame:
        """Return a formatted descriptive-statistics table."""
        raise NotImplementedError


_DESC_TABLE_MODULE = cast(_DescTableModuleProtocol, desc_table_module)


# =============================================================================
# Deterministic duplicate-label fixture
# =============================================================================

_ASSET_LABELS = pd.Index(
    ['Duplicate Asset', 'Control Asset', 'Duplicate Asset', 'Duplicate Asset'],
    name='Asset',
)
_DATES = pd.date_range('2024-01-31', periods=5, freq='ME')


def _duplicate_score_returns() -> pd.DataFrame:
    """Create distinct physical histories under nonadjacent duplicate labels.

    Returns:
        Five-row return panel with three distinct physical ``Duplicate Asset`` columns.
    """
    return pd.DataFrame(
        [
            [1.0, 0.0, 10.0, np.nan],
            [4.0, 2.0, np.nan, np.nan],
            [np.nan, 4.0, 5.0, np.nan],
            [2.0, 6.0, 15.0, np.nan],
            [3.0, 8.0, 5.0, np.nan],
        ],
        index=_DATES,
        columns=_ASSET_LABELS,
    )


# =============================================================================
# Positional score behavior
# =============================================================================

@pytest.mark.parametrize(
    'use_nullable_dtype',
    [False, True],
    ids=['float64', 'nullable-float64'],
)
def test_compute_desc_table_scores_duplicate_labels_by_physical_column(
        use_nullable_dtype: bool) -> None:
    """Score duplicate histories independently for standard and nullable floating data.

    The first duplicate has mean ``10 / 4 = 2.5``, sample standard deviation
    ``sqrt(5 / 3)``, and final-value rank ``3 / 4 = 75%``. The control has mean 4,
    sample standard deviation ``sqrt(10)``, and rank 100%. The second duplicate has mean
    ``35 / 4 = 8.75``, sample standard deviation ``sqrt(68.75 / 3)``, and a final value tied
    at ranks 1 and 2, whose average percentile is ``1.5 / 4 = 37.5%`` and displays as 38%.
    The final duplicate is all missing, so each of its statistics remains undefined.
    """
    returns = _duplicate_score_returns()
    if use_nullable_dtype:
        returns = returns.astype(pd.Float64Dtype())
    original_returns = returns.copy(deep=True)
    expected = pd.DataFrame(
        {
            'Avg': ['2.50', '4.00', '8.75', 'nan'],
            'Std': ['1.29', '3.16', '4.79', 'nan'],
            'Last': ['3.00', '8.00', '5.00', 'nan'],
            'Rank': ['75%', '100%', '38%', 'nan%'],
        },
        index=_ASSET_LABELS,
    )

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        actual = _DESC_TABLE_MODULE.compute_desc_table(
            df=returns,
            desc_table_type=DescTableType.WITH_SCORE,
        )

    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(returns, original_returns)

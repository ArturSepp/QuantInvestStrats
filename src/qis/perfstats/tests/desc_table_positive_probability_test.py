"""Regression coverage for positive probabilities with incomplete return histories.

``compute_desc_table`` documents that statistics use available observations. Positive
probability must therefore divide each asset's positive-return count by that asset's non-missing
observation count, rather than by the common number of DataFrame rows. A missing return is an
absent observation, while a supplied zero return is observed and non-positive.

The deterministic fixtures below calculate those ratios directly and exercise both descriptive
table modes that report them. Series/DataFrame consistency, output labels and order, custom names,
and caller ownership are included because the function returns a display-ready string table.
"""

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
# Shared deterministic fixtures
# =============================================================================

_DATES = pd.date_range('2024-01-31', periods=4, freq='ME')

_ALL_MISSING_ASSET = 'All Missing Asset'
_COMPLETE_ASSET = 'Complete Asset'
_POSITIVE_COLUMN = 'Positive'
_RAGGED_ASSET = 'Ragged Asset'
_SPARSE_POSITIVE_ASSET = 'Sparse Positive Asset'


def _mixed_history_returns() -> pd.DataFrame:
    """Create complete and ragged histories with directly countable signs.

    ``Complete Asset`` has two positive returns among four observations, with a supplied zero
    remaining in the denominator. ``Ragged Asset`` has one positive return among two observed
    returns. ``Sparse Positive Asset`` has two positive returns and no other observed return.
    ``All Missing Asset`` has no observations and therefore no positive probability.

    Returns:
        Four-row return panel in the reporting order used by expected output.
    """
    return pd.DataFrame(
        {
            _RAGGED_ASSET: (np.nan, 0.02, -0.01, np.nan),
            _COMPLETE_ASSET: (0.01, -0.02, 0.00, 0.03),
            _SPARSE_POSITIVE_ASSET: (np.nan, 0.04, 0.05, np.nan),
            _ALL_MISSING_ASSET: (np.nan, np.nan, np.nan, np.nan),
        },
        index=_DATES,
    )


def _expected_positive_probabilities() -> pd.Series:
    """Return independently calculated display values for the shared panel.

    Returns:
        Positive-probability strings indexed in the original asset order.
    """
    return pd.Series(
        ('50.0%', '50.0%', '100.0%', 'nan%'),
        index=(
            _RAGGED_ASSET,
            _COMPLETE_ASSET,
            _SPARSE_POSITIVE_ASSET,
            _ALL_MISSING_ASSET,
        ),
        name=_POSITIVE_COLUMN,
    )


# =============================================================================
# Available-observation denominator contract
# =============================================================================

@pytest.mark.filterwarnings('ignore:Mean of empty slice:RuntimeWarning')
@pytest.mark.filterwarnings('ignore:Degrees of freedom <= 0 for slice:RuntimeWarning')
@pytest.mark.parametrize(
    ('desc_table_type', 'expected_columns'),
    (
        (
            DescTableType.WITH_POSITIVE_PROB,
            ('Avg', 'Std', _POSITIVE_COLUMN),
        ),
        (
            DescTableType.AVG_WITH_POSITIVE_PROB,
            (_POSITIVE_COLUMN,),
        ),
    ),
)
def test_compute_desc_table_positive_probability_uses_available_observations(
        desc_table_type: DescTableType,
        expected_columns: tuple[str, ...]) -> None:
    """Exclude missing returns from each asset's probability denominator.

    The direct ratios are ``1 / 2`` for the ragged asset, ``2 / 4`` for the complete asset, and
    ``2 / 2`` for the sparse positive asset. The all-missing asset has a zero denominator and
    therefore displays ``nan%``. The complete fixture also proves that a supplied zero is observed
    but not positive. Both public table modes must report the same positive column, retain their
    established surrounding columns, and leave the input unchanged. Existing all-missing mean and
    standard-deviation warnings are outside this probability contract and filtered at test scope.

    Args:
        desc_table_type: Positive-probability table mode under test.
        expected_columns: Complete expected display-column schema for that mode.
    """
    returns = _mixed_history_returns()
    original_returns = returns.copy(deep=True)

    actual = _DESC_TABLE_MODULE.compute_desc_table(
        df=returns,
        desc_table_type=desc_table_type,
    )

    assert tuple(actual.columns) == expected_columns
    pd.testing.assert_series_equal(
        actual[_POSITIVE_COLUMN],
        _expected_positive_probabilities(),
    )
    pd.testing.assert_frame_equal(returns, original_returns)


# =============================================================================
# Series/DataFrame consistency and custom naming
# =============================================================================

def test_compute_desc_table_positive_probability_preserves_named_series_contract() -> None:
    """Return the same labeled result for equivalent Series and DataFrame inputs.

    The named Series contains one positive and one negative observed return plus two missing
    dates, giving an independently calculated probability of 50%. Converting that Series to a
    one-column DataFrame must not change the display table, and neither input may be modified.
    """
    returns = pd.Series(
        (np.nan, 0.02, -0.01, np.nan),
        index=_DATES,
        name='Custom Asset',
    )
    returns_frame = returns.to_frame()
    original_returns = returns.copy(deep=True)
    original_frame = returns_frame.copy(deep=True)

    series_result = _DESC_TABLE_MODULE.compute_desc_table(
        df=returns,
        desc_table_type=DescTableType.WITH_POSITIVE_PROB,
    )
    frame_result = _DESC_TABLE_MODULE.compute_desc_table(
        df=returns_frame,
        desc_table_type=DescTableType.WITH_POSITIVE_PROB,
    )

    assert series_result.index.tolist() == ['Custom Asset']
    assert series_result.at['Custom Asset', _POSITIVE_COLUMN] == '50.0%'
    pd.testing.assert_frame_equal(series_result, frame_result)
    pd.testing.assert_series_equal(returns, original_returns)
    pd.testing.assert_frame_equal(returns_frame, original_frame)

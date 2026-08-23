"""Regression tests for annualized reduced descriptive-table modes.

``compute_desc_table`` initially creates an average and either a periodic ``Std`` column or an
annualized ``Std An`` column. ``AVG_WITH_POSITIVE_PROB`` and ``SKEW_KURTOSIS`` then remove those
setup columns before reporting their reduced schemas. The selected volatility convention must
therefore change which label is removed, not whether these otherwise valid modes can be used.

The deterministic monthly fixture has directly countable signs and symmetric arithmetic
progressions. Its positive shares, skewness, and excess kurtosis are calculated independently in
the expected tables below. The tests cover annualized regressions, unchanged periodic controls,
Series/DataFrame consistency, display formatting, labels, warnings, and caller ownership.
"""

import warnings
from typing import Protocol, cast

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
            desc_table_type: DescTableType,
            annualize_vol: bool,
            norm_variable_display_type: str = '{:.1f}') -> pd.DataFrame:
        """Return a formatted descriptive-statistics table."""
        raise NotImplementedError


_DESC_TABLE_MODULE = cast(_DescTableModuleProtocol, desc_table_module)


# =============================================================================
# Shared deterministic fixtures
# =============================================================================

_DATES = pd.date_range('2024-01-31', periods=5, freq='ME')

_KURTOSIS_COLUMN = 'Kurt'
_NORM_FORMAT = '{:.3f}'
_POSITIVE_ASSET = 'Positive Asset'
_POSITIVE_COLUMN = 'Positive'
_SKEW_COLUMN = 'Skew'
_SYMMETRIC_ASSET = 'Symmetric Asset'


def _monthly_returns() -> pd.DataFrame:
    """Create two five-observation arithmetic progressions.

    ``Symmetric Asset`` has two positive observations among five, while ``Positive Asset`` has
    five. Both columns are symmetric around their means, so their skewness is zero. Their second
    and fourth central moments give excess kurtosis ``1.7 - 3.0 = -1.3``.

    Returns:
        Monthly return panel in the expected display order.
    """
    return pd.DataFrame(
        {
            _SYMMETRIC_ASSET: (-0.50, -0.25, 0.00, 0.25, 0.50),
            _POSITIVE_ASSET: (0.25, 0.50, 0.75, 1.00, 1.25),
        },
        index=_DATES,
    )


def _expected_positive_table() -> pd.DataFrame:
    """Return the direct positive-count ratios as display strings.

    Returns:
        One-column table containing ``2 / 5`` and ``5 / 5``.
    """
    return pd.DataFrame(
        {_POSITIVE_COLUMN: ('40.0%', '100.0%')},
        index=[_SYMMETRIC_ASSET, _POSITIVE_ASSET],
    )


def _expected_skew_kurtosis_table() -> pd.DataFrame:
    """Return the independently calculated centered-moment statistics.

    Returns:
        Two-column table with zero skewness and ``-1.3`` excess kurtosis.
    """
    return pd.DataFrame(
        {
            _SKEW_COLUMN: ('0.000', '0.000'),
            _KURTOSIS_COLUMN: ('-1.300', '-1.300'),
        },
        index=[_SYMMETRIC_ASSET, _POSITIVE_ASSET],
    )


def _compute_without_warnings(
        data: pd.DataFrame | pd.Series,
        desc_table_type: DescTableType,
        annualize_vol: bool) -> pd.DataFrame:
    """Call the public function while treating every emitted warning as a failure.

    Args:
        data: Finite monthly Series or DataFrame supplied to the public function.
        desc_table_type: Reduced descriptive-table mode under test.
        annualize_vol: Whether the setup volatility uses the annualized label.

    Returns:
        Formatted descriptive-statistics table.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        return _DESC_TABLE_MODULE.compute_desc_table(
            df=data,
            desc_table_type=desc_table_type,
            annualize_vol=annualize_vol,
            norm_variable_display_type=_NORM_FORMAT,
        )


# =============================================================================
# Annualized reduced-mode regressions
# =============================================================================

def test_compute_desc_table_annualized_positive_probability_omits_volatility() -> None:
    """Drop ``Std An`` and return only independently counted positive shares.

    Five monthly dates imply annualization factor 12, so the setup schema contains ``Std An``
    rather than ``Std``. The reduced output must remove that selected label and retain only the
    direct ``2 / 5`` and ``5 / 5`` positive probabilities without mutating the input frame.
    """
    returns = _monthly_returns()
    original_returns = returns.copy(deep=True)

    actual = _compute_without_warnings(
        data=returns,
        desc_table_type=DescTableType.AVG_WITH_POSITIVE_PROB,
        annualize_vol=True,
    )

    pd.testing.assert_frame_equal(actual, _expected_positive_table())
    pd.testing.assert_frame_equal(returns, original_returns)


def test_compute_desc_table_annualized_skew_kurtosis_omits_volatility() -> None:
    """Drop ``Std An`` and return the independently calculated centered moments.

    Both arithmetic progressions have zero third central moment. Their fourth standardized
    moment is 1.7, giving Fisher excess kurtosis ``-1.3``. The custom three-decimal format, asset
    order, and statistic order must survive annualization without modifying the input frame.
    """
    returns = _monthly_returns()
    original_returns = returns.copy(deep=True)

    actual = _compute_without_warnings(
        data=returns,
        desc_table_type=DescTableType.SKEW_KURTOSIS,
        annualize_vol=True,
    )

    pd.testing.assert_frame_equal(actual, _expected_skew_kurtosis_table())
    pd.testing.assert_frame_equal(returns, original_returns)


# =============================================================================
# Existing periodic-mode controls
# =============================================================================

@pytest.mark.parametrize(
    ('desc_table_type', 'expected'),
    (
        (DescTableType.AVG_WITH_POSITIVE_PROB, _expected_positive_table()),
        (DescTableType.SKEW_KURTOSIS, _expected_skew_kurtosis_table()),
    ),
)
def test_compute_desc_table_periodic_reduced_modes_remain_unchanged(
        desc_table_type: DescTableType,
        expected: pd.DataFrame) -> None:
    """Preserve the established reduced tables when volatility is not annualized.

    Args:
        desc_table_type: Reduced descriptive-table mode under test.
        expected: Independently calculated complete display table for that mode.
    """
    returns = _monthly_returns()
    original_returns = returns.copy(deep=True)

    actual = _compute_without_warnings(
        data=returns,
        desc_table_type=desc_table_type,
        annualize_vol=False,
    )

    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(returns, original_returns)


# =============================================================================
# Series/DataFrame consistency
# =============================================================================

@pytest.mark.parametrize(
    ('desc_table_type', 'expected'),
    (
        (DescTableType.AVG_WITH_POSITIVE_PROB, _expected_positive_table()),
        (DescTableType.SKEW_KURTOSIS, _expected_skew_kurtosis_table()),
    ),
)
def test_compute_desc_table_annualized_reduced_modes_preserve_named_series_contract(
        desc_table_type: DescTableType,
        expected: pd.DataFrame) -> None:
    """Return the same one-row annualized table for equivalent pandas shapes.

    Args:
        desc_table_type: Reduced descriptive-table mode under test.
        expected: Independently calculated complete display table for that mode.
    """
    returns = _monthly_returns()[_SYMMETRIC_ASSET]
    returns_frame = returns.to_frame()
    original_returns = returns.copy(deep=True)
    original_frame = returns_frame.copy(deep=True)

    series_result = _compute_without_warnings(
        data=returns,
        desc_table_type=desc_table_type,
        annualize_vol=True,
    )
    frame_result = _compute_without_warnings(
        data=returns_frame,
        desc_table_type=desc_table_type,
        annualize_vol=True,
    )

    expected_series_table = expected.loc[[_SYMMETRIC_ASSET]]
    pd.testing.assert_frame_equal(series_result, expected_series_table)
    pd.testing.assert_frame_equal(frame_result, expected_series_table)
    pd.testing.assert_frame_equal(series_result, frame_result)
    pd.testing.assert_series_equal(returns, original_returns)
    pd.testing.assert_frame_equal(returns_frame, original_frame)

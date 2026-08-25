"""Regression coverage for zero-volatility descriptive t-statistic boundaries.

``compute_desc_table`` conditionally adds a display statistic based on adjusted mean and
volatility. A constant history has no usable volatility denominator, so its statistic should use
the established ``nan`` display without emitting a divide-by-zero warning. The deterministic
panel below covers positive, zero, negative, and ragged constants alongside a finite positive
control in periodic and monthly annualized modes. Named Series/DataFrame consistency and caller
ownership are checked separately.
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
            desc_table_type: DescTableType,
            annualize_vol: bool,
            is_add_tstat: bool,
            norm_variable_display_type: str) -> pd.DataFrame:
        """Return a formatted descriptive-statistics table."""
        raise NotImplementedError


_DESC_TABLE_MODULE = cast(_DescTableModuleProtocol, desc_table_module)


# =============================================================================
# Deterministic fixtures and independent expectations
# =============================================================================

_ASSETS = pd.Index(
    [
        'Positive Constant',
        'Zero Constant',
        'Negative Constant',
        'Positive Variable',
        'Ragged Positive Constant',
    ],
    name='Asset',
)
_DATES = pd.date_range('2024-01-31', periods=4, freq='ME')


def _tstat_boundary_returns() -> pd.DataFrame:
    """Create constant boundaries and one finite positive-volatility control.

    Returns:
        Four-row monthly panel in the expected reporting order.
    """
    return pd.DataFrame(
        {
            'Positive Constant': [2.0, 2.0, 2.0, 2.0],
            'Zero Constant': [0.0, 0.0, 0.0, 0.0],
            'Negative Constant': [-2.0, -2.0, -2.0, -2.0],
            'Positive Variable': [1.0, 2.0, 3.0, 4.0],
            'Ragged Positive Constant': [2.0, np.nan, 2.0, 2.0],
        },
        index=_DATES,
    ).set_axis(_ASSETS, axis='columns')


def _expected_table(
        volatility_label: str,
        variable_volatility: str,
        variable_tstat: str) -> pd.DataFrame:
    """Build the exact display table from independently calculated strings.

    Args:
        volatility_label: Periodic or annualized output-column label.
        variable_volatility: Displayed volatility for the finite control.
        variable_tstat: Displayed statistic for the finite control.

    Returns:
        Complete expected table for all constant boundaries and the finite control.
    """
    return pd.DataFrame(
        {
            'Avg': ['2.00', '0.00', '-2.00', '2.50', '2.00'],
            volatility_label: ['0.00', '0.00', '0.00', variable_volatility, '0.00'],
            'T-stat': ['nan', 'nan', 'nan', variable_tstat, 'nan'],
        },
        index=_ASSETS,
    )


def _compute_without_warnings(
        data: pd.DataFrame | pd.Series,
        annualize_vol: bool) -> pd.DataFrame:
    """Compute a descriptive table while treating every warning as a failure.

    Args:
        data: Monthly return Series or DataFrame supplied to the public function.
        annualize_vol: Whether to annualize the mean and volatility ratio.

    Returns:
        Formatted descriptive table including the optional statistic.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        return _DESC_TABLE_MODULE.compute_desc_table(
            df=data,
            desc_table_type=DescTableType.SHORT,
            annualize_vol=annualize_vol,
            is_add_tstat=True,
            norm_variable_display_type='{:.3f}',
        )


# =============================================================================
# Zero-volatility and finite-neighbor behavior
# =============================================================================

@pytest.mark.parametrize(
    ('annualize_vol', 'volatility_label', 'variable_volatility', 'variable_tstat'),
    [
        (False, 'Std', '1.29', '1.936'),
        (True, 'Std An', '4.47', '6.708'),
    ],
)
@pytest.mark.parametrize(
    'use_nullable_dtype',
    [False, True],
    ids=['float64', 'nullable-float64'],
)
def test_compute_desc_table_returns_undefined_tstat_for_zero_volatility(
        annualize_vol: bool,
        volatility_label: str,
        variable_volatility: str,
        variable_tstat: str,
        use_nullable_dtype: bool) -> None:
    """Return undefined constant statistics while preserving the finite ratio exactly.

    For the control ``[1, 2, 3, 4]``, the mean is 2.5 and sample volatility is
    ``sqrt(5 / 3)``. The periodic ratio is therefore 1.936491.... Monthly annualization gives
    volatility ``sqrt(12) * sqrt(5 / 3) = sqrt(20)`` and adjusted mean 30, producing
    ``30 / sqrt(20) = 6.708204...``. Every constant has zero volatility and an undefined
    displayed statistic, including the ragged positive constant.

    Args:
        annualize_vol: Whether to test the periodic or monthly annualized calculation.
        volatility_label: Expected output label for the selected volatility convention.
        variable_volatility: Expected formatted finite-control volatility.
        variable_tstat: Expected formatted finite-control statistic.
        use_nullable_dtype: Whether to exercise pandas ``Float64`` and ``pd.NA`` inputs.
    """
    returns = _tstat_boundary_returns()
    if use_nullable_dtype:
        returns = returns.astype(pd.Float64Dtype())
    original_returns = returns.copy(deep=True)

    actual = _compute_without_warnings(returns, annualize_vol)

    expected = _expected_table(volatility_label, variable_volatility, variable_tstat)
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(returns, original_returns)


# =============================================================================
# Named Series/DataFrame consistency
# =============================================================================

@pytest.mark.parametrize(
    ('annualize_vol', 'volatility_label'),
    [
        (False, 'Std'),
        (True, 'Std An'),
    ],
)
def test_compute_desc_table_constant_tstat_named_series_matches_dataframe(
        annualize_vol: bool,
        volatility_label: str) -> None:
    """Return the same named positive-constant row for equivalent pandas input shapes.

    Args:
        annualize_vol: Whether to test the periodic or monthly annualized calculation.
        volatility_label: Expected output label for the selected volatility convention.
    """
    returns = _tstat_boundary_returns()['Positive Constant']
    returns_frame = returns.to_frame()
    original_returns = returns.copy(deep=True)
    original_frame = returns_frame.copy(deep=True)

    series_result = _compute_without_warnings(returns, annualize_vol)
    frame_result = _compute_without_warnings(returns_frame, annualize_vol)

    expected = pd.DataFrame(
        {
            'Avg': ['2.00'],
            volatility_label: ['0.00'],
            'T-stat': ['nan'],
        },
        index=['Positive Constant'],
    )
    pd.testing.assert_frame_equal(series_result, expected)
    pd.testing.assert_frame_equal(frame_result, expected)
    pd.testing.assert_frame_equal(series_result, frame_result)
    pd.testing.assert_series_equal(returns, original_returns)
    pd.testing.assert_frame_equal(returns_frame, original_frame)

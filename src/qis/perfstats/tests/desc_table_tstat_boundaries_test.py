"""Regression coverage for descriptive-table t-statistic conventions and boundaries.

``compute_desc_table`` labels its optional statistic as a t-statistic of the sample mean. It must
therefore divide the mean by its standard error, retain the sign, and depend on the non-missing
sample count rather than the displayed volatility's annualization. A constant or undersized
history has no usable standard error and should use the established ``nan`` display without a
warning. The mixed panel below exercises all of those states simultaneously under periodic and
monthly annualized display conventions. Named Series/DataFrame consistency and caller ownership
are checked separately.
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
        'Negative Variable',
        'Zero Mean Variable',
        'Ragged Positive Variable',
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
            'Negative Variable': [-1.0, -2.0, -3.0, -4.0],
            'Zero Mean Variable': [-1.0, 1.0, -1.0, 1.0],
            'Ragged Positive Variable': [1.0, np.nan, 2.0, 3.0],
            'Ragged Positive Constant': [2.0, np.nan, 2.0, 2.0],
        },
        index=_DATES,
    ).set_axis(_ASSETS, axis='columns')


def _expected_table(
        volatility_label: str,
        variable_volatilities: tuple[str, str, str, str]) -> pd.DataFrame:
    """Build the exact display table from independently calculated strings.

    Args:
        volatility_label: Periodic or annualized output-column label.
        variable_volatilities: Displayed volatility for the positive, negative, zero-mean, and
            ragged variable histories.

    Returns:
        Complete expected table for all constant boundaries and the finite control.
    """
    return pd.DataFrame(
        {
            'Avg': ['2.00', '0.00', '-2.00', '2.50', '-2.50', '0.00', '2.00', '2.00'],
            volatility_label: [
                '0.00',
                '0.00',
                '0.00',
                *variable_volatilities,
                '0.00',
            ],
            'T-stat': [
                'nan',
                'nan',
                'nan',
                '3.873',
                '-3.873',
                '0.000',
                '3.464',
                'nan',
            ],
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
    ('annualize_vol', 'volatility_label', 'variable_volatilities'),
    [
        (False, 'Std', ('1.29', '1.29', '1.15', '1.00')),
        (True, 'Std An', ('4.47', '4.47', '4.00', '3.46')),
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
        variable_volatilities: tuple[str, str, str, str],
        use_nullable_dtype: bool) -> None:
    """Apply the signed sample-mean statistic independently to every column state.

    For the control ``[1, 2, 3, 4]``, the mean is 2.5 and sample volatility is
    ``sqrt(5 / 3)``. Its standard error is ``sqrt(5 / 3) / sqrt(4)``, so its t-statistic is
    ``sqrt(15) = 3.872983...``; the negative mirror keeps the opposite sign. The zero-mean
    variable has statistic zero, and the three-point variable has statistic
    ``2 * sqrt(3) = 3.464101...``. Annualization changes only the displayed volatility. Every
    constant has zero volatility and an undefined statistic, including the ragged constant.

    Args:
        annualize_vol: Whether to test the periodic or monthly annualized calculation.
        volatility_label: Expected output label for the selected volatility convention.
        variable_volatilities: Expected formatted variable-column volatilities.
        use_nullable_dtype: Whether to exercise pandas ``Float64`` and ``pd.NA`` inputs.
    """
    returns = _tstat_boundary_returns()
    if use_nullable_dtype:
        returns = returns.astype(pd.Float64Dtype())
    original_returns = returns.copy(deep=True)

    actual = _compute_without_warnings(returns, annualize_vol)

    expected = _expected_table(volatility_label, variable_volatilities)
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

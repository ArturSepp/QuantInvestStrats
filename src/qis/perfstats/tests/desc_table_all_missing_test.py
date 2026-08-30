"""Regression coverage for all-missing descriptive-statistics columns.

``compute_desc_table`` reports display-ready statistics for every input column. An asset with
dates but no observed values still belongs in that table: each undefined statistic should use the
established formatted missing representation without emitting reduction warnings or preventing
other assets from being reported.

The base eight-date panel pairs an all-missing asset with a symmetric finite control. Normality
cases use a separate symmetric 20-date panel so every supported SciPy version remains above its
small-sample warning boundary. The finite expectations below come from direct counts and centered
moments; the normality p-value is an unchanged accepted-output control. Tests cover every
implemented table mode, exact schemas and strings, named Series/DataFrame consistency, formatting,
optional t-statistics, nullable ``Float64`` compatibility, and caller ownership.
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
            var_format: str = '{:.2f}',
            annualize_vol: bool = False,
            is_add_tstat: bool = False,
            norm_variable_display_type: str = '{:.1f}') -> pd.DataFrame:
        """Return a formatted descriptive-statistics table."""
        raise NotImplementedError


_DESC_TABLE_MODULE = cast(_DescTableModuleProtocol, desc_table_module)


# =============================================================================
# Shared deterministic fixtures and independent expectations
# =============================================================================

_DATES = pd.date_range('2024-01-31', periods=8, freq='ME')

_ALL_MISSING_ASSET = 'All Missing Asset'
_FINITE_ASSET = 'Finite Asset'
_FINITE_VALUES = (-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0)
_NORMALITY_FINITE_VALUES = tuple(float(value) for value in range(-10, 0)) + tuple(
    float(value) for value in range(1, 11))
_SECOND_FINITE_ASSET = 'Second Finite Asset'

_IMPLEMENTED_MODES = tuple(
    mode for mode in DescTableType if mode is not DescTableType.NONE)
_REDUCED_MODES = (
    DescTableType.AVG_WITH_POSITIVE_PROB,
    DescTableType.SKEW_KURTOSIS,
)

_EXPECTED_COLUMNS: dict[DescTableType, tuple[tuple[str, str, str], ...]] = {
    DescTableType.SHORT: (
        ('Avg', '0.00', 'nan'),
        ('Std', '2.93', 'nan'),
    ),
    DescTableType.AVG_WITH_POSITIVE_PROB: (
        ('Positive', '50.0%', 'nan%'),
    ),
    DescTableType.WITH_POSITIVE_PROB: (
        ('Avg', '0.00', 'nan'),
        ('Std', '2.93', 'nan'),
        ('Positive', '50.0%', 'nan%'),
    ),
    DescTableType.WITH_KURTOSIS: (
        ('Avg', '0.00', 'nan'),
        ('Std', '2.93', 'nan'),
        ('Skew', '0.0', 'nan'),
        ('Kurt', '-1.4', 'nan'),
    ),
    DescTableType.WITH_NORMAL_PVAL: (
        ('Avg', '0.00', 'nan'),
        ('Std', '6.37', 'nan'),
        ('Skew', '0.0', 'nan'),
        ('Kurt', '-1.3', 'nan'),
        ('P-val', '0.08', 'nan'),
    ),
    DescTableType.WITH_SCORE: (
        ('Avg', '0.00', 'nan'),
        ('Std', '2.93', 'nan'),
        ('Last', '4.00', 'nan'),
        ('Rank', '100%', 'nan%'),
    ),
    DescTableType.EXTENSIVE: (
        ('Avg', '0.00', 'nan'),
        ('Std', '2.93', 'nan'),
        ('Skew', '0.0', 'nan'),
        ('Kurt', '-1.4', 'nan'),
        ('Min', '-4.00', 'nan'),
        ('-1std', '-2.88', 'nan'),
        ('Median', '0.00', 'nan'),
        ('+1std', '2.88', 'nan'),
        ('Max', '4.00', 'nan'),
    ),
    DescTableType.SKEW_KURTOSIS: (
        ('Skew', '0.0', 'nan'),
        ('Kurt', '-1.4', 'nan'),
    ),
    DescTableType.WITH_MEDIAN: (
        ('Avg', '0.00', 'nan'),
        ('Std', '2.93', 'nan'),
        ('Median', '0.00', 'nan'),
        ('Skew', '0.0', 'nan'),
        ('Kurt', '-1.4', 'nan'),
    ),
}


def _finite_values(desc_table_type: DescTableType) -> tuple[float, ...]:
    """Select a finite fixture that is warning-free for the requested mode.

    The 20 normality observations have squared deviations totaling 770, so their sample standard
    deviation is ``sqrt(770 / 19)``. Their second and fourth central moments are 38.5 and 2533.3,
    giving Fisher excess kurtosis ``2533.3 / 38.5**2 - 3 = -1.290909...``.

    Args:
        desc_table_type: Descriptive-table mode under test.

    Returns:
        Symmetric finite observations used by that mode.
    """
    if desc_table_type is DescTableType.WITH_NORMAL_PVAL:
        return _NORMALITY_FINITE_VALUES
    return _FINITE_VALUES


def _mixed_returns(desc_table_type: DescTableType) -> pd.DataFrame:
    """Create one finite and one all-missing asset over the same dates.

    The base finite values sum to zero, have squared deviations totaling 60, and contain four
    positives among eight observations. Their sample standard deviation is therefore
    ``sqrt(60 / 7)``, while symmetry makes skewness zero. Their second and fourth central
    moments are 7.5 and 88.5, giving excess kurtosis ``88.5 / 7.5**2 - 3``. The normality mode
    substitutes the 20-point fixture documented by ``_finite_values``.

    Args:
        desc_table_type: Descriptive-table mode that determines the finite fixture length.

    Returns:
        Return panel in the expected reporting order and mode-specific sample length.
    """
    finite_values = _finite_values(desc_table_type)
    dates = pd.date_range('2024-01-31', periods=len(finite_values), freq='ME')
    return pd.DataFrame(
        {
            _FINITE_ASSET: finite_values,
            _ALL_MISSING_ASSET: (np.nan,) * len(dates),
        },
        index=dates,
    )


def _nullable_returns(
        desc_table_type: DescTableType,
        *,
        include_all_missing: bool) -> pd.DataFrame:
    """Create a multi-column nullable panel with optional all-missing history.

    Two independently stored finite columns exercise the extension-array conversion across a
    genuine two-dimensional panel. The optional third column adds ``pd.NA`` observations so the
    same fixture covers both finite-only and mixed finite/all-missing nullable boundaries.

    Args:
        desc_table_type: Descriptive-table mode that determines the finite fixture length.
        include_all_missing: Whether to append the all-missing nullable column.

    Returns:
        Return panel whose columns use pandas nullable ``Float64`` and the mode-specific length.
    """
    finite_values = _finite_values(desc_table_type)
    dates = pd.date_range('2024-01-31', periods=len(finite_values), freq='ME')
    values: dict[str, tuple[float, ...]] = {
        _FINITE_ASSET: finite_values,
        _SECOND_FINITE_ASSET: finite_values,
    }
    if include_all_missing:
        values[_ALL_MISSING_ASSET] = (np.nan,) * len(dates)
    return pd.DataFrame(values, index=dates).astype('Float64')


def _expected_table(desc_table_type: DescTableType) -> pd.DataFrame:
    """Build the exact display table from independently specified strings.

    Args:
        desc_table_type: Implemented descriptive-table mode under test.

    Returns:
        Complete expected table for the shared finite and all-missing assets.
    """
    return pd.DataFrame(
        {
            column: (finite_value, missing_value)
            for column, finite_value, missing_value in _EXPECTED_COLUMNS[desc_table_type]
        },
        index=[_FINITE_ASSET, _ALL_MISSING_ASSET],
    )


def _expected_nullable_table(
        desc_table_type: DescTableType,
        *,
        include_all_missing: bool) -> pd.DataFrame:
    """Build exact expected strings for the multi-column nullable fixture.

    Args:
        desc_table_type: Implemented descriptive-table mode under test.
        include_all_missing: Whether the expected table includes the missing asset.

    Returns:
        Complete expected table for two finite assets and the optional missing asset.
    """
    index = [_FINITE_ASSET, _SECOND_FINITE_ASSET]
    if include_all_missing:
        index.append(_ALL_MISSING_ASSET)
    return pd.DataFrame(
        {
            column: (finite_value, finite_value, missing_value)
            if include_all_missing else (finite_value, finite_value)
            for column, finite_value, missing_value in _EXPECTED_COLUMNS[desc_table_type]
        },
        index=index,
    )


def _compute_without_warnings(
        data: pd.DataFrame | pd.Series,
        desc_table_type: DescTableType,
        *,
        var_format: str = '{:.2f}',
        annualize_vol: bool = False,
        is_add_tstat: bool = False,
        norm_variable_display_type: str = '{:.1f}') -> pd.DataFrame:
    """Call the public function while treating every emitted warning as a failure.

    Args:
        data: Finite/all-missing Series or DataFrame supplied to the public function.
        desc_table_type: Descriptive-table mode under test.
        var_format: Display format for mean, volatility, extrema, quantiles, and last value.
        annualize_vol: Whether volatility uses the annualized output convention.
        is_add_tstat: Whether to include the optional t-statistic column.
        norm_variable_display_type: Display format for moments and t-statistics.

    Returns:
        Formatted descriptive-statistics table.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        return _DESC_TABLE_MODULE.compute_desc_table(
            df=data,
            desc_table_type=desc_table_type,
            var_format=var_format,
            annualize_vol=annualize_vol,
            is_add_tstat=is_add_tstat,
            norm_variable_display_type=norm_variable_display_type,
        )


# =============================================================================
# Warning-free all-missing behavior across every table mode
# =============================================================================

@pytest.mark.parametrize('desc_table_type', _IMPLEMENTED_MODES)
def test_compute_desc_table_all_missing_column_returns_undefined_statistics(
        desc_table_type: DescTableType) -> None:
    """Return each complete mode-specific table without warnings or input mutation.

    The all-missing asset has no population from which any statistic can be calculated, so every
    reported value is the established ``nan`` string (or ``nan%`` for percentage displays). The
    finite neighbor proves exact schemas, labels, order, values, and formatting remain unchanged.

    Args:
        desc_table_type: Implemented descriptive-table mode under test.
    """
    returns = _mixed_returns(desc_table_type)
    original_returns = returns.copy(deep=True)

    actual = _compute_without_warnings(returns, desc_table_type)

    pd.testing.assert_frame_equal(actual, _expected_table(desc_table_type))
    pd.testing.assert_frame_equal(returns, original_returns)


# =============================================================================
# Nullable Float64 compatibility across finite and all-missing panels
# =============================================================================

@pytest.mark.parametrize('desc_table_type', _IMPLEMENTED_MODES)
@pytest.mark.parametrize('include_all_missing', (False, True), ids=('finite', 'mixed'))
def test_compute_desc_table_nullable_float64_returns_exact_statistics(
        desc_table_type: DescTableType,
        include_all_missing: bool) -> None:
    """Support nullable finite-only and mixed panels without warnings or mutation.

    The two finite columns deliberately repeat the independently specified distribution so every
    expected display string remains directly traceable to the counts and centered moments in the
    module docstring. The mixed case simultaneously adds the materially different all-``pd.NA``
    state and proves it remains undefined across every implemented table mode.

    Args:
        desc_table_type: Implemented descriptive-table mode under test.
        include_all_missing: Whether the nullable panel contains the all-missing asset.
    """
    returns = _nullable_returns(
        desc_table_type, include_all_missing=include_all_missing)
    original_returns = returns.copy(deep=True)
    assert returns.dtypes.astype(str).tolist() == ['Float64'] * len(returns.columns)

    actual = _compute_without_warnings(returns, desc_table_type)

    expected = _expected_nullable_table(
        desc_table_type, include_all_missing=include_all_missing)
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(returns, original_returns)


@pytest.mark.parametrize('desc_table_type', _REDUCED_MODES)
def test_compute_desc_table_nullable_float64_supports_annualized_reduced_modes(
        desc_table_type: DescTableType) -> None:
    """Preserve reduced schemas when nullable volatility uses the annualized convention.

    Both reduced modes discard volatility from their final schemas, so their exact expected
    displays are convention-independent. Exercising them with ``annualize_vol=True`` nevertheless
    proves the nullable conversion composes with the selected ``Std An`` setup column introduced
    by the accepted annualized-mode correction.

    Args:
        desc_table_type: Reduced descriptive-table mode under test.
    """
    returns = _nullable_returns(desc_table_type, include_all_missing=True)
    original_returns = returns.copy(deep=True)

    actual = _compute_without_warnings(
        returns, desc_table_type, annualize_vol=True)

    expected = _expected_nullable_table(
        desc_table_type, include_all_missing=True)
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(returns, original_returns)


# =============================================================================
# Optional formatting and t-statistic path
# =============================================================================

def test_compute_desc_table_all_missing_column_preserves_custom_formats_and_tstat() -> None:
    """Format defined neighboring statistics while retaining undefined missing statistics.

    The finite control ``1, ..., 8`` has mean 4.5, sample standard deviation ``sqrt(6)``, and
    standard error ``sqrt(6) / sqrt(8)``. Its t-statistic is therefore
    ``4.5 * sqrt(8) / sqrt(6) = 5.196152...``. The all-missing asset has none of those statistics,
    and custom formatting must not turn its missing representations into values or warnings.
    """
    returns = pd.DataFrame(
        {
            _FINITE_ASSET: tuple(float(value) for value in range(1, 9)),
            _ALL_MISSING_ASSET: (np.nan,) * len(_DATES),
        },
        index=_DATES,
    )
    original_returns = returns.copy(deep=True)
    expected = pd.DataFrame(
        {
            'Avg': ('4.500', 'nan'),
            'Std': ('2.449', 'nan'),
            'T-stat': ('5.196', 'nan'),
        },
        index=[_FINITE_ASSET, _ALL_MISSING_ASSET],
    )

    actual = _compute_without_warnings(
        returns,
        DescTableType.SHORT,
        var_format='{:.3f}',
        is_add_tstat=True,
        norm_variable_display_type='{:.3f}',
    )

    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(returns, original_returns)


# =============================================================================
# Named Series/DataFrame consistency
# =============================================================================

@pytest.mark.parametrize('desc_table_type', _IMPLEMENTED_MODES)
def test_compute_desc_table_all_missing_named_series_matches_dataframe(
        desc_table_type: DescTableType) -> None:
    """Return the same one-row missing table for equivalent pandas input shapes.

    Args:
        desc_table_type: Implemented descriptive-table mode under test.
    """
    returns = _mixed_returns(desc_table_type)[_ALL_MISSING_ASSET]
    returns_frame = returns.to_frame()
    original_returns = returns.copy(deep=True)
    original_frame = returns_frame.copy(deep=True)

    series_result = _compute_without_warnings(returns, desc_table_type)
    frame_result = _compute_without_warnings(returns_frame, desc_table_type)

    expected = _expected_table(desc_table_type).loc[[_ALL_MISSING_ASSET]]
    pd.testing.assert_frame_equal(series_result, expected)
    pd.testing.assert_frame_equal(frame_result, expected)
    pd.testing.assert_frame_equal(series_result, frame_result)
    pd.testing.assert_series_equal(returns, original_returns)
    pd.testing.assert_frame_equal(returns_frame, original_frame)

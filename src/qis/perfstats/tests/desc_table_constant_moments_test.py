"""Regression coverage for constant descriptive-table moments.

Finite constant histories have useful level statistics but no standardized central moments:
their means, zero sample volatility, positive probability, extrema, quantiles, median, last value,
and average tie rank are defined, while skewness, excess kurtosis, and a normality p-value are not.
The public table should therefore retain the defined strings and report the three moment results
as ``nan`` without invoking SciPy reducers that warn about catastrophic cancellation.

One 24-month mixed panel combines positive, zero, negative, and ragged constant histories with a
symmetric finite-variable neighbor. The expectations below come from literal counts and central
moments rather than another QIS calculation. Matched ordinary ``float64``/``np.nan`` and nullable
``Float64``/``pd.NA`` inputs also verify labels, column order, named Series/DataFrame consistency,
warning behavior, and caller ownership.
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
    ) -> pd.DataFrame:
        """Return a formatted descriptive-statistics table."""
        raise NotImplementedError


_DESC_TABLE_MODULE = cast(_DescTableModuleProtocol, desc_table_module)


# =============================================================================
# Shared deterministic fixtures and independent expectations
# =============================================================================

_DATES = pd.date_range("2024-01-31", periods=24, freq="ME")

_POSITIVE_CONSTANT = "Positive Constant"
_ZERO_CONSTANT = "Zero Constant"
_NEGATIVE_CONSTANT = "Negative Constant"
_RAGGED_CONSTANT = "Ragged Constant"
_FINITE_VARIABLE = "Finite Variable"

_ASSETS = (
    _POSITIVE_CONSTANT,
    _ZERO_CONSTANT,
    _NEGATIVE_CONSTANT,
    _RAGGED_CONSTANT,
    _FINITE_VARIABLE,
)

_MOMENT_MODES = (
    DescTableType.WITH_KURTOSIS,
    DescTableType.WITH_NORMAL_PVAL,
    DescTableType.SKEW_KURTOSIS,
    DescTableType.EXTENSIVE,
    DescTableType.WITH_MEDIAN,
)

_DEFINED_CONTROL_MODES = (
    DescTableType.WITH_POSITIVE_PROB,
    DescTableType.WITH_SCORE,
)

_EXPECTED_VALUES: dict[str, tuple[str, ...]] = {
    "Avg": ("2.00", "0.00", "-3.00", "4.00", "0.00"),
    "Std": ("0.00", "0.00", "0.00", "0.00", "7.07"),
    "Positive": ("100.0%", "0.0%", "0.0%", "100.0%", "50.0%"),
    "Skew": ("nan", "nan", "nan", "nan", "0.0"),
    "Kurt": ("nan", "nan", "nan", "nan", "-1.2"),
    "P-val": ("nan", "nan", "nan", "nan", "0.08"),
    "Last": ("2.00", "0.00", "-3.00", "4.00", "11.50"),
    "Rank": ("52%", "52%", "52%", "52%", "100%"),
    "Min": ("2.00", "0.00", "-3.00", "4.00", "-11.50"),
    "-1std": ("2.00", "0.00", "-3.00", "4.00", "-7.82"),
    "Median": ("2.00", "0.00", "-3.00", "4.00", "0.00"),
    "+1std": ("2.00", "0.00", "-3.00", "4.00", "7.82"),
    "Max": ("2.00", "0.00", "-3.00", "4.00", "11.50"),
}

_EXPECTED_MODE_COLUMNS: dict[DescTableType, tuple[str, ...]] = {
    DescTableType.WITH_POSITIVE_PROB: ("Avg", "Std", "Positive"),
    DescTableType.WITH_KURTOSIS: ("Avg", "Std", "Skew", "Kurt"),
    DescTableType.WITH_NORMAL_PVAL: ("Avg", "Std", "Skew", "Kurt", "P-val"),
    DescTableType.WITH_SCORE: ("Avg", "Std", "Last", "Rank"),
    DescTableType.EXTENSIVE: (
        "Avg",
        "Std",
        "Skew",
        "Kurt",
        "Min",
        "-1std",
        "Median",
        "+1std",
        "Max",
    ),
    DescTableType.SKEW_KURTOSIS: ("Skew", "Kurt"),
    DescTableType.WITH_MEDIAN: ("Avg", "Std", "Median", "Skew", "Kurt"),
}


def _mixed_returns(*, nullable: bool) -> pd.DataFrame:
    """Create the mixed constant and finite-variable panel.

    The finite neighbor is ``[-11.5, -10.5, ..., 11.5]``. Its mean is zero, its sample variance
    is 50, its biased skewness is zero, and its biased excess kurtosis is
    ``-1.204173913043478``. Linear interpolation at positions ``0.16 * 23`` and ``0.84 * 23``
    gives the displayed quantiles ``-7.82`` and ``7.82``.

    Args:
        nullable: Whether every column uses pandas nullable ``Float64`` storage.

    Returns:
        Twenty-four-month panel containing every zero-spread state and a finite neighbor.
    """
    returns = pd.DataFrame(
        {
            _POSITIVE_CONSTANT: (2.0,) * len(_DATES),
            _ZERO_CONSTANT: (0.0,) * len(_DATES),
            _NEGATIVE_CONSTANT: (-3.0,) * len(_DATES),
            _RAGGED_CONSTANT: (np.nan,) * 4 + (4.0,) * 20,
            _FINITE_VARIABLE: tuple(float(value) - 11.5 for value in range(len(_DATES))),
        },
        index=_DATES,
    )
    if nullable:
        return returns.astype(pd.Float64Dtype())
    return returns


def _expected_table(desc_table_type: DescTableType) -> pd.DataFrame:
    """Build the expected formatted table for one public mode.

    Args:
        desc_table_type: Descriptive-table mode under test.

    Returns:
        Expected schema, row order, and independently specified display strings.
    """
    return pd.DataFrame(
        {column: _EXPECTED_VALUES[column] for column in _EXPECTED_MODE_COLUMNS[desc_table_type]},
        index=pd.Index(_ASSETS),
    )


def _compute_without_warnings(
    data: pd.DataFrame | pd.Series,
    desc_table_type: DescTableType,
) -> pd.DataFrame:
    """Call the public function while treating every warning as a failure.

    Args:
        data: Series or DataFrame supplied to the public function.
        desc_table_type: Descriptive-table mode under test.

    Returns:
        Formatted descriptive-statistics table.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return _DESC_TABLE_MODULE.compute_desc_table(
            df=data,
            desc_table_type=desc_table_type,
        )


# =============================================================================
# Constant-moment regression and defined-statistic controls
# =============================================================================


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
@pytest.mark.parametrize("desc_table_type", _MOMENT_MODES, ids=lambda mode: mode.name.lower())
def test_compute_desc_table_constant_moments_remain_undefined_without_warnings(
    desc_table_type: DescTableType,
    nullable: bool,
) -> None:
    """Retain defined neighbors while zero-spread moments remain warning-free ``nan``.

    Args:
        desc_table_type: Moment-reporting table schema under test.
        nullable: Whether the panel uses nullable ``Float64``/``pd.NA`` storage.
    """
    returns = _mixed_returns(nullable=nullable)
    original = returns.copy()

    actual = _compute_without_warnings(returns, desc_table_type)

    pd.testing.assert_frame_equal(actual, _expected_table(desc_table_type))
    pd.testing.assert_frame_equal(returns, original)


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
@pytest.mark.parametrize(
    "desc_table_type", _DEFINED_CONTROL_MODES, ids=lambda mode: mode.name.lower()
)
def test_compute_desc_table_preserves_defined_constant_statistics(
    desc_table_type: DescTableType,
    nullable: bool,
) -> None:
    """Preserve constant positive probabilities, last values, and average tie ranks.

    A 24-value constant has average rank ``(1 + 24) / 2 = 12.5`` and therefore percentile
    ``12.5 / 24``, displayed as ``52%``. The ragged 20-value constant similarly displays ``52%``.

    Args:
        desc_table_type: Defined-statistic control schema under test.
        nullable: Whether the panel uses nullable ``Float64``/``pd.NA`` storage.
    """
    returns = _mixed_returns(nullable=nullable)
    original = returns.copy()

    actual = _compute_without_warnings(returns, desc_table_type)

    pd.testing.assert_frame_equal(actual, _expected_table(desc_table_type))
    pd.testing.assert_frame_equal(returns, original)


# =============================================================================
# Public input-shape consistency
# =============================================================================


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
def test_compute_desc_table_preserves_constant_series_dataframe_consistency(
    nullable: bool,
) -> None:
    """Return the same warning-free constant row for a named Series and one-column frame.

    Args:
        nullable: Whether both inputs use nullable ``Float64`` storage.
    """
    frame = _mixed_returns(nullable=nullable).filter(items=[_POSITIVE_CONSTANT])
    series = frame[_POSITIVE_CONSTANT]
    assert isinstance(series, pd.Series)
    original_frame = frame.copy()
    original_series = series.copy()

    frame_result = _compute_without_warnings(frame, DescTableType.WITH_NORMAL_PVAL)
    series_result = _compute_without_warnings(series, DescTableType.WITH_NORMAL_PVAL)
    expected = _expected_table(DescTableType.WITH_NORMAL_PVAL).filter(
        items=[_POSITIVE_CONSTANT], axis="index"
    )

    pd.testing.assert_frame_equal(frame_result, expected)
    pd.testing.assert_frame_equal(series_result, expected)
    pd.testing.assert_frame_equal(frame, original_frame)
    pd.testing.assert_series_equal(series, original_series)

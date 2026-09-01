"""Regression coverage for non-finite descriptive-table observations.

``compute_desc_table`` treats ordinary ``NaN`` and nullable ``pd.NA`` as missing observations,
but positive and negative infinity are observed values outside the domain of finite descriptive
statistics. Every public table mode should reject either infinity before reducers can emit
warnings or return internally inconsistent combinations of infinite and undefined statistics.

The primary 24-date panel combines a finite control, positive infinity, negative infinity, both
signs, and an all-missing neighbor in one call. Matched ordinary ``float64`` and nullable
``Float64`` inputs cover every implemented table mode, while named Series cases establish shape
parity. Separate valid controls use literal counts and central moments to prove that finite and
missing-only behavior, labels, formatting, warning handling, and caller ownership remain intact.
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
        var_format: str = "{:.2f}",
        annualize_vol: bool = False,
        is_add_tstat: bool = False,
        norm_variable_display_type: str = "{:.1f}",
    ) -> pd.DataFrame:
        """Return a formatted descriptive-statistics table."""
        raise NotImplementedError


_DESC_TABLE_MODULE = cast(_DescTableModuleProtocol, desc_table_module)


# =============================================================================
# Shared deterministic fixtures and independent expectations
# =============================================================================

_DATES = pd.date_range("2024-01-01", periods=24, freq="B")

_FINITE = "Finite Control"
_POSITIVE_INFINITY = "Positive Infinity"
_NEGATIVE_INFINITY = "Negative Infinity"
_BOTH_INFINITIES = "Both Infinities"
_ALL_MISSING = "All Missing"

_FINITE_VALUES = tuple(float(value) / 100.0 for value in range(-12, 12))
_IMPLEMENTED_MODES = tuple(mode for mode in DescTableType if mode is not DescTableType.NONE)

_EXPECTED_VALID_TABLES: dict[DescTableType, pd.DataFrame] = {
    DescTableType.WITH_POSITIVE_PROB: pd.DataFrame(
        {
            "Avg": ("-0.0050", "nan"),
            "Std": ("0.0707", "nan"),
            "Positive": ("45.8%", "nan%"),
        },
        index=(_FINITE, _ALL_MISSING),
    ),
    DescTableType.EXTENSIVE: pd.DataFrame(
        {
            "Avg": ("-0.0050", "nan"),
            "Std": ("0.0707", "nan"),
            "Skew": ("0.0000", "nan"),
            "Kurt": ("-1.2042", "nan"),
            "Min": ("-0.1200", "nan"),
            "-1std": ("-0.0832", "nan"),
            "Median": ("-0.0050", "nan"),
            "+1std": ("0.0732", "nan"),
            "Max": ("0.1100", "nan"),
        },
        index=(_FINITE, _ALL_MISSING),
    ),
}


def _mixed_infinite_returns(*, nullable: bool) -> pd.DataFrame:
    """Create one panel containing every materially distinct infinity state.

    Args:
        nullable: Whether every column uses pandas nullable ``Float64`` storage.

    Returns:
        Finite, missing, positive-infinity, negative-infinity, and both-sign columns.
    """
    positive_infinity = np.asarray(_FINITE_VALUES, dtype=float).copy()
    negative_infinity = np.asarray(_FINITE_VALUES, dtype=float).copy()
    both_infinities = np.asarray(_FINITE_VALUES, dtype=float).copy()
    positive_infinity[5] = np.inf
    negative_infinity[7] = -np.inf
    both_infinities[3] = -np.inf
    both_infinities[20] = np.inf

    returns = pd.DataFrame(
        {
            _FINITE: _FINITE_VALUES,
            _POSITIVE_INFINITY: positive_infinity,
            _NEGATIVE_INFINITY: negative_infinity,
            _BOTH_INFINITIES: both_infinities,
            _ALL_MISSING: (np.nan,) * len(_DATES),
        },
        index=_DATES,
    )
    if nullable:
        return returns.astype(pd.Float64Dtype())
    return returns


def _infinite_series(*, infinity: float | tuple[float, float], nullable: bool) -> pd.Series:
    """Create a named Series containing the requested infinity state.

    Args:
        infinity: One signed infinity or both negative and positive infinity.
        nullable: Whether the Series uses pandas nullable ``Float64`` storage.

    Returns:
        Named 24-date Series with finite neighbors around the selected infinity state.
    """
    values = np.asarray(_FINITE_VALUES, dtype=float).copy()
    if isinstance(infinity, tuple):
        values[3], values[20] = infinity
    else:
        values[5] = infinity
    dtype = pd.Float64Dtype() if nullable else float
    return pd.Series(values, index=_DATES, name="Asset", dtype=dtype)


def _valid_returns(*, nullable: bool) -> pd.DataFrame:
    """Create the finite and all-missing unchanged-behavior control.

    The finite sequence ``[-0.12, -0.11, ..., 0.11]`` has mean ``-0.005`` and sample variance
    ``0.005``. Its positive share is ``11 / 24``; symmetry gives zero biased skewness; its biased
    excess kurtosis is ``-1.204173913...``. Linear interpolation at positions ``3.68`` and
    ``19.32`` gives the 16% and 84% quantiles ``-0.0832`` and ``0.0732``.

    Args:
        nullable: Whether every column uses pandas nullable ``Float64`` storage.

    Returns:
        One finite and one all-missing column over the shared dates.
    """
    returns = pd.DataFrame(
        {
            _FINITE: _FINITE_VALUES,
            _ALL_MISSING: (np.nan,) * len(_DATES),
        },
        index=_DATES,
    )
    if nullable:
        return returns.astype(pd.Float64Dtype())
    return returns


def _compute_without_warnings(
    data: pd.DataFrame | pd.Series,
    desc_table_type: DescTableType,
    *,
    annualize_vol: bool = False,
    is_add_tstat: bool = False,
) -> pd.DataFrame:
    """Call the public function while treating every warning as a failure.

    Args:
        data: Series or DataFrame supplied to the public function.
        desc_table_type: Descriptive-table mode under test.
        annualize_vol: Whether to exercise the annualized-volatility entry path.
        is_add_tstat: Whether to exercise the optional t-statistic entry path.

    Returns:
        Formatted descriptive-statistics table.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return _DESC_TABLE_MODULE.compute_desc_table(
            df=data,
            desc_table_type=desc_table_type,
            var_format="{:.4f}",
            annualize_vol=annualize_vol,
            is_add_tstat=is_add_tstat,
            norm_variable_display_type="{:.4f}",
        )


# =============================================================================
# Infinity rejection and valid-input controls
# =============================================================================


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
@pytest.mark.parametrize(
    "desc_table_type",
    _IMPLEMENTED_MODES,
    ids=tuple(mode.name.lower() for mode in _IMPLEMENTED_MODES),
)
def test_compute_desc_table_rejects_mixed_panel_infinities(
    desc_table_type: DescTableType,
    nullable: bool,
) -> None:
    """Reject every infinity state before mode-specific reductions can warn.

    Args:
        desc_table_type: Implemented public table mode under test.
        nullable: Whether the panel uses nullable ``Float64``/``pd.NA`` storage.
    """
    returns = _mixed_infinite_returns(nullable=nullable)
    original = returns.copy()

    with pytest.raises(ValueError, match="^data contains infinite values$"):
        _compute_without_warnings(
            returns,
            desc_table_type,
            annualize_vol=True,
            is_add_tstat=True,
        )

    pd.testing.assert_frame_equal(returns, original)


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
@pytest.mark.parametrize(
    "infinity",
    (np.inf, -np.inf, (-np.inf, np.inf)),
    ids=("positive", "negative", "both-signs"),
)
def test_compute_desc_table_rejects_named_series_infinities(
    infinity: float | tuple[float, float],
    nullable: bool,
) -> None:
    """Apply the same exact rejection contract to named Series inputs.

    Args:
        infinity: Signed infinity state inserted into the Series.
        nullable: Whether the Series uses nullable ``Float64`` storage.
    """
    returns = _infinite_series(infinity=infinity, nullable=nullable)
    original = returns.copy()

    with pytest.raises(ValueError, match="^data contains infinite values$"):
        _compute_without_warnings(returns, DescTableType.SHORT)

    pd.testing.assert_series_equal(returns, original)


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
@pytest.mark.parametrize(
    "desc_table_type",
    tuple(_EXPECTED_VALID_TABLES),
    ids=tuple(mode.name.lower() for mode in _EXPECTED_VALID_TABLES),
)
def test_compute_desc_table_preserves_valid_finite_and_missing_statistics(
    desc_table_type: DescTableType,
    nullable: bool,
) -> None:
    """Preserve independently calculated finite and all-missing results.

    Args:
        desc_table_type: Positive-probability or extensive control schema.
        nullable: Whether the panel uses nullable ``Float64``/``pd.NA`` storage.
    """
    returns = _valid_returns(nullable=nullable)
    original = returns.copy()

    actual = _compute_without_warnings(returns, desc_table_type)

    pd.testing.assert_frame_equal(actual, _EXPECTED_VALID_TABLES[desc_table_type])
    pd.testing.assert_frame_equal(returns, original)

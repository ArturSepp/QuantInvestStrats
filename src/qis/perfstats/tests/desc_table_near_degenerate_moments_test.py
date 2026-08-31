"""Regression coverage for near-degenerate descriptive-table moments.

Skewness, excess kurtosis, and the normality statistic are invariant under translation. Floating-
point samples whose spread is only a few representable steps relative to their level must
therefore report the same moments as their zero-based counterparts. Raw SciPy reducers can instead
emit catastrophic-cancellation warnings and return offset-dependent values.

The deterministic panel below combines positive, negative, and large offsets; symmetric and
asymmetric shapes; multiple ULP widths; ragged and all-missing histories; an exact constant; and a
regular finite control. For an equally weighted two-level sample, direct central-moment formulas
give skewness zero and excess kurtosis ``-2``. The asymmetric reference uses the literal centered
pattern ``[0, 1, 3, 4, 7, 9]`` repeated four times. Ordinary ``float64`` and nullable ``Float64``
inputs verify every public moment mode, translation equivalence, Series/DataFrame consistency,
schema, formatting, warning behavior, and caller ownership.
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
        norm_variable_display_type: str,
    ) -> pd.DataFrame:
        """Return a formatted descriptive-statistics table."""
        raise NotImplementedError


_DESC_TABLE_MODULE = cast(_DescTableModuleProtocol, desc_table_module)


# =============================================================================
# Shared deterministic fixtures and independent expectations
# =============================================================================

_DATES = pd.date_range("2024-01-31", periods=24, freq="ME")

_POSITIVE_NEAR = "Positive Near"
_NEGATIVE_NEAR = "Negative Near"
_LARGE_NEAR = "Large Near"
_ASYMMETRIC_NEAR = "Asymmetric Near"
_RAGGED_NEAR = "Ragged Near"
_EXACT_CONSTANT = "Exact Constant"
_REGULAR_FINITE = "Regular Finite"
_ALL_MISSING = "All Missing"

_ASSETS = (
    _POSITIVE_NEAR,
    _NEGATIVE_NEAR,
    _LARGE_NEAR,
    _ASYMMETRIC_NEAR,
    _RAGGED_NEAR,
    _EXACT_CONSTANT,
    _REGULAR_FINITE,
    _ALL_MISSING,
)

_MOMENT_MODES = (
    DescTableType.WITH_KURTOSIS,
    DescTableType.WITH_NORMAL_PVAL,
    DescTableType.SKEW_KURTOSIS,
    DescTableType.EXTENSIVE,
    DescTableType.WITH_MEDIAN,
)

_EXPECTED_MODE_COLUMNS: dict[DescTableType, tuple[str, ...]] = {
    DescTableType.WITH_KURTOSIS: ("Avg", "Std", "Skew", "Kurt"),
    DescTableType.WITH_NORMAL_PVAL: ("Avg", "Std", "Skew", "Kurt", "P-val"),
    DescTableType.SKEW_KURTOSIS: ("Skew", "Kurt"),
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
    DescTableType.WITH_MEDIAN: ("Avg", "Std", "Median", "Skew", "Kurt"),
}

_EXPECTED_MOMENTS = pd.DataFrame(
    {
        "Skew": (
            "0",
            "0",
            "0",
            "0.316227766017",
            "0",
            "nan",
            "0",
            "nan",
        ),
        "Kurt": (
            "-2",
            "-2",
            "-2",
            "-1.26",
            "-2",
            "nan",
            "-1.20417391304",
            "nan",
        ),
    },
    index=pd.Index(_ASSETS),
)

_EXPECTED_NORMALITY = pd.Series(
    ("0.00", "0.00", "0.00", "0.07", "0.00", "nan", "0.08", "nan"),
    index=pd.Index(_ASSETS),
    name="P-val",
)

_MOMENT_FORMAT = "{:.12g}"


def _next_float(value: float, steps: int) -> float:
    """Move a finite value upward by an exact number of representable steps.

    Args:
        value: Starting floating-point value.
        steps: Number of calls to ``np.nextafter`` toward positive infinity.

    Returns:
        Representable value exactly ``steps`` ULP transitions above the start.
    """
    result = value
    for _ in range(steps):
        result = float(np.nextafter(result, np.inf))
    return result


def _mixed_samples(*, nullable: bool) -> pd.DataFrame:
    """Create the mixed near-degenerate and control panel.

    Args:
        nullable: Whether every column uses pandas nullable ``Float64`` storage.

    Returns:
        Twenty-four-row panel containing every material moment boundary.
    """
    positive_pair = (1.0, _next_float(1.0, 1))
    negative_pair = (-1.0, _next_float(-1.0, 3))
    large_pair = (1.0e8, _next_float(1.0e8, 1))
    asymmetric_pattern = tuple(_next_float(1.0e-8, step) for step in (0, 1, 3, 4, 7, 9))

    samples = pd.DataFrame(
        {
            _POSITIVE_NEAR: positive_pair * 12,
            _NEGATIVE_NEAR: negative_pair * 12,
            _LARGE_NEAR: large_pair * 12,
            _ASYMMETRIC_NEAR: asymmetric_pattern * 4,
            _RAGGED_NEAR: (np.nan,) * 4 + positive_pair * 10,
            _EXACT_CONSTANT: (2.0,) * 24,
            _REGULAR_FINITE: tuple(float(value) - 11.5 for value in range(24)),
            _ALL_MISSING: (np.nan,) * 24,
        },
        index=_DATES,
    )
    if nullable:
        return samples.astype(pd.Float64Dtype())
    return samples


def _translated_asymmetric_samples(*, nullable: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create equivalent zero-based and offset asymmetric samples.

    Args:
        nullable: Whether both frames use pandas nullable ``Float64`` storage.

    Returns:
        Canonical and translated one-column frames with identical standardized moments.
    """
    canonical_pattern = (0.0, 1.0, 3.0, 4.0, 7.0, 9.0)
    offset_pattern = tuple(_next_float(1.0e-8, step) for step in (0, 1, 3, 4, 7, 9))
    canonical = pd.DataFrame({_ASYMMETRIC_NEAR: canonical_pattern * 4}, index=_DATES)
    translated = pd.DataFrame({_ASYMMETRIC_NEAR: offset_pattern * 4}, index=_DATES)
    if nullable:
        dtype = pd.Float64Dtype()
        return canonical.astype(dtype), translated.astype(dtype)
    return canonical, translated


def _compute_without_warnings(
    data: pd.DataFrame | pd.Series,
    desc_table_type: DescTableType,
) -> pd.DataFrame:
    """Call the public function while treating every warning as a failure.

    Args:
        data: Series or DataFrame supplied to the public function.
        desc_table_type: Moment-reporting table mode under test.

    Returns:
        Formatted descriptive-statistics table.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return _DESC_TABLE_MODULE.compute_desc_table(
            df=data,
            desc_table_type=desc_table_type,
            norm_variable_display_type=_MOMENT_FORMAT,
        )


# =============================================================================
# Mixed-panel moment stability
# =============================================================================


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
@pytest.mark.parametrize("desc_table_type", _MOMENT_MODES, ids=lambda mode: mode.name.lower())
def test_compute_desc_table_stabilizes_near_degenerate_moments(
    desc_table_type: DescTableType,
    nullable: bool,
) -> None:
    """Return independently specified moments without cancellation warnings.

    For each symmetric two-level column, half the centered deviations are ``-d/2`` and half are
    ``d/2``. Its third central moment is therefore zero, while its fourth standardized moment is
    one and SciPy's excess-kurtosis convention subtracts three, giving ``-2``. Exact constants
    remain undefined, and the regular finite neighbor protects ordinary-scale behavior.

    Args:
        desc_table_type: Public moment-reporting schema under test.
        nullable: Whether the panel uses nullable ``Float64``/``pd.NA`` storage.
    """
    samples = _mixed_samples(nullable=nullable)
    original = samples.copy()

    actual = _compute_without_warnings(samples, desc_table_type)

    assert tuple(actual.index) == _ASSETS
    assert tuple(actual.columns) == _EXPECTED_MODE_COLUMNS[desc_table_type]
    pd.testing.assert_frame_equal(actual.filter(items=["Skew", "Kurt"]), _EXPECTED_MOMENTS)
    if desc_table_type is DescTableType.WITH_NORMAL_PVAL:
        pd.testing.assert_series_equal(actual["P-val"], _EXPECTED_NORMALITY)
    pd.testing.assert_frame_equal(samples, original)


# =============================================================================
# Translation and public input-shape consistency
# =============================================================================


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
def test_compute_desc_table_preserves_moments_under_translation(nullable: bool) -> None:
    """Return identical moments for canonical and offset copies of one finite shape.

    The canonical pattern has independently calculated biased skewness
    ``0.31622776601683794`` and excess kurtosis ``-1.26``. SciPy's normality test on that literal
    zero-based sample gives p-value ``0.07108735195801984``; translating its represented levels
    cannot change any of these standardized statistics.

    Args:
        nullable: Whether both inputs use nullable ``Float64`` storage.
    """
    canonical, translated = _translated_asymmetric_samples(nullable=nullable)

    canonical_result = _compute_without_warnings(canonical, DescTableType.WITH_NORMAL_PVAL)
    translated_result = _compute_without_warnings(translated, DescTableType.WITH_NORMAL_PVAL)

    pd.testing.assert_frame_equal(
        translated_result.filter(items=["Skew", "Kurt", "P-val"]),
        canonical_result.filter(items=["Skew", "Kurt", "P-val"]),
    )


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
def test_compute_desc_table_preserves_near_degenerate_series_dataframe_consistency(
    nullable: bool,
) -> None:
    """Return the same warning-free row for a named Series and one-column frame.

    Args:
        nullable: Whether both inputs use nullable ``Float64`` storage.
    """
    frame = _mixed_samples(nullable=nullable).filter(items=[_ASYMMETRIC_NEAR])
    series = frame[_ASYMMETRIC_NEAR]
    assert isinstance(series, pd.Series)
    original_frame = frame.copy()
    original_series = series.copy()

    frame_result = _compute_without_warnings(frame, DescTableType.WITH_NORMAL_PVAL)
    series_result = _compute_without_warnings(series, DescTableType.WITH_NORMAL_PVAL)

    pd.testing.assert_frame_equal(series_result, frame_result)
    pd.testing.assert_frame_equal(frame, original_frame)
    pd.testing.assert_series_equal(series, original_series)

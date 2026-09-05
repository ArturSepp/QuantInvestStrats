"""Regression coverage for near-degenerate descriptive spread and t-statistics.

Sample standard deviation is translation invariant, but a direct floating-point reduction can
lose precision when a sample's spread is only a few representable steps relative to its level.
That denominator error also changes the optional signed sample-mean t-statistic and annualized
volatility. The expected values below were calculated independently with 80-digit ``Decimal``
arithmetic applied to the exact stored binary64 observations, retaining the established binary64
sample mean only for the t-statistic numerator.

The mixed panel exercises positive, negative, and large offsets; one-, three-, and seven-ULP
widths; an asymmetric shape; a ragged history; an exact constant; an ordinary centered sample;
and an all-missing column in one public call. Ordinary ``float64`` and nullable ``Float64`` inputs
verify exact formatted results, periodic and annualized display, warnings, schema, ordering, and
caller ownership. Separate controls prove translation stability and named Series/DataFrame
consistency without broadening the established mean, missing-data, or standardized-moment
contracts.
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
        var_format: str,
        annualize_vol: bool,
        is_add_tstat: bool,
        norm_variable_display_type: str,
    ) -> pd.DataFrame:
        """Return a formatted descriptive-statistics table."""
        raise NotImplementedError


_DESC_TABLE_MODULE = cast(_DescTableModuleProtocol, desc_table_module)


# =============================================================================
# Shared deterministic fixtures and independent expectations
# =============================================================================

_DATES = pd.date_range("2024-01-31", periods=24, freq="ME")

_POSITIVE_ONE_ULP = "Positive One ULP"
_NEGATIVE_THREE_ULP = "Negative Three ULP"
_LARGE_ONE_ULP = "Large One ULP"
_POSITIVE_THREE_ULP = "Positive Three ULP"
_POSITIVE_SEVEN_ULP = "Positive Seven ULP"
_ASYMMETRIC_NEAR = "Asymmetric Near"
_RAGGED_ONE_ULP = "Ragged One ULP"
_EXACT_CONSTANT = "Exact Constant"
_REGULAR_CENTERED = "Regular Centered"
_ALL_MISSING = "All Missing"

_ASSETS = (
    _POSITIVE_ONE_ULP,
    _NEGATIVE_THREE_ULP,
    _LARGE_ONE_ULP,
    _POSITIVE_THREE_ULP,
    _POSITIVE_SEVEN_ULP,
    _ASYMMETRIC_NEAR,
    _RAGGED_ONE_ULP,
    _EXACT_CONSTANT,
    _REGULAR_CENTERED,
    _ALL_MISSING,
)

_EXPECTED_AVERAGES = (
    "1",
    "-1",
    "100000000",
    "1e-08",
    "1e-08",
    "1e-08",
    "1",
    "2",
    "0",
    "nan",
)
_EXPECTED_PERIODIC_SPREADS = (
    "1.13410152037e-16",
    "1.70115228056e-16",
    "7.61082646929e-09",
    "2.53491443479e-24",
    "5.91480034784e-24",
    "5.34406885838e-24",
    "1.13906478925e-16",
    "0",
    "7.07106781187",
    "nan",
)
_EXPECTED_ANNUALIZED_SPREADS = (
    "3.92864290845e-16",
    "5.89296436268e-16",
    "2.63646762648e-08",
    "8.78120118779e-24",
    "2.04894694382e-23",
    "1.85123975637e-23",
    "3.94583617619e-16",
    "0",
    "24.4948974278",
    "nan",
)
_EXPECTED_TSTATS = (
    "4.31970101226e+16",
    "-2.87980067484e+16",
    "6.4368561093e+16",
    "1.93260151835e+16",
    "8.28257793579e+15",
    "9.16713391124e+15",
    "3.92614713158e+16",
    "nan",
    "0",
    "nan",
)

_VALUE_FORMAT = "{:.12g}"


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
    """Create the mixed spread, missing-data, and ordinary-scale panel.

    Args:
        nullable: Whether every column uses pandas nullable ``Float64`` storage.

    Returns:
        Twenty-four-row panel containing every material spread boundary.
    """
    positive_pair = (1.0, _next_float(1.0, 1))
    negative_pair = (-1.0, _next_float(-1.0, 3))
    large_pair = (1.0e8, _next_float(1.0e8, 1))
    positive_three_pair = (1.0e-8, _next_float(1.0e-8, 3))
    positive_seven_pair = (1.0e-8, _next_float(1.0e-8, 7))
    asymmetric_pattern = tuple(_next_float(1.0e-8, step) for step in (0, 1, 3, 4, 7, 9))

    samples = pd.DataFrame(
        {
            _POSITIVE_ONE_ULP: positive_pair * 12,
            _NEGATIVE_THREE_ULP: negative_pair * 12,
            _LARGE_ONE_ULP: large_pair * 12,
            _POSITIVE_THREE_ULP: positive_three_pair * 12,
            _POSITIVE_SEVEN_ULP: positive_seven_pair * 12,
            _ASYMMETRIC_NEAR: asymmetric_pattern * 4,
            _RAGGED_ONE_ULP: (np.nan,) * 4 + positive_pair * 10,
            _EXACT_CONSTANT: (2.0,) * 24,
            _REGULAR_CENTERED: tuple(float(value) - 11.5 for value in range(24)),
            _ALL_MISSING: (np.nan,) * 24,
        },
        index=_DATES,
    )
    if nullable:
        return samples.astype(pd.Float64Dtype())
    return samples


def _translated_shape_samples(*, nullable: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create zero-based and offset copies of the same represented shape.

    Args:
        nullable: Whether both frames use pandas nullable ``Float64`` storage.

    Returns:
        Canonical and translated one-column frames with identical sample spread.
    """
    steps = (0, 1, 3, 4, 7, 9)
    origin = 1.0e-8
    unit = _next_float(origin, 1) - origin
    canonical_pattern = tuple(step * unit for step in steps)
    offset_pattern = tuple(_next_float(origin, step) for step in steps)
    canonical = pd.DataFrame({_ASYMMETRIC_NEAR: canonical_pattern * 4}, index=_DATES)
    translated = pd.DataFrame({_ASYMMETRIC_NEAR: offset_pattern * 4}, index=_DATES)
    if nullable:
        dtype = pd.Float64Dtype()
        return canonical.astype(dtype), translated.astype(dtype)
    return canonical, translated


def _expected_table(*, annualize_vol: bool) -> pd.DataFrame:
    """Return the literal high-precision reference formatted for public display.

    Args:
        annualize_vol: Whether to select periodic or monthly annualized spread references.

    Returns:
        Exact expected table for the mixed public-call regression.
    """
    volatility_label = "Std An" if annualize_vol else "Std"
    spreads = _EXPECTED_ANNUALIZED_SPREADS if annualize_vol else _EXPECTED_PERIODIC_SPREADS
    return pd.DataFrame(
        {
            "Avg": _EXPECTED_AVERAGES,
            volatility_label: spreads,
            "T-stat": _EXPECTED_TSTATS,
        },
        index=pd.Index(_ASSETS),
    )


def _compute_without_warnings(
    data: pd.DataFrame | pd.Series,
    *,
    annualize_vol: bool,
    is_add_tstat: bool,
) -> pd.DataFrame:
    """Call the public function while treating every warning as a failure.

    Args:
        data: Series or DataFrame supplied to the public function.
        annualize_vol: Whether to report monthly annualized volatility.
        is_add_tstat: Whether to include the signed sample-mean t-statistic.

    Returns:
        Formatted descriptive-statistics table.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return _DESC_TABLE_MODULE.compute_desc_table(
            df=data,
            desc_table_type=DescTableType.SHORT,
            var_format=_VALUE_FORMAT,
            annualize_vol=annualize_vol,
            is_add_tstat=is_add_tstat,
            norm_variable_display_type=_VALUE_FORMAT,
        )


# =============================================================================
# Mixed-panel spread and t-statistic precision
# =============================================================================


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
@pytest.mark.parametrize("annualize_vol", (False, True), ids=("periodic", "annualized"))
def test_compute_desc_table_stabilizes_near_degenerate_spread(
    annualize_vol: bool,
    nullable: bool,
) -> None:
    """Return Decimal-referenced spread and t-statistics for every column state.

    The periodic references use sample variance with ``ddof=1`` over the exact represented
    binary64 observations. Annualized references multiply only that spread by ``sqrt(12)``; the
    signed t-statistic retains the periodic standard error and observed sample count. Constants
    and all-missing histories preserve the established undefined-statistic behavior.

    Args:
        annualize_vol: Whether to test periodic or monthly annualized display.
        nullable: Whether the panel uses nullable ``Float64``/``pd.NA`` storage.
    """
    samples = _mixed_samples(nullable=nullable)
    original = samples.copy()

    actual = _compute_without_warnings(
        samples,
        annualize_vol=annualize_vol,
        is_add_tstat=True,
    )

    pd.testing.assert_frame_equal(actual, _expected_table(annualize_vol=annualize_vol))
    pd.testing.assert_frame_equal(samples, original)


# =============================================================================
# Translation and public input-shape consistency
# =============================================================================


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
def test_compute_desc_table_preserves_sample_spread_under_translation(nullable: bool) -> None:
    """Return identical sample spread for zero-based and offset copies of one shape.

    Each represented offset is an exact integer number of local ULPs above ``1e-8``. Subtracting
    that common level produces the canonical sample, so their independently defined sample
    standard deviations are identical even though their means are intentionally different.

    Args:
        nullable: Whether both inputs use nullable ``Float64`` storage.
    """
    canonical, translated = _translated_shape_samples(nullable=nullable)

    canonical_result = _compute_without_warnings(
        canonical,
        annualize_vol=False,
        is_add_tstat=False,
    )
    translated_result = _compute_without_warnings(
        translated,
        annualize_vol=False,
        is_add_tstat=False,
    )

    assert canonical_result.loc[_ASYMMETRIC_NEAR, "Std"] == _EXPECTED_PERIODIC_SPREADS[5]
    assert translated_result.loc[_ASYMMETRIC_NEAR, "Std"] == _EXPECTED_PERIODIC_SPREADS[5]


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
def test_compute_desc_table_near_degenerate_series_matches_dataframe(nullable: bool) -> None:
    """Return the same formatted row for a named Series and one-column DataFrame.

    Args:
        nullable: Whether both inputs use nullable ``Float64`` storage.
    """
    frame = _mixed_samples(nullable=nullable).filter(items=[_ASYMMETRIC_NEAR])
    series = frame[_ASYMMETRIC_NEAR]
    assert isinstance(series, pd.Series)
    original_frame = frame.copy()
    original_series = series.copy()

    frame_result = _compute_without_warnings(
        frame,
        annualize_vol=False,
        is_add_tstat=True,
    )
    series_result = _compute_without_warnings(
        series,
        annualize_vol=False,
        is_add_tstat=True,
    )

    pd.testing.assert_frame_equal(series_result, frame_result)
    pd.testing.assert_frame_equal(frame, original_frame)
    pd.testing.assert_series_equal(series, original_series)

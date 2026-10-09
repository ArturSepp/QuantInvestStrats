"""Regression coverage for extreme-scale descriptive sample spread.

Sample standard deviation is invariant to multiplication by a finite nonzero scale, but directly
squaring binary64 deviations can underflow to zero or overflow to infinity. A finite range can
also overflow while it is translated away from a large negative origin. These failures corrupt
periodic spread, annualized volatility, and the optional sample-mean t-statistic even though the
mathematical periodic result is representable.

The mixed panel below exercises tiny positive, tiny negative, zero-origin, huge same-sign, huge
mixed-sign, tiny-normal, subnormal, wide signed, ragged, constant, ordinary, and all-missing
columns in one public call. Literal expectations were calculated independently with high-precision
arithmetic on the exact stored binary64 observations using ``ddof=1``. The controls distinguish a
representable finite spread from both unavoidable subnormal rounding to zero and a mathematical
result larger than binary64 can represent.
"""

import sys
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

_TINY_POSITIVE = "Tiny Positive"
_TINY_NEGATIVE = "Tiny Negative"
_TINY_ZERO_ORIGIN = "Tiny Zero Origin"
_LARGE_POSITIVE = "Large Positive"
_LARGE_MIXED = "Large Mixed"
_NEAR_MINIMUM_NORMAL = "Near Minimum Normal"
_SUBNORMAL_REPRESENTABLE = "Subnormal Representable"
_SUBNORMAL_ROUNDS_ZERO = "Subnormal Rounds Zero"
_WIDE_FINITE = "Wide Finite"
_PERIODIC_OVERFLOW = "Periodic Overflow"
_RAGGED_TINY = "Ragged Tiny"
_EXACT_CONSTANT = "Exact Constant"
_ORDINARY_CENTERED = "Ordinary Centered"
_ALL_MISSING = "All Missing"

_ASSETS = (
    _TINY_POSITIVE,
    _TINY_NEGATIVE,
    _TINY_ZERO_ORIGIN,
    _LARGE_POSITIVE,
    _LARGE_MIXED,
    _NEAR_MINIMUM_NORMAL,
    _SUBNORMAL_REPRESENTABLE,
    _SUBNORMAL_ROUNDS_ZERO,
    _WIDE_FINITE,
    _PERIODIC_OVERFLOW,
    _RAGGED_TINY,
    _EXACT_CONSTANT,
    _ORDINARY_CENTERED,
    _ALL_MISSING,
)

_EXPECTED_AVERAGES = (
    "2.5e-200",
    "-2.5e-200",
    "1.5e-200",
    "2.5e+200",
    "0",
    "5.56268464627e-308",
    "1.48219693752e-323",
    "0",
    "0",
    "0",
    "2.5e-200",
    "2",
    "0",
    "nan",
)
_EXPECTED_PERIODIC_SPREADS = (
    "1.14208048144e-200",
    "1.14208048144e-200",
    "1.14208048144e-200",
    "1.14208048144e+200",
    "2.97818152862e+200",
    "2.54121342356e-308",
    "9.88131291682e-324",
    "0",
    "1.41421356237e+308",
    "inf",
    "1.14707866935e-200",
    "0",
    "7.07106781187",
    "nan",
)
_EXPECTED_ANNUALIZED_SPREADS = (
    "3.95628284037e-200",
    "3.95628284037e-200",
    "3.95628284037e-200",
    "3.95628284037e+200",
    "1.03167234435e+201",
    "8.80302152498e-308",
    "3.45845952089e-323",
    "0",
    "inf",
    "inf",
    "3.9735970712e-200",
    "0",
    "24.4948974278",
    "nan",
)
_EXPECTED_TSTATS = (
    "10.7238052948",
    "-10.7238052948",
    "6.43428317686",
    "10.7238052948",
    "0",
    "10.7238052948",
    "7.5",
    "nan",
    "0",
    "nan",
    "9.74679434481",
    "nan",
    "0",
    "nan",
)

_VALUE_FORMAT = "{:.12g}"


def _extreme_scale_samples(*, nullable: bool) -> pd.DataFrame:
    """Create one panel containing every material sample-spread scale boundary.

    Six repetitions of the four-value shapes give 24 observations. Their high-precision sample
    variances reduce to ``30 / 23`` times scale squared for ``[1, 2, 3, 4]`` and ``204 / 23``
    times scale squared for ``[-4, -1, 1, 4]``. The ragged shape has 20 observations and variance
    ``25 / 19`` times scale squared. Wide signed columns retain only two observations so their
    means remain representable independently of the separate extreme-mean contract.

    Args:
        nullable: Whether every column uses pandas nullable ``Float64`` storage.

    Returns:
        Twenty-four-row monthly panel in expected output order.
    """
    minimum_subnormal = np.nextafter(0.0, 1.0)
    maximum_finite = sys.float_info.max
    minimum_normal = np.finfo(float).tiny
    missing_tail = np.full(22, np.nan)

    samples = pd.DataFrame(
        {
            _TINY_POSITIVE: np.tile(np.array((1.0, 2.0, 3.0, 4.0)) * 1.0e-200, 6),
            _TINY_NEGATIVE: np.tile(-np.array((1.0, 2.0, 3.0, 4.0)) * 1.0e-200, 6),
            _TINY_ZERO_ORIGIN: np.tile(np.array((0.0, 1.0, 2.0, 3.0)) * 1.0e-200, 6),
            _LARGE_POSITIVE: np.tile(np.array((1.0, 2.0, 3.0, 4.0)) * 1.0e200, 6),
            _LARGE_MIXED: np.tile(np.array((-4.0, -1.0, 1.0, 4.0)) * 1.0e200, 6),
            _NEAR_MINIMUM_NORMAL: np.tile(np.array((1.0, 2.0, 3.0, 4.0)) * minimum_normal, 6),
            _SUBNORMAL_REPRESENTABLE: np.tile(
                np.array((0.0, 2.0, 4.0, 6.0)) * minimum_subnormal, 6
            ),
            _SUBNORMAL_ROUNDS_ZERO: np.concatenate(((minimum_subnormal,), np.zeros(23))),
            _WIDE_FINITE: np.concatenate(((-1.0e308, 1.0e308), missing_tail)),
            _PERIODIC_OVERFLOW: np.concatenate(((-maximum_finite, maximum_finite), missing_tail)),
            _RAGGED_TINY: np.concatenate(
                (np.full(4, np.nan), np.tile(np.array((1.0, 2.0, 3.0, 4.0)) * 1.0e-200, 5))
            ),
            _EXACT_CONSTANT: np.full(24, 2.0),
            _ORDINARY_CENTERED: np.arange(24, dtype=float) - 11.5,
            _ALL_MISSING: np.full(24, np.nan),
        },
        index=_DATES,
    )
    if nullable:
        return samples.astype(pd.Float64Dtype())
    return samples


def _expected_table(*, annualize_vol: bool) -> pd.DataFrame:
    """Build the complete literal result calculated independently at high precision.

    The t-statistic retains the established calculation from the represented periodic binary64
    spread. Consequently, the coarsely represented subnormal denominator produces ``7.5`` even
    though scaling the displayed spread annually remains independent of inference.

    Args:
        annualize_vol: Whether to select periodic or monthly annualized spread references.

    Returns:
        Exact formatted table for the mixed-panel regression.
    """
    spreads = _EXPECTED_ANNUALIZED_SPREADS if annualize_vol else _EXPECTED_PERIODIC_SPREADS
    volatility_label = "Std An" if annualize_vol else "Std"
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
) -> pd.DataFrame:
    """Call the public function while treating every warning as a regression failure.

    Args:
        data: Series or DataFrame supplied to the public function.
        annualize_vol: Whether to display monthly annualized rather than periodic spread.

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
            is_add_tstat=True,
            norm_variable_display_type=_VALUE_FORMAT,
        )


# =============================================================================
# Mixed-panel extreme-scale spread and dependent statistics
# =============================================================================


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
@pytest.mark.parametrize("annualize_vol", (False, True), ids=("periodic", "annualized"))
def test_compute_desc_table_stabilizes_extreme_scale_sample_spread(
    annualize_vol: bool,
    nullable: bool,
) -> None:
    """Return exact formatted spread results across every material column state.

    Representable periodic spreads remain finite at tiny, subnormal, huge, and wide signed scales.
    A true periodic overflow and annualization of the wide finite spread return ``inf`` without a
    reduction warning. Exact constants and all-missing histories preserve their established
    undefined-inference behavior, and the public call does not mutate either pandas storage type.

    Args:
        annualize_vol: Whether to verify periodic or monthly annualized display.
        nullable: Whether the panel uses nullable ``Float64``/``pd.NA`` storage.
    """
    samples = _extreme_scale_samples(nullable=nullable)
    original = samples.copy()

    actual = _compute_without_warnings(samples, annualize_vol=annualize_vol)

    pd.testing.assert_frame_equal(actual, _expected_table(annualize_vol=annualize_vol))
    pd.testing.assert_frame_equal(samples, original)


# =============================================================================
# Public input-shape and accepted safe-scale controls
# =============================================================================


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
def test_compute_desc_table_extreme_scale_series_matches_dataframe(nullable: bool) -> None:
    """Return the same huge finite spread for a named Series and one-column frame.

    Args:
        nullable: Whether both equivalent inputs use nullable ``Float64`` storage.
    """
    frame = _extreme_scale_samples(nullable=nullable).filter(items=[_LARGE_MIXED])
    series = frame[_LARGE_MIXED]
    assert isinstance(series, pd.Series)

    frame_result = _compute_without_warnings(frame, annualize_vol=False)
    series_result = _compute_without_warnings(series, annualize_vol=False)

    pd.testing.assert_frame_equal(series_result, frame_result)


def test_compute_desc_table_preserves_safe_scale_sample_spread() -> None:
    """Leave accepted ordinary and near-degenerate translated reductions unchanged.

    The ordinary sample has exact spread ``sqrt(50)``. Alternating 1.0 with its next represented
    value has the independently established PR 67 spread ``1.13410152037e-16``. These controls
    ensure extreme-scale normalization is guarded rather than applied to every finite sample.
    """
    near_value = np.nextafter(1.0, np.inf)
    samples = pd.DataFrame(
        {
            "Near Degenerate": np.tile((1.0, near_value), 12),
            _ORDINARY_CENTERED: np.arange(24, dtype=float) - 11.5,
        },
        index=_DATES,
    )
    expected = pd.DataFrame(
        {
            "Avg": ("1", "0"),
            "Std": ("1.13410152037e-16", "7.07106781187"),
            "T-stat": ("4.31970101226e+16", "0"),
        },
        index=pd.Index(("Near Degenerate", _ORDINARY_CENTERED)),
    )

    actual = _compute_without_warnings(samples, annualize_vol=False)

    pd.testing.assert_frame_equal(actual, expected)

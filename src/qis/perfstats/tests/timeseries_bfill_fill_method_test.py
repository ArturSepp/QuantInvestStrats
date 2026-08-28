"""Regression tests for ``bfill_timeseries`` fill-method validation.

The public contract supports exactly three return-gap policies: ``None`` preserves missing
returns, ``to_zero`` replaces missing returns with zero after a column begins, and ``ffill``
carries the last observed return forward. Unsupported values must fail before either provider is
processed instead of silently selecting the forward-fill branch.

The valid-policy controls use mixed panels so each call exercises complete, interior-gap,
leading-gap, and all-missing histories together. The price panel additionally contains an older
fallback and an absent all-missing column. Expected values are literal or calculated directly by
scaling an older price history to the newer provider's initial level; no QIS transformation
helper constructs an expected result.
"""

from typing import cast
import warnings

import numpy as np
import pandas as pd
import pytest

from qis.perfstats.timeseries_bfill import bfill_timeseries


# =============================================================================
# Shared deterministic timeline and labels
# =============================================================================

_DATES = pd.date_range("2024-01-01", periods=5, freq="D")

_ABSENT = "Absent"
_ALL_MISSING = "All missing"
_COMPLETE = "Complete"
_FALLBACK = "Fallback"
_INTERIOR_GAP = "Interior gap"
_LEADING_GAP = "Leading gap"

_TOLERANCE = 1.0e-12


# =============================================================================
# Independently specified mixed-panel fixtures
# =============================================================================

def _price_histories() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create price panels containing every materially different column state.

    Returns:
        Older and newer price DataFrames. The newer column order is deliberately different from
        the older order and defines the expected output schema.
    """
    older = pd.DataFrame(
        {
            _COMPLETE: (100.0, 110.0, 121.0),
            _INTERIOR_GAP: (200.0, np.nan, 242.0),
            _LEADING_GAP: (np.nan, 300.0, 330.0),
            _FALLBACK: (50.0, 55.0, 60.5),
        },
        index=_DATES[:3],
    )
    newer = pd.DataFrame(
        {
            _ABSENT: (np.nan, np.nan),
            _FALLBACK: (np.nan, np.nan),
            _LEADING_GAP: (400.0, 440.0),
            _INTERIOR_GAP: (300.0, 330.0),
            _COMPLETE: (150.0, 165.0),
        },
        index=_DATES[3:],
    )
    return older, newer


def _return_histories() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create return panels containing every materially different column state.

    Returns:
        Older and newer return DataFrames. Reversed newer labels make output column order part of
        every valid-policy assertion.
    """
    older = pd.DataFrame(
        {
            _COMPLETE: (0.01, 0.02, 0.03),
            _INTERIOR_GAP: (0.10, np.nan, 0.30),
            _LEADING_GAP: (np.nan, np.nan, 0.30),
            _ALL_MISSING: (np.nan, np.nan, np.nan),
        },
        index=_DATES[:3],
    )
    newer = pd.DataFrame(
        {
            _ALL_MISSING: (np.nan, np.nan),
            _LEADING_GAP: (0.40, 0.50),
            _INTERIOR_GAP: (np.nan, 0.50),
            _COMPLETE: (0.04, 0.05),
        },
        index=_DATES[3:],
    )
    return older, newer


def _expected_prices() -> pd.DataFrame:
    """Calculate mixed price expectations without a QIS transformation helper.

    Returns:
        Price DataFrame anchored to each newer provider level while retaining fallback and absent
        column semantics.
    """
    complete_scale = 150.0 / 121.0
    interior_scale = 300.0 / 242.0
    leading_scale = 400.0 / 330.0
    return pd.DataFrame(
        {
            _ABSENT: (np.nan, np.nan, np.nan, np.nan, np.nan),
            _FALLBACK: (50.0, 55.0, 60.5, 60.5, 60.5),
            _LEADING_GAP: (np.nan, 300.0 * leading_scale, 400.0, 400.0, 440.0),
            _INTERIOR_GAP: (
                200.0 * interior_scale,
                200.0 * interior_scale,
                300.0,
                300.0,
                330.0,
            ),
            _COMPLETE: (
                100.0 * complete_scale,
                110.0 * complete_scale,
                150.0,
                150.0,
                165.0,
            ),
        },
        index=_DATES,
    )


def _expected_returns(fill_method: str | None) -> pd.DataFrame:
    """Select the literal mixed return path for one valid policy.

    Args:
        fill_method: One of the three documented return-gap policies.

    Returns:
        Expected return DataFrame with policy-specific interior gaps and preserved leading gaps.
    """
    interior_values: dict[
        str | None,
        tuple[float, float, float, float, float],
    ] = {
        None: (0.10, np.nan, 0.30, np.nan, 0.50),
        "to_zero": (0.10, 0.00, 0.30, 0.00, 0.50),
        "ffill": (0.10, 0.10, 0.30, 0.30, 0.50),
    }
    return pd.DataFrame(
        {
            _ALL_MISSING: (np.nan, np.nan, np.nan, np.nan, np.nan),
            _LEADING_GAP: (np.nan, np.nan, 0.30, 0.40, 0.50),
            _INTERIOR_GAP: interior_values[fill_method],
            _COMPLETE: (0.01, 0.02, 0.03, 0.04, 0.05),
        },
        index=_DATES,
    )


# =============================================================================
# Unsupported-policy validation
# =============================================================================

@pytest.mark.parametrize("is_prices", (False, True), ids=("returns", "prices"))
@pytest.mark.parametrize(
    "invalid_fill_method",
    (
        "",
        "misspelled",
        "FFILL",
        " ffill ",
        0,
        False,
        ["ffill"],
        pd.NA,
        np.array(("ffill", "to_zero"), dtype=object),
    ),
    ids=(
        "empty-string",
        "misspelled",
        "wrong-case",
        "surrounding-whitespace",
        "integer",
        "boolean",
        "list",
        "pandas-na",
        "numpy-array",
    ),
)
def test_bfill_timeseries_rejects_unsupported_fill_method(
        invalid_fill_method: object,
        is_prices: bool,
) -> None:
    """Reject unsupported policies before processing either caller-owned input.

    Args:
        invalid_fill_method: Invalid string or non-string value supplied deliberately at runtime.
        is_prices: Whether to exercise the return or price processing path.
    """
    dates = _DATES[:3]
    if is_prices:
        older = pd.Series((100.0, np.nan), index=dates[:2], name="Older provider")
        newer = pd.Series((121.0,), index=dates[2:], name="Asset")
    else:
        older = pd.Series((0.10, np.nan), index=dates[:2], name="Older provider")
        newer = pd.Series((0.30,), index=dates[2:], name="Asset")
    original_older = older.copy(deep=True)
    original_newer = newer.copy(deep=True)
    expected_message = (
        "fill_method must be None, 'to_zero', or 'ffill', "
        f"got {invalid_fill_method!r}"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError) as exc_info:
            bfill_timeseries(
                df_newer=newer,
                df_older=older,
                freq="D",
                fill_method=cast(str | None, invalid_fill_method),
                is_prices=is_prices,
            )

    assert str(exc_info.value) == expected_message
    pd.testing.assert_series_equal(older, original_older, check_exact=True)
    pd.testing.assert_series_equal(newer, original_newer, check_exact=True)


# =============================================================================
# Valid mixed-panel preservation controls
# =============================================================================

@pytest.mark.parametrize("fill_method", (None, "to_zero", "ffill"))
def test_bfill_timeseries_preserves_valid_price_fill_methods(
        fill_method: str | None,
) -> None:
    """Preserve established price behavior for every documented policy.

    Args:
        fill_method: Valid policy applied in return space before price reconstruction.
    """
    older, newer = _price_histories()
    original_older = older.copy(deep=True)
    original_newer = newer.copy(deep=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = bfill_timeseries(
            df_newer=newer,
            df_older=older,
            freq="D",
            fill_method=fill_method,
            is_prices=True,
        )

    assert isinstance(actual, pd.DataFrame)
    pd.testing.assert_frame_equal(
        actual,
        _expected_prices(),
        check_exact=False,
        check_freq=False,
        rtol=0.0,
        atol=_TOLERANCE,
    )
    pd.testing.assert_frame_equal(older, original_older, check_exact=True)
    pd.testing.assert_frame_equal(newer, original_newer, check_exact=True)


@pytest.mark.parametrize("fill_method", (None, "to_zero", "ffill"))
def test_bfill_timeseries_preserves_valid_return_fill_methods(
        fill_method: str | None,
) -> None:
    """Apply each documented return policy independently across a mixed panel.

    Args:
        fill_method: Valid policy selecting preserved, zero-filled, or forward-filled gaps.
    """
    older, newer = _return_histories()
    original_older = older.copy(deep=True)
    original_newer = newer.copy(deep=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = bfill_timeseries(
            df_newer=newer,
            df_older=older,
            freq="D",
            fill_method=fill_method,
        )

    assert isinstance(actual, pd.DataFrame)
    pd.testing.assert_frame_equal(
        actual,
        _expected_returns(fill_method),
        check_exact=True,
    )
    pd.testing.assert_frame_equal(older, original_older, check_exact=True)
    pd.testing.assert_frame_equal(newer, original_newer, check_exact=True)

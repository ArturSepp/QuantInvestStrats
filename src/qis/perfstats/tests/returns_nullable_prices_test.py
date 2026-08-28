"""Regression coverage for nullable price-to-return conversion.

``to_returns`` accepts pandas Series and DataFrames, so a multi-column nullable ``Float64`` panel
must follow the same return conventions as its NumPy-backed ``float64`` equivalent. These tests
use literal references for every return mode and first-observation option. A mixed panel combines
two finite assets, ragged and all-missing histories, and terminal non-positive observations in one
call so pandas' physical multi-column representation cannot hide behind one-column coverage.

The downstream backfill control also verifies that nullable conversion retains an independently
supplied first-price anchor without changing frequency or fill conventions.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from qis.perfstats.config import ReturnTypes
from qis.perfstats.returns import to_returns
from qis.perfstats.timeseries_bfill import bfill_timeseries


# =============================================================================
# Shared deterministic fixtures and comparison helpers
# =============================================================================

_DATES = pd.date_range("2024-01-01", periods=4, freq="D")

_ALL_MISSING = "All missing"
_FINITE_A = "Finite A"
_FINITE_B = "Finite B"
_NEGATIVE_TERMINAL = "Negative terminal"
_RAGGED = "Ragged"
_ZERO_TERMINAL = "Zero terminal"

_TOLERANCE = 1.0e-12


def _as_nullable(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert every physical column to pandas nullable floating point.

    Args:
        frame: NumPy-backed price or expected-return panel.

    Returns:
        A new panel whose columns all use ``Float64``.
    """
    return frame.astype(pd.Float64Dtype())


def _assert_frame_close(
    actual: pd.Series | pd.DataFrame,
    expected: pd.DataFrame,
) -> None:
    """Assert DataFrame shape, dtype, labels, missingness, and values.

    Args:
        actual: Result under the public Series-or-DataFrame annotation.
        expected: Independently constructed DataFrame reference.
    """
    assert isinstance(actual, pd.DataFrame)
    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_exact=False,
        rtol=0.0,
        atol=_TOLERANCE,
    )


def _make_positive_panel() -> pd.DataFrame:
    """Create finite, ragged, and all-missing positive price histories.

    Returns:
        NumPy-backed panel used across every return convention.
    """
    return pd.DataFrame(
        {
            _FINITE_A: (100.0, 110.0, 121.0, 133.1),
            _FINITE_B: (50.0, 55.0, 60.5, 66.55),
            _RAGGED: (np.nan, 200.0, 220.0, 242.0),
            _ALL_MISSING: (np.nan, np.nan, np.nan, np.nan),
        },
        index=_DATES,
    )


def _make_expected_mode_returns(return_type: ReturnTypes) -> pd.DataFrame:
    """Return literal references for one documented return convention.

    Args:
        return_type: Convention whose independently calculated values are required.

    Returns:
        Expected NumPy-backed return panel.

    Raises:
        AssertionError: If the test requests an unknown enum member.
    """
    if return_type == ReturnTypes.RELATIVE:
        finite_a = (np.nan, 0.10, 0.10, 0.10)
        finite_b = (np.nan, 0.10, 0.10, 0.10)
        ragged = (np.nan, np.nan, 0.10, 0.10)
    elif return_type == ReturnTypes.LOG:
        log_ten_percent = np.log(1.10)
        finite_a = (np.nan, log_ten_percent, log_ten_percent, log_ten_percent)
        finite_b = (np.nan, log_ten_percent, log_ten_percent, log_ten_percent)
        ragged = (np.nan, np.nan, log_ten_percent, log_ten_percent)
    elif return_type == ReturnTypes.DIFFERENCE:
        finite_a = (np.nan, 10.0, 11.0, 12.1)
        finite_b = (np.nan, 5.0, 5.5, 6.05)
        ragged = (np.nan, np.nan, 20.0, 22.0)
    elif return_type == ReturnTypes.LEVEL:
        finite_a = (100.0, 110.0, 121.0, 133.1)
        finite_b = (50.0, 55.0, 60.5, 66.55)
        ragged = (np.nan, 200.0, 220.0, 242.0)
    elif return_type == ReturnTypes.LEVEL0:
        finite_a = (np.nan, 100.0, 110.0, 121.0)
        finite_b = (np.nan, 50.0, 55.0, 60.5)
        ragged = (np.nan, np.nan, 200.0, 220.0)
    else:
        raise AssertionError(f"Unhandled return type: {return_type}")

    return pd.DataFrame(
        {
            _FINITE_A: finite_a,
            _FINITE_B: finite_b,
            _RAGGED: ragged,
            _ALL_MISSING: (np.nan, np.nan, np.nan, np.nan),
        },
        index=_DATES,
    )


# =============================================================================
# Return-mode and mixed-panel regressions
# =============================================================================


@pytest.mark.parametrize("return_type", tuple(ReturnTypes))
def test_to_returns_supports_nullable_panel_for_each_return_mode(
    return_type: ReturnTypes,
) -> None:
    """Match literal ordinary and nullable results for every return mode.

    Args:
        return_type: Return convention under test.
    """
    float_prices = _make_positive_panel()
    nullable_prices = _as_nullable(float_prices)
    original_float_prices = float_prices.copy(deep=True)
    original_nullable_prices = nullable_prices.copy(deep=True)
    expected_float = _make_expected_mode_returns(return_type)
    expected_nullable = _as_nullable(expected_float)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual_float = to_returns(
            prices=float_prices,
            return_type=return_type,
            ffill_nans=False,
        )
        actual_nullable = to_returns(
            prices=nullable_prices,
            return_type=return_type,
            ffill_nans=False,
        )

    _assert_frame_close(actual_float, expected_float)
    _assert_frame_close(actual_nullable, expected_nullable)
    pd.testing.assert_frame_equal(float_prices, original_float_prices, check_exact=True)
    pd.testing.assert_frame_equal(nullable_prices, original_nullable_prices, check_exact=True)


def test_to_returns_supports_nullable_mixed_price_states() -> None:
    """Validate every material column state in one physical nullable panel.

    Zero and negative observations occur only on the terminal date. Their expected return is
    therefore unambiguously missing without establishing how a later valid price should treat an
    invalid predecessor; PR 48 owns that separate numerical policy.
    """
    float_prices = _make_positive_panel().assign(
        **{
            _ZERO_TERMINAL: (100.0, 110.0, 121.0, 0.0),
            _NEGATIVE_TERMINAL: (50.0, 55.0, 60.5, -1.0),
        }
    )
    nullable_prices = _as_nullable(float_prices)
    original_nullable_prices = nullable_prices.copy(deep=True)
    expected_float = pd.DataFrame(
        {
            _FINITE_A: (np.nan, 0.10, 0.10, 0.10),
            _FINITE_B: (np.nan, 0.10, 0.10, 0.10),
            _RAGGED: (np.nan, np.nan, 0.10, 0.10),
            _ALL_MISSING: (np.nan, np.nan, np.nan, np.nan),
            _ZERO_TERMINAL: (np.nan, 0.10, 0.10, np.nan),
            _NEGATIVE_TERMINAL: (np.nan, 0.10, 0.10, np.nan),
        },
        index=_DATES,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual_float = to_returns(prices=float_prices, ffill_nans=False)
        actual_nullable = to_returns(prices=nullable_prices, ffill_nans=False)

    _assert_frame_close(actual_float, expected_float)
    _assert_frame_close(actual_nullable, _as_nullable(expected_float))
    pd.testing.assert_frame_equal(nullable_prices, original_nullable_prices, check_exact=True)


# =============================================================================
# First-observation, fill, and shape controls
# =============================================================================


def test_to_returns_sets_nullable_first_price_anchor_to_zero() -> None:
    """Apply ``is_first_zero`` independently at each nullable column's start."""
    prices = _as_nullable(_make_positive_panel().drop(columns=[_FINITE_B, _ALL_MISSING]))
    expected = _as_nullable(
        pd.DataFrame(
            {
                _FINITE_A: (0.0, 0.10, 0.10, 0.10),
                _RAGGED: (np.nan, 0.0, 0.10, 0.10),
            },
            index=_DATES,
        )
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = to_returns(prices=prices, ffill_nans=False, is_first_zero=True)

    _assert_frame_close(actual, expected)


def test_to_returns_drops_first_nullable_observation() -> None:
    """Drop the first physical row without altering later nullable results."""
    prices = _as_nullable(_make_positive_panel().drop(columns=[_FINITE_B, _ALL_MISSING]))
    expected = _as_nullable(
        pd.DataFrame(
            {
                _FINITE_A: (0.10, 0.10, 0.10),
                _RAGGED: (np.nan, 0.10, 0.10),
            },
            index=_DATES[1:],
        )
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = to_returns(prices=prices, ffill_nans=False, drop_first=True)

    _assert_frame_close(actual, expected)


def test_to_returns_log_flag_overrides_nullable_level_mode() -> None:
    """Honor the documented log-return override for nullable multi-column input."""
    prices = _as_nullable(_make_positive_panel().drop(columns=[_RAGGED, _ALL_MISSING]))
    log_ten_percent = np.log(1.10)
    expected = _as_nullable(
        pd.DataFrame(
            {
                _FINITE_A: (np.nan, log_ten_percent, log_ten_percent, log_ten_percent),
                _FINITE_B: (np.nan, log_ten_percent, log_ten_percent, log_ten_percent),
            },
            index=_DATES,
        )
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = to_returns(
            prices=prices,
            is_log_returns=True,
            return_type=ReturnTypes.LEVEL,
            ffill_nans=False,
        )

    _assert_frame_close(actual, expected)


@pytest.mark.parametrize(
    ("ffill_nans", "expected_ragged"),
    (
        pytest.param(True, (np.nan, 0.0, 0.21, 0.10), id="forward-fill"),
        pytest.param(False, (np.nan, np.nan, np.nan, 0.10), id="preserve-gap"),
    ),
)
def test_to_returns_honors_nullable_price_fill_policy(
    ffill_nans: bool,
    expected_ragged: tuple[float, ...],
) -> None:
    """Apply the existing fill policy before nullable return calculation.

    Args:
        ffill_nans: Whether the January 2 price gap carries January 1's level.
        expected_ragged: Independently calculated ragged-column returns.
    """
    prices = _as_nullable(
        pd.DataFrame(
            {
                _FINITE_A: (100.0, 110.0, 121.0, 133.1),
                _RAGGED: (100.0, np.nan, 121.0, 133.1),
            },
            index=_DATES,
        )
    )
    expected = _as_nullable(
        pd.DataFrame(
            {
                _FINITE_A: (np.nan, 0.10, 0.10, 0.10),
                _RAGGED: expected_ragged,
            },
            index=_DATES,
        )
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = to_returns(prices=prices, ffill_nans=ffill_nans)

    _assert_frame_close(actual, expected)


def test_to_returns_retains_nullable_series_and_one_column_behavior() -> None:
    """Preserve the nullable shapes that already work on accepted main."""
    prices = pd.Series(
        pd.array((100.0, 110.0, 121.0, 133.1), dtype="Float64"),
        index=_DATES,
        name="Asset",
    )
    expected = pd.Series(
        pd.array((pd.NA, 0.10, 0.10, 0.10), dtype="Float64"),
        index=_DATES,
        name="Asset",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        series_actual = to_returns(prices=prices, ffill_nans=False)
        frame_actual = to_returns(prices=prices.to_frame(), ffill_nans=False)

    assert isinstance(series_actual, pd.Series)
    assert isinstance(frame_actual, pd.DataFrame)
    pd.testing.assert_series_equal(
        series_actual,
        expected,
        check_exact=False,
        rtol=0.0,
        atol=_TOLERANCE,
    )
    pd.testing.assert_frame_equal(
        frame_actual,
        expected.to_frame(),
        check_exact=False,
        rtol=0.0,
        atol=_TOLERANCE,
    )


# =============================================================================
# Downstream nullable price-backfill preservation
# =============================================================================


def test_bfill_timeseries_supports_nullable_mixed_price_panel() -> None:
    """Retain nullable conversion and the coincident ragged price anchor.

    Fallback, shared, and absent columns pin the mixed-state, off-grid, schema, warning, and
    ownership behavior around the Tuesday ragged anchor.
    """
    older = _as_nullable(
        pd.DataFrame(
            {
                _RAGGED: (np.nan, 210.0),
                "Fallback": (50.0, 55.0),
                "Shared": (100.0, 110.0),
            },
            index=pd.DatetimeIndex(("2024-01-06", "2024-01-09")),
        )
    )
    newer = _as_nullable(
        pd.DataFrame(
            {
                "Absent": (np.nan, np.nan),
                _RAGGED: (210.0, 231.0),
                "Fallback": (np.nan, np.nan),
                "Shared": (110.0, 121.0),
            },
            index=pd.DatetimeIndex(("2024-01-09", "2024-01-10")),
        )
    )
    original_older = older.copy(deep=True)
    original_newer = newer.copy(deep=True)
    expected = _as_nullable(
        pd.DataFrame(
            {
                "Absent": (np.nan, np.nan, np.nan),
                _RAGGED: (np.nan, 210.0, 231.0),
                "Fallback": (50.0, 55.0, 55.0),
                "Shared": (100.0, 110.0, 121.0),
            },
            index=pd.bdate_range("2024-01-08", "2024-01-10"),
        )
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = bfill_timeseries(
            df_newer=newer,
            df_older=older,
            freq="B",
            is_prices=True,
        )

    _assert_frame_close(actual, expected)
    pd.testing.assert_frame_equal(older, original_older, check_exact=True)
    pd.testing.assert_frame_equal(newer, original_newer, check_exact=True)

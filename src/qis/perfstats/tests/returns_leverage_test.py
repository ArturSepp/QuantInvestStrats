"""Boundary tests for leverage transformations with financing costs.

The tests use small deterministic pandas objects so that every expected value can
be calculated directly from the documented leverage equations.  For leverage
``L`` and a per-period financing return ``f``, the independently calculated
relationships are::

    levered_return = (1 + L) * return - L * f
    return = (levered_return + L * f) / (1 + L)

The cases below verify that a time-varying financing Series is aligned to the
date index, pandas labels are preserved, unavailable leading financing data is
not backfilled, and zero leverage is an exact identity operation.
"""

import numpy as np
import pandas as pd

from qis.perfstats.returns import delever_returns, lever_returns


# =============================================================================
# Shared deterministic fixtures and independently calculated expectations
# =============================================================================

_DATES = pd.date_range("2024-01-31", periods=4, freq="ME")
_TOLERANCE = 1.0e-12


def _expected_levered_frame() -> pd.DataFrame:
    """Return manually calculated values for leverage one and variable funding."""
    return pd.DataFrame(
        {
            "Asset A": [0.03, -0.03, 0.04, -0.02],
            "Asset B": [-0.05, 0.07, 0.00, -0.08],
        },
        index=_DATES,
    )


def _funding_series() -> pd.Series:
    """Return annual funding observations that forward-fill to 1%, 1%, 2%, 2%."""
    return pd.Series(
        [0.12, 0.24],
        index=pd.DatetimeIndex([_DATES[0], _DATES[2]]),
        name="Annual funding",
    )


def _returns_frame() -> pd.DataFrame:
    """Return the deterministic two-asset monthly return fixture."""
    return pd.DataFrame(
        {
            "Asset A": [0.02, -0.01, 0.03, 0.00],
            "Asset B": [-0.02, 0.04, 0.01, -0.03],
        },
        index=_DATES,
    )


# =============================================================================
# Date-axis alignment and pandas metadata
# =============================================================================

def test_lever_returns_aligns_time_varying_funding_by_dataframe_rows() -> None:
    """Apply each financing observation to every asset on the same date.

    The annual funding Series becomes monthly returns of 1%, 1%, 2%, and 2%.
    With leverage one, the expected result is therefore ``2 * return - funding``.
    Literal expected values make the test independent of the implementation's
    pandas alignment operations.
    """
    returns = _returns_frame()
    funding = _funding_series()
    original_returns = returns.copy(deep=True)
    original_funding = funding.copy(deep=True)

    actual = lever_returns(
        returns=returns,
        leverage=1.0,
        financing_rate=funding,
        periods_per_year=12,
    )

    assert isinstance(actual, pd.DataFrame)
    pd.testing.assert_frame_equal(actual, _expected_levered_frame(), atol=_TOLERANCE)
    pd.testing.assert_frame_equal(returns, original_returns)
    pd.testing.assert_series_equal(funding, original_funding)


def test_delever_returns_aligns_time_varying_funding_by_dataframe_rows() -> None:
    """Recover each asset return using financing aligned by observation date.

    The levered input is supplied from literal expected values instead of being
    produced by ``lever_returns``.  This avoids using one production function as
    the oracle for the other and independently exercises the inverse equation.
    """
    levered_returns = _expected_levered_frame()
    funding = _funding_series()
    original_levered_returns = levered_returns.copy(deep=True)
    original_funding = funding.copy(deep=True)

    actual = delever_returns(
        returns=levered_returns,
        leverage=1.0,
        financing_rate=funding,
        periods_per_year=12,
    )

    assert isinstance(actual, pd.DataFrame)
    pd.testing.assert_frame_equal(actual, _returns_frame(), atol=_TOLERANCE)
    pd.testing.assert_frame_equal(levered_returns, original_levered_returns)
    pd.testing.assert_series_equal(funding, original_funding)


def test_leverage_transforms_preserve_series_name_and_invert_values() -> None:
    """Preserve a return Series name through both leverage transformations."""
    returns = _returns_frame()["Asset A"]
    funding = _funding_series().rename("Different funding name")
    expected_levered = _expected_levered_frame()["Asset A"]
    original_returns = returns.copy(deep=True)
    original_funding = funding.copy(deep=True)

    actual_levered = lever_returns(
        returns=returns,
        leverage=1.0,
        financing_rate=funding,
        periods_per_year=12,
    )

    assert isinstance(actual_levered, pd.Series)
    pd.testing.assert_series_equal(actual_levered, expected_levered, atol=_TOLERANCE)

    actual_delevered = delever_returns(
        returns=actual_levered,
        leverage=1.0,
        financing_rate=funding,
        periods_per_year=12,
    )

    assert isinstance(actual_delevered, pd.Series)
    pd.testing.assert_series_equal(actual_delevered, returns, atol=_TOLERANCE)
    pd.testing.assert_series_equal(returns, original_returns)
    pd.testing.assert_series_equal(funding, original_funding)


# =============================================================================
# Missing financing boundaries and zero-leverage identity
# =============================================================================

def test_leverage_transforms_are_exact_identity_at_zero_leverage() -> None:
    """Ignore unavailable financing when leverage makes its contribution zero.

    Financing begins after the first return.  Mathematically, both transformations
    reduce exactly to the original returns at zero leverage, so a missing financing
    observation must not turn the first return into NaN.  Returned pandas objects
    must also be independent copies rather than aliases of the caller's input.
    """
    returns = _returns_frame()
    funding = pd.Series(
        [0.12, 0.24],
        index=pd.DatetimeIndex([_DATES[1], _DATES[3]]),
        name="Annual funding",
    )

    levered_frame = lever_returns(
        returns=returns,
        leverage=0.0,
        financing_rate=funding,
        periods_per_year=12,
    )
    delevered_frame = delever_returns(
        returns=returns,
        leverage=0.0,
        financing_rate=funding,
        periods_per_year=12,
    )

    assert isinstance(levered_frame, pd.DataFrame)
    assert isinstance(delevered_frame, pd.DataFrame)
    pd.testing.assert_frame_equal(levered_frame, returns, check_exact=True)
    pd.testing.assert_frame_equal(delevered_frame, returns, check_exact=True)
    assert levered_frame is not returns
    assert delevered_frame is not returns

    returns_series = returns["Asset A"]
    levered_series = lever_returns(
        returns=returns_series,
        leverage=0.0,
        financing_rate=funding,
        periods_per_year=12,
    )
    delevered_series = delever_returns(
        returns=returns_series,
        leverage=0.0,
        financing_rate=funding,
        periods_per_year=12,
    )

    assert isinstance(levered_series, pd.Series)
    assert isinstance(delevered_series, pd.Series)
    pd.testing.assert_series_equal(levered_series, returns_series, check_exact=True)
    pd.testing.assert_series_equal(delevered_series, returns_series, check_exact=True)
    assert levered_series is not returns_series
    assert delevered_series is not returns_series


def test_lever_returns_does_not_backfill_leading_unavailable_funding() -> None:
    """Leave a leading result unavailable when no prior financing rate exists."""
    returns = _returns_frame()["Asset A"]
    funding = pd.Series(
        [0.12, 0.24],
        index=pd.DatetimeIndex([_DATES[1], _DATES[3]]),
        name="Annual funding",
    )
    expected = pd.Series(
        [np.nan, -0.03, 0.05, -0.02],
        index=_DATES,
        name="Asset A",
    )

    actual = lever_returns(
        returns=returns,
        leverage=1.0,
        financing_rate=funding,
        periods_per_year=12,
    )

    assert isinstance(actual, pd.Series)
    pd.testing.assert_series_equal(actual, expected, atol=_TOLERANCE)


def test_delever_returns_does_not_backfill_leading_unavailable_funding() -> None:
    """Keep a leading delevered return unavailable without prior financing data."""
    levered_returns = pd.Series(
        [0.04, -0.03, 0.05, -0.02],
        index=_DATES,
        name="Asset A",
    )
    funding = pd.Series(
        [0.12, 0.24],
        index=pd.DatetimeIndex([_DATES[1], _DATES[3]]),
        name="Annual funding",
    )
    expected = pd.Series(
        [np.nan, -0.01, 0.03, 0.00],
        index=_DATES,
        name="Asset A",
    )

    actual = delever_returns(
        returns=levered_returns,
        leverage=1.0,
        financing_rate=funding,
        periods_per_year=12,
    )

    assert isinstance(actual, pd.Series)
    pd.testing.assert_series_equal(actual, expected, atol=_TOLERANCE)


# =============================================================================
# Existing scalar-financing convention
# =============================================================================

def test_scalar_and_constant_series_funding_produce_the_same_values() -> None:
    """Keep scalar financing behavior while adding date-aligned Series support."""
    returns = _returns_frame()
    constant_funding = pd.Series(0.12, index=_DATES, name="Annual funding")
    expected_levered = pd.DataFrame(
        {
            "Asset A": [0.03, -0.03, 0.05, -0.01],
            "Asset B": [-0.05, 0.07, 0.01, -0.07],
        },
        index=_DATES,
    )

    scalar_levered = lever_returns(
        returns=returns,
        leverage=1.0,
        financing_rate=0.12,
        periods_per_year=12,
    )
    series_levered = lever_returns(
        returns=returns,
        leverage=1.0,
        financing_rate=constant_funding,
        periods_per_year=12,
    )

    assert isinstance(scalar_levered, pd.DataFrame)
    assert isinstance(series_levered, pd.DataFrame)
    pd.testing.assert_frame_equal(scalar_levered, expected_levered, atol=_TOLERANCE)
    pd.testing.assert_frame_equal(series_levered, expected_levered, atol=_TOLERANCE)

    scalar_delevered = delever_returns(
        returns=expected_levered,
        leverage=1.0,
        financing_rate=0.12,
        periods_per_year=12,
    )
    series_delevered = delever_returns(
        returns=expected_levered,
        leverage=1.0,
        financing_rate=constant_funding,
        periods_per_year=12,
    )

    assert isinstance(scalar_delevered, pd.DataFrame)
    assert isinstance(series_delevered, pd.DataFrame)
    pd.testing.assert_frame_equal(scalar_delevered, returns, atol=_TOLERANCE)
    pd.testing.assert_frame_equal(series_delevered, returns, atol=_TOLERANCE)

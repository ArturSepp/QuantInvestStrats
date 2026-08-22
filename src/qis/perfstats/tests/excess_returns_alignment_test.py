"""Regression coverage for excess-return alignment and compounded panel output.

The tests separate periodic funding alignment from the later geometric reduction. Short daily
histories make the lagged funding cost and expected returns independently calculable, while
distinct and reordered columns expose any loss of DataFrame shape or positional alignment.
"""

from typing import cast
import numpy as np
import pandas as pd
from numpy.typing import NDArray

# qis
from qis.perfstats.returns import compute_excess_returns, compute_pa_excess_compounded_returns


# =============================================================================
# Shared deterministic fixtures
# =============================================================================

_DATES = pd.date_range('2024-01-01', periods=5, freq='D')
_RATE_DATES = pd.date_range('2023-12-31', periods=6, freq='D')

_ANNUAL_FUNDING_RATE = 0.365
_ASSET_A_RETURNS = (0.000, 0.101, 0.001, 0.001, 0.001)
_ASSET_B_RETURNS = (np.nan, 0.201, 0.001, 0.001, 0.001)

_TOLERANCE = 1e-12


def _constant_daily_funding_rates() -> pd.Series:
    """Create a rate history equivalent to 0.1% per calendar day.

    Returns:
        A new Series containing a constant 36.5% annual funding rate.
    """
    return pd.Series(_ANNUAL_FUNDING_RATE, index=_RATE_DATES, name='risk_free_rate')


def _two_asset_returns() -> pd.DataFrame:
    """Create two distinct funded paths, including a ragged second history.

    Returns:
        A new DataFrame with Asset B starting one observation after Asset A.
    """
    return pd.DataFrame(
        {
            'Asset A': _ASSET_A_RETURNS,
            'Asset B': _ASSET_B_RETURNS,
        },
        index=_DATES,
    )


def _assert_array_close(actual: object, expected: NDArray[np.float64]) -> None:
    """Compare a public result with a shape-sensitive numerical reference.

    Args:
        actual: Value returned by the public excess-return function.
        expected: Independently calculated array with the required output shape.
    """
    assert isinstance(actual, np.ndarray)
    actual_array = cast(NDArray[np.float64], actual)
    assert actual_array.shape == expected.shape
    np.testing.assert_allclose(
        actual_array,
        expected,
        rtol=0.0,
        atol=_TOLERANCE,
        equal_nan=True,
    )


# =============================================================================
# Periodic excess-return alignment
# =============================================================================

def test_compute_excess_returns_aligns_late_rates_without_lookahead() -> None:
    """Align a late funding history and charge only previously observable rates.

    The annual rate is first observed on January 2. With the documented one-observation lag,
    neither January 1 nor January 2 has an available funding cost. From January 3 onward, the
    36.5% annual rate costs exactly 0.1% for each one-day interval. The final return date extends
    past the rate history and therefore uses the last previously observed rate.
    """
    returns = pd.DataFrame(
        {
            'Asset B': [0.01, 0.02, -0.03, 0.04, 0.05],
            'Asset A': [0.02, np.nan, 0.01, -0.02, 0.03],
        },
        index=_DATES,
    )
    rates = pd.Series(
        _ANNUAL_FUNDING_RATE,
        index=pd.date_range('2024-01-02', periods=3, freq='D'),
        name='risk_free_rate',
    )
    original_returns = returns.copy(deep=True)
    original_rates = rates.copy(deep=True)
    expected = pd.DataFrame(
        {
            'Asset B': [np.nan, np.nan, -0.031, 0.039, 0.049],
            'Asset A': [np.nan, np.nan, 0.009, -0.021, 0.029],
        },
        index=_DATES,
    )

    actual = compute_excess_returns(returns=returns, rates_data=rates)

    assert isinstance(actual, pd.DataFrame)
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(returns, original_returns)
    pd.testing.assert_series_equal(rates, original_rates)


def test_compute_excess_returns_matches_series_and_dataframe_paths() -> None:
    """Keep labels, missing observations, and values consistent across pandas shapes.

    Asset A earns 10.1% on January 2 and 0.1% thereafter. Subtracting the independently
    calculated 0.1% daily funding cost produces 10% followed by zeroes. Running the same named
    history as a Series and as a one-column DataFrame must not change its values or metadata.
    """
    series_returns = pd.Series(
        _ASSET_A_RETURNS,
        index=_DATES,
        name='Asset A',
    )
    returns = series_returns.to_frame()
    rates = _constant_daily_funding_rates()
    expected = pd.Series([0.0, 0.10, 0.0, 0.0, 0.0], index=_DATES, name='Asset A')

    frame_result = compute_excess_returns(returns=returns, rates_data=rates)
    series_result = compute_excess_returns(returns=series_returns, rates_data=rates)

    assert isinstance(frame_result, pd.DataFrame)
    assert isinstance(series_result, pd.Series)
    pd.testing.assert_series_equal(frame_result['Asset A'], expected)
    pd.testing.assert_series_equal(series_result, expected)


# =============================================================================
# Compounded panel output
# =============================================================================

def test_compute_pa_excess_compounded_returns_preserves_every_dataframe_column() -> None:
    """Return one independently calculated compounded result per DataFrame column.

    After funding, Asset A compounds from 1.00 to 1.10 and ragged Asset B from 1.00 to 1.20.
    Because the sample is shorter than one year and ``annualize_less_1y`` is false, the function
    should return the exact total excess returns ``[0.10, 0.20]`` instead of only Asset A's value.
    """
    returns = _two_asset_returns()
    original_returns = returns.copy(deep=True)
    rates = _constant_daily_funding_rates()
    expected = np.array([0.10, 0.20], dtype=float)

    actual = compute_pa_excess_compounded_returns(returns=returns, rates_data=rates)

    _assert_array_close(actual, expected)
    pd.testing.assert_frame_equal(returns, original_returns)


def test_compute_pa_excess_compounded_returns_follows_dataframe_column_order() -> None:
    """Associate each compounded result with the corresponding input column position.

    The public function returns an ndarray for a DataFrame, so column position is its alignment
    contract. Reversing the input columns must reverse the two expected values; it must not keep
    returning whichever asset happened to occupy the first position.
    """
    returns = _two_asset_returns()[['Asset B', 'Asset A']]
    rates = _constant_daily_funding_rates()
    expected = np.array([0.20, 0.10], dtype=float)

    actual = compute_pa_excess_compounded_returns(returns=returns, rates_data=rates)

    _assert_array_close(actual, expected)

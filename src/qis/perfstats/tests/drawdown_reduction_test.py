"""Regression coverage for maximum and current drawdown reductions.

Maximum drawdown is the minimum running drawdown over an observed price history, while current
drawdown is the final running drawdown after the established forward-fill convention. All-missing
histories have neither value, so both results are NaN and must not emit a reduction warning. The
short deterministic paths below keep every expected reduction independently calculable.
"""

import warnings
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# qis
from qis.perfstats.perf_stats import compute_max_current_drawdown


# =============================================================================
# Shared deterministic fixtures
# =============================================================================

_DATES = pd.date_range('2024-01-01', periods=4, freq='D')

_TOLERANCE = 1e-12


def _mixed_prices() -> pd.DataFrame:
    """Create finite, missing, ragged, trailing-gap, and constant price histories.

    Returns:
        New price DataFrame whose columns exercise distinct drawdown reductions.
    """
    return pd.DataFrame(
        {
            'recovered': (100.0, 80.0, 100.0, 90.0),
            'missing': (np.nan, np.nan, np.nan, np.nan),
            'ragged': (np.nan, 200.0, 100.0, 150.0),
            'trailing_gap': (100.0, 80.0, np.nan, np.nan),
            'constant': (5.0, 5.0, 5.0, 5.0),
        },
        index=_DATES,
    )


def _assert_array_close(actual: object, expected: NDArray[np.float64]) -> None:
    """Compare a public result with a shape-sensitive numerical reference.

    Args:
        actual: Value returned by ``compute_max_current_drawdown``.
        expected: Independently calculated array in input-column order.
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
# Missing-history reductions
# =============================================================================

def test_compute_max_current_drawdown_all_nan_series_is_warning_free() -> None:
    """Return scalar NaNs without warning when no price has ever been observed.

    An all-missing Series has no running peak, historical drawdown, or final drawdown. Both public
    results are therefore scalar NaNs. Treating ``RuntimeWarning`` as an error ensures an all-NaN
    reduction cannot silently reappear while the original input verifies caller ownership.
    """
    prices = pd.Series(np.nan, index=_DATES, name='missing')
    original_prices = prices.copy(deep=True)

    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        maximum_drawdown, current_drawdown = compute_max_current_drawdown(prices=prices)

    assert isinstance(maximum_drawdown, float)
    assert isinstance(current_drawdown, float)
    assert np.isnan(maximum_drawdown)
    assert np.isnan(current_drawdown)
    pd.testing.assert_series_equal(prices, original_prices)


def test_compute_max_current_drawdown_reduces_mixed_dataframe_columns() -> None:
    """Preserve finite and missing reductions when one column is entirely NaN.

    The recovered path has maximum/current drawdowns of -20%/-10%; the ragged path has
    -50%/-25%; the trailing gap retains -20% for both values; and the constant path remains at
    zero. The missing column contributes NaN in the same column position without warning.
    """
    prices = _mixed_prices()
    original_prices = prices.copy(deep=True)
    expected_maximum = np.asarray((-0.20, np.nan, -0.50, -0.20, 0.0), dtype=float)
    expected_current = np.asarray((-0.10, np.nan, -0.25, -0.20, 0.0), dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        maximum_drawdowns, current_drawdowns = compute_max_current_drawdown(prices=prices)

    _assert_array_close(maximum_drawdowns, expected_maximum)
    _assert_array_close(current_drawdowns, expected_current)
    pd.testing.assert_frame_equal(prices, original_prices)


# =============================================================================
# Shape consistency
# =============================================================================

def test_compute_max_current_drawdown_matches_series_and_dataframe_values() -> None:
    """Keep finite Series and one-column DataFrame values identical across return shapes.

    Prices ``100, 80, 100, 90`` produce running drawdowns ``0%, -20%, 0%, -10%`` directly. The
    Series contract returns scalar floats and the one-column DataFrame contract returns one-element
    arrays, but both must report the same -20% maximum and -10% current drawdown.
    """
    prices = _mixed_prices()['recovered']

    series_maximum, series_current = compute_max_current_drawdown(prices=prices)
    frame_maximum, frame_current = compute_max_current_drawdown(prices=prices.to_frame())

    assert isinstance(series_maximum, float)
    assert isinstance(series_current, float)
    np.testing.assert_allclose(series_maximum, -0.20, rtol=0.0, atol=_TOLERANCE)
    np.testing.assert_allclose(series_current, -0.10, rtol=0.0, atol=_TOLERANCE)
    _assert_array_close(frame_maximum, np.asarray((-0.20,), dtype=float))
    _assert_array_close(frame_current, np.asarray((-0.10,), dtype=float))

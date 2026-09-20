"""Nullable endpoint regressions for the public total-return reduction.

``compute_total_return`` selects each history's first and last finite observations. Pandas
nullable ``Float64`` inputs must therefore match ordinary ``float64`` inputs across Series and
mixed DataFrame boundaries without mutating the caller-owned price history.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from qis.perfstats.returns import compute_total_return


_TOLERANCE = 1.0e-12


@pytest.mark.parametrize("storage", ["float64", "Float64"])
@pytest.mark.parametrize(
    "values",
    [
        (np.nan, 100.0, 110.0),
        (100.0, 110.0, np.nan),
    ],
    ids=["leading-missing", "trailing-missing"],
)
def test_compute_total_return_supports_nullable_series_endpoints(
    storage: str,
    values: tuple[float, float, float],
) -> None:
    """Return 10% from either missing endpoint representation without changing the Series."""
    prices = pd.Series(
        values,
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        dtype=storage,
        name="Asset",
    )
    original = prices.copy(deep=True)

    with pytest.warns(UserWarning) as recorded:
        actual = compute_total_return(prices)

    assert np.isclose(float(actual), 0.10, rtol=0.0, atol=_TOLERANCE)
    assert len(recorded) == 1
    assert not any(issubclass(item.category, RuntimeWarning) for item in recorded)
    pd.testing.assert_series_equal(prices, original, check_exact=True)


@pytest.mark.parametrize("storage", ["float64", "Float64"])
def test_compute_total_return_supports_nullable_mixed_panel(storage: str) -> None:
    """Match literal per-column returns across every material missing-data state."""
    prices = pd.DataFrame(
        {
            "Complete": (120.0, 100.0, 110.0, 105.0),
            "Leading": (120.0, np.nan, 110.0, 100.0),
            "Interior": (110.0, 100.0, np.nan, 105.0),
            "Trailing": (np.nan, 100.0, 110.0, 105.0),
            "All missing": (np.nan, np.nan, np.nan, np.nan),
        },
        index=pd.to_datetime(["2024-01-04", "2024-01-01", "2024-01-03", "2024-01-02"]),
        dtype=storage,
    )
    original = prices.copy(deep=True)
    expected = np.array([0.20, 0.20, 0.10, 0.10, np.nan])

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        actual = compute_total_return(prices)

    assert isinstance(actual, np.ndarray)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=_TOLERANCE, equal_nan=True)
    assert len(recorded) == 3
    assert all(issubclass(item.category, UserWarning) for item in recorded)
    assert not any(issubclass(item.category, RuntimeWarning) for item in recorded)
    pd.testing.assert_frame_equal(prices, original, check_exact=True)

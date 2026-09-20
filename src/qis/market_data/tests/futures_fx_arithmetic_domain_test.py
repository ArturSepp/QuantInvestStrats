"""Arithmetic futures FX returns preserve losses outside the logarithmic domain."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qis.market_data.fx_hedging import compute_futures_fx_adjusted_returns


@pytest.mark.filterwarnings("error")
def test_compute_futures_fx_adjusted_returns_preserves_large_arithmetic_losses() -> None:
    """Simple output returns exact and below-minus-one futures payoffs without warnings."""
    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    prices = pd.DataFrame(
        {
            "EXACT_MINUS_ONE": pd.Series([100.0, 50.0], index=dates),
            "BELOW_MINUS_ONE": pd.Series([100.0, 10.0], index=dates, dtype="Float64"),
        }
    )
    fx_spots = pd.DataFrame(
        {
            "EXACT_MINUS_ONE": pd.Series([1.0, 2.0], index=dates),
            "BELOW_MINUS_ONE": pd.Series([1.0, 2.0], index=dates, dtype="Float64"),
        }
    )
    original_prices = prices.copy(deep=True)
    original_fx_spots = fx_spots.copy(deep=True)

    actual = compute_futures_fx_adjusted_returns(prices, fx_spots, is_log_returns=False)

    # Futures translate only P&L: (-50% * 2) = -100% and (-90% * 2) = -180%.
    expected = pd.DataFrame(
        {
            "EXACT_MINUS_ONE": pd.Series([np.nan, -1.0], index=dates),
            "BELOW_MINUS_ONE": pd.Series([pd.NA, -1.8], index=dates, dtype="Float64"),
        }
    )
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(prices, original_prices)
    pd.testing.assert_frame_equal(fx_spots, original_fx_spots)

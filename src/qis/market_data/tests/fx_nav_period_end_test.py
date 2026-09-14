"""Period-end sampling regressions for public FX NAV methods.

Reporting boundaries can fall on dates absent from a business-day market. These tests require
completed periods to use their final available NAV without including an incomplete terminal period.
"""

from __future__ import annotations

import pandas as pd
import pytest

from qis.market_data import FxRatesData


def _make_market(
    *, nullable: bool, timezone: str | None, local_rate: float
) -> tuple[FxRatesData, pd.DataFrame, pd.DataFrame]:
    """Return a deterministic market spanning an unobserved weekend month end."""
    dates = pd.bdate_range("2024-07-31", "2024-09-02", name="date", tz=timezone)
    eur_spot = pd.Series(1.0, index=dates)
    eur_spot.loc["2024-08-29":] = 1.2
    spots = pd.DataFrame({"USD": 1.0, "EUR": eur_spot}, index=dates)
    rates = pd.DataFrame({"USD": 0.0, "EUR": local_rate}, index=dates)
    if nullable:
        spots = spots.astype("Float64")
        rates = rates.astype("Float64")
    return FxRatesData(spots.copy(), rates.copy()), spots, rates


@pytest.mark.parametrize("nullable", [False, True], ids=["ordinary", "nullable"])
@pytest.mark.parametrize("timezone", [None, "UTC"], ids=["naive", "timezone-aware"])
def test_get_fx_total_return_nav_uses_final_available_completed_period_value(
    nullable: bool, timezone: str | None
) -> None:
    """An absent month-end uses the current month's final available total-return NAV."""
    data, _, _ = _make_market(nullable=nullable, timezone=timezone, local_rate=0.0)
    original_spots = data.fx_spots.copy(deep=True)
    original_rates = data.domestic_rates.copy(deep=True)

    actual = data.get_fx_total_return_nav("EUR", "USD", freq="ME")

    expected_index = pd.date_range("2024-07-31", periods=2, freq="ME", name="date", tz=timezone)
    expected = pd.Series([1.0, 1.2], index=expected_index, name="EUR-USD")
    pd.testing.assert_series_equal(actual, expected, check_dtype=False)
    assert actual.dtype == data.get_fx_total_return_nav("EUR", "USD").dtype
    pd.testing.assert_frame_equal(data.fx_spots, original_spots)
    pd.testing.assert_frame_equal(data.domestic_rates, original_rates)


def test_get_carry_fx_return_nav_uses_final_available_completed_period_value() -> None:
    """An absent month-end uses the current month's final available carry NAV."""
    data, _, _ = _make_market(nullable=False, timezone=None, local_rate=2.52)

    actual = data.get_carry_fx_return_nav("EUR", "USD", is_normalise_by_spot_vol=False, freq="ME")

    expected_index = pd.date_range("2024-07-31", periods=2, freq="ME", name="date")
    # The first return is the synthetic zero; the next 22 business-day returns earn exactly 1%.
    expected = pd.Series([1.0, 1.01**22], index=expected_index, name="EUR-USD")
    pd.testing.assert_series_equal(actual, expected, rtol=1e-13, atol=1e-14)

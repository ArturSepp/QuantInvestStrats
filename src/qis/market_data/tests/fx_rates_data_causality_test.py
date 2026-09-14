"""Causal construction regressions for FX spot and domestic-rate panels.

Spot and rate inputs may arrive on different calendars or in a different physical row order.
Construction must select only observations available on or before each stored spot date because
every downstream cross, carry, and currency-conversion calculation consumes these aligned panels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qis.market_data import FxRatesData


@pytest.mark.parametrize("nullable", [False, True], ids=["numpy-missing", "nullable-missing"])
@pytest.mark.parametrize(
    "source_order",
    [(2, 0, 1), (1, 2, 0)],
    ids=["descending-source", "unsorted-source"],
)
def test_fx_rates_data_sorts_spots_before_forward_fill(
    nullable: bool,
    source_order: tuple[int, ...],
) -> None:
    """A later spot quote cannot populate an earlier missing date through row-order fill."""
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    spot_values = (np.nan, 1.10, 1.20)
    source_dates = pd.DatetimeIndex([dates[position] for position in source_order])
    spots = pd.DataFrame(
        {
            "EUR": [spot_values[position] for position in source_order],
            "USD": [1.0] * len(source_order),
        },
        index=source_dates,
    )
    rates = pd.DataFrame({"EUR": [0.01, 0.01, 0.01], "USD": [0.04, 0.04, 0.04]}, index=dates)
    if nullable:
        spots = spots.astype("Float64")
        rates = rates.astype("Float64")
    original_spots = spots.copy(deep=True)
    original_rates = rates.copy(deep=True)

    actual = FxRatesData(fx_spots=spots, domestic_rates=rates)

    expected_index = pd.DatetimeIndex(dates.to_numpy())
    expected_spots = pd.DataFrame(
        {"EUR": [np.nan, 1.10, 1.20], "USD": [1.0, 1.0, 1.0]}, index=expected_index
    )
    if nullable:
        expected_spots = expected_spots.astype("Float64")
    pd.testing.assert_frame_equal(actual.fx_spots, expected_spots)
    pd.testing.assert_frame_equal(spots, original_spots)
    pd.testing.assert_frame_equal(rates, original_rates)


@pytest.mark.parametrize("nullable", [False, True], ids=["numpy-missing", "nullable-missing"])
def test_fx_rates_data_retains_off_grid_rates_for_public_carry(nullable: bool) -> None:
    """An off-grid rate update remains available to later spot dates and public CIP carry."""
    spot_dates = pd.DatetimeIndex(
        ["2024-01-02", "2024-01-03", "2024-01-06", "2024-01-07"], tz="UTC"
    )
    rate_dates = pd.DatetimeIndex(["2024-01-05", "2024-01-03"], tz="UTC")
    spots = pd.DataFrame({"USD": 1.0, "EUR": 1.10}, index=spot_dates)
    rates = pd.DataFrame({"USD": [0.05, 0.04], "EUR": [0.01, 0.01]}, index=rate_dates)
    if nullable:
        spots = spots.astype("Float64")
        rates = rates.astype("Float64")
    original_spots = spots.copy(deep=True)
    original_rates = rates.copy(deep=True)

    actual = FxRatesData(fx_spots=spots, domestic_rates=rates)

    expected_rates = pd.DataFrame(
        {
            "USD": [np.nan, 0.04, 0.05, 0.05],
            "EUR": [np.nan, 0.01, 0.01, 0.01],
        },
        index=spot_dates,
    )
    if nullable:
        expected_rates = expected_rates.astype("Float64")
    periods_per_year = 252.0
    initial_carry = (1.0 + 0.04 / periods_per_year) / (1.0 + 0.01 / periods_per_year) - 1.0
    updated_carry = (1.0 + 0.05 / periods_per_year) / (1.0 + 0.01 / periods_per_year) - 1.0
    expected_carry = pd.Series(
        [np.nan, initial_carry, updated_carry, updated_carry],
        index=spot_dates,
        name="USD-EUR",
    )
    if nullable:
        expected_carry = expected_carry.astype("Float64")
    actual_carry = actual.get_forward_rate_for_local_ccy("USD", "EUR", freq="B")

    pd.testing.assert_series_equal(actual_carry, expected_carry)
    pd.testing.assert_frame_equal(actual.domestic_rates, expected_rates)
    pd.testing.assert_frame_equal(spots, original_spots)
    pd.testing.assert_frame_equal(rates, original_rates)

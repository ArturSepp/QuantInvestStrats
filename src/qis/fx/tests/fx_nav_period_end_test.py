"""Period-end sampling regressions for public FX NAV methods.

Reporting boundaries can fall on dates absent from a business-day market. These tests require
completed periods to use their final available NAV without including an incomplete terminal period.
"""

from __future__ import annotations

import numpy as np
import qis
import pandas as pd
import pytest

from qis.fx import FxRatesData


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
def test_normalized_spot_index_uses_final_available_completed_period_value(
    nullable: bool, timezone: str | None
) -> None:
    """An absent month-end uses the current month's final available total-return NAV."""
    data, _, _ = _make_market(nullable=nullable, timezone=timezone, local_rate=0.0)
    original_spots = data.fx_spots.copy(deep=True)
    original_rates = data.domestic_rates.copy(deep=True)

    result = qis.compute_fx_returns(
        data, qis.FxCurrencyReturnsRequest(("EUR",), "USD", frequency="B")
    )["B"]
    daily = result.to_index(
        qis.FxIndexNormalization(
            1.0, 1.0, "capital_proportional", "B", "each_interval", "as_returned", "stop"
        )
    )
    actual = qis.df_asfreq(daily, freq="ME").EUR

    expected_index = pd.date_range("2024-07-31", periods=2, freq="ME", name="date", tz=timezone)
    expected = pd.Series([1.0, 1.2], index=expected_index, name="EUR")
    pd.testing.assert_series_equal(actual, expected, check_dtype=False)
    assert actual.dtype == daily.EUR.dtype
    pd.testing.assert_frame_equal(data.fx_spots, original_spots)
    pd.testing.assert_frame_equal(data.domestic_rates, original_rates)


def test_normalized_forward_index_uses_final_available_completed_period_value() -> None:
    """A missing month-end uses the final available explicitly normalized forward index."""
    data, _, _ = _make_market(nullable=False, timezone=None, local_rate=2.52)

    data.fx_spots["EUR"] = 1.0
    result = qis.compute_fx_returns(
        data,
        qis.FxCurrencyReturnsRequest(("EUR",), "USD", qis.FxReturnType.FX_FORWARD, frequency="B"),
    )["B"]
    daily = result.to_index(
        qis.FxIndexNormalization(
            1.0, 1.0, "capital_proportional", "B", "each_interval", "as_returned", "stop"
        )
    )
    actual = qis.df_asfreq(daily, freq="ME").EUR
    expected_index = pd.date_range("2024-07-31", periods=2, freq="ME", name="date")
    # A long-local forward pays 1 - F/S = c_local / (1 + c_local) at flat spot.
    # Business-day resets: 18 one-day and four three-day intervals through 30 August.
    one_day = 2.52 / 365.0
    three_days = 3 * one_day
    expected = pd.Series(
        [1.0, (1 + one_day / (1 + one_day)) ** 18 * (1 + three_days / (1 + three_days)) ** 4],
        index=expected_index,
        name="EUR",
    )
    pd.testing.assert_series_equal(actual, expected, rtol=1e-13, atol=1e-14)
    assert actual.index[-1] < daily.index[-1]  # Incomplete September is excluded.
    np.testing.assert_allclose(result.values.EUR.iloc[1], one_day / (1 + one_day))

"""Causal alignment regressions for public FX spot and return helpers."""

from __future__ import annotations

from collections.abc import Callable
import math

import numpy as np
import pandas as pd
import pytest

from qis.market_data.fx_hedging import (
    compute_cash_fx_adjusted_returns,
    compute_futures_fx_adjusted_returns,
    get_aligned_fx_spots,
)


@pytest.mark.parametrize("nullable", [False, True], ids=["numpy-missing", "nullable-missing"])
def test_fx_spot_alignment_uses_only_current_and_prior_observations(nullable: bool) -> None:
    """Mixed FX panels preserve causal gaps, controls, labels, and caller-owned inputs."""
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    prices = pd.DataFrame(
        {
            "EUR_ASSET": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "GBP_ASSET": [200.0, 201.0, np.nan, 203.0, 204.0, 205.0],
            "JPY_ASSET": [300.0, 301.0, 302.0, 303.0, 304.0, 305.0],
            "USD_ASSET": [50.0, 51.0, 52.0, 53.0, np.nan, 55.0],
        },
        index=dates,
    )
    fx_prices = pd.DataFrame(
        {
            "EUR": [1.10, 1.20, 1.30],
            "GBP": [1.30, np.nan, 1.40],
            "JPY": [np.nan, np.nan, np.nan],
        },
        index=dates[[1, 3, 4]],
    )
    if nullable:
        prices = prices.astype("Float64")
        fx_prices = fx_prices.astype("Float64")
    asset_ccy_map = {
        "EUR_ASSET": "EUR",
        "GBP_ASSET": "GBP",
        "JPY_ASSET": "JPY",
        "USD_ASSET": "USD",
    }
    original_prices = prices.copy(deep=True)
    original_fx_prices = fx_prices.copy(deep=True)

    actual = get_aligned_fx_spots(prices, asset_ccy_map, fx_prices)

    # This explicit table is an as-of oracle independent of the production fill operations.
    expected = pd.DataFrame(
        {
            "EUR_ASSET": [np.nan, 1.10, 1.10, 1.20, 1.30, 1.30],
            "GBP_ASSET": [np.nan, 1.30, np.nan, 1.30, 1.40, 1.40],
            "JPY_ASSET": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            "USD_ASSET": [1.0, 1.0, 1.0, 1.0, np.nan, 1.0],
        },
        index=dates,
    )
    if nullable:
        actual = actual.astype("Float64")
        expected = expected.astype("Float64")
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    pd.testing.assert_frame_equal(prices, original_prices)
    pd.testing.assert_frame_equal(fx_prices, original_fx_prices)


@pytest.mark.parametrize(
    "source_order",
    [(0, 1, 2), (2, 1, 0), (1, 2, 0)],
    ids=["ascending-source", "descending-source", "unsorted-source"],
)
@pytest.mark.parametrize(
    "target_order",
    [(0, 1, 2, 3), (3, 2, 1, 0), (2, 0, 3, 1)],
    ids=["ascending-target", "descending-target", "unsorted-target"],
)
def test_fx_spot_alignment_is_independent_of_input_row_order(
    source_order: tuple[int, ...],
    target_order: tuple[int, ...],
) -> None:
    """Source and target row order cannot change the timestamp-based as-of result."""
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    source_values = (1.10, 1.15, 1.20)
    price_values = (100.0, 101.0, 102.0, 103.0)
    expected_values = (np.nan, 1.10, 1.15, 1.20)
    source_dates = pd.DatetimeIndex([dates[position + 1] for position in source_order])
    target_dates = pd.DatetimeIndex([dates[position] for position in target_order])
    fx_prices = pd.DataFrame(
        {"EUR": [source_values[position] for position in source_order]}, index=source_dates
    )
    prices = pd.DataFrame(
        {"ASSET": [price_values[position] for position in target_order]}, index=target_dates
    )

    actual = get_aligned_fx_spots(prices, {"ASSET": "EUR"}, fx_prices)

    expected = pd.DataFrame(
        {"ASSET": [expected_values[position] for position in target_order]}, index=target_dates
    )
    pd.testing.assert_frame_equal(actual, expected)


FxReturnConverter = Callable[[pd.DataFrame, pd.DataFrame, int, bool], pd.DataFrame]


@pytest.mark.parametrize(
    ("converter", "includes_fx_notional"),
    [
        (compute_cash_fx_adjusted_returns, True),
        (compute_futures_fx_adjusted_returns, False),
    ],
    ids=["cash", "futures"],
)
@pytest.mark.parametrize("is_log_returns", [False, True], ids=["simple", "log"])
def test_fx_adjusted_returns_require_two_causally_available_spots(
    converter: FxReturnConverter,
    includes_fx_notional: bool,
    is_log_returns: bool,
) -> None:
    """Cash and futures returns stay missing until both endpoint spots are observable."""
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    prices = pd.DataFrame({"ASSET": [100.0, 101.0, 102.0, 103.0]}, index=dates)
    fx_prices = pd.DataFrame({"EUR": [1.10, 1.20]}, index=dates[[1, 3]])
    aligned_spots = get_aligned_fx_spots(prices, {"ASSET": "EUR"}, fx_prices)

    actual = converter(prices, aligned_spots, 1, is_log_returns)

    # Derive the observable returns directly from endpoint prices and spots, without the helper.
    expected_d2 = 102.0 / 101.0 - 1.0
    local_d3 = 103.0 / 102.0 - 1.0
    fx_d3 = 1.20 / 1.10 - 1.0
    expected_d3 = local_d3 * (1.0 + fx_d3)
    if includes_fx_notional:
        expected_d3 += fx_d3
    if is_log_returns:
        expected_values = [np.nan, np.nan, math.log1p(expected_d2), math.log1p(expected_d3)]
    else:
        expected_values = [np.nan, np.nan, expected_d2, expected_d3]
    expected = pd.DataFrame({"ASSET": expected_values}, index=dates)
    pd.testing.assert_frame_equal(actual, expected)

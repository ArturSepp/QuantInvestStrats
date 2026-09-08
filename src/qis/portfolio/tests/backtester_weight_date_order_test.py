"""Regression tests for chronological dated-weight scheduling in the portfolio backtester.

The backtester maps decision dates onto the price grid and then consumes one target row per
trade. Permuting a dated schedule must not change which target is applied at each date, including
when execution is delayed by an observation. These fixtures make the held-unit path calculable by
hand and also protect the existing pre-price validation boundary.
"""

import pandas as pd
import pytest

import qis


def _prices() -> pd.DataFrame:
    """Return a two-asset panel with a hand-checkable held-unit path."""
    dates = pd.bdate_range("2024-01-02", periods=6)
    return pd.DataFrame(
        {
            "A": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
            "B": [100.0] * len(dates),
        },
        index=dates,
    )


def _weight_schedules(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return equivalent chronological and deliberately permuted target schedules."""
    chronological = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
        index=prices.index[[0, 2, 4]],
        columns=prices.columns,
    )
    return chronological, chronological.iloc[[2, 0, 1]]


@pytest.mark.parametrize(
    ("lag", "trade_positions", "expected_nav", "expected_units"),
    [
        pytest.param(
            0,
            [0, 2, 4],
            [100.0, 110.0, 120.0, 120.0, 120.0, 124.28571428571428],
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.2],
                [0.0, 1.2],
                [0.42857142857142855, 0.6],
                [0.42857142857142855, 0.6],
            ],
            id="same-observation",
        ),
        pytest.param(
            1,
            [1, 3, 5],
            [
                100.0,
                100.0,
                109.0909090909091,
                118.18181818181817,
                118.18181818181817,
                118.18181818181816,
            ],
            [
                [0.0, 0.0],
                [0.9090909090909091, 0.0],
                [0.9090909090909091, 0.0],
                [0.0, 1.1818181818181817],
                [0.0, 1.1818181818181817],
                [0.3939393939393939, 0.5909090909090908],
            ],
            id="next-observation",
        ),
    ],
)
def test_backtest_model_portfolio_normalizes_dated_weight_order(
    lag: int,
    trade_positions: list[int],
    expected_nav: list[float],
    expected_units: list[list[float]],
) -> None:
    """A row permutation preserves the chronological target and held-unit paths."""
    prices = _prices()
    chronological, permuted = _weight_schedules(prices)
    caller_weights = permuted.copy()

    result = qis.backtest_model_portfolio(
        prices=prices,
        weights=permuted,
        weight_implementation_lag=lag,
    )

    # These values follow directly from nav_t * target_t / price_t at each dated trade.
    expected_nav_series = pd.Series(expected_nav, index=prices.index, name="Portfolio")
    expected_units_frame = pd.DataFrame(expected_units, index=prices.index, columns=prices.columns)
    expected_realised_weights = expected_units_frame.multiply(prices).divide(
        expected_nav_series,
        axis=0,
    )
    expected_rebalancing = pd.Series(False, index=prices.index)
    expected_rebalancing.iloc[trade_positions] = True

    pd.testing.assert_series_equal(result.nav, expected_nav_series)
    pd.testing.assert_frame_equal(result.units, expected_units_frame)
    pd.testing.assert_frame_equal(result.weights, expected_realised_weights)
    pd.testing.assert_series_equal(result.is_rebalancing, expected_rebalancing)

    assert isinstance(result.input_weights, pd.DataFrame)
    pd.testing.assert_frame_equal(result.input_weights, chronological)
    target_turnover = result.get_turnover(
        is_unit_based_traded_volume=False,
        roll_period=None,
        add_total=False,
    )
    assert isinstance(target_turnover, pd.DataFrame)
    pd.testing.assert_frame_equal(target_turnover, chronological.diff().abs())

    # Normalizing the local schedule must not reorder the caller-owned DataFrame.
    pd.testing.assert_frame_equal(permuted, caller_weights)


def test_backtest_model_portfolio_checks_earliest_weight_date_after_normalizing() -> None:
    """A pre-price target cannot evade validation by appearing after a later row."""
    prices = _prices()
    weights = pd.DataFrame(
        [[0.0, 1.0], [1.0, 0.0]],
        index=[prices.index[2], prices.index[0] - pd.Timedelta(days=1)],
        columns=prices.columns,
    )
    caller_weights = weights.copy()

    with pytest.raises(ValueError, match="price dates .* are after weights start date"):
        qis.backtest_model_portfolio(prices=prices, weights=weights)

    pd.testing.assert_frame_equal(weights, caller_weights)

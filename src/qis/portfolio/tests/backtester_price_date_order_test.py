"""Regression tests for chronological portfolio-backtester input handling.

The backtester is a stateful recursion over its price rows, so equivalent timestamped mappings
must first share one chronological grid. These tests cover the held-unit path, dated decisions,
calendar validation, missing-history classification, and every dated financial companion aligned
by the public wrapper.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import qis


DATES = pd.DatetimeIndex(
    ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
    tz="America/New_York",
    name="date",
)
PERMUTATION = [2, 0, 3, 1]


def _prices() -> pd.DataFrame:
    """Return a hand-checkable two-asset price path on a named timezone-aware grid."""
    return pd.DataFrame(
        {"A": [100.0, 110.0, 121.0, 133.1], "B": [100.0] * len(DATES)},
        index=DATES,
    )


def _permuted_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame rows in a deterministic nonchronological order."""
    return data.iloc[PERMUTATION, :]


def _permuted_series(data: pd.Series) -> pd.Series:
    """Return Series rows in a deterministic nonchronological order."""
    return data.iloc[PERMUTATION]


def test_backtest_model_portfolio_normalizes_fixed_weight_price_order() -> None:
    """A price-row permutation preserves the chronological one-unit economic path."""
    chronological = pd.DataFrame({"A": [100.0, 110.0, 121.0, 133.1]}, index=DATES)
    prices = _permuted_frame(chronological)
    caller_prices = prices.copy()

    result = qis.backtest_model_portfolio(
        prices=prices,
        weights={"A": 1.0},
        rebalancing_freq="YE",
        is_rebalanced_at_first_date=True,
    )

    # Investing the initial 100 at the first chronological price of 100 buys exactly one unit.
    expected_nav = pd.Series(
        [100.0, 110.0, 121.0, 133.1],
        index=DATES,
        name="Portfolio",
    )
    expected_units = pd.DataFrame(1.0, index=DATES, columns=["A"])
    expected_rebalancing = pd.Series([True, False, False, False], index=DATES)
    pd.testing.assert_series_equal(result.nav, expected_nav)
    pd.testing.assert_frame_equal(result.units, expected_units)
    pd.testing.assert_frame_equal(result.weights, expected_units)
    pd.testing.assert_series_equal(result.is_rebalancing, expected_rebalancing)
    pd.testing.assert_frame_equal(result.prices, chronological)
    pd.testing.assert_frame_equal(prices, caller_prices)


def test_backtest_model_portfolio_does_not_revise_a_chronological_prefix() -> None:
    """Appending a later price cannot change an earlier normalized portfolio path."""
    full_prices = _permuted_frame(pd.DataFrame({"A": [100.0, 110.0, 121.0, 133.1]}, index=DATES))
    prefix_dates = pd.DatetimeIndex(DATES[:3])
    prefix_prices = pd.DataFrame(
        {"A": [110.0, 100.0, 121.0]},
        index=pd.DatetimeIndex(
            [prefix_dates[1], prefix_dates[0], prefix_dates[2]], name=DATES.name
        ),
    )

    full_result = qis.backtest_model_portfolio(
        prices=full_prices,
        weights={"A": 1.0},
        rebalancing_freq="YE",
        is_rebalanced_at_first_date=True,
    )
    prefix_result = qis.backtest_model_portfolio(
        prices=prefix_prices,
        weights={"A": 1.0},
        rebalancing_freq="YE",
        is_rebalanced_at_first_date=True,
    )

    pd.testing.assert_series_equal(full_result.nav.head(3), prefix_result.nav)
    pd.testing.assert_frame_equal(full_result.units.head(3), prefix_result.units)


@pytest.mark.parametrize(
    ("lag", "decision_dates", "trade_positions", "expected_nav", "expected_units"),
    [
        pytest.param(
            0,
            ("2024-01-02", "2024-01-03 12:00"),
            (0, 2),
            [100.0, 110.0, 121.0, 121.0],
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.21], [0.0, 1.21]],
            id="same-observation-off-grid",
        ),
        pytest.param(
            1,
            ("2024-01-02", "2024-01-03 12:00"),
            (1, 3),
            [100.0, 100.0, 110.0, 121.0],
            [
                [0.0, 0.0],
                [100.0 / 110.0, 0.0],
                [100.0 / 110.0, 0.0],
                [0.0, 1.21],
            ],
            id="next-observation-off-grid",
        ),
    ],
)
def test_backtest_model_portfolio_normalizes_prices_before_mapping_dated_weights(
    lag: int,
    decision_dates: tuple[str, str],
    trade_positions: tuple[int, int],
    expected_nav: list[float],
    expected_units: list[list[float]],
) -> None:
    """Exact and off-grid decisions use the sorted price grid before implementation lag."""
    chronological = _prices()
    prices = _permuted_frame(chronological)
    dates = pd.DatetimeIndex(decision_dates, tz=DATES.tz)
    weights = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=dates, columns=prices.columns)
    caller_prices = prices.copy()
    caller_weights = weights.copy()

    result = qis.backtest_model_portfolio(
        prices=prices,
        weights=weights,
        weight_implementation_lag=lag,
    )

    expected_nav_series = pd.Series(expected_nav, index=DATES, name="Portfolio")
    expected_units_frame = pd.DataFrame(
        expected_units,
        index=DATES,
        columns=chronological.columns,
    )
    expected_weights = expected_units_frame.multiply(chronological).divide(
        expected_nav_series,
        axis=0,
    )
    expected_rebalancing = pd.Series(False, index=DATES)
    expected_rebalancing.iloc[list(trade_positions)] = True
    pd.testing.assert_series_equal(result.nav, expected_nav_series)
    pd.testing.assert_frame_equal(result.units, expected_units_frame)
    pd.testing.assert_frame_equal(result.weights, expected_weights)
    pd.testing.assert_series_equal(result.is_rebalancing, expected_rebalancing)
    pd.testing.assert_frame_equal(prices, caller_prices)
    pd.testing.assert_frame_equal(weights, caller_weights)


def test_backtest_model_portfolio_aligns_permuted_dated_companions() -> None:
    """Costs, funding, and carry retain their dated meaning on the normalized price grid."""
    chronological = pd.DataFrame({"A": [100.0] * len(DATES)}, index=DATES)
    prices = _permuted_frame(chronological)
    funding = pd.Series([0.0, 0.365, 0.73, 1.095], index=DATES, name="funding")
    carry = pd.DataFrame({"A": [0.0, 0.365, 0.73, 1.095]}, index=DATES)
    costs = pd.DataFrame({"A": [0.01, 0.02, 0.03, 0.04]}, index=DATES)
    permuted_funding = _permuted_series(funding)
    permuted_carry = _permuted_frame(carry)
    permuted_costs = _permuted_frame(costs)
    caller_prices = prices.copy()
    caller_funding = permuted_funding.copy()
    caller_carry = permuted_carry.copy()
    caller_costs = permuted_costs.copy()

    result = qis.backtest_model_portfolio(
        prices=prices,
        weights={"A": 0.5},
        rebalancing_freq="YE",
        is_rebalanced_at_first_date=True,
        funding_rate=permuted_funding,
        instruments_carry=permuted_carry,
        rebalancing_costs=permuted_costs,
    )

    # Opening costs leave 49.5 cash. Each later cash balance earns its dated funding rate,
    # then receives carry on the unchanged 50 notional before it is added back to that notional.
    expected_nav = pd.Series(
        [99.5, 99.5995, 99.798699, 100.098095097],
        index=DATES,
        name="Portfolio",
    )
    expected_costs = pd.DataFrame(
        {"A": [0.5, 0.0, 0.0, 0.0]},
        index=DATES,
    )
    pd.testing.assert_series_equal(result.nav, expected_nav)
    pd.testing.assert_frame_equal(result.realized_costs, expected_costs)
    pd.testing.assert_frame_equal(prices, caller_prices)
    pd.testing.assert_series_equal(permuted_funding, caller_funding)
    pd.testing.assert_frame_equal(permuted_carry, caller_carry)
    pd.testing.assert_frame_equal(permuted_costs, caller_costs)


@pytest.mark.parametrize(
    ("index", "error", "message"),
    [
        pytest.param(pd.Index([0, 1, 2]), TypeError, "prices must use a DatetimeIndex", id="type"),
        pytest.param(
            pd.DatetimeIndex(["2024-01-02", pd.NaT, "2024-01-04"]),
            ValueError,
            "prices index must not contain NaT",
            id="nat",
        ),
        pytest.param(
            pd.DatetimeIndex(["2024-01-02", "2024-01-02", "2024-01-04"]),
            ValueError,
            "prices index must be unique",
            id="duplicate",
        ),
    ],
)
def test_backtest_model_portfolio_rejects_ambiguous_price_calendars(
    index: pd.Index,
    error: type[Exception],
    message: str,
) -> None:
    """Malformed authoritative calendars fail before portfolio state is constructed."""
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0]}, index=index)
    caller_prices = prices.copy()

    with pytest.raises(error, match=message):
        qis.backtest_model_portfolio(prices=prices, weights={"A": 1.0})

    pd.testing.assert_frame_equal(prices, caller_prices)


def test_backtest_model_portfolio_classifies_missing_history_after_price_sorting() -> None:
    """A leading unavailable price does not become a false interior hole through row order."""
    chronological = _prices()
    chronological.loc[chronological.index[0], "A"] = np.nan
    prices = _permuted_frame(chronological)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = qis.backtest_model_portfolio(
            prices=prices,
            weights={"A": 0.0, "B": 1.0},
            rebalancing_freq="YE",
            is_rebalanced_at_first_date=True,
        )

    expected_nav = pd.Series(100.0, index=DATES, name="Portfolio")
    pd.testing.assert_series_equal(result.nav, expected_nav)

"""
Initial portfolio trades use the same absolute-traded-notional cost convention as later trades.

The regression covers every documented cost container, an economically identical delayed trade,
a cost schedule that starts after inception, and one mixed panel containing long, short, unpriced,
and zero-weight instruments. These boundaries distinguish a real opening trade from cash or an
unexecutable target without relying on the backtester's own accounting for expected values.
"""

from typing import Literal, TypeAlias

import numpy as np
import pandas as pd
import pytest

import qis


CostForm = Literal["scalar", "series", "frame"]
CostInput: TypeAlias = float | pd.Series | pd.DataFrame

INITIAL_NAV = 100.0
COST_RATE = 0.01


def _constant_prices(columns: list[str] | None = None) -> pd.DataFrame:
    """Return a short constant-price panel so all NAV movement comes from trading costs."""
    tickers = columns if columns is not None else ["A", "B"]
    dates = pd.bdate_range("2025-01-02", periods=4)
    return pd.DataFrame(100.0, index=dates, columns=tickers)


def _dated_weights(prices: pd.DataFrame, values: list[float], position: int = 0) -> pd.DataFrame:
    """Return one target row traded at the selected price observation."""
    return pd.DataFrame([values], index=[prices.index[position]], columns=prices.columns)


def _cost_input(prices: pd.DataFrame, cost_form: CostForm) -> CostInput:
    """Express one constant rate through each documented public cost container."""
    if cost_form == "scalar":
        return COST_RATE
    if cost_form == "series":
        return pd.Series(COST_RATE, index=prices.columns)
    return pd.DataFrame(COST_RATE, index=prices.index, columns=prices.columns)


@pytest.mark.parametrize("cost_form", ["scalar", "series", "frame"])
def test_backtest_model_portfolio_charges_initial_trade(cost_form: CostForm) -> None:
    """Each cost container charges 1% of both 50-unit opening notionals."""
    prices = _constant_prices()
    weights = _dated_weights(prices, [0.5, 0.5])

    portfolio = qis.backtest_model_portfolio(
        prices=prices,
        weights=weights,
        initial_nav=INITIAL_NAV,
        rebalancing_costs=_cost_input(prices, cost_form),
    )

    expected_units = np.array([0.5, 0.5])
    expected_costs = np.array([0.5, 0.5])
    expected_nav = INITIAL_NAV - float(expected_costs.sum())
    np.testing.assert_allclose(portfolio.units.to_numpy(dtype=float)[0], expected_units)
    np.testing.assert_allclose(portfolio.realized_costs.to_numpy(dtype=float)[0], expected_costs)
    np.testing.assert_allclose(
        portfolio.nav.to_numpy(dtype=float), np.full(len(prices.index), expected_nav)
    )


def test_backtest_model_portfolio_initial_trade_matches_delayed_trade() -> None:
    """Moving the same opening trade one date later does not change its costs or post-trade NAV."""
    prices = _constant_prices()
    initial = qis.backtest_model_portfolio(
        prices=prices,
        weights=_dated_weights(prices, [0.5, 0.5]),
        initial_nav=INITIAL_NAV,
        rebalancing_costs=COST_RATE,
    )
    delayed = qis.backtest_model_portfolio(
        prices=prices,
        weights=_dated_weights(prices, [0.5, 0.5], position=1),
        initial_nav=INITIAL_NAV,
        rebalancing_costs=COST_RATE,
    )

    np.testing.assert_allclose(
        initial.realized_costs.to_numpy(dtype=float)[0],
        delayed.realized_costs.to_numpy(dtype=float)[1],
    )
    np.testing.assert_allclose(
        initial.nav.to_numpy(dtype=float)[:-1], delayed.nav.to_numpy(dtype=float)[1:]
    )
    assert delayed.nav.to_numpy(dtype=float)[0] == INITIAL_NAV
    np.testing.assert_array_equal(delayed.realized_costs.to_numpy(dtype=float)[0], np.zeros(2))


def test_backtest_model_portfolio_preserves_costless_initial_schedule_gap() -> None:
    """A future first cost row cannot charge an earlier opening trade."""
    prices = _constant_prices()
    costs = pd.DataFrame(COST_RATE, index=[prices.index[1]], columns=prices.columns)

    portfolio = qis.backtest_model_portfolio(
        prices=prices,
        weights=_dated_weights(prices, [0.5, 0.5]),
        initial_nav=INITIAL_NAV,
        rebalancing_costs=costs,
    )

    np.testing.assert_allclose(portfolio.nav.to_numpy(dtype=float), INITIAL_NAV)
    np.testing.assert_array_equal(portfolio.realized_costs.to_numpy(dtype=float), np.zeros((4, 2)))


def test_backtest_model_portfolio_charges_only_executable_initial_legs() -> None:
    """Long and short legs pay costs while unavailable and zero-weight legs remain costless."""
    prices = _constant_prices(["long", "short", "unpriced", "zero_unpriced"])
    prices.iloc[0, 2:] = np.nan
    weights = _dated_weights(prices, [0.5, -0.2, 0.7, 0.0])
    costs = pd.DataFrame(
        [[0.01, 0.02, 0.03, 0.04]] * len(prices.index),
        index=prices.index,
        columns=prices.columns,
    )

    with pytest.warns(UserWarning, match="stays in the cash balance"):
        portfolio = qis.backtest_model_portfolio(
            prices=prices,
            weights=weights,
            initial_nav=INITIAL_NAV,
            rebalancing_costs=costs,
        )

    expected_costs = np.array([0.5, 0.4, 0.0, 0.0])
    np.testing.assert_allclose(portfolio.realized_costs.to_numpy(dtype=float)[0], expected_costs)
    assert portfolio.nav.to_numpy(dtype=float)[0] == pytest.approx(
        INITIAL_NAV - float(expected_costs.sum())
    )

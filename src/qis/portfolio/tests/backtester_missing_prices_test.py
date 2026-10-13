"""Missing execution quotes retain holdings valued at the last observed mark (#189)."""

import warnings

import numpy as np
import pandas as pd
import pytest

import qis
from qis.portfolio.backtester import backtest_rebalanced_portfolio


def _backtest(prices, targets, **kwargs):
    # Warning wording is tested separately; accounting must fail independently of it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return qis.backtest_model_portfolio(prices, targets, **kwargs)


@pytest.mark.parametrize("position", [0.5, -0.5])
@pytest.mark.parametrize("quotes", [[100., 100., np.nan], [100., np.nan, 110.]])
def test_missing_long_or_short_is_marked_until_price_resumes(position, quotes):
    prices = pd.DataFrame(
        {"A": 100., "B": quotes}, index=pd.bdate_range("2024-01-01", periods=3)
    )
    original = prices.copy(deep=True)
    targets = pd.DataFrame([[1. - position, position]],
                           index=prices.index[:1], columns=prices.columns)
    result = _backtest(prices, targets)
    marks = np.array([100., 100., 100. if np.isnan(quotes[-1]) else quotes[-1]])
    expected_nav = 100. + position * (marks - 100.)
    np.testing.assert_allclose(result.nav, expected_nav)
    np.testing.assert_allclose(result.units["B"], position)
    np.testing.assert_allclose(result.weights["B"], position * marks / expected_nav)
    np.testing.assert_allclose(result.instrument_pnl.sum(axis=1).iloc[1:],
                               expected_nav[1:] / expected_nav[:-1] - 1., atol=1e-14)
    pd.testing.assert_frame_equal(prices, original)
    pd.testing.assert_frame_equal(result.prices, original)


@pytest.mark.parametrize("new_weight", [0., 0.8, -0.5])
@pytest.mark.parametrize("position", [0.5, -0.5])
def test_rebalance_during_gap_freezes_leg_without_cost_or_cash_proceeds(position, new_weight):
    prices = pd.DataFrame({"A": 100., "B": [100., np.nan, 110., 110.]},
                          index=pd.bdate_range("2024-01-01", periods=4))
    targets = pd.DataFrame([[1. - position, position], [1. - new_weight, new_weight],
                            [1., 0.]], index=prices.index[[0, 1, 3]], columns=prices.columns)
    result = _backtest(prices, targets, rebalancing_costs=0.01)
    opening_cost = abs(1. - position) + abs(position)
    nav0 = 100. - opening_cost
    a1 = nav0 * (1. - new_weight) / 100.
    cost1 = abs(a1 - (1. - position))
    nav1 = nav0 - cost1
    nav2 = nav1 + position * 10.
    cost3 = abs(nav2 - a1 * 100.) * 0.01 + abs(position * 110.) * 0.01
    np.testing.assert_allclose(result.nav, [nav0, nav1, nav2, nav2 - cost3])
    np.testing.assert_allclose(result.units["B"], [position, position, position, 0.])
    assert result.realized_costs["B"].iloc[1] == 0.
    cash = result.nav - result.units.mul(prices.ffill()).sum(axis=1)
    cash1 = -opening_cost - (a1 - (1. - position)) * 100. - cost1
    np.testing.assert_allclose(cash.iloc[1:3], cash1)
    np.testing.assert_allclose(result.get_instruments_pnl(is_net=True).sum(axis=1).iloc[1:],
                               result.nav.pct_change().iloc[1:], atol=1e-14)


def test_unavailable_assets_do_not_open_or_trade_on_price_resumption():
    prices = pd.DataFrame({"A": 100., "late": [np.nan, 100., 110., 120.],
                           "absent": np.nan},
                          index=pd.bdate_range("2024-01-01", periods=4))
    targets = pd.DataFrame([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]],
                           index=prices.index[[0, 2]], columns=prices.columns)
    result = _backtest(prices, targets, rebalancing_costs=0.01)
    np.testing.assert_allclose(result.units["late"].iloc[:2], 0.)
    np.testing.assert_allclose(result.units["absent"], 0.)
    np.testing.assert_allclose(result.weights["absent"], 0.)
    np.testing.assert_allclose(result.realized_costs["absent"], 0.)
    np.testing.assert_allclose(result.nav.iloc[:2], 99.5)
    assert result.units["late"].iloc[2] == pytest.approx(99.5 * 0.3 / 110.)
    np.testing.assert_allclose(result.get_instruments_pnl(is_net=True).sum(axis=1).iloc[1:],
                               result.nav.pct_change().iloc[1:], atol=1e-14)


@pytest.mark.parametrize("settlement", [0., 40.])
def test_explicit_settlement_and_zero_target_realise_terminal_loss(settlement):
    prices = pd.DataFrame({"A": 100., "B": [100., np.nan, settlement, np.nan]},
                          index=pd.bdate_range("2024-01-01", periods=4))
    targets = pd.DataFrame([[0.5, 0.5], [1., 0.]],
                           index=prices.index[[0, 2]], columns=prices.columns)
    result = _backtest(prices, targets)
    np.testing.assert_allclose(result.nav, [100., 100., 50. + 0.5 * settlement,
                                           50. + 0.5 * settlement])
    np.testing.assert_allclose(result.units["B"], [0.5, 0.5, 0., 0.])
    np.testing.assert_allclose(result.instrument_pnl.sum(axis=1).iloc[1:],
                               result.nav.pct_change().iloc[1:], atol=1e-14)


def _ledger_reference(prices, targets, trade_flags, funding, fees, carry, costs, trade_level):
    """Advance wealth by P&L and a scalar trade ledger, independently of kernel NAV sums."""
    marks = pd.DataFrame(prices).ffill().fillna(0.).to_numpy()
    held = np.zeros(prices.shape[1])
    cash, wealth, target_index = 100., 100., 0
    nav, units, expenses, balances = [], [], [], []
    for t in range(len(prices)):
        if t:
            accrual = cash * funding[t] - wealth * fees[t] + sum(held * marks[t] * carry[t])
            wealth += sum(held * (marks[t] - marks[t - 1])) + accrual
            cash += accrual
        period_costs = np.zeros(len(held))
        if trade_flags[t]:
            budget = wealth if trade_level is None else trade_level
            for j, quote in enumerate(prices[t]):
                if np.isnan(quote):
                    continue
                trade_notional = budget * targets[target_index, j] - held[j] * quote
                period_costs[j] = abs(trade_notional) * costs[t, j]
                cash -= trade_notional + period_costs[j]
                held[j] += trade_notional / quote
            wealth -= sum(period_costs)
            target_index += 1
        nav.append(wealth)
        units.append(held.copy())
        expenses.append(period_costs)
        balances.append(cash)
    return np.array(nav), np.array(units), np.array(expenses), np.array(balances), marks


@pytest.mark.parametrize("trade_level", [None, 80.])
@pytest.mark.parametrize("opening_trade", [False, True])
@pytest.mark.parametrize("with_gaps", [False, True])
def test_kernel_matches_independent_cash_ledger(trade_level, opening_trade, with_gaps):
    prices = np.array([[100., 50.], [102., 51.], [103., 52.], [101., 48.], [105., 55.]])
    if with_gaps:
        prices[2:4, 1] = np.nan
    flags = np.array([opening_trade, not opening_trade, True, False, True])
    targets = np.array([[0.7, -0.2], [1., 0.], [0.6, 0.4]])
    funding = np.array([0., 0.001, 0.002, 0.003, 0.004])
    fees = np.full(5, 0.0001)
    carry = np.tile([0.0003, -0.0002], (5, 1))
    costs = np.tile([0.001, 0.002], (5, 1))
    nav, units, effective, actual_costs = backtest_rebalanced_portfolio(
        prices, targets, flags, funding, fees, carry,
        constant_trade_level=trade_level, rebalancing_costs=costs,
    )
    expected_nav, expected_units, expected_costs, cash, marks = _ledger_reference(
        prices, targets, flags, funding, fees, carry, costs, trade_level
    )
    np.testing.assert_allclose(nav, expected_nav, atol=1e-12, rtol=0.)
    np.testing.assert_allclose(units, expected_units, atol=1e-14, rtol=0.)
    np.testing.assert_allclose(actual_costs, expected_costs, atol=1e-14, rtol=0.)
    np.testing.assert_allclose(effective, expected_units * marks / expected_nav[:, None],
                               atol=1e-14, rtol=0.)
    np.testing.assert_allclose(nav - (units * marks).sum(axis=1), cash, atol=1e-12, rtol=0.)


def test_future_prices_cannot_change_past_gap_accounting():
    prices = pd.DataFrame({"A": [100., 101., 102., 103.], "B": [100., np.nan, np.nan, 110.]},
                          index=pd.bdate_range("2024-01-01", periods=4))
    targets = pd.DataFrame([[0.5, 0.5], [1., 0.]],
                           index=prices.index[[0, 2]], columns=prices.columns)
    full = _backtest(prices, targets)
    prefix = _backtest(prices.iloc[:3], targets)
    for field in ("nav", "units", "weights", "realized_costs", "instrument_pnl"):
        actual, expected = getattr(full, field).iloc[:3], getattr(prefix, field)
        if isinstance(actual, pd.Series):
            pd.testing.assert_series_equal(actual, expected)
        else:
            pd.testing.assert_frame_equal(actual, expected)

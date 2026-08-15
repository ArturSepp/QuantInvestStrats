"""Contract tests for the deterministic discrete-portfolio replay boundary."""

# packages
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

# qis
from qis.discrete_portfolio import (
    DiscretePortfolioState,
    FullFillExecution,
    Order,
    OrderStatus,
    Trade,
    backtest_discrete_portfolio,
)
from qis.portfolio.backtester import backtest_model_portfolio


def test_order_is_immutable_and_rejects_invalid_quantity() -> None:
    """An order is a fixed, non-zero, finite signed instruction."""
    order = Order(
        order_id='order-1',
        decision_time=pd.Timestamp('2026-01-05 09:30'),
        ticker='SPY',
        quantity=2.0,
        reason='entry',
    )

    with pytest.raises(FrozenInstanceError):
        order.quantity = 3.0  # type: ignore[misc]
    with pytest.raises(ValueError, match='finite and non-zero'):
        Order('zero', pd.Timestamp('2026-01-05'), 'SPY', 0.0)
    with pytest.raises(ValueError, match='finite and non-zero'):
        Order('nan', pd.Timestamp('2026-01-05'), 'SPY', np.nan)


def test_full_fill_execution_applies_adverse_slippage_and_separate_cost() -> None:
    """Slippage changes price adversely while fees remain a separate cash charge."""
    execution = FullFillExecution(transaction_cost_rate=0.001, slippage_rate=0.01)
    decision_time = pd.Timestamp('2026-01-05 09:30')
    fill_time = pd.Timestamp('2026-01-05 09:35')

    buy = execution(Order('buy', decision_time, 'SPY', 2.0), fill_time, 100.0)
    sell = execution(Order('sell', decision_time, 'SPY', -2.0), fill_time, 100.0)

    assert buy.executed_price == pytest.approx(101.0)
    assert buy.notional == pytest.approx(202.0)
    assert buy.transaction_cost == pytest.approx(0.202)
    assert buy.slippage == pytest.approx(2.0)
    assert sell.executed_price == pytest.approx(99.0)
    assert sell.notional == pytest.approx(-198.0)
    assert sell.transaction_cost == pytest.approx(0.198)
    assert sell.slippage == pytest.approx(2.0)


def test_strategy_decision_fills_only_on_the_next_observation() -> None:
    """A strategy seeing p[t] cannot receive a fill at p[t]."""
    index = pd.date_range('2026-01-05 09:30', periods=2, freq='5min')
    prices = pd.DataFrame({'SPY': [100.0, 110.0]}, index=index)

    class BuyOnce:
        def __init__(self) -> None:
            self.units_seen: list[float] = []

        def on_bar(
                self,
                timestamp: pd.Timestamp,
                current_prices: pd.Series,
                state: DiscretePortfolioState,
        ) -> list[Order]:
            self.units_seen.append(float(state.units['SPY']))
            if timestamp == index[0]:
                assert current_prices['SPY'] == 100.0
                return [Order('entry', timestamp, 'SPY', 2.0, reason='entry')]
            return []

    strategy = BuyOnce()
    result = backtest_discrete_portfolio(
        prices=prices,
        strategy=strategy,
        initial_cash=1_000.0,
    )

    assert strategy.units_seen == [0.0, 2.0]
    assert len(result.trade_ledger.index) == 1
    trade = result.trade_ledger.iloc[0]
    assert trade['decision_time'] == index[0]
    assert trade['fill_time'] == index[1]
    assert trade['reference_price'] == 110.0
    assert trade['executed_price'] == 110.0
    assert result.cash.loc[index[0]] == 1_000.0
    assert result.cash.loc[index[1]] == 780.0
    assert result.states[0].units['SPY'] == 0.0
    assert result.states[1].units['SPY'] == 2.0
    assert result.order_ledger.set_index('order_id').loc['entry', 'status'] == OrderStatus.FILLED


@pytest.mark.parametrize(
    'prices, message',
    [
        (pd.Series([100.0]), 'prices must be a pandas DataFrame'),
        (
            pd.DataFrame(
                {'SPY': [100.0, 101.0]},
                index=pd.to_datetime(['2026-01-06', '2026-01-05']),
            ),
            'prices index must be sorted',
        ),
        (
            pd.DataFrame(
                {'SPY': [100.0, 101.0]},
                index=pd.to_datetime(['2026-01-05', '2026-01-05']),
            ),
            'prices index must not contain duplicates',
        ),
    ],
)
def test_price_grid_validation(prices: object, message: str) -> None:
    """The deterministic replay grid rejects ambiguous ordering and shape."""
    with pytest.raises((TypeError, ValueError), match=message):
        backtest_discrete_portfolio(prices=prices, strategy=lambda *_: [])


def test_duplicate_order_identifiers_are_rejected() -> None:
    """Order identifiers uniquely join the order and trade ledgers."""
    index = pd.date_range('2026-01-05', periods=2, freq='D')
    prices = pd.DataFrame({'SPY': [100.0, 101.0]}, index=index)

    class DuplicateIds:
        def on_bar(
                self,
                timestamp: pd.Timestamp,
                current_prices: pd.Series,
                state: DiscretePortfolioState,
        ) -> list[Order]:
            del current_prices, state
            if timestamp == index[0]:
                return [
                    Order('duplicate', timestamp, 'SPY', 1.0),
                    Order('duplicate', timestamp, 'SPY', 1.0),
                ]
            return []

    with pytest.raises(ValueError, match='duplicate order_id'):
        backtest_discrete_portfolio(prices=prices, strategy=DuplicateIds())


def test_final_observation_order_is_recorded_as_unfilled() -> None:
    """An order with no later observation remains visible in the order ledger."""
    timestamp = pd.Timestamp('2026-01-05 09:30')
    prices = pd.DataFrame({'SPY': [100.0]}, index=pd.DatetimeIndex([timestamp]))

    class FinalBarOrder:
        def on_bar(
                self,
                current_time: pd.Timestamp,
                current_prices: pd.Series,
                state: DiscretePortfolioState,
        ) -> list[Order]:
            del current_prices, state
            return [Order('final', current_time, 'SPY', 1.0)]

    result = backtest_discrete_portfolio(prices=prices, strategy=FinalBarOrder())

    final_order = result.order_ledger.set_index('order_id').loc['final']
    assert final_order['status'] == OrderStatus.UNFILLED_END_OF_DATA
    assert pd.isna(final_order['fill_time'])
    assert result.trade_ledger.empty


def test_two_asset_accounting_matches_an_independent_scalar_ledger() -> None:
    """Buys, sells, shorting, and fees reconcile without calling engine helpers."""
    index = pd.date_range('2026-01-05 09:30', periods=3, freq='5min')
    prices = pd.DataFrame(
        {'AAA': [100.0, 110.0, 105.0], 'BBB': [50.0, 48.0, 52.0]},
        index=index,
    )

    class TwoAssetOrders:
        def on_bar(
                self,
                timestamp: pd.Timestamp,
                current_prices: pd.Series,
                state: DiscretePortfolioState,
        ) -> list[Order]:
            del current_prices, state
            if timestamp == index[0]:
                return [
                    Order('buy-aaa', timestamp, 'AAA', 2.0),
                    Order('short-bbb', timestamp, 'BBB', -3.0),
                ]
            if timestamp == index[1]:
                return [
                    Order('sell-aaa', timestamp, 'AAA', -3.0),
                    Order('buy-bbb', timestamp, 'BBB', 1.0),
                ]
            return []

    result = backtest_discrete_portfolio(
        prices=prices,
        strategy=TwoAssetOrders(),
        initial_cash=1_000.0,
        execution_model=FullFillExecution(transaction_cost_rate=0.01),
    )

    expected_cash_t1 = 1_000.0 - (2.0 * 110.0) - 2.20 - (-3.0 * 48.0) - 1.44
    expected_cash_t2 = expected_cash_t1 - (-3.0 * 105.0) - 3.15 - (1.0 * 52.0) - 0.52
    expected_nav_t1 = expected_cash_t1 + 2.0 * 110.0 - 3.0 * 48.0
    expected_nav_t2 = expected_cash_t2 - 1.0 * 105.0 - 2.0 * 52.0

    assert result.cash.to_list() == pytest.approx([1_000.0, expected_cash_t1, expected_cash_t2])
    assert result.states[1].units.to_dict() == {'AAA': 2.0, 'BBB': -3.0}
    assert result.states[2].units.to_dict() == {'AAA': -1.0, 'BBB': -2.0}
    assert result.states[1].nav == pytest.approx(expected_nav_t1)
    assert result.states[2].nav == pytest.approx(expected_nav_t2)
    assert result.trade_ledger['transaction_cost'].sum() == pytest.approx(7.31)
    for state in result.states:
        assert state.nav == pytest.approx(state.cash + state.position_values.sum())
        assert state.position_values.to_numpy() == pytest.approx(
            state.units.multiply(state.prices).to_numpy()
        )

    portfolio_data = result.portfolio_data
    assert portfolio_data is not None
    assert portfolio_data.nav.to_list() == pytest.approx(
        [1_000.0, expected_nav_t1, expected_nav_t2]
    )
    assert portfolio_data.units.loc[index[1]].to_dict() == {'AAA': 2.0, 'BBB': -3.0}
    assert portfolio_data.units.loc[index[2]].to_dict() == {'AAA': -1.0, 'BBB': -2.0}
    assert portfolio_data.realized_costs.loc[index[1]].to_dict() == pytest.approx(
        {'AAA': 2.20, 'BBB': 1.44}
    )
    assert portfolio_data.realized_costs.loc[index[2]].to_dict() == pytest.approx(
        {'AAA': 3.15, 'BBB': 0.52}
    )
    assert portfolio_data.is_rebalancing.to_list() == [False, True, True]
    expected_instrument_pnl = pd.DataFrame(
        [
            [0.0, 0.0],
            [-2.20 / 1_000.0, -1.44 / 1_000.0],
            [(-10.0 - 3.15) / expected_nav_t1, (-12.0 - 0.52) / expected_nav_t1],
        ],
        index=index,
        columns=prices.columns,
    )
    pd.testing.assert_frame_equal(portfolio_data.instrument_pnl, expected_instrument_pnl)
    pd.testing.assert_series_equal(
        portfolio_data.instrument_pnl.sum(axis=1),
        portfolio_data.nav.pct_change(fill_method=None).fillna(0.0),
        check_names=False,
    )


def test_multiple_same_ticker_orders_fill_and_aggregate_in_units() -> None:
    """Separate orders remain separate trades while holdings aggregate their quantities."""
    index = pd.date_range('2026-01-05', periods=2, freq='D')
    prices = pd.DataFrame({'SPY': [100.0, 105.0]}, index=index)

    class LayeredEntry:
        def on_bar(
                self,
                timestamp: pd.Timestamp,
                current_prices: pd.Series,
                state: DiscretePortfolioState,
        ) -> list[Order]:
            del current_prices, state
            if timestamp == index[0]:
                return [
                    Order('layer-1', timestamp, 'SPY', 1.0),
                    Order('layer-2', timestamp, 'SPY', 2.0),
                ]
            return []

    result = backtest_discrete_portfolio(
        prices,
        LayeredEntry(),
        initial_cash=1_000.0,
        execution_model=FullFillExecution(transaction_cost_rate=0.01),
    )

    assert result.trade_ledger['order_id'].to_list() == ['layer-1', 'layer-2']
    assert result.trade_ledger['filled_quantity'].sum() == 3.0
    assert result.states[-1].units['SPY'] == 3.0
    assert result.cash.iloc[-1] == pytest.approx(681.85)
    assert result.portfolio_data is not None
    assert result.portfolio_data.realized_costs.iloc[-1, 0] == pytest.approx(3.15)
    assert result.portfolio_data.instrument_pnl.iloc[-1, 0] == pytest.approx(-3.15 / 1_000.0)


def test_portfolio_adapter_attributes_slippage_and_fees_to_the_fill_instrument() -> None:
    """Execution-price impact and explicit fees both reconcile to the NAV return."""
    index = pd.date_range('2026-01-05', periods=2, freq='D')
    prices = pd.DataFrame({'SPY': [100.0, 100.0]}, index=index)

    class BuyOnce:
        def on_bar(
                self,
                timestamp: pd.Timestamp,
                current_prices: pd.Series,
                state: DiscretePortfolioState,
        ) -> list[Order]:
            del current_prices, state
            return [Order('entry', timestamp, 'SPY', 1.0)] if timestamp == index[0] else []

    result = backtest_discrete_portfolio(
        prices,
        BuyOnce(),
        initial_cash=1_000.0,
        execution_model=FullFillExecution(transaction_cost_rate=0.001, slippage_rate=0.01),
    )

    portfolio_data = result.portfolio_data
    assert portfolio_data is not None
    assert portfolio_data.nav.iloc[-1] == pytest.approx(998.899)
    assert portfolio_data.realized_costs.iloc[-1, 0] == pytest.approx(0.101)
    assert portfolio_data.instrument_pnl.iloc[-1, 0] == pytest.approx(-1.101 / 1_000.0)
    assert portfolio_data.instrument_pnl.sum(axis=1).to_numpy() == pytest.approx(
        portfolio_data.nav.pct_change(fill_method=None).fillna(0.0).to_numpy()
    )


def test_scheduled_trade_matches_existing_weight_backtester_after_one_bar_lag() -> None:
    """A matching scheduled trade reconciles to the established unit-based backtester."""
    index = pd.date_range('2026-01-05', periods=3, freq='D')
    prices = pd.DataFrame({'SPY': [100.0, 110.0, 120.0]}, index=index)

    class BuyTwoUnits:
        def on_bar(
                self,
                timestamp: pd.Timestamp,
                current_prices: pd.Series,
                state: DiscretePortfolioState,
        ) -> list[Order]:
            del current_prices, state
            return [Order('entry', timestamp, 'SPY', 2.0)] if timestamp == index[0] else []

    discrete = backtest_discrete_portfolio(
        prices,
        BuyTwoUnits(),
        initial_cash=1_000.0,
        execution_model=FullFillExecution(transaction_cost_rate=0.01),
    ).portfolio_data
    scheduled = backtest_model_portfolio(
        prices=prices,
        weights=pd.DataFrame({'SPY': [0.22]}, index=[index[0]]),
        initial_nav=1_000.0,
        rebalancing_costs=0.01,
        weight_implementation_lag=1,
        ticker='ScheduledPortfolio',
    )

    assert discrete is not None
    pd.testing.assert_frame_equal(discrete.units, scheduled.units)
    pd.testing.assert_frame_equal(discrete.weights, scheduled.weights)
    pd.testing.assert_frame_equal(discrete.realized_costs, scheduled.realized_costs)
    pd.testing.assert_series_equal(discrete.nav, scheduled.nav, check_names=False)
    pd.testing.assert_series_equal(
        discrete.is_rebalancing,
        scheduled.is_rebalancing.rename('is_rebalancing'),
    )


def test_missing_execution_price_is_logged_without_a_fill() -> None:
    """A missing next-bar price produces a terminal unfilled status and no zero fill."""
    index = pd.date_range('2026-01-05', periods=3, freq='D')
    prices = pd.DataFrame({'SPY': [100.0, np.nan, 102.0]}, index=index)
    observed: list[float] = []

    class BuyOnce:
        def on_bar(
                self,
                timestamp: pd.Timestamp,
                current_prices: pd.Series,
                state: DiscretePortfolioState,
        ) -> list[Order]:
            del state
            observed.append(float(current_prices['SPY']))
            if timestamp == index[0]:
                return [Order('missing-price', timestamp, 'SPY', 1.0)]
            return []

    result = backtest_discrete_portfolio(prices, BuyOnce(), initial_cash=1_000.0)

    order = result.order_ledger.set_index('order_id').loc['missing-price']
    assert order['status'] == OrderStatus.UNFILLED_MISSING_PRICE
    assert result.trade_ledger.empty
    assert result.states[1].prices['SPY'] == 100.0
    assert np.isnan(observed[1])
    assert result.states[-1].units['SPY'] == 0.0


def test_execution_response_must_preserve_order_identity() -> None:
    """An injected execution response cannot silently fill another instrument."""
    index = pd.date_range('2026-01-05', periods=2, freq='D')
    prices = pd.DataFrame({'SPY': [100.0, 101.0]}, index=index)

    class BuyOnce:
        def on_bar(
                self,
                timestamp: pd.Timestamp,
                current_prices: pd.Series,
                state: DiscretePortfolioState,
        ) -> list[Order]:
            del current_prices, state
            return [Order('entry', timestamp, 'SPY', 1.0)] if timestamp == index[0] else []

    def wrong_ticker(order: Order, fill_time: pd.Timestamp, reference_price: float) -> Trade:
        return Trade(
            order_id=order.order_id,
            decision_time=order.decision_time,
            fill_time=fill_time,
            ticker='NOT-SPY',
            filled_quantity=order.quantity,
            reference_price=reference_price,
            executed_price=reference_price,
            notional=order.quantity * reference_price,
            transaction_cost=0.0,
        )

    with pytest.raises(ValueError, match='order_id and ticker must match'):
        backtest_discrete_portfolio(prices, BuyOnce(), execution_model=wrong_ticker)

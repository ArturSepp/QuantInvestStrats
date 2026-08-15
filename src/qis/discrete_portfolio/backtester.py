"""
orders and observed prices in, deterministic ledgers out: the event loop for discrete trading.

``replay_discrete_portfolio`` is the raw entry point. It deliberately stops at orders, fills,
cash, units, and marked states; the reporting conversion lives in ``portfolio_adapter.py`` so
the same replay core can be used in research or live deployment without depending on
``PortfolioData``.

Each timestamp is processed in one fixed order:

1. orders decided at the preceding observation fill from the current observed price;
2. cash and signed units are updated from those fills;
3. holdings are marked and an immutable ``DiscretePortfolioState`` is created;
4. the strategy sees only the current observation and post-fill state, then emits new orders;
5. those new orders wait for the next observation.

This ordering is the no-look-ahead contract. An order decided at *t* can first fill at *t+1*;
an order emitted on the final observation is recorded as ``UNFILLED_END_OF_DATA``. A missing or
non-positive next-observation price terminates that order as ``UNFILLED_MISSING_PRICE`` rather
than silently delaying it to a more favourable later bar.

The engine holds units, not weights. Cash changes on a full fill as

    cash_after = cash_before - quantity * executed_price - transaction_cost

where quantity is signed, so a sale adds its notional to cash. Marked NAV is cash plus current
position values. Valuation carries the last finite positive observation for an existing
position, while the strategy still receives the unmodified current price row and can therefore
distinguish a stale mark from a newly observed price.

Execution is injected through ``ExecutionModel`` and version one requires a complete fill that
preserves the submitted order identity, ticker, quantity, and timestamps. Order identifiers are
unique across a replay and provide the stable join between the order and trade ledgers.
"""

from __future__ import annotations

# packages
from dataclasses import asdict
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

# qis
from qis.discrete_portfolio.execution import FullFillExecution
from qis.discrete_portfolio.types import (
    DiscreteBacktestResult,
    DiscretePortfolioState,
    DiscreteStrategy,
    ExecutionModel,
    Order,
    OrderStatus,
    Trade,
)

ORDER_COLUMNS = [
    'order_id',
    'decision_time',
    'ticker',
    'quantity',
    'reason',
    'status',
    'fill_time',
    'status_reason',
]
TRADE_COLUMNS = [
    'order_id',
    'decision_time',
    'fill_time',
    'ticker',
    'filled_quantity',
    'reference_price',
    'executed_price',
    'notional',
    'transaction_cost',
    'slippage',
]


def _validate_prices(prices: pd.DataFrame) -> None:
    """Validate the deterministic observation grid.

    Args:
        prices: Candidate timestamp-by-ticker price panel.

    Raises:
        TypeError: If ``prices`` is not a DataFrame or its index is not a DatetimeIndex.
        ValueError: If timestamps or tickers are duplicated, timestamps are unsorted, or the
            panel has no observations or instruments.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError('prices must be a pandas DataFrame')
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError('prices index must be a pandas DatetimeIndex')
    if prices.index.has_duplicates:
        raise ValueError('prices index must not contain duplicates')
    if not prices.index.is_monotonic_increasing:
        raise ValueError('prices index must be sorted in increasing order')
    if prices.columns.has_duplicates:
        raise ValueError('prices columns must not contain duplicate tickers')
    if len(prices.index) == 0 or len(prices.columns) == 0:
        raise ValueError('prices must contain at least one timestamp and ticker')


def _validate_execution(order: Order, timestamp: pd.Timestamp, trade: Trade) -> None:
    """Ensure an injected execution model preserves the version-one fill contract.

    Args:
        order: Submitted order awaiting execution.
        timestamp: Current observation, which is the required fill timestamp.
        trade: Response returned by the injected execution model.

    Raises:
        TypeError: If the response is not a ``Trade``.
        ValueError: If identity, timestamps, or filled quantity differ from the order.
    """
    if not isinstance(trade, Trade):
        raise TypeError('execution_model must return a Trade')
    if trade.order_id != order.order_id or trade.ticker != order.ticker:
        raise ValueError('execution response order_id and ticker must match the submitted order')
    if trade.decision_time != order.decision_time or trade.fill_time != timestamp:
        raise ValueError('execution response timestamps must match the submitted order and bar')
    if not np.isclose(
            trade.filled_quantity, order.quantity, rtol=1e-12, atol=1e-12,
    ):
        raise ValueError('version-one execution must fill the complete submitted quantity')


def _create_state(
        timestamp: pd.Timestamp,
        valuation_prices: pd.Series,
        units: pd.Series,
        cash: float,
) -> DiscretePortfolioState:
    """Mark a post-fill state using prices observed no later than the timestamp.

    Args:
        timestamp: Current observation timestamp.
        valuation_prices: Latest valid price observed for each instrument.
        units: Post-fill signed units by instrument.
        cash: Post-fill cash balance.

    Returns:
        Immutable point-in-time state with position values, NAV, and realised weights.

    Raises:
        ValueError: If a held instrument has no valuation price or marked NAV is not finite.
    """
    current_prices = valuation_prices.astype(float).copy()
    invalid_held_prices = current_prices.isna() & units.ne(0.0)
    if invalid_held_prices.any():
        tickers = current_prices.index[invalid_held_prices].to_list()
        raise ValueError(f'cannot value non-zero positions without prior prices: {tickers}')
    position_values = units.multiply(current_prices).mask(
        current_prices.isna() & units.eq(0.0), 0.0
    )
    nav = float(cash + position_values.sum())
    if not np.isfinite(nav):
        raise ValueError('marked nav must be finite')
    if nav == 0.0:
        weights = pd.Series(np.nan, index=units.index, dtype=float)
    else:
        weights = position_values.divide(nav)
    return DiscretePortfolioState(
        timestamp=timestamp,
        cash=cash,
        units=units,
        prices=current_prices,
        position_values=position_values,
        nav=nav,
        weights=weights,
    )


def replay_discrete_portfolio(
        prices: pd.DataFrame,
        strategy: DiscreteStrategy,
        initial_cash: float = 1_000_000.0,
        execution_model: Optional[ExecutionModel] = None,
) -> DiscreteBacktestResult:
    """Replay a strategy on a price grid with strict next-observation execution.

    At timestamp ``t`` the engine first executes orders created at the preceding observation,
    then marks cash and holdings, and only then calls the strategy with the state at ``t``.
    Orders returned by that call are queued for the next timestamp. An order created at the final
    timestamp is recorded as ``UNFILLED_END_OF_DATA``.

    Args:
        prices: Valuation and execution prices indexed by unique, increasing timestamps.
        strategy: Object implementing ``on_bar(timestamp, prices, state)``.
        initial_cash: Finite starting cash balance.
        execution_model: Callable mapping a pending order and later price to a full fill. The
            default fills at the unadjusted next observed price without fees.

    Returns:
        Raw ledgers, cash series, and point-in-time states. ``portfolio_data`` is ``None`` so
        this live-deployable core does not depend on the reporting layer.

    Raises:
        TypeError: If the price grid, strategy response, or execution response has wrong type.
        ValueError: If the grid, cash, orders, prices, or execution response is invalid.
    """
    _validate_prices(prices)
    if not np.isfinite(initial_cash):
        raise ValueError('initial_cash must be finite')
    if not hasattr(strategy, 'on_bar'):
        raise TypeError('strategy must define on_bar(timestamp, prices, state)')
    execution = FullFillExecution() if execution_model is None else execution_model
    units = pd.Series(0.0, index=prices.columns, dtype=float, name='units')
    valuation_prices = pd.Series(np.nan, index=prices.columns, dtype=float, name='prices')
    cash = float(initial_cash)
    pending_orders: List[Order] = []
    seen_order_ids: Set[str] = set()
    order_records: List[Dict[str, object]] = []
    records_by_id: Dict[str, Dict[str, object]] = {}
    trades: List[Trade] = []
    states: List[DiscretePortfolioState] = []
    cash_by_time: Dict[pd.Timestamp, float] = {}

    for timestamp, current_prices in prices.iterrows():
        # Mark from information available at this timestamp. A missing current observation does
        # not erase the last valid valuation, but remains missing in the row shown to strategy.
        observed_prices = current_prices.astype(float).copy()
        valid_prices = np.isfinite(observed_prices) & observed_prices.gt(0.0)
        valuation_prices.loc[valid_prices] = observed_prices.loc[valid_prices]

        # Orders from the preceding decision point fill before the new state is published.
        for order in pending_orders:
            reference_price = float(observed_prices.loc[order.ticker])
            if not np.isfinite(reference_price) or reference_price <= 0.0:
                record = records_by_id[order.order_id]
                record['status'] = OrderStatus.UNFILLED_MISSING_PRICE
                record['status_reason'] = 'no finite positive execution price'
                continue
            trade = execution(order, timestamp, reference_price)
            _validate_execution(order, timestamp, trade)
            units.loc[trade.ticker] += trade.filled_quantity
            cash -= trade.notional + trade.transaction_cost
            trades.append(trade)
            record = records_by_id[order.order_id]
            record['status'] = OrderStatus.FILLED
            record['fill_time'] = timestamp
        pending_orders = []

        state = _create_state(timestamp, valuation_prices, units, cash)
        states.append(state)
        cash_by_time[timestamp] = cash

        # The strategy receives a read-only price row and immutable state, so it cannot mutate
        # the engine's accounting history. Its orders are eligible only on the next observation.
        observed_prices.to_numpy(copy=False).flags.writeable = False
        new_orders = strategy.on_bar(timestamp, observed_prices, state)
        if new_orders is None:
            raise TypeError('strategy.on_bar must return a sequence of Order objects')
        for order in new_orders:
            if not isinstance(order, Order):
                raise TypeError('strategy.on_bar must return only Order objects')
            if order.decision_time != timestamp:
                raise ValueError('order decision_time must equal the current strategy timestamp')
            if order.ticker not in prices.columns:
                raise ValueError(f'order ticker {order.ticker!r} is outside the price universe')
            if order.order_id in seen_order_ids:
                raise ValueError(f'duplicate order_id {order.order_id!r}')
            seen_order_ids.add(order.order_id)
            record = {
                'order_id': order.order_id,
                'decision_time': order.decision_time,
                'ticker': order.ticker,
                'quantity': order.quantity,
                'reason': order.reason,
                'status': OrderStatus.PENDING,
                'fill_time': pd.NaT,
                'status_reason': None,
            }
            order_records.append(record)
            records_by_id[order.order_id] = record
            pending_orders.append(order)

    # No synthetic terminal fill: every order without a later observation stays visible.
    for order in pending_orders:
        record = records_by_id[order.order_id]
        record['status'] = OrderStatus.UNFILLED_END_OF_DATA
        record['status_reason'] = 'no later market observation'

    trade_ledger = pd.DataFrame([asdict(trade) for trade in trades], columns=TRADE_COLUMNS)
    order_ledger = pd.DataFrame(order_records, columns=ORDER_COLUMNS)
    cash_series = pd.Series(cash_by_time, index=prices.index, dtype=float, name='cash')
    return DiscreteBacktestResult(
        portfolio_data=None,
        trade_ledger=trade_ledger,
        order_ledger=order_ledger,
        cash=cash_series,
        states=tuple(states),
    )

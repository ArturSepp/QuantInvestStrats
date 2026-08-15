"""
the domain language of discrete trading: immutable orders, fills, states, and callable borders.

The replay pipeline exchanges explicit records rather than loosely shaped dictionaries.
``Order`` is a strategy decision, ``Trade`` is a later execution response, and
``DiscretePortfolioState`` is the post-fill mark shown to the next strategy decision. Frozen
dataclasses validate accounting relationships at construction so an invalid record cannot enter
a ledger and fail much later in reporting.

Pandas Series stored on a state are copied and their current NumPy buffers made read-only. This
prevents a strategy from mutating the engine's point-in-time history through a shared object.
The state holds signed units, not target weights; realised weights are derived from position
values and NAV at the mark.

``DiscreteStrategy`` and ``ExecutionModel`` are structural protocols. Research classes and live
adapters need only implement the documented call signature; inheritance from a QIS base class is
not required. ``DiscreteBacktestResult`` keeps raw ledgers available even after the optional
reporting adapter attaches ``PortfolioData``.
"""

from __future__ import annotations

# packages
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional, Protocol, Sequence, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from qis.portfolio.portfolio_data import PortfolioData


class OrderStatus(str, Enum):
    """Lifecycle state recorded in the order ledger.

    Attributes:
        PENDING: Submitted after a decision and awaiting the next observation.
        FILLED: Fully executed at the next observation.
        UNFILLED_MISSING_PRICE: Terminated because the next execution price was invalid.
        UNFILLED_END_OF_DATA: Submitted at the final observation with no later fill point.
    """

    PENDING = 'pending'
    FILLED = 'filled'
    UNFILLED_MISSING_PRICE = 'unfilled_missing_price'
    UNFILLED_END_OF_DATA = 'unfilled_end_of_data'


def _timestamp(value: pd.Timestamp, field: str) -> pd.Timestamp:
    """Return a valid pandas timestamp for a domain record.

    Args:
        value: Timestamp-like value to normalize.
        field: Field name used in validation errors.

    Returns:
        Normalized pandas timestamp.

    Raises:
        ValueError: If the value resolves to ``NaT``.
    """
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f'{field} must be a valid timestamp')
    return timestamp


def _immutable_float_series(value: pd.Series, field: str) -> pd.Series:
    """Copy a numeric series and make its current value buffer read-only.

    Args:
        value: Candidate indexed values.
        field: Field name used in validation errors.

    Returns:
        Independent float Series with a read-only value buffer.

    Raises:
        TypeError: If ``value`` is not a Series.
        ValueError: If its index contains duplicates.
    """
    if not isinstance(value, pd.Series):
        raise TypeError(f'{field} must be a pandas Series')
    if value.index.has_duplicates:
        raise ValueError(f'{field} index must not contain duplicates')
    output = value.astype(float).copy(deep=True)
    output.to_numpy(copy=False).flags.writeable = False
    return output


@dataclass(frozen=True)
class Order:
    """A signed instruction created from one observed bar.

    Attributes:
        order_id: Unique identifier joining the order and trade ledgers.
        decision_time: Timestamp of the observation used to create the order.
        ticker: Instrument to trade.
        quantity: Signed units; positive buys and negative sells.
        reason: Optional strategy label for the decision.
    """

    order_id: str
    decision_time: pd.Timestamp
    ticker: str
    quantity: float
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.order_id, str) or not self.order_id.strip():
            raise ValueError('order_id must be a non-empty string')
        if not isinstance(self.ticker, str) or not self.ticker.strip():
            raise ValueError('ticker must be a non-empty string')
        if not np.isfinite(self.quantity) or self.quantity == 0.0:
            raise ValueError('quantity must be finite and non-zero')
        if self.reason is not None and not isinstance(self.reason, str):
            raise TypeError('reason must be a string or None')
        object.__setattr__(self, 'decision_time', _timestamp(self.decision_time, 'decision_time'))
        object.__setattr__(self, 'quantity', float(self.quantity))


@dataclass(frozen=True)
class Trade:
    """A full fill produced from an order and a later market observation.

    Attributes:
        order_id: Identifier of the submitted order.
        decision_time: Timestamp when the strategy created the order.
        fill_time: Timestamp of the later observation used for execution.
        ticker: Filled instrument.
        filled_quantity: Signed filled units.
        reference_price: Observed market price supplied to execution.
        executed_price: Price after applying the execution model.
        notional: Signed filled quantity times executed price.
        transaction_cost: Non-negative fee charged separately to cash.
        slippage: Non-negative currency impact embedded in executed price.
    """

    order_id: str
    decision_time: pd.Timestamp
    fill_time: pd.Timestamp
    ticker: str
    filled_quantity: float
    reference_price: float
    executed_price: float
    notional: float
    transaction_cost: float
    slippage: float = 0.0

    def __post_init__(self) -> None:
        numeric = {
            'filled_quantity': self.filled_quantity,
            'reference_price': self.reference_price,
            'executed_price': self.executed_price,
            'notional': self.notional,
            'transaction_cost': self.transaction_cost,
            'slippage': self.slippage,
        }
        if not np.all(np.isfinite(tuple(numeric.values()))):
            raise ValueError('trade quantities, prices, notional, and costs must be finite')
        if self.filled_quantity == 0.0:
            raise ValueError('filled_quantity must be non-zero')
        if self.reference_price <= 0.0 or self.executed_price <= 0.0:
            raise ValueError('reference_price and executed_price must be positive')
        if self.transaction_cost < 0.0 or self.slippage < 0.0:
            raise ValueError('transaction_cost and slippage must be non-negative')
        expected_notional = self.filled_quantity * self.executed_price
        if not np.isclose(self.notional, expected_notional, rtol=1e-12, atol=1e-12):
            raise ValueError('notional must equal filled_quantity times executed_price')
        object.__setattr__(self, 'decision_time', _timestamp(self.decision_time, 'decision_time'))
        object.__setattr__(self, 'fill_time', _timestamp(self.fill_time, 'fill_time'))


@dataclass(frozen=True)
class DiscretePortfolioState:
    """Immutable strategy-facing state after fills and marking at one timestamp.

    Attributes:
        timestamp: Current market observation timestamp.
        cash: Cash balance after fills at the timestamp.
        units: Post-fill signed units by ticker.
        prices: Current valuation prices by ticker.
        position_values: Post-fill units times current valuation prices.
        nav: Cash plus the sum of position values.
        weights: Realised position values divided by NAV.
    """

    timestamp: pd.Timestamp
    cash: float
    units: pd.Series
    prices: pd.Series
    position_values: pd.Series
    nav: float
    weights: pd.Series

    def __post_init__(self) -> None:
        if not np.isfinite(self.cash) or not np.isfinite(self.nav):
            raise ValueError('cash and nav must be finite')
        units = _immutable_float_series(self.units, 'units')
        prices = _immutable_float_series(self.prices, 'prices')
        position_values = _immutable_float_series(self.position_values, 'position_values')
        weights = _immutable_float_series(self.weights, 'weights')
        if not units.index.equals(prices.index):
            raise ValueError('units and prices must have identical indexes')
        if not units.index.equals(position_values.index) or not units.index.equals(weights.index):
            raise ValueError('state panels must have identical indexes')
        invalid_held_prices = prices.isna() & units.ne(0.0)
        if invalid_held_prices.any():
            raise ValueError('a non-zero position must have a finite valuation price')
        expected_values = units.multiply(prices).mask(prices.isna() & units.eq(0.0), 0.0)
        if not np.allclose(
                position_values, expected_values, rtol=1e-12, atol=1e-12, equal_nan=True,
        ):
            raise ValueError('position_values must equal units times prices')
        expected_nav = float(self.cash + position_values.sum())
        if not np.isclose(self.nav, expected_nav, rtol=1e-12, atol=1e-12):
            raise ValueError('nav must equal cash plus position values')
        expected_weights = position_values.divide(self.nav) if self.nav != 0.0 else weights
        if self.nav != 0.0 and not np.allclose(
                weights, expected_weights, rtol=1e-12, atol=1e-12,
        ):
            raise ValueError('weights must equal position values divided by nav')
        object.__setattr__(self, 'timestamp', _timestamp(self.timestamp, 'timestamp'))
        object.__setattr__(self, 'cash', float(self.cash))
        object.__setattr__(self, 'nav', float(self.nav))
        object.__setattr__(self, 'units', units)
        object.__setattr__(self, 'prices', prices)
        object.__setattr__(self, 'position_values', position_values)
        object.__setattr__(self, 'weights', weights)


class DiscreteStrategy(Protocol):
    """Structural strategy interface evaluated once after each marked observation."""

    def on_bar(
            self,
            timestamp: pd.Timestamp,
            prices: pd.Series,
            state: DiscretePortfolioState,
    ) -> Sequence[Order]:
        """Return zero or more orders using only the supplied point-in-time state.

        Args:
            timestamp: Current observation timestamp.
            prices: Read-only current observed prices; missing values are not forward-filled.
            state: Immutable post-fill state marked at ``timestamp``.

        Returns:
            Orders decided at ``timestamp`` and eligible at the next observation.
        """
        ...


class ExecutionModel(Protocol):
    """Structural execution interface mapping one pending order to one full fill."""

    def __call__(
            self,
            order: Order,
            fill_time: pd.Timestamp,
            reference_price: float,
    ) -> Trade:
        """Execute an order using the supplied later market observation.

        Args:
            order: Pending order from the preceding decision point.
            fill_time: Timestamp of the later execution observation.
            reference_price: Positive market price observed at ``fill_time``.

        Returns:
            Complete fill preserving the submitted order identity and quantity.
        """
        ...


@dataclass(frozen=True)
class DiscreteBacktestResult:
    """Raw replay output, optionally enriched with the QIS reporting adapter.

    Attributes:
        portfolio_data: Reporting object added by the Stage 3 adapter; otherwise ``None``.
        trade_ledger: One row per full fill.
        order_ledger: One row per submitted order with its final status.
        cash: Post-fill cash balance at every observation.
        states: Point-in-time strategy states in observation order.
    """

    portfolio_data: Optional[PortfolioData]
    trade_ledger: pd.DataFrame
    order_ledger: pd.DataFrame
    cash: pd.Series
    states: Tuple[DiscretePortfolioState, ...]

"""
an order and a later market observation in, one explicit full fill out: execution policies.

The replay engine depends on the ``ExecutionModel`` protocol rather than on this implementation,
so production deployment can inject venue-specific execution while retaining the same event and
accounting contracts. ``FullFillExecution`` is the deterministic research default.

For signed quantity *q*, reference price *p*, slippage rate *s*, and fee rate *c*:

    executed_price = p * (1 + sign(q) * s)
    notional = q * executed_price
    transaction_cost = abs(notional) * c

Slippage is therefore adverse on both sides: buys pay above the reference and sells receive
below it. Its currency impact is embedded in ``executed_price`` and also recorded separately for
diagnostics; ``transaction_cost`` is an additional cash charge. Rates are fractional, so
``0.0010`` is 10 bp.

Version one is intentionally all-or-nothing. Partial fills, order cancellation, latency beyond
one observation, and market impact belong in alternative execution models and require an
explicit extension of the replay contract rather than an implicit approximation here.
"""

from __future__ import annotations

# packages
from dataclasses import dataclass

import numpy as np
import pandas as pd

# qis
from qis.discrete_portfolio.types import Order, Trade


@dataclass(frozen=True)
class FullFillExecution:
    """Fill every order at the next reference price with optional costs.

    Slippage is adverse: a buy executes above the reference price and a sell below it.
    Slippage is embedded in ``executed_price``; ``transaction_cost`` is charged separately.

    Attributes:
        transaction_cost_rate: Non-negative fee as a fraction of absolute executed notional.
        slippage_rate: Non-negative adverse price move as a fraction of reference price. Must be
            below one so a sell execution remains positive.
    """

    transaction_cost_rate: float = 0.0
    slippage_rate: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.transaction_cost_rate) or self.transaction_cost_rate < 0.0:
            raise ValueError('transaction_cost_rate must be finite and non-negative')
        if not np.isfinite(self.slippage_rate) or not 0.0 <= self.slippage_rate < 1.0:
            raise ValueError('slippage_rate must be finite and in [0, 1)')

    def __call__(
            self,
            order: Order,
            fill_time: pd.Timestamp,
            reference_price: float,
    ) -> Trade:
        """Execute ``order`` in full using a strictly later observation.

        Args:
            order: Pending signed order.
            fill_time: Timestamp of the execution observation.
            reference_price: Positive observed market price.

        Returns:
            Full fill with signed notional and separately recorded costs.

        Raises:
            ValueError: If the timestamp is not later or the reference price is invalid.
        """
        fill_time = pd.Timestamp(fill_time)
        if pd.isna(fill_time) or fill_time <= order.decision_time:
            raise ValueError('fill_time must be later than order.decision_time')
        if not np.isfinite(reference_price) or reference_price <= 0.0:
            raise ValueError('reference_price must be finite and positive')
        direction = 1.0 if order.quantity > 0.0 else -1.0
        executed_price = float(reference_price * (1.0 + direction * self.slippage_rate))
        notional = float(order.quantity * executed_price)
        transaction_cost = float(abs(notional) * self.transaction_cost_rate)
        slippage = float(abs(order.quantity * (executed_price - reference_price)))
        return Trade(
            order_id=order.order_id,
            decision_time=order.decision_time,
            fill_time=fill_time,
            ticker=order.ticker,
            filled_quantity=order.quantity,
            reference_price=float(reference_price),
            executed_price=executed_price,
            notional=notional,
            transaction_cost=transaction_cost,
            slippage=slippage,
        )

"""
event-driven trading replay with explicit orders, fills, holdings, and QIS reporting output.

Use ``replay_discrete_portfolio`` for the deployment-oriented core result or
``backtest_discrete_portfolio`` for the same replay enriched with ``PortfolioData``. Strategies
implement ``DiscreteStrategy.on_bar`` and execution policies implement ``ExecutionModel``;
``FullFillExecution`` supplies the deterministic next-observation default.

The event contract is decision at *t*, execution at *t+1*. The engine holds signed units between
events and exposes both order- and trade-level ledgers, while the adapter aggregates identical
accounting into the existing QIS portfolio reporting surface.
"""

from qis.discrete_portfolio.backtester import replay_discrete_portfolio
from qis.discrete_portfolio.execution import FullFillExecution
from qis.discrete_portfolio.portfolio_adapter import (
    backtest_discrete_portfolio,
    to_portfolio_data,
)
from qis.discrete_portfolio.types import (
    DiscreteBacktestResult,
    DiscretePortfolioState,
    DiscreteStrategy,
    ExecutionModel,
    Order,
    OrderStatus,
    Trade,
)

__all__ = [
    'DiscreteBacktestResult',
    'DiscretePortfolioState',
    'DiscreteStrategy',
    'ExecutionModel',
    'FullFillExecution',
    'Order',
    'OrderStatus',
    'Trade',
    'backtest_discrete_portfolio',
    'replay_discrete_portfolio',
    'to_portfolio_data',
]

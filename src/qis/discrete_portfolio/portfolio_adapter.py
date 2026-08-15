"""
discrete states and fills in, a ``PortfolioData`` out: the reporting adapter for event replay.

``replay_discrete_portfolio`` stays independent of the portfolio reporting layer. This module is
the upstream link to established QIS factsheets and performance analytics:
``backtest_discrete_portfolio`` runs the raw replay and ``to_portfolio_data`` converts its state
and trade ledgers into aligned NAV, unit, weight, cost, and instrument-contribution panels.

For instrument *i* between observations *t-1* and *t*, currency P&L is

    units[i, t-1] * (mark[i, t] - mark[i, t-1])
    + fill_quantity[i, t] * (mark[i, t] - executed_price[i, t])
    - transaction_cost[i, t]

The first term belongs to units held across the interval. The second captures execution-price
impact for units acquired or sold at *t* without pretending post-fill units were held for the
whole interval. Dividing by prior NAV produces the contribution return expected by
``PortfolioData.instrument_pnl``.

The adapter treats reconciliation as an invariant, not a reporting convenience: aggregated
trade quantities must equal state-unit changes, and instrument contributions must sum to the
actual NAV return at every observation. A mismatch raises instead of passing numerically wrong
data downstream to a factsheet.
"""

from __future__ import annotations

# packages
from dataclasses import replace
from typing import Optional

import numpy as np
import pandas as pd

# qis
from qis.discrete_portfolio.backtester import replay_discrete_portfolio
from qis.discrete_portfolio.types import (
    DiscreteBacktestResult,
    DiscreteStrategy,
    ExecutionModel,
)
from qis.portfolio.portfolio_data import PortfolioData


def to_portfolio_data(
        result: DiscreteBacktestResult,
        ticker: str = 'DiscretePortfolio',
) -> PortfolioData:
    """Aggregate a raw discrete replay into aligned QIS reporting panels.

    Instrument contribution at each observation is the P&L of units held from the preceding
    mark, plus the difference between the current mark and each fill price, minus explicit
    transaction costs. This captures fills inside the close-to-close interval without treating
    post-fill units as if they had been held for the whole interval.

    Args:
        result: Raw result returned by :func:`replay_discrete_portfolio`.
        ticker: Portfolio name assigned to the NAV and reporting object.

    Returns:
        PortfolioData with explicit NAV, holdings, prices, costs, and contribution returns.

    Raises:
        ValueError: If the state sequence is empty, timestamps are duplicated, a contribution
            return follows zero NAV, or the aggregated contributions do not reconcile to NAV.
    """
    if len(result.states) == 0:
        raise ValueError('a discrete replay must contain at least one state')
    state_index = pd.DatetimeIndex([state.timestamp for state in result.states])
    index = result.cash.index.copy()
    if not state_index.equals(index):
        raise ValueError('state timestamps must equal the cash-series index')
    if index.has_duplicates:
        raise ValueError('state timestamps must not contain duplicates')
    columns = result.states[0].units.index
    units = pd.DataFrame(
        [state.units.to_numpy(copy=True) for state in result.states],
        index=index,
        columns=columns,
        dtype=float,
    )
    prices = pd.DataFrame(
        [state.prices.to_numpy(copy=True) for state in result.states],
        index=index,
        columns=columns,
        dtype=float,
    )
    weights = pd.DataFrame(
        [state.weights.to_numpy(copy=True) for state in result.states],
        index=index,
        columns=columns,
        dtype=float,
    )
    nav = pd.Series(
        [state.nav for state in result.states], index=index, dtype=float, name=ticker,
    )
    realized_costs = pd.DataFrame(0.0, index=index, columns=columns)
    fill_quantities = pd.DataFrame(0.0, index=index, columns=columns)
    fill_price_pnl = pd.DataFrame(0.0, index=index, columns=columns)
    is_rebalancing = pd.Series(False, index=index, dtype=bool, name='is_rebalancing')

    # Fold individual fills into observation-by-instrument panels while retaining the raw trade
    # ledger on the result for order-level analysis.
    for trade in result.trade_ledger.itertuples(index=False):
        if trade.fill_time not in index or trade.ticker not in columns:
            raise ValueError('trade ledger fill timestamp and ticker must exist in the states')
        mark_price = float(prices.loc[trade.fill_time, trade.ticker])
        realized_costs.loc[trade.fill_time, trade.ticker] += float(trade.transaction_cost)
        fill_quantities.loc[trade.fill_time, trade.ticker] += float(trade.filled_quantity)
        fill_price_pnl.loc[trade.fill_time, trade.ticker] += float(
            trade.filled_quantity * (mark_price - trade.executed_price)
        )
        is_rebalancing.loc[trade.fill_time] = True

    expected_unit_change = units.diff().fillna(units.iloc[0])
    if not np.allclose(
            fill_quantities, expected_unit_change, rtol=1e-12, atol=1e-12, equal_nan=True,
    ):
        raise ValueError('trade ledger quantities do not reconcile to state unit changes')

    # Reconstruct contribution returns independently from the engine's marked NAV. This is the
    # accounting cross-check that prevents a plausible but mis-timed unit return from passing.
    instrument_pnl = pd.DataFrame(0.0, index=index, columns=columns)
    for row in range(1, len(index)):
        prior_units = units.iloc[row - 1]
        price_change = prices.iloc[row].subtract(prices.iloc[row - 1])
        holding_pnl = prior_units.multiply(price_change)
        holding_pnl = holding_pnl.mask(prior_units.eq(0.0) & holding_pnl.isna(), 0.0)
        currency_pnl = (
            holding_pnl
            + fill_price_pnl.iloc[row]
            - realized_costs.iloc[row]
        )
        prior_nav = float(nav.iloc[row - 1])
        if prior_nav == 0.0:
            if not np.allclose(currency_pnl, 0.0, rtol=1e-12, atol=1e-12):
                raise ValueError('cannot compute contribution returns after zero NAV')
        else:
            instrument_pnl.iloc[row] = currency_pnl.divide(prior_nav)

    nav_returns = nav.pct_change(fill_method=None).fillna(0.0)
    contribution_returns = instrument_pnl.sum(axis=1)
    if not np.allclose(
            contribution_returns, nav_returns, rtol=1e-11, atol=1e-12, equal_nan=True,
    ):
        max_error = float((contribution_returns - nav_returns).abs().max())
        raise ValueError(f'instrument contributions do not reconcile to NAV: max error={max_error}')

    return PortfolioData(
        nav=nav,
        prices=prices,
        units=units,
        weights=weights,
        instrument_pnl=instrument_pnl,
        realized_costs=realized_costs,
        is_rebalancing=is_rebalancing,
        ticker=ticker,
    )


def backtest_discrete_portfolio(
        prices: pd.DataFrame,
        strategy: DiscreteStrategy,
        initial_cash: float = 1_000_000.0,
        execution_model: Optional[ExecutionModel] = None,
        ticker: str = 'DiscretePortfolio',
) -> DiscreteBacktestResult:
    """Replay a discrete strategy and aggregate its result for QIS reporting.

    Args:
        prices: Valuation and execution prices indexed by unique, increasing timestamps.
        strategy: Object implementing ``on_bar(timestamp, prices, state)``.
        initial_cash: Finite starting cash balance.
        execution_model: Optional callable mapping a pending order and later price to a full fill.
        ticker: Portfolio name assigned to the NAV and reporting object.

    Returns:
        Discrete backtest result containing raw ledgers and a populated PortfolioData object.

    Raises:
        TypeError: If replay inputs or strategy/execution responses have invalid types.
        ValueError: If replay validation or reporting reconciliation fails.
    """
    result = replay_discrete_portfolio(
        prices=prices,
        strategy=strategy,
        initial_cash=initial_cash,
        execution_model=execution_model,
    )
    return replace(result, portfolio_data=to_portfolio_data(result, ticker=ticker))

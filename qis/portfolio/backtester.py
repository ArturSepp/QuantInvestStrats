"""
create total return blended portfolios based on weights
"""

# packages
import numpy as np
import pandas as pd
from numba import njit
from typing import Union, Dict, Tuple, List, Optional
# qis
from qis.utils.dates import generate_rebalancing_indicators, set_rebalancing_timeindex_on_given_timeindex
from qis.utils.df_ops import multiply_df_by_dt
from qis.utils.np_ops import repeat_by_columns, repeat_by_rows
from qis.utils.struct_ops import assert_list_subset
from qis.portfolio.portfolio_data import PortfolioData


def backtest_model_portfolio(prices: pd.DataFrame,
                             weights: Union[Dict[str, float], List[float], np.ndarray, pd.DataFrame, pd.Series],
                             rebalancing_freq: Optional[str] = 'QE',
                             initial_nav: float = 100,
                             funding_rate: pd.Series = None,  # annualised on positive / negative cash balances
                             management_fee: float = None,  # annualised
                             instruments_carry: pd.DataFrame = None,  # on nav
                             rebalancing_costs: Union[float, pd.Series, pd.DataFrame] = None,
                             weight_implementation_lag: Optional[int] = None,  # applies for weight is pd.Dataframe
                             constant_trade_level: float = None,
                             is_rebalanced_at_first_date: bool = False,
                             ticker: str = 'Portfolio'  # default ticker
                             ) -> PortfolioData:
    """
    simulate a rebalanced portfolio from prices and target weights.

    The simulation holds units, not weights: target weights are converted to units at each
    rebalancing and held until the next one, so the realised weights drift with prices in
    between and reported turnover is the turnover actually traded. ``PortfolioData.weights``
    returns those realised weights, not the targets.

    Args:
        prices: instrument prices, columns are tickers. Costs, carry and fees are applied on
            this grid
        weights: target weights. A Dict or pd.DataFrame is safest, since both are aligned to
            ``prices.columns`` by name; a list or array is positional and must match the
            column count. A fixed weight vector is applied at every date in
            ``rebalancing_freq``; a pd.DataFrame supplies its own rebalancing dates and
            ``rebalancing_freq`` is then ignored
        rebalancing_freq: calendar anchor for rebalancing when ``weights`` is a fixed vector,
            passed to :func:`generate_rebalancing_indicators`
        initial_nav: starting nav
        funding_rate: annualised rate applied to positive and negative cash balances
        management_fee: annualised fee accrued on nav
        instruments_carry: per-instrument carry, expressed on nav
        rebalancing_costs: proportional cost on traded notional, fractional units (0.0010 is
            10 bp). A float applies to every instrument and date; a Series indexed by ticker
            is per-instrument and constant in time; a DataFrame of dates x tickers is
            reindexed to ``prices`` dates taking the last schedule row at or before each
            date, so a cost schedule stated on era boundaries applies from each boundary
            onward. Costs are read at the trade date; dates before the first schedule row
            are costless
        weight_implementation_lag: days between a weight being observed and traded. Applies
            only when ``weights`` is a pd.DataFrame, since a fixed vector has no signal date
        constant_trade_level: size each rebalancing off this notional rather than off current
            nav, so trade size does not compound with performance
        is_rebalanced_at_first_date: rebalance on the first price date as well as on the
            schedule
        ticker: name carried on the resulting portfolio

    Returns:
        PortfolioData holding the nav, realised weights, units, instrument pnl and realised
        costs

    Raises:
        ValueError: if ``prices`` is not a pd.DataFrame, if a weight vector does not match the
            number of price columns, if the price history starts after the weights do, if a
            ``rebalancing_costs`` DataFrame is missing a price column, or if a
            ``rebalancing_costs`` Series is indexed by dates rather than tickers
        NotImplementedError: if ``weights`` is of an unsupported type
    """
    if not isinstance(prices, pd.DataFrame):
        raise ValueError(f"prices type={type(prices)} must be pd.Dataframe")
    if isinstance(weights, pd.Series):  # map to dict
        weights = weights.to_dict()

    if isinstance(weights, Dict):  # map to np
        assert_list_subset(large_list=prices.columns.to_list(),
                              list_sample=list(weights.keys()),
                              message=f"weights columns must be aligned with price columns")
        weights = prices.columns.map(weights).to_numpy()
    elif isinstance(weights, List):
        weights = np.array(weights)

    # align weights with prices
    if isinstance(weights, np.ndarray):
        if weights.shape[0] != len(prices.columns):
            raise ValueError(f"number of weights must be aligned with number of price columns")
        if len(weights.shape) > 1:
            raise ValueError(f"only single aray is allowed")

        is_rebalancing = generate_rebalancing_indicators(df=prices,
                                                            freq=rebalancing_freq,
                                                            include_start_date=is_rebalanced_at_first_date)

        portfolio_rebalance_dates = is_rebalancing[is_rebalancing == True]
        portfolio_weights = pd.DataFrame(data=repeat_by_rows(weights, n=len(portfolio_rebalance_dates)),
                                         index=portfolio_rebalance_dates,
                                         columns=prices.columns)

    elif isinstance(weights, pd.DataFrame):
        assert_list_subset(large_list=prices.columns.to_list(),
                              list_sample=weights.columns.to_list(),
                              message=f"weights columns must be aligned with price columns")
        if prices.index[0] > weights.index[0]:
            raise ValueError(f"price dates {prices.index[0]} are after weights start date {weights.index[0]}")
        portfolio_weights = weights[prices.columns]  # alighn

        # implementation lag is only valid for quant-generated weights
        if weight_implementation_lag is not None and weight_implementation_lag > 0:
            rebalancing_index = portfolio_weights.index + pd.Timedelta(days=weight_implementation_lag)
        else:
            rebalancing_index = portfolio_weights.index
        is_rebalancing = set_rebalancing_timeindex_on_given_timeindex(given_index=prices.index,
                                                                         rebalancing_index=rebalancing_index)
    else:
        raise NotImplementedError(f"unsupported weights type = {type(weights)}")

    # adjust rates at rebealncing
    if funding_rate is not None:
        funding_rate_dt = multiply_df_by_dt(df=funding_rate, dates=prices.index, lag=0)
    else:
        funding_rate_dt = pd.Series(0.0, index=prices.index)

    if management_fee is not None:
        management_fee_dt = multiply_df_by_dt(df=pd.Series(management_fee, index=prices.index), dates=prices.index, lag=0)
    else:
        management_fee_dt = pd.Series(0.0, index=prices.index)

    if instruments_carry is not None:
        instruments_carry_dt = multiply_df_by_dt(df=instruments_carry, dates=prices.index, lag=0)
    else:
        instruments_carry_dt = pd.Series(0.0, index=prices.index)

    if rebalancing_costs is not None:
        if isinstance(rebalancing_costs, pd.DataFrame):
            missing_columns = prices.columns.difference(rebalancing_costs.columns)
            if len(missing_columns) > 0:
                raise ValueError(f"rebalancing_costs is missing price columns "
                                 f"{list(missing_columns)}")
            rebalancing_costs = rebalancing_costs.sort_index().reindex(index=prices.index,
                                                                       method='ffill')
            rebalancing_costs = rebalancing_costs[prices.columns].fillna(0.0).to_numpy()
        elif isinstance(rebalancing_costs, pd.Series):
            if isinstance(rebalancing_costs.index, pd.DatetimeIndex):
                raise ValueError("a date-indexed rebalancing_costs Series is ambiguous: "
                                 "pass a DataFrame of dates x tickers")
            costs_vector = rebalancing_costs[prices.columns].to_numpy()
            rebalancing_costs = np.tile(costs_vector, (len(prices.index), 1))
        else:
            rebalancing_costs = np.full((len(prices.index), len(prices.columns)),
                                        float(rebalancing_costs))

    nav, units, effective_weights, realized_costs = backtest_rebalanced_portfolio(prices=prices.to_numpy(),
                                                                                  weights=portfolio_weights.to_numpy(),
                                                                                  is_rebalancing=is_rebalancing.to_numpy(),
                                                                                  funding_rate_dt=funding_rate_dt.to_numpy(),
                                                                                  management_fee_dt=management_fee_dt.to_numpy(),
                                                                                  instruments_carry_dt=instruments_carry_dt.to_numpy(),
                                                                                  initial_nav=initial_nav,
                                                                                  constant_trade_level=constant_trade_level,
                                                                                  rebalancing_costs=rebalancing_costs)

    portfolio_nav = pd.Series(nav, index=prices.index)
    if ticker is not None:
        portfolio_nav = portfolio_nav.rename(ticker)

    output_portfolio_data = PortfolioData(nav=portfolio_nav,
                                          units=pd.DataFrame(units, index=prices.index, columns=prices.columns),
                                          weights=pd.DataFrame(effective_weights, index=prices.index, columns=prices.columns),
                                          input_weights=weights,
                                          is_rebalancing=is_rebalancing,
                                          prices=prices,
                                          realized_costs=pd.DataFrame(realized_costs, index=prices.index, columns=prices.columns),
                                          ticker=ticker)

    return output_portfolio_data


@njit
def backtest_rebalanced_portfolio(prices: np.ndarray,
                                  weights: np.ndarray,
                                  is_rebalancing: np.ndarray,
                                  funding_rate_dt: np.ndarray = None,
                                  management_fee_dt: np.ndarray = None,
                                  instruments_carry_dt: np.ndarray = None,
                                  initial_nav: float = 100.0,
                                  constant_trade_level: float = None,
                                  rebalancing_costs: np.ndarray = None  # (t, n) proportional costs
                                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    path-dependent portfolio simulation kernel.

    The numba layer under :func:`backtest_model_portfolio`. Between rebalancings it holds
    units fixed and lets nav follow prices, so the effective weights it returns are those
    implied by the held units rather than the targets. This is a genuine recursion — each
    step depends on the previous nav and cash balance — which is why it is not vectorized.

    Args:
        prices: price panel, shape (t, n), time along the first axis
        weights: target weights, shape (num_rebalancings, n), one row consumed per True in
            ``is_rebalancing``
        is_rebalancing: boolean flag per date, shape (t,)
        funding_rate_dt: per-period funding rate on the cash balance, shape (t,); zeros when
            None
        management_fee_dt: per-period fee on nav, shape (t,); zeros when None
        instruments_carry_dt: per-period carry per instrument, shape (t, n)
        initial_nav: starting nav
        constant_trade_level: size trades off this notional instead of current nav
        rebalancing_costs: proportional costs on traded notional, shape (t, n); the wrapper
            broadcasts the scalar and per-instrument forms to this shape. None trades costless

    Returns:
        (nav, units, effective_weights, realized_costs)

    Raises:
        ValueError: if ``prices`` and ``is_rebalancing`` disagree on length, or if ``weights``
            and ``prices`` disagree on the number of instruments
    """
    if prices.shape[0] != is_rebalancing.shape[0]:
        raise ValueError(f"prices.shape[0] != is_rebalancing.shape[0]")
    if weights.shape[1] != prices.shape[1]:
        raise ValueError(f"weights.shape[1] != prices.shape[1]")

    if funding_rate_dt is None:
        funding_rate_dt = np.zeros(prices.shape[0])

    if management_fee_dt is None:
        management_fee_dt = np.zeros(prices.shape[0])

    # initialize
    current_rebalancing_idx = 0
    units = np.zeros_like(prices)
    realized_costs = np.zeros_like(prices)
    nav = np.zeros(prices.shape[0])
    cash_balances = np.zeros(prices.shape[0])

    # build initial portfolio
    current_prices = prices[0, :]
    if is_rebalancing[0]:
        if constant_trade_level is not None:
            current_units = (constant_trade_level * weights[current_rebalancing_idx]) / current_prices
        else:
            current_units = (initial_nav * weights[current_rebalancing_idx]) / current_prices

        current_units[np.isnan(current_units)] = 0
        current_cash_balance = initial_nav - np.nansum(current_units * current_prices)
        current_rebalancing_idx += 1
    else:
        current_units = np.zeros(prices.shape[1])
        current_cash_balance = initial_nav
    units[0, :] = current_units
    nav[0] = np.nansum(current_units * current_prices) + current_cash_balance  # need to be adjusted when cost are present for is_rebalancing[0] = True
    cash_balances[0] = current_cash_balance

    # loop over t
    for t in np.arange(1, prices.shape[0]):
        current_units = units[t - 1]
        current_prices = prices[t, :]
        management_fee = management_fee_dt[t]*nav[t-1]
        current_cash_balance = cash_balances[t-1] * (1.0 + funding_rate_dt[t]) - management_fee

        if instruments_carry_dt is not None:
            carry = np.nansum(current_units * current_prices * instruments_carry_dt[t])
            current_cash_balance += carry

        if is_rebalancing[t]:
            if constant_trade_level:
                current_nav0 = constant_trade_level
            else:
                current_nav0 = np.nansum(current_units * current_prices) + current_cash_balance
            current_units = (current_nav0 * weights[current_rebalancing_idx]) / current_prices
            current_units[np.isnan(current_units)] = 0
            units_change = current_units - units[t-1]
            change_in_cash = -np.nansum(units_change*current_prices)
            if rebalancing_costs is not None:
                realized_costs_t = rebalancing_costs[t, :]*current_prices*np.abs(units_change)
                realized_costs[t, :] = realized_costs_t
                change_in_cash -= np.nansum(realized_costs_t)
            current_cash_balance = current_cash_balance + change_in_cash
            current_rebalancing_idx += 1

        # store
        units[t, :] = current_units
        nav[t] = np.nansum(current_units * current_prices) + current_cash_balance
        cash_balances[t] = current_cash_balance

    effective_weights = np.divide(units * prices, repeat_by_columns(a=nav, n=prices.shape[1]))

    return nav, units, effective_weights, realized_costs

"""
create total return blended portfolios based on weights
"""

# packages
import numpy as np
import pandas as pd
from numba import njit
from enum import Enum
from typing import Union, Dict, Tuple, List

# qis
import qis.utils as qu
from qis.portfolio.portfolio_data import PortfolioData


def backtest_model_portfolio(prices: pd.DataFrame,
                             weights: Union[Dict[str, float], List[float], np.ndarray, pd.DataFrame, pd.Series],
                             rebalance_freq: str = 'QE',
                             initial_nav: float = 100,
                             funding_rate: pd.Series = None,  # on positive / negative cash balances
                             instruments_carry: pd.DataFrame = None,  # on nav
                             rebalancing_costs: Union[float, pd.Series] = None,  # rebalancing costs in bp
                             constant_trade_level: float = None,
                             is_rebalanced_at_first_date: bool = False,
                             ticker: str = None,
                             is_output_portfolio_data: bool = False
                             ) -> Union[pd.Series, PortfolioData]:
    """
    simulate portfolio given prices and weights
    include_start_date if index rebalanced at start date
    the safest weight is to pass weights as Dict or pd.Dataframe - this enforces the alignment with prices
    """
    if not isinstance(prices, pd.DataFrame):
        raise ValueError(f"prices type={type(prices)} must be pd.Dataframe")
    if isinstance(weights, pd.Series):  # map to dict
        weights = weights.to_dict()

    if isinstance(weights, Dict):  # map to np
        qu.assert_list_subset(large_list=prices.columns.to_list(),
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

        is_rebalancing = qu.generate_rebalancing_indicators(df=prices,
                                                            freq=rebalance_freq,
                                                            include_start_date=is_rebalanced_at_first_date)

        portfolio_rebalance_dates = is_rebalancing[is_rebalancing == True]
        portfolio_weights = pd.DataFrame(data=qu.repeat_by_rows(weights, n=len(portfolio_rebalance_dates)),
                                         index=portfolio_rebalance_dates,
                                         columns=prices.columns)

    elif isinstance(weights, pd.DataFrame):
        qu.assert_list_subset(large_list=prices.columns.to_list(),
                              list_sample=weights.columns.to_list(),
                              message=f"weights columns must be aligned with price columns")
        if prices.index[0] > weights.index[0]:
            raise ValueError(f"price dates {prices.index[0]} are after weights start date {weights.index[0]}")
        portfolio_weights = weights[prices.columns]  # alighn
        # rebalancing is set on portfolio weight index
        is_rebalancing = pd.Series(1, index=portfolio_weights.index, dtype=int).reindex(index=prices.index
                                                                                        ).replace(np.nan, 0).astype(bool)

    else:
        raise NotImplementedError(f"unsupported weights type = {type(weights)}")

    # adjust rates at rebealncing
    if funding_rate is not None:
        funding_rate_dt = qu.multiply_df_by_dt(df=funding_rate, dates=prices.index, lag=0)
    else:
        funding_rate_dt = pd.Series(0.0, index=prices.index)
    if instruments_carry is not None:
        instruments_carry_dt = qu.multiply_df_by_dt(df=instruments_carry, dates=prices.index, lag=0)
    else:
        instruments_carry_dt = pd.Series(0.0, index=prices.index)

    if rebalancing_costs is not None:
        if isinstance(rebalancing_costs, pd.Series):
            rebalancing_costs = rebalancing_costs[prices.columns].to_numpy()

    nav, units, effective_weights, realized_costs = backtest_rebalanced_portfolio(prices=prices.to_numpy(),
                                                                                  weights=portfolio_weights.to_numpy(),
                                                                                  is_rebalancing=is_rebalancing.to_numpy(),
                                                                                  funding_rate_dt=funding_rate_dt.to_numpy(),
                                                                                  instruments_carry_dt=instruments_carry_dt.to_numpy(),
                                                                                  initial_nav=initial_nav,
                                                                                  constant_trade_level=constant_trade_level,
                                                                                  rebalancing_costs=rebalancing_costs)

    portfolio_nav = pd.Series(nav, index=prices.index)
    if ticker is not None:
        portfolio_nav = portfolio_nav.rename(ticker)

    if is_output_portfolio_data:
        output_portfolio_data = PortfolioData(nav=portfolio_nav,
                                              units=pd.DataFrame(units, index=prices.index, columns=prices.columns),
                                              weights=pd.DataFrame(effective_weights, index=prices.index, columns=prices.columns),
                                              input_weights=weights,
                                              is_rebalancing=is_rebalancing,
                                              prices=prices,
                                              realized_costs=pd.DataFrame(realized_costs, index=prices.index, columns=prices.columns),
                                              ticker=ticker)
    else:
        output_portfolio_data = portfolio_nav
    return output_portfolio_data


@njit
def backtest_rebalanced_portfolio(prices: np.ndarray,
                                  weights: np.ndarray,
                                  is_rebalancing: np.ndarray,
                                  funding_rate_dt: np.ndarray = None,
                                  instruments_carry_dt: np.ndarray = None,
                                  initial_nav: float = 100.0,
                                  constant_trade_level: float = None,
                                  rebalancing_costs: Union[float, np.ndarray] = None  # proportional rebalancing costs
                                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if prices.shape[0] != is_rebalancing.shape[0]:
        raise ValueError(f"prices.shape[0] != is_rebalancing.shape[0]")
    if weights.shape[1] != prices.shape[1]:
        raise ValueError(f"weights.shape[1] != prices.shape[1]")

    if funding_rate_dt is None:
        funding_rate_dt = np.zeros(prices.shape[0])

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
            current_units = (constant_trade_level * weights[current_rebalancing_idx]) / prices[0, :]
        else:
            current_units = (initial_nav * weights[current_rebalancing_idx]) / prices[0, :]

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
        current_cash_balance = cash_balances[t-1] * (1.0 + funding_rate_dt[t])
        
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
                realized_costs_t = rebalancing_costs*current_prices*np.abs(units_change)
                realized_costs[t, :] = realized_costs_t
                change_in_cash -= np.nansum(realized_costs_t)
            current_cash_balance = current_cash_balance + change_in_cash
            current_rebalancing_idx += 1

        # store
        units[t, :] = current_units
        nav[t] = np.nansum(current_units * current_prices) + current_cash_balance
        cash_balances[t] = current_cash_balance

    effective_weights = np.divide(units * prices, qu.repeat_by_columns(a=nav, n=prices.shape[1]))

    return nav, units, effective_weights, realized_costs


class UnitTests(Enum):
    BLENDED = 1
    COSTS = 2


def run_unit_test(unit_test: UnitTests):

    import matplotlib.pyplot as plt
    import qis.plots.derived.prices as ppd

    from qis.test_data import load_etf_data
    prices = load_etf_data().dropna()

    prices = prices[['SPY', 'TLT']]
    # prices.iloc[:200, :] = np.nan
    print(prices)
    
    if unit_test == UnitTests.BLENDED:

        portfolio_nav_1_0 = backtest_model_portfolio(prices=prices,
                                                     weights=np.array([1.0, 0.0]),
                                                     rebalance_freq='QE')

        portfolio_nav_5_5 = backtest_model_portfolio(prices=prices,
                                                     weights=np.array([1.0, 0.5]),
                                                     rebalance_freq='QE')

        portfolio_nav_0_1 = backtest_model_portfolio(prices=prices,
                                                     weights=np.array([1.0, 1.0]),
                                                     rebalance_freq='QE')

        portfolio_nav = pd.concat([portfolio_nav_1_0, portfolio_nav_5_5, portfolio_nav_0_1], axis=1)
        portfolio_nav.columns = ['x1=100, x2=0', 'x1=100, x2=50', 'x1=100, x2=100']
        print(portfolio_nav)
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ppd.plot_prices(prices=portfolio_nav, ax=ax)

    elif unit_test == UnitTests.COSTS:
        portfolio_nav = backtest_model_portfolio(prices=prices,
                                                 weights=np.array([1.0, 1.0]),
                                                 rebalance_freq='QE',
                                                 is_output_portfolio_data=True)

        portfolio_nav.plot_pnl()

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.BLENDED

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

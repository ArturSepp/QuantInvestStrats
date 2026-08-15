"""
what ``weight_implementation_lag`` does, and what it deliberately does not do.

The lag selects the entry price for the units and nothing else. A weight observed at *t* is
traded at the price ``weight_implementation_lag`` observations of the price index later, so on a
business-day panel a lag of 1 trades the next business day and a lag of 20 trades roughly a month
on. Prices are not shifted, resampled or otherwise touched, so instrument returns are identical
under every lag - only the dates at which units change hands move.

The unit is observations, not calendar days. Those differ: a Friday weight at a lag of 1 is
traded on the following Monday, three calendar days later. Counting in observations is also what
makes the lag safe, because two weight dates can never resolve onto the same traded date and so
the weight rows cannot slip out of step with the rebalancing schedule.

The strategy is a long-only trend book on the seeded synthetic universe, rebalanced monthly, so
the signal is autocorrelated and delaying its implementation costs something measurable. Run it
with ``python -m examples.portfolios.lagged_weight_implementation``. No network, no data file.
"""
# packages
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum
# qis / project
import qis as qis
from qis.datasets.synthetic import generate_synthetic_prices

TICKERS = ['SEQ_US', 'SBD_TSY', 'SBD_IG', 'SBD_HY']  # four full histories, no missing prices
LAGS = (0, 1, 5, 20)  # observations of the price index
SIGNAL_SPAN = 60  # ewm span of the trend signal, in observations
REBALANCING_FREQ = 'ME'
REBALANCING_COSTS = 0.0010  # fractional units: 10 bp


class LocalTest(Enum):
    ENTRY_DATES = 1  # where the lag puts the trade
    PERFORMANCE = 2  # what the delay costs
    RETURNS_UNCHANGED = 3  # what the lag does not touch


def fetch_prices() -> pd.DataFrame:
    """the seeded panel, restricted to instruments reported over the whole window."""
    return generate_synthetic_prices()[TICKERS]


def compute_trend_weights(prices: pd.DataFrame) -> pd.DataFrame:
    """
    long-only trend weights on the rebalancing dates.

    An ewm of log returns gives the trend, its sign gives the position and
    :func:`df_to_long_only_allocation_sum1` normalises the positive legs to sum to one, leaving a
    month with no positive trend fully in cash.

    Args:
        prices: instrument prices, columns are tickers

    Returns:
        weights indexed by the rebalancing dates, columns as ``prices.columns``
    """
    returns = qis.to_returns(prices=prices, is_log_returns=True)
    trend = qis.compute_ewm(data=returns, span=SIGNAL_SPAN)
    weights = qis.df_to_long_only_allocation_sum1(df=np.sign(trend))
    rebalancing_dates = qis.generate_rebalancing_indicators(df=prices,
                                                            freq=REBALANCING_FREQ,
                                                            include_start_date=True,
                                                            return_true_only=True).index
    return weights.loc[rebalancing_dates, :]


def backtest_at_lag(prices: pd.DataFrame,
                    weights: pd.DataFrame,
                    lag: int
                    ) -> qis.PortfolioData:
    """
    run the same weights at one implementation lag.

    Args:
        prices: instrument prices, columns are tickers
        weights: trend weights on the rebalancing dates
        lag: observations of the price index between the weight and the trade

    Returns:
        the simulated portfolio, named for its lag
    """
    with warnings.catch_warnings():  # the tail weights that cannot be traded are the point here
        warnings.simplefilter('ignore')
        return qis.backtest_model_portfolio(prices=prices,
                                            weights=weights,
                                            weight_implementation_lag=lag,
                                            rebalancing_costs=REBALANCING_COSTS,
                                            ticker=f"lag={lag}")


def run_local_test(local_test: LocalTest) -> None:
    """
    run one case.

    Args:
        local_test: which case to run
    """
    prices = fetch_prices()
    weights = compute_trend_weights(prices=prices)

    if local_test == LocalTest.ENTRY_DATES:
        print(f"{len(weights.index)} monthly weight dates on a {len(prices.index)}-observation "
              f"business-day panel\n")
        print(f"{'lag':>5s}{'traded':>9s}{'dropped':>9s}{'calendar days weight -> trade':>32s}")
        for lag in LAGS:
            portfolio_data = backtest_at_lag(prices=prices, weights=weights, lag=lag)
            is_rebalancing = portfolio_data.is_rebalancing
            traded_dates = is_rebalancing[is_rebalancing == True].index
            gaps = (traded_dates - weights.index[:len(traded_dates)]).days
            print(f"{lag:5d}{len(traded_dates):9d}{len(weights.index) - len(traded_dates):9d}"
                  f"{f'{gaps.min()} to {gaps.max()}, median {int(np.median(gaps))}':>32s}")
        print("\nEvery weight row is traded exactly once, at the observation the lag names. The "
              "calendar\ngap varies with weekends and holidays, which is why the unit is "
              "observations: a lag of 1\nover a Friday is three calendar days, and counting in "
              "days would land two weight dates on\nthe same Monday. The dropped rows are the "
              "tail weights whose traded date would fall past\nthe end of the price history; "
              "backtest_model_portfolio warns when it drops them.")

    elif local_test == LocalTest.PERFORMANCE:
        navs, turnovers, costs = [], {}, {}
        for lag in LAGS:
            portfolio_data = backtest_at_lag(prices=prices, weights=weights, lag=lag)
            nav = portfolio_data.get_portfolio_nav()
            navs.append(nav)
            num_years = qis.get_time_period(df=nav).get_time_period_an()
            # weight turnover rather than unit turnover: the book is fully in cash in some months
            # and unit turnover divides by the absolute exposure
            turnovers[nav.name] = portfolio_data.get_turnover(is_agg=True,
                                                              roll_period=None,
                                                              is_unit_based_traded_volume=False
                                                              ).sum() / num_years
            costs[nav.name] = portfolio_data.get_costs(is_agg=True,
                                                       roll_period=None).sum() / num_years
        navs = pd.concat(navs, axis=1, sort=True)
        perf_params = qis.PerfParams(freq='ME')
        columns = [qis.PerfStat.PA_RETURN.to_str(), qis.PerfStat.VOL.to_str(),
                   qis.PerfStat.SHARPE_RF0.to_str(), qis.PerfStat.MAX_DD.to_str()]
        table = qis.compute_ra_perf_table(prices=navs, perf_params=perf_params)[columns]
        table['Turnover p.a.'] = pd.Series(turnovers)
        table['Costs p.a.'] = pd.Series(costs)
        print(table.round(3).to_string())
        print("\nTurnover is identical across the lags and realised costs move by less than a "
              "basis point\np.a.: the same weights are traded, only later. The spread in return "
              "and Sharpe is therefore\nentirely about when the signal is implemented, and on "
              "this seeded panel a longer delay\nhappens to help. That direction is a property "
              "of the data, not a rule - the point is that\nthe choice of lag moves the answer "
              "materially, so a backtest has to state which one it used.")

        with sns.axes_style('darkgrid'):
            fig, axs = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)
            qis.plot_prices_with_dd(prices=navs, perf_params=perf_params, axs=axs,
                                   title='Monthly trend book at four implementation lags')
        plt.show()

    elif local_test == LocalTest.RETURNS_UNCHANGED:
        unlagged = backtest_at_lag(prices=prices, weights=weights, lag=0)
        print(f"{'lag':>5s}{'prices identical':>19s}{'instrument returns identical':>31s}")
        for lag in LAGS:
            lagged = backtest_at_lag(prices=prices, weights=weights, lag=lag)
            unlagged_returns = qis.to_returns(prices=unlagged.prices, drop_first=True)
            lagged_returns = qis.to_returns(prices=lagged.prices, drop_first=True)
            same_returns = np.allclose(unlagged_returns.to_numpy(), lagged_returns.to_numpy())
            print(f"{lag:5d}{str(unlagged.prices.equals(lagged.prices)):>19s}{str(same_returns):>31s}")
        print("\nThe lag moves the dates at which units change hands. It does not shift, resample "
              "or\notherwise touch the price panel, so nothing an instrument earned moves with it.")

    else:
        raise ValueError(f"unknown case: {local_test!r}")


if __name__ == '__main__':

    for case in LocalTest:
        print(f"\n===== {case.name} =====")
        run_local_test(local_test=case)

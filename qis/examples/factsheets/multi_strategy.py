"""
multi strategy backtest is generated using same strategy with a set of different model parameters
"""
# packages
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List
from enum import Enum
import yfinance as yf
import qis
from qis import TimePeriod, MultiPortfolioData, generate_multi_portfolio_factsheet, fetch_default_report_kwargs


def fetch_universe_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    define custom universe with asset class grouping
    """
    universe_data = dict(SPY='Equities',
                         QQQ='Equities',
                         EEM='Equities',
                         TLT='Bonds',
                         IEF='Bonds',
                         LQD='Credit',
                         HYG='HighYield',
                         GLD='Gold')
    tickers = list(universe_data.keys())
    group_data = pd.Series(universe_data)  # for portfolio reporting
    prices = yf.download(tickers=tickers, start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True)['Close'][tickers]
    prices = prices.asfreq('B', method='ffill')
    benchmark_prices = prices[['SPY', 'TLT']]
    return prices, benchmark_prices, group_data


def generate_volparity_multi_strategy(prices: pd.DataFrame,
                                      benchmark_prices: pd.DataFrame,
                                      group_data: pd.Series,
                                      time_period: TimePeriod,
                                      spans: List[int] = (5, 10, 20, 40, 60, 120),
                                      vol_target: float = 0.15,
                                      rebalancing_costs: float = 0.0010
                                      ) -> MultiPortfolioData:
    """
    generate volparity sensitivity to span
    """
    returns = qis.to_returns(prices=prices, is_log_returns=True)

    portfolio_datas = []
    for span in spans:
        ra_returns, weights, ewm_vol = qis.compute_ra_returns(returns=returns, span=span, vol_target=vol_target)
        weights = weights.divide(weights.sum(1), axis=0)
        portfolio_data = qis.backtest_model_portfolio(prices=prices,
                                                      weights=time_period.locate(weights),
                                                      rebalancing_costs=rebalancing_costs,
                                                      ticker=f"VP span={span}")
        portfolio_data.set_group_data(group_data=group_data, group_order=list(group_data.unique()))
        portfolio_datas.append(portfolio_data)

    multi_portfolio_data = MultiPortfolioData(portfolio_datas, benchmark_prices=benchmark_prices)
    return multi_portfolio_data


class UnitTests(Enum):
    VOLPARITY_SPAN = 1


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.VOLPARITY_SPAN:
        # time period for portfolio reporting
        time_period = qis.TimePeriod('31Dec2005', '21Apr2025')

        prices, benchmark_prices, group_data = fetch_universe_data()
        multi_portfolio_data = generate_volparity_multi_strategy(prices=prices,
                                                                 benchmark_prices=benchmark_prices,
                                                                 group_data=group_data,
                                                                 time_period=time_period,
                                                                 vol_target=0.15,
                                                                 rebalancing_costs=0.0010  # per traded volume
                                                                 )
        weights = multi_portfolio_data.get_grouped_weights(group_data=group_data)
        print(weights)

        figs = generate_multi_portfolio_factsheet(multi_portfolio_data=multi_portfolio_data,
                                                  time_period=time_period,
                                                  add_group_exposures_and_pnl=True,
                                                  **fetch_default_report_kwargs(time_period=time_period))

        qis.save_fig(fig=figs[0], file_name=f"multi_strategy", local_path=qis.local_path.get_output_path())

        qis.save_figs_to_pdf(figs=figs,
                             file_name=f"volparity_span_factsheet_long",
                             orientation='landscape',
                             local_path=qis.local_path.get_output_path())

        time_period_short = TimePeriod('31Dec2019', time_period.end)
        figs = generate_multi_portfolio_factsheet(multi_portfolio_data=multi_portfolio_data,
                                                  time_period=time_period_short,
                                                  add_group_exposures_and_pnl=True,
                                                  **fetch_default_report_kwargs(time_period=time_period_short))
        qis.save_figs_to_pdf(figs=figs,
                             file_name=f"volparity_span_factsheet_short",
                             orientation='landscape',
                             local_path=qis.local_path.get_output_path())
    # plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.VOLPARITY_SPAN

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

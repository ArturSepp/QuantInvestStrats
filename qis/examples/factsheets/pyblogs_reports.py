
import pandas as pd
from typing import Tuple, List
from enum import Enum
import yfinance as yf
import qis
from qis import TimePeriod, MultiPortfolioData
from qis.portfolio.reports.config import fetch_default_report_kwargs

# multiportfolio
from qis.portfolio.reports.multi_strategy_factseet_pybloqs import generate_multi_portfolio_factsheet_with_pyblogs
from qis.portfolio.reports.strategy_benchmark_factsheet_pybloqs import generate_strategy_benchmark_factsheet_with_pyblogs


def fetch_riskparity_universe_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
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
    prices = yf.download(tickers=tickers, start=None, end=None, ignore_tz=True)['Adj Close'][tickers]
    prices = prices.asfreq('B', method='ffill').loc['2003': ]
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
                                                      is_output_portfolio_data=True,
                                                      ticker=f"VP span={span}")
        portfolio_data.set_group_data(group_data=group_data, group_order=list(group_data.unique()))
        portfolio_datas.append(portfolio_data)

    multi_portfolio_data = MultiPortfolioData(portfolio_datas, benchmark_prices=benchmark_prices)
    return multi_portfolio_data


class UnitTests(Enum):
        MULTI_PORTFOLIO = 1
        STRATEGY_BENCHMARK = 2


def run_unit_test(unit_test: UnitTests):

    time_period = qis.TimePeriod('31Dec2005', '25Apr2024')  # time period for portfolio reporting

    prices, benchmark_prices, group_data = fetch_riskparity_universe_data()
    multi_portfolio_data = generate_volparity_multi_strategy(prices=prices,
                                                             benchmark_prices=benchmark_prices,
                                                             group_data=group_data,
                                                             time_period=time_period,
                                                             vol_target=0.15,
                                                             rebalancing_costs=0.0010  # per traded volume
                                                             )

    if unit_test == UnitTests.MULTI_PORTFOLIO:

        report = generate_multi_portfolio_factsheet_with_pyblogs(multi_portfolio_data=multi_portfolio_data,
                                                  time_period=time_period,
                                                  **fetch_default_report_kwargs(time_period=time_period))

        filename = f"{qis.local_path.get_output_path()}_volparity_span_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf"
        report.save(filename)
        print(f"saved allocation report to {filename}")

    elif unit_test == UnitTests.STRATEGY_BENCHMARK:

        report = generate_strategy_benchmark_factsheet_with_pyblogs(multi_portfolio_data=multi_portfolio_data,
                                                                    strategy_idx=-1,
                                                                    benchmark_idx=0,
                                                                    time_period=time_period,
                                                                    **fetch_default_report_kwargs(time_period=time_period))

        filename = f"{qis.local_path.get_output_path()}_volparity_pybloq_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf"
        report.save(filename)
        print(f"saved allocation report to {filename}")


if __name__ == '__main__':

    unit_test = UnitTests.STRATEGY_BENCHMARK

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

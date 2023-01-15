"""
generate and load test data using yf library
"""

import pandas as pd
import yfinance as yf
from typing import List
from enum import Enum

import qis.file_utils as fu
import qis.local_path
from qis.utils.dates import TimePeriod


LOCAL_RESOURCE_PATH = qis.local_path.get_paths()['LOCAL_RESOURCE_PATH']


def fetch_prices(tickers: List[str] = ('BTC-USD', ),
                 time_period: TimePeriod = None,
                 freq: str = None
                 ) -> pd.DataFrame:

    if len(tickers) == 1:
        prices = yf.download(tickers[0], start=None, end=None)
        prices = prices.rename({'Adj Close': 'close'}, axis=1)
    else:
        prices = {}
        for ticker in tickers:
            prices[ticker] = yf.download(ticker, start=None, end=None)['Adj Close']
        prices = pd.DataFrame.from_dict(prices, orient='columns')
    if time_period is not None:
        prices = time_period.locate(prices)
    if freq is not None:
        prices = prices.resample(freq).last().fillna(method='ffill')
    return prices


def load_etf_data() -> pd.DataFrame:
    prices = fu.load_df_from_csv(file_name='etf_prices', local_path=LOCAL_RESOURCE_PATH)
    return prices


class UnitTests(Enum):
    ETF_PRICES = 1
    TEST_LOADING = 2


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.ETF_PRICES:
        prices = fetch_prices(tickers=['SPY', 'QQQ', 'EEM', 'TLT', 'IEF', 'LQD', 'HYG', 'SHY', 'GLD'])
        print(prices)
        fu.save_df_to_csv(df=prices, file_name='etf_prices', local_path=LOCAL_RESOURCE_PATH)

    elif unit_test == UnitTests.TEST_LOADING:
        prices = load_etf_data()
        print(prices)


if __name__ == '__main__':

    unit_test = UnitTests.TEST_LOADING

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

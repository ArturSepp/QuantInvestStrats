"""
generate and load test data using yf library
"""

import pandas as pd
import yfinance as yf
from enum import Enum

import qis.file_utils as fu
import qis.local_path as local_path


RESOURCE_PATH = local_path.get_paths()['RESOURCE_PATH']


def load_etf_data() -> pd.DataFrame:
    prices = fu.load_df_from_csv(file_name='etf_prices', local_path=RESOURCE_PATH)
    return prices


class LocalTests(Enum):
    ETF_PRICES = 1
    TEST_LOADING = 2


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    if local_test == LocalTests.ETF_PRICES:
        prices = yf.download(tickers=['SPY', 'QQQ', 'EEM', 'TLT', 'IEF', 'LQD', 'HYG', 'SHY', 'GLD'],
                             start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True)['Close']
        print(prices)
        fu.save_df_to_csv(df=prices, file_name='etf_prices', local_path=RESOURCE_PATH)

    elif local_test == LocalTests.TEST_LOADING:
        prices = load_etf_data()
        print(prices)


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.ETF_PRICES)

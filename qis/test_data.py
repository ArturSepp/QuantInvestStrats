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


class UnitTests(Enum):
    ETF_PRICES = 1
    TEST_LOADING = 2


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.ETF_PRICES:
        prices = yf.download(tickers=['SPY', 'QQQ', 'EEM', 'TLT', 'IEF', 'LQD', 'HYG', 'SHY', 'GLD'],
                             start=None, end=None,
                             ignore_tz=True)['Adj Close']
        print(prices)
        fu.save_df_to_csv(df=prices, file_name='etf_prices', local_path=RESOURCE_PATH)

    elif unit_test == UnitTests.TEST_LOADING:
        prices = load_etf_data()
        print(prices)


if __name__ == '__main__':

    unit_test = UnitTests.ETF_PRICES

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

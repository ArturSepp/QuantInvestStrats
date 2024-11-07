import pandas as pd
import qis as qis
from typing import List
import yfinance as yf
from enum import Enum


LOCAL_PATH = f"{qis.get_resource_path()}sp500//"
SP500_FILE = "S&P 500 Historical Components & Changes(08-17-2024).csv" # source: https://github.com/fja05680/sp500


def read_universe_data() -> pd.DataFrame:
    df = pd.read_csv(f"{LOCAL_PATH}{SP500_FILE}", index_col='date')
    return df


def fetch_universe_prices(tickers: List[str]) -> pd.DataFrame:
    prices = yf.download(tickers=tickers, start=None, end=None, ignore_tz=True)['Adj Close']
    return prices[tickers]


def fetch_universe_industry(tickers: List[str]) -> pd.Series:
    group_data = {}
    for ticker in tickers:
        this = yf.Ticker(ticker).info
        if 'sector' in this:
            group_data[ticker] = this['sector']
        else:
            group_data[ticker] = 'unclassified'
    return pd.Series(group_data)


def create_inclusion_indicators(universe: pd.DataFrame) -> pd.DataFrame:
    inclusion_indicators = {}
    for date in universe.index:
        tickers = universe.loc[date, :].apply(lambda x: sorted(x.split(','))).to_list()[0]
        inclusion_indicators[date] = pd.Series(1.0, index=tickers)
    inclusion_indicators = pd.DataFrame.from_dict(inclusion_indicators, orient='index').sort_index()
    return inclusion_indicators


def create_sp500_universe():
    universe = pd.read_csv(f"{LOCAL_PATH}{SP500_FILE}", index_col='date')
    inclusion_indicators = create_inclusion_indicators(universe)
    prices = fetch_universe_prices(tickers=inclusion_indicators.columns.to_list())
    # remove all nans
    prices = prices.dropna(axis=1, how='all').asfreq('B', method='ffill')
    group_data = fetch_universe_industry(tickers=prices.columns.to_list())
    inclusion_indicators = inclusion_indicators[prices.columns]
    print(prices)
    print(group_data)
    qis.save_df_to_csv(df=prices, file_name='sp500_prices', local_path=LOCAL_PATH)
    qis.save_df_to_csv(df=inclusion_indicators, file_name='sp500_inclusions', local_path=LOCAL_PATH)
    qis.save_df_to_csv(df=group_data.to_frame(), file_name='sp500_groups', local_path=LOCAL_PATH)


class UnitTests(Enum):
    CREATE_inclusion_indicators = 1
    CREATE_UNIVERSE_DATA = 2


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.CREATE_inclusion_indicators:
        universe = read_universe_data()
        print(universe)
        inclusion_indicators = create_inclusion_indicators(universe)
        print(inclusion_indicators)
        qis.save_df_to_csv(df=inclusion_indicators, file_name='sp500_inclusion_indicators', local_path=LOCAL_PATH)
        #group_data = fetch_universe_industry(tickers=inclusion_indicators.columns.to_list())
        #print(group_data)

        prices = fetch_universe_prices(tickers=inclusion_indicators.columns.to_list())
        print(prices)
        qis.save_df_to_csv(df=prices, file_name='sp500_prices', local_path=LOCAL_PATH)

    elif unit_test == UnitTests.CREATE_UNIVERSE_DATA:
        create_sp500_universe()


if __name__ == '__main__':

    unit_test = UnitTests.CREATE_UNIVERSE_DATA

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

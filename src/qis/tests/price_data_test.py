"""
generate and load the local etf price file used by the run_local_test entry points

`load_etf_data` reads a csv from RESOURCE_PATH that is not distributed with the package: it is
produced locally by running LocalTest.FETCH_ETF_PRICES, which needs the [data] extra. Code that
has to run for anyone — the test suite, CI, documented examples — uses
`qis.datasets.synthetic` instead, which needs no file, no network and no vendor licence.
"""

# packages
import pandas as pd
from enum import Enum

# qis
import qis.file_utils as fu
import qis.local_path as local_path

ETF_TICKERS = ['SPY', 'QQQ', 'EEM', 'TLT', 'IEF', 'LQD', 'HYG', 'SHY', 'GLD']
ETF_PRICES_FILE = 'etf_prices'


def load_etf_data() -> pd.DataFrame:
    """
    Load the locally cached etf price panel.

    Returns:
        price panel indexed by date with one column per ticker in ``ETF_TICKERS``

    Raises:
        FileNotFoundError: if the cache has not been produced on this machine
    """
    resource_path = local_path.get_paths()['RESOURCE_PATH']
    prices = fu.load_df_from_csv(file_name=ETF_PRICES_FILE, local_path=resource_path)
    if prices is None or len(prices.index) == 0:
        raise FileNotFoundError(f"no cached etf prices under {resource_path!r}; produce them with "
                                f"run_local_test(LocalTest.FETCH_ETF_PRICES), which needs "
                                f"pip install qis[data], or use qis.datasets.synthetic instead")
    return prices


class LocalTest(Enum):
    """Enumeration of available local test cases."""
    FETCH_ETF_PRICES = 1
    LOAD_ETF_PRICES = 2


def run_local_test(local_test: LocalTest) -> None:
    """
    Run local tests for development and debugging purposes.

    FETCH_ETF_PRICES downloads real data and overwrites the local cache. It is never run by the
    test suite or by CI, and yfinance is imported inside the branch so that importing this
    module does not require the [data] extra.

    Args:
        local_test: which test case to run

    Raises:
        ImportError: if FETCH_ETF_PRICES runs without the [data] extra installed
    """
    if local_test == LocalTest.FETCH_ETF_PRICES:
        try:
            import yfinance as yf
        except ImportError as error:
            raise ImportError("fetching etf prices needs yfinance: "
                              "pip install qis[data]") from error
        resource_path = local_path.get_paths()['RESOURCE_PATH']
        prices = yf.download(tickers=ETF_TICKERS, start="2003-12-31", end=None,
                             ignore_tz=True, auto_adjust=True)['Close']
        print(prices)
        fu.save_df_to_csv(df=prices, file_name=ETF_PRICES_FILE, local_path=resource_path)

    elif local_test == LocalTest.LOAD_ETF_PRICES:
        prices = load_etf_data()
        print(prices)


if __name__ == '__main__':

    run_local_test(local_test=LocalTest.LOAD_ETF_PRICES)

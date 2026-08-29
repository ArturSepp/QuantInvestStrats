"""
Generate and load the local ETF price file used by development runners.

`load_etf_data` reads a csv from RESOURCE_PATH, falling back to the operating-system user cache
when the shipped path placeholder is unchanged. The file is not distributed with the package: it
is produced locally by running Locals.FETCH_ETF_PRICES, which needs the [data] extra. Code that
has to run for anyone — the test suite, CI, documented examples — uses
`qis.datasets.synthetic` instead, which needs no file, no network and no vendor licence.
"""

# packages
import os
import pandas as pd
from enum import Enum
from pathlib import Path

# qis
import qis.file_utils as fu
import qis.local_path as local_path

ETF_TICKERS = ['SPY', 'QQQ', 'EEM', 'TLT', 'IEF', 'LQD', 'HYG', 'SHY', 'GLD']
ETF_PRICES_FILE = 'etf_prices'


def _get_etf_resource_path() -> Path:
    """Resolve the configured resource directory or a machine-local cache fallback.

    Returns:
        Configured resource directory, unless it is the shipped ``...`` placeholder; in that
        case, a QIS resource directory under the operating-system user cache.
    """
    configured_path = local_path.get_paths()['RESOURCE_PATH']
    if configured_path and '...' not in configured_path:
        return Path(configured_path).expanduser()

    cache_root = os.environ.get('LOCALAPPDATA') or os.environ.get('XDG_CACHE_HOME')
    if cache_root is None:
        cache_root = Path.home().joinpath('.cache')
    return Path(cache_root).joinpath('qis', 'resources')


def load_etf_data() -> pd.DataFrame:
    """
    Load the locally cached etf price panel.

    Returns:
        price panel indexed by date with one column per ticker in ``ETF_TICKERS``

    Raises:
        FileNotFoundError: if the cache has not been produced on this machine
    """
    resource_path = _get_etf_resource_path()
    prices = fu.load_df_from_csv(file_name=ETF_PRICES_FILE, local_path=str(resource_path))
    if prices is None or len(prices.index) == 0:
        raise FileNotFoundError(f"no cached etf prices under {resource_path!r}; produce them with "
                                f"run_local(Locals.FETCH_ETF_PRICES), which needs "
                                f"pip install qis[data], or use qis.datasets.synthetic instead")
    return prices


class Locals(Enum):
    """Available local ETF-data diagnostics."""
    FETCH_ETF_PRICES = 1
    LOAD_ETF_PRICES = 2


def run_local(local: Locals) -> None:
    """
    Run the selected ETF-data development diagnostic.

    FETCH_ETF_PRICES downloads real data and overwrites the local cache. It is never run by the
    test suite or by CI, and yfinance is imported inside the branch so that importing this
    module does not require the [data] extra.

    Args:
        local: Development case to run.

    Raises:
        ImportError: if FETCH_ETF_PRICES runs without the [data] extra installed
    """
    if local == Locals.FETCH_ETF_PRICES:
        resource_path = _get_etf_resource_path()
        resource_path.mkdir(parents=True, exist_ok=True)
        try:
            import yfinance as yf
        except ImportError as error:
            raise ImportError("fetching etf prices needs yfinance: "
                              "pip install qis[data]") from error
        prices = yf.download(tickers=ETF_TICKERS, start="2003-12-31", end=None,
                             ignore_tz=True, auto_adjust=True)['Close']
        print(prices)
        fu.save_df_to_csv(df=prices, file_name=ETF_PRICES_FILE, local_path=str(resource_path))
        print(f"Saved ETF prices to {resource_path.joinpath(f'{ETF_PRICES_FILE}.csv')}")

    elif local == Locals.LOAD_ETF_PRICES:
        prices = load_etf_data()
        print(prices)


if __name__ == '__main__':

    run_local(local=Locals.FETCH_ETF_PRICES)

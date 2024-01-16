import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from enum import Enum
from typing import Union
import yfinance as yf
import qis


def fetch_hourly_data(ticker: str = 'BTC-USD') -> pd.DataFrame:
    """
    yf reports timestamps of bars at the start of the period: we shift it to the end of the period
    """
    asset = yf.Ticker(ticker)
    ohlc_data = asset.history(period="730d", interval="1h")
    ohlc_data.index = [t + pd.Timedelta(minutes=60) for t in ohlc_data.index]
    ohlc_data = ohlc_data.rename({'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'}, axis=1)
    ohlc_data.index = ohlc_data.index.tz_convert('UTC')
    ohlc_data.index.name = 'timestamp'
    return ohlc_data


def compute_vols(prices: pd.Series,
                 span: int = 5*24
                 ) -> pd.DataFrame:
    """
    compute rolling ewma with and without weekends
    need to adjust annualisation factor
    """
    returns = np.log(prices).diff(1)
    init_value = np.nanvar(returns, axis=0)  # set initial value to average variance
    vol = qis.compute_ewm_vol(data=returns, span=span, af=365*24, init_value=init_value)
    vol1 = qis.compute_ewm_vol(data=returns, span=span, af=260*24, init_value=init_value, is_exlude_weekends=True)
    vols = pd.concat([vol.rename('including weekends'),
                      vol1.rename('excluding weekends')], axis=1)
    return vols


class UnitTests(Enum):
    VOLS = 1


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.VOLS:
        prices = fetch_hourly_data(ticker='BTC-USD')['close']
        vol = compute_vols(prices=prices)
        with sns.axes_style("darkgrid"):
            fig, ax = plt.subplots(1, 1, figsize=(10, 7))
            qis.plot_time_series(df=vol, var_format='{:,.2%}', ax=ax)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.VOLS

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

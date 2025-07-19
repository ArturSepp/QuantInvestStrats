"""
compute split overnight and intraday returns
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple
import yfinance as yf
import qis


def compute_returns(ticker: str = 'SPY', time_period: qis.TimePeriod = None) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ohlc_data = yf.download(tickers=ticker, start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True)
    if time_period is not None:
        ohlc_data = time_period.locate(ohlc_data)
    # need to adjust open price for dividends
    adjustment_ratio = ohlc_data['Close'] / ohlc_data['Close']
    adjusted_open = adjustment_ratio.multiply(ohlc_data['Open'])
    adjusted_close = ohlc_data['Close']
    overnight_return = adjusted_open.divide(adjusted_close.shift(1)) - 1.0
    intraday_return = adjusted_close.divide(adjusted_open) - 1.0
    close_to_close_return = adjusted_close.divide(adjusted_close.shift(1)) - 1.0
    return overnight_return.rename('Overnight'), intraday_return.rename('Intraday'), close_to_close_return.rename('Close-to-Close')


def plot_split_returns(ticker: str = 'SPY',
                       time_period: qis.TimePeriod = None,
                       is_check_total: bool = False,
                       ax: plt.Subplot = None
                       ) -> None:
    overnight_return, intraday_return, close_to_close_return = compute_returns(ticker=ticker, time_period=time_period)

    if is_check_total:
        cum_performance = pd.concat([close_to_close_return.rename('Close-to-Close'),
                                     overnight_return.add(intraday_return).rename('Overnight+Intraday')
                                     ], axis=1).cumsum(0)
    else:
        cum_performance = pd.concat([overnight_return, intraday_return], axis=1).cumsum(0)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(15, 8), tight_layout=True)
    qis.plot_time_series(df=cum_performance,
                         var_format='{:,.0%}',
                         title=f"{ticker} Performance",
                         x_date_freq='YE',
                         date_format='%Y',
                         framealpha=0.9,
                         legend_stats=qis.LegendStats.LAST,
                         ax=ax)


# tickers = ['SPY', 'QQQ', 'GLD', 'TLT', 'HYG', 'MSFT']
tickers = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOG']
time_period = qis.TimePeriod('31Dec2004', None)

with sns.axes_style("darkgrid"):
    fig, axs = plt.subplots(2, len(tickers)//2, figsize=(15, 8), tight_layout=True)

for ticker, ax in zip(tickers, qis.to_flat_list(axs)):
    plot_split_returns(ticker=ticker, time_period=time_period, ax=ax)

plt.show()

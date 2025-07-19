"""
example of creating 60/40 equity bon portfolio
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import qis as qis
import yfinance as yf

tickers_weights = dict(SPY=0.6, IEF=0.4)
tickers = list(tickers_weights.keys())
prices = yf.download(tickers, start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True)['Close'][tickers]
prices = prices.asfreq('B', method='ffill').dropna()

balanced_60_40a = qis.backtest_model_portfolio(prices=prices, weights=tickers_weights, rebalancing_freq='QE',
                                              ticker='Zero Cost').get_portfolio_nav()
balanced_60_40b = qis.backtest_model_portfolio(prices=prices, weights=tickers_weights, rebalancing_freq='QE',
                                              ticker='2% Cost',
                                               management_fee=0.02).get_portfolio_nav()
navs = pd.concat([balanced_60_40a, balanced_60_40b], axis=1)

with sns.axes_style("darkgrid"):
    fig, axs = plt.subplots(2, 1, figsize=(15, 12), tight_layout=True)
    qis.plot_prices_with_dd(prices=navs, axs=axs)

plt.show()

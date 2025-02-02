
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import qis as qis
import yfinance as yf

tickers_weights = dict(SPY=0.6, IEF=0.4)
tickers = list(tickers_weights.keys())
prices = yf.download(tickers, start=None, end=None)['Close'][tickers]
prices = prices.asfreq('B', method='ffill').dropna()

balanced_60_40a = qis.backtest_model_portfolio(prices=prices, weights=tickers_weights, rebalance_freq='QE',
                                              ticker='Zero Cost').get_portfolio_nav()
balanced_60_40b = qis.backtest_model_portfolio(prices=prices, weights=tickers_weights, rebalance_freq='QE',
                                              ticker='2% Cost',
                                               management_fee=0.02).get_portfolio_nav()
navs = pd.concat([balanced_60_40a, balanced_60_40b], axis=1)
qis.plot_prices_with_dd(prices=navs)

plt.show()

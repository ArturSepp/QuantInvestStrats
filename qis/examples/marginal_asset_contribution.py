"""
example of creating 60/40 equity with and without BTC
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import qis as qis
import yfinance as yf

btc_weight = 0.02
tickers_weights_wo = {'SPY': 0.6, 'IEF': 0.4, 'BTC-USD': 0.0}
tickers_weights_with = {'SPY': 0.6-0.5*btc_weight, 'IEF': 0.4-0.5*btc_weight, 'BTC-USD': btc_weight}


tickers = list(tickers_weights_wo.keys())
prices = yf.download(tickers, start=None, end=None)['Close'][tickers].asfreq('B', method='ffill').dropna()
prices = prices.loc['31Dec2018':, :]

balanced_60_40_wo = qis.backtest_model_portfolio(prices=prices, weights=tickers_weights_wo, rebalancing_freq='QE',
                                                 ticker='100% Balanced').get_portfolio_nav()
balanced_60_40b = qis.backtest_model_portfolio(prices=prices, weights=tickers_weights_with, rebalancing_freq='QE',
                                               ticker=f"{1.0-btc_weight:0.0%} Balanced / {btc_weight:0.0%} BTC").get_portfolio_nav()
navs = pd.concat([balanced_60_40_wo, balanced_60_40b], axis=1)

perf_params = qis.PerfParams(freq='ME')
with sns.axes_style("darkgrid"):
    fig = plt.figure(figsize=(15, 12), constrained_layout=True)
    gs = fig.add_gridspec(nrows=7, ncols=1, wspace=0.0, hspace=0.0)
    axs = [fig.add_subplot(gs[1:4, 0]), fig.add_subplot(gs[4:, 0])]
    qis.plot_prices_with_dd(prices=navs, perf_params=perf_params, axs=axs)
    qis.plot_ra_perf_table_benchmark(prices=navs,
                                     benchmark=balanced_60_40_wo.name,
                                     perf_params=perf_params,
                                     title='Risk-adjusted Performance Table',
                                     digits_to_show=1,
                                     ax=fig.add_subplot(gs[0, 0]))

plt.show()

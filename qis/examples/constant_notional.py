"""
run simulation of short exposure strategies
follow the discussion in
https://twitter.com/ArturSepp/status/1614551490093359104
to run the code, install qis package (Quantitative Investment Strategies):
pip install qis
"""

# imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
import qis

# load dprice data for given ticker
ticker = 'SPY'
price = yf.download(ticker, start=None, end=None)['Adj Close'].rename(ticker)
price = price.loc['2016':]
price_np = price.to_numpy()

# specify position
long_or_short = -1.0
constant_notional = 1.0

# start backtest
# this will track constant notional strategy
constant_notional_units, constant_notional_cum_nav = np.zeros_like(price_np), np.zeros_like(price_np)
constant_notional_units[0], constant_notional_cum_nav[0] = long_or_short * constant_notional / price_np[0], constant_notional

# this will track constant nav exposure strategy which is equivalent to long or short leverage ETF
short_etf_units, short_etf_cum_nav = np.zeros_like(price_np), np.zeros_like(price_np)
short_etf_units[0], short_etf_cum_nav[0] = long_or_short * constant_notional / price_np[0], constant_notional

for idx, (price0, price1) in enumerate(zip(price_np[:-1], price_np[1:])):
    # time_t = idx+1
    constant_notional_cum_nav[idx+1] = constant_notional_cum_nav[idx] + constant_notional_units[idx] * (price1-price0)
    # exposure in unit is constant for the next step and computed using eod price
    constant_notional_units[idx+1] = long_or_short*constant_notional / price1

    short_etf_cum_nav[idx+1] = short_etf_cum_nav[idx] + short_etf_units[idx] * (price1-price0)
    # etf is rebalanced proportionally to the current eod nav
    short_etf_units[idx+1] = long_or_short*short_etf_cum_nav[idx+1]/price1

# store
constant_notional_cum_nav = pd.Series(constant_notional_cum_nav, index=price.index, name='constant_notional_cum_nav')
short_etf_cum_nav = pd.Series(short_etf_cum_nav, index=price.index, name='short_etf_cum_nav')

# add simple byu and hold starts
buy_and_hold_units = np.ones_like(price_np) * constant_notional / price_np[0]
buy_and_hold_cum_nav = pd.Series(constant_notional + buy_and_hold_units*(price_np-price_np[0]), index=price.index, name='buy_and_hold_cum_nav')

sell_and_hold_units = - np.ones_like(price_np) * constant_notional / price_np[0]
sell_and_hold_cum_nav = pd.Series(constant_notional + sell_and_hold_units*(price_np-price_np[0]), index=price.index, name='sell_and_hold_cum_nav')

# prices
prices = pd.concat([buy_and_hold_cum_nav, sell_and_hold_cum_nav, constant_notional_cum_nav, short_etf_cum_nav], axis=1)

# portfolio units
buy_and_hold_units = pd.Series(buy_and_hold_units, index=price.index, name='buy_and_hold_units')
sell_and_hold_units = pd.Series(sell_and_hold_units, index=price.index, name='sell_and_hold_units')
constant_notional_units = pd.Series(constant_notional_units, index=price.index, name='constant_notional_units')
short_etf_units = pd.Series(short_etf_units, index=price.index, name='short_etf_units')
portfolio_units = pd.concat([buy_and_hold_units, sell_and_hold_units, constant_notional_units, short_etf_units], axis=1)


with sns.axes_style("darkgrid"):
    fig, axs = plt.subplots(2, 1, figsize=(9, 7), tight_layout=True)

    # plot performance
    qis.plot_prices_with_dd(prices=prices,
                            perf_stats_labels=qis.PerfStatsLabels.TOTAL.value,
                            title=f"Realized performance of strategies with short exposure to {ticker}",
                            axs=axs)

    # plot units
    fig, ax = plt.subplots(1, 1, figsize=(9, 7), tight_layout=True)
    qis.plot_time_series(df=portfolio_units,
                         title='Portfolio Units',
                         ax=ax)

    fig, ax = plt.subplots(1, 1, figsize=(9, 7), tight_layout=True)
    qis.plot_returns_scatter(prices=prices,
                             benchmark=prices.columns[0],
                             freq='W-MON',
                             return_type=qis.ReturnTypes.DIFFERENCE,  # cannot use log and relative returns
                             title='Scatterplot of weekly P&L relative to buy-and-hold',
                             ylabel='Daily P&L of short strategies',
                             xlabel='Daily P&L of buy-and-hold',
                             var_format='{:,.0%}',
                             ax=ax)

plt.show()

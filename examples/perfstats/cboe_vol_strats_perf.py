"""
Performance of Cboe's S&P 500 Market-Neutral Volatility Risk Premia
Optimized Index (SVRPO) vs SPY. Index description and methodology:
https://www.cboe.com/us/indices/dashboard/svrpo/

Downloads the SVRPO history from Cboe's public CSV at
https://cdn.cboe.com/api/global/us_indices/daily_prices/SVRPO_History.csv
and produces a price/drawdown plot and a weekly return scatter
vs SPY using a UST 3m rate as the risk-free leg.
"""

# imports
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yfinance as yf
import qis


# download from https://cdn.cboe.com/api/global/us_indices/daily_prices/SVRPO_History.csv
svrpo = pd.read_csv(
    'https://cdn.cboe.com/api/global/us_indices/daily_prices/SVRPO_History.csv',
    index_col='DATE', parse_dates=['DATE'], date_format='%m/%d/%Y',
)[['SVRPO']].sort_index()
# spy etf as benchmark
benchmark = 'SPY'
spy = yf.download([benchmark], start='2003-12-31', end=None, ignore_tz=True, auto_adjust=True)[
    'Close'
][benchmark]
# merge
prices = pd.concat([spy, svrpo], axis=1).dropna()
# Use the history since 2020.
prices = prices.loc['2020':, :]

# set parameters for computing performance stats including returns vols and regressions
ust_3m_rate = (
    yf.download('^IRX', start='2003-12-31', end=None, ignore_tz=True, auto_adjust=True)['Close'][
        '^IRX'
    ].dropna()
    / 100.0
)
perf_params = qis.PerfParams(freq='ME', freq_reg='W-WED', rates_data=ust_3m_rate)

# price perf
with sns.axes_style("darkgrid"):
    fig1, axs = plt.subplots(2, 1, figsize=(10, 7))
    qis.plot_prices_with_dd(prices=prices,
                            regime_benchmark=benchmark,
                            x_date_freq='QE',
                            framealpha=0.9,
                            perf_params=perf_params,
                            axs=axs)
    fig2, ax = plt.subplots(1, 1, figsize=(10, 7))
    qis.plot_returns_scatter(prices=prices,
                             benchmark=benchmark,
                             ylabel=svrpo.columns[0],
                             title='Regression of weekly returns',
                             freq='W-WED',
                             ax=ax)


plt.show()

"""
analyse performance of CBOE vol strats
use downloaded data
"""

# imports
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yfinance as yf
import qis


# download from https://cdn.cboe.com/api/global/us_indices/daily_prices/SVRPO_History.csv
svrpo = qis.load_df_from_csv(file_name='SVRPO_History', local_path=qis.get_resource_path())
# spy etf as benchmark
benchmark = 'SPY'
spy = yf.download([benchmark], start=None, end=None)['Adj Close'].rename(benchmark)
# merge
prices = pd.concat([spy, svrpo], axis=1).dropna()
# take last 3 years
prices = prices.loc['2020':, :]

# set parameters for computing performance stats including returns vols and regressions
ust_3m_rate = yf.download('^IRX', start=None, end=None)['Adj Close'].dropna() / 100.0
perf_params = qis.PerfParams(freq='ME', freq_reg='W-WED', alpha_an_factor=52.0, rates_data=ust_3m_rate)

# price perf
with sns.axes_style("darkgrid"):
    fig1, axs = plt.subplots(2, 1, figsize=(10, 7))
    qis.plot_prices_with_dd(prices=prices,
                            regime_benchmark_str=benchmark,
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
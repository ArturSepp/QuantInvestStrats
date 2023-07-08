# imports
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import qis

# skip
import qis.file_utils as fu

# define tickers and fetch price data
tickers = ['SPY', 'QQQ', 'EEM', 'TLT', 'IEF', 'LQD', 'HYG', 'GLD']
prices = yf.download(tickers, start=None, end=None)['Adj Close'][tickers].dropna()

# minimum usage
with sns.axes_style("darkgrid"):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    qis.plot_prices(prices=prices, x_date_freq='A', ax=ax)

# skip
fu.save_fig(fig, file_name='perf1', local_path="figures/")

# with drawdowns using sns styles
with sns.axes_style("darkgrid"):
    fig, axs = plt.subplots(2, 1, figsize=(10, 7))
    qis.plot_prices_with_dd(prices=prices, x_date_freq='A', axs=axs)

# skip
fu.save_fig(fig, file_name='perf2', local_path="figures/")

# risk-adjusted performance table with specified data entries
# add rates for excess Sharpe
from qis import PerfStat
ust_3m_rate = yf.download('^IRX', start=None, end=None)['Adj Close'].dropna() / 100.0

# set parameters for computing performance stats including returns vols and regressions
perf_params = qis.PerfParams(freq='M', freq_reg='Q', rates_data=ust_3m_rate)

fig = qis.plot_ra_perf_table(prices=prices,
                             perf_columns=[PerfStat.TOTAL_RETURN, PerfStat.PA_RETURN, PerfStat.VOL, PerfStat.SHARPE,
                                           PerfStat.SHARPE_EXCESS, PerfStat.MAX_DD, PerfStat.MAX_DD_VOL,
                                           PerfStat.SKEWNESS, PerfStat.KURTOSIS],
                             title=f"Risk-adjusted performance: {qis.get_time_period_label(prices, date_separator='-')}",
                             perf_params=perf_params)

# skip
fu.save_fig(fig, file_name='perf3', local_path="figures/")


plt.show()

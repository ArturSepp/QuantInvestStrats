# imports
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import qis


# define tickers and fetch price data
tickers = ['SPY', 'QQQ', 'EEM', 'TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'GLD']
prices = yf.download(tickers, start=None, end=None)['Adj Close'][tickers].dropna()

# minimum usage
with sns.axes_style("darkgrid"):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    qis.plot_prices(prices=prices, x_date_freq='YE', ax=ax)

# skip
qis.save_fig(fig, file_name='perf1', local_path="figures/")

# with drawdowns using sns styles
with sns.axes_style("darkgrid"):
    fig, axs = plt.subplots(2, 1, figsize=(10, 7))
    qis.plot_prices_with_dd(prices=prices, x_date_freq='YE', axs=axs)

# skip
qis.save_fig(fig, file_name='perf2', local_path="figures/")

# risk-adjusted performance table with specified data entries
# add rates for excess Sharpe
from qis import PerfStat
ust_3m_rate = yf.download('^IRX', start=None, end=None)['Adj Close'].dropna() / 100.0

# set parameters for computing performance stats including returns vols and regressions
perf_params = qis.PerfParams(freq='ME', freq_reg='QE', alpha_an_factor=4.0, rates_data=ust_3m_rate)
# perf_columns is list to display different perfomance metrics from enumeration PerfStat
fig = qis.plot_ra_perf_table(prices=prices,
                             perf_columns=[PerfStat.TOTAL_RETURN, PerfStat.PA_RETURN, PerfStat.PA_EXCESS_RETURN,
                                           PerfStat.VOL, PerfStat.SHARPE_RF0,
                                           PerfStat.SHARPE_EXCESS, PerfStat.SORTINO_RATIO, PerfStat.CALMAR_RATIO,
                                           PerfStat.MAX_DD, PerfStat.MAX_DD_VOL,
                                           PerfStat.SKEWNESS, PerfStat.KURTOSIS],
                             title=f"Risk-adjusted performance: {qis.get_time_period_label(prices, date_separator='-')}",
                             perf_params=perf_params)

# skip
qis.save_fig(fig, file_name='perf3', local_path="figures/")

# add benchmark regression using excess returns for linear beta
# regression frequency is specified using perf_params.freq_reg
# regression alpha is multiplied using perf_params.alpha_an_factor
fig = qis.plot_ra_perf_table_benchmark(prices=prices,
                                       benchmark='SPY',
                                       perf_columns=[PerfStat.TOTAL_RETURN, PerfStat.PA_RETURN, PerfStat.PA_EXCESS_RETURN,
                                                     PerfStat.VOL, PerfStat.SHARPE_RF0,
                                                     PerfStat.SHARPE_EXCESS, PerfStat.SORTINO_RATIO, PerfStat.CALMAR_RATIO,
                                                     PerfStat.MAX_DD, PerfStat.MAX_DD_VOL,
                                                     PerfStat.SKEWNESS, PerfStat.KURTOSIS,
                                                     PerfStat.ALPHA_AN, PerfStat.BETA, PerfStat.R2],
                                       title=f"Risk-adjusted performance: {qis.get_time_period_label(prices, date_separator='-')} benchmarked with SPY",
                                       perf_params=perf_params)
# skip
qis.save_fig(fig, file_name='perf4', local_path="figures/")

plt.show()

# imports
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import qis as qis
from qis import PerfStat

# define tickers and fetch price data
tickers = ['SPY', 'QQQ', 'EEM', 'TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'GLD']
prices = yf.download(tickers, start=None, end=None)['Adj Close'][tickers].dropna()

# define frequencies for vol computations
freqs = ['B', 'W-WED', 'ME', 'QE']

# plot scatter plot of PerfStat.SHARPE_RF0 vs PerfStat.SORTINO_RATIO
with sns.axes_style('darkgrid'):
    fig, axs = plt.subplots(len(freqs)//2, len(freqs)//2, figsize=(10, 5), tight_layout=True)
    qis.set_suptitle(fig, f"Sortino ratio vs Sharpe ratio with rf=0 as function of returns frequency")
    axs = qis.to_flat_list(axs)
    for idx, freq in enumerate(freqs):
        perf_table = qis.get_ra_perf_columns(prices=prices,
                                             perf_params=qis.PerfParams(freq=freq),
                                             perf_columns=[PerfStat.SHARPE_RF0, PerfStat.SORTINO_RATIO],
                                             is_to_str=False)
        qis.plot_scatter(df=perf_table,
                         x=PerfStat.SHARPE_RF0.to_str(),
                         y=PerfStat.SORTINO_RATIO.to_str(),
                         title=f"returns freq={freq}",
                         annotation_labels=perf_table.index.to_list(),
                         xvar_format='{:.2f}', yvar_format='{:.2f}',
                         full_sample_color='blue',
                         full_sample_order=1,
                         fit_intercept=False,
                         ax=axs[idx])

plt.show()

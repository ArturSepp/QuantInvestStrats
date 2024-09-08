"""
run bond etf risk return scatter
"""
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import qis as qis
from qis import PerfStat

# define bond etfs
bond_etfs = {'SHV': '3m UST',
             'SHY': '1-3y UST',
             'IEI': '3-7y UST',
             'IEF': '7-10y UST',
             'TLT': '20y+ UST',
             'TIP': 'TIPS',
             'MUB': 'Munis',
             'MBB': 'MBS',
             'LQD': 'IG',
             'HYG': 'HY',
             'EMB': 'EM'
             }
# fetch prices and rename to dict values
tickers = list(bond_etfs.keys())
prices = yf.download(tickers, start=None, end=None)['Adj Close'][tickers].rename(bond_etfs, axis=1)

# set parameters for computing performance stats including returns vols and regressions
ust_3m_rate = yf.download('^IRX', start=None, end=None)['Adj Close'].dropna() / 100.0
perf_params = qis.PerfParams(freq='W-WED', rates_data=ust_3m_rate)

# time period for performance measurement
time_period = qis.TimePeriod('31Dec2007', '04Sep2024')

# plot scatter plot of x vs y performance variables
xy1 = [PerfStat.VOL, PerfStat.PA_RETURN]
xy2 = [PerfStat.VOL, PerfStat.SHARPE_EXCESS]
xys = [xy1, xy2]
# xys = [xy2]

with sns.axes_style('darkgrid'):
    fig, axs = plt.subplots(1, len(xys), figsize=(12, 8), tight_layout=True)
    qis.set_suptitle(fig, f"P.a. returns vs Vol and Excess Sharpe ratio vs Vol for Fixed Income ETFs")
    axs = qis.to_flat_list(axs)
    for idx, xy in enumerate(xys):
        perf_table = qis.get_ra_perf_columns(prices=prices,
                                             perf_params=perf_params,
                                             perf_columns=xy,
                                             is_to_str=False)
        qis.plot_scatter(df=perf_table,
                         x=xy[0].to_str(), y=xy[1].to_str(),
                         title=f"{xy[0].to_str()} vs {xy[1].to_str()}",
                         annotation_labels=perf_table.index.to_list(),
                         xvar_format=xy[0].to_format(), yvar_format=xy[1].to_format(),
                         full_sample_color='blue',
                         full_sample_order=2,
                         x_limits=(0.0, None),
                         ax=axs[idx])

plt.show()
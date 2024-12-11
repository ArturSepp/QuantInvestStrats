# packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import qis as qis
from bbg_fetch import fetch_field_timeseries_per_tickers

prices = fetch_field_timeseries_per_tickers(tickers={'SPTR Index': 'SPTR'}, freq='B', field='PX_LAST').ffill()

yields = fetch_field_timeseries_per_tickers(tickers={'USGGBE10 Index': '10Y BE'}, freq='B', field='PX_LAST').ffill() / 100.0

roll_periodss = {'5y': 5*252, '10y': 10*252}
roll_periodss = {'10y': 10*252}
perfs = {}
for key, roll_periods in roll_periodss.items():
    perf, _ = qis.compute_rolling_perf_stat(prices=prices,
                                            rolling_perf_stat=qis.RollingPerfStat.PA_RETURNS,
                                            roll_freq='B',
                                            roll_periods=roll_periods)
    perfs[key] = perf.iloc[:, 0]
perfs = pd.DataFrame.from_dict(perfs, orient='columns').dropna(axis=0, how='all')

with sns.axes_style("darkgrid"):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)
    yvar_major_ticks1 = np.linspace(-0.05, 0.3, 8)
    yvar_major_ticks2 = np.linspace(0.0, 0.03, 6)
    qis.plot_time_series_2ax(df1=perfs,
                             df2=yields,
                             legend_stats=qis.LegendStats.AVG_LAST,
                             legend_stats2=qis.LegendStats.AVG_LAST,
                             title='Rolling 10y p.a. returns of SPTR Index vs US 10Y BE',
                             var_format='{:.1%}',
                             var_format_yax2='{:.1%}',
                             yvar_major_ticks1=yvar_major_ticks1,
                             yvar_major_ticks2=yvar_major_ticks2,
                             trend_line1=qis.TrendLine.ZERO_SHADOWS,
                             framealpha=0.9,
                             ax=ax)

plt.show()
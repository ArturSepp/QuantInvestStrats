
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import qis as qis
from qis import PerfStat
from bbg_fetch import fetch_field_timeseries_per_tickers

assets = {'BASPCAAA Index': 'AAA10y',
          'BICLA10Y Index': 'A10y',
          'BICLB10Y Index': 'BAA10y',
          'CSI BARC Index': 'HY 10Y'}

spreads = fetch_field_timeseries_per_tickers(tickers=list(assets.keys()), field='PX_LAST', CshAdjNormal=True)
spreads = spreads.rename(assets, axis=1)
spreads['HY 10Y'] = 100.0*spreads['HY 10Y']
# to %
spreads = spreads / 10000.0

predictors = {'SPXT Index': 'SPX', 'gt10 Govt': 'UST10y'}
benchmarks = fetch_field_timeseries_per_tickers(tickers=list(predictors.keys()), field='PX_LAST', CshAdjNormal=True)
benchmarks = benchmarks.rename(predictors, axis=1)
print(benchmarks)
qis.plot_time_series(spreads,
                     var_format='{:,.2%}', x_date_freq='YE')

freq = 'ME'

df1 = pd.concat([benchmarks['SPX'].asfreq(freq, method='ffill').pct_change(),
                 spreads.asfreq(freq, method='ffill').diff()], axis=1).dropna()

qis.plot_scatter(df=df1, x='SPX')

df2 = pd.concat([benchmarks['UST10y'].asfreq(freq, method='ffill').diff()/100.0,
                 spreads.asfreq(freq, method='ffill').diff()], axis=1).dropna()

qis.plot_scatter(df=df2, x='UST10y')

plt.show()





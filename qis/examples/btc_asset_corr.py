"""
compute rolling correlations between crypto and asset
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import qis as qis
import yfinance as yf

# define asset and cryptocurrency
ASSET = 'QQQ'
CRYPTO = 'BTC-USD'
tickers = [ASSET, CRYPTO]
# fetch yahoo data
prices = yf.download(tickers, start=None, end=None)['Adj Close'][tickers]
# resample to business days
prices = prices.asfreq('B', method='ffill').dropna()
# % returns
returns = prices.pct_change()
# returns = np.log(prices).diff()

# compute rolling correlations
corr_3m = returns[CRYPTO].rolling(63).corr(returns[ASSET]).rename('3m')
corr_1y = returns[CRYPTO].rolling(252).corr(returns[ASSET]).rename('1y')
corr_3y = returns[CRYPTO].rolling(3*252).corr(returns[ASSET]).rename('3y')
corrs = pd.concat([corr_3m, corr_1y, corr_3y], axis=1).dropna()

# select period
time_period = qis.TimePeriod('01Jan2016', None)
corrs = time_period.locate(corrs)
# qis.save_df_to_excel(data=corrs, file_name='btc_spy_corr')

with sns.axes_style("darkgrid"):
    fig, ax = plt.subplots(1, 1, figsize=(15, 8), tight_layout=True)

qis.plot_time_series(df=corrs,
                     trend_line=qis.TrendLine.ZERO_SHADOWS,
                     legend_stats=qis.LegendStats.AVG_LAST_SCORE,
                     var_format='{:.0%}',
                     fontsize=14,
                     title=f"Rolling correlation of daily returns between {CRYPTO} and {ASSET} as function of rolling window",
                     ax=ax)
# qis.save_fig(fig=fig, file_name='btc_all_corr')

plt.show()

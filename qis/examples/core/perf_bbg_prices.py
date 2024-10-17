# packages
import matplotlib.pyplot as plt
import seaborn as sns
import qis as qis

from bbg_fetch import fetch_field_timeseries_per_tickers
tickers = {'TY1 Comdty': '10y',
           'UXY1 Comdty': '10y Ultra'}

prices = fetch_field_timeseries_per_tickers(tickers=tickers, freq='B', field='PX_LAST').ffill().dropna()
print(prices)

with sns.axes_style("darkgrid"):
    fig, axs = plt.subplots(2, 1, figsize=(7, 7))
    qis.plot_prices_with_dd(prices=prices,
                            regime_benchmark_str=prices.columns[0],
                            axs=axs)
plt.show()

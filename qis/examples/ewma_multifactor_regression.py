"""
Example of using multivariate ewma betas
"""
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
import qis as qis
from qis import EwmLinearModel

tickers = ['SPY', 'TLT', 'GLD']
prices = yf.download(tickers=tickers, start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True)['Close'][tickers]
prices = prices.asfreq('B', method='ffill').dropna()  # make B frequency and align
returns = qis.to_returns(prices, is_log_returns=True, drop_first=True, freq='ME')
x = returns[['SPY', 'TLT']]
y = returns[['GLD']]


ewm_linear_model = EwmLinearModel(x=x, y=y)
ewm_linear_model.fit(span=12, is_x_correlated=True, mean_adj_type=qis.MeanAdjType.EWMA)
betas = ewm_linear_model.get_asset_factor_betas(asset=y.columns[0])
print(betas)

with sns.axes_style("darkgrid"):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    qis.plot_time_series(df=betas, ax=ax)

plt.show()

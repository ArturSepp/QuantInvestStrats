import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import qis


# select tickers
tickers = ['SPY', 'TLT', 'GLD']

# fetch prices
prices = yf.download(tickers=tickers, start=None, end=None, ignore_tz=True)['Adj Close'][tickers]
prices = prices.asfreq('B', method='ffill').dropna() # make B frequency
returns = qis.to_returns(prices, freq='ME', drop_first=True)
returns['month'] = returns.index.month
df = returns.set_index('month', drop=True)
qis.df_boxplot_by_hue_var(df=df, hue_var_name='asset', x_index_var_name='month',
                          add_hue_to_legend_title=True,
                          labels=tickers,
                          add_zero_line=True)
seasonal_returns = returns.groupby('month').agg(['mean', 'std', 'min', 'max'])
print(seasonal_returns)

plt.show()


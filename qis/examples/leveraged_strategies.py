"""
backtest example of leveraged strategies
"""

# packages
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import qis

from qis.portfolio.reports.multi_assets_factsheet import generate_multi_asset_factsheet
from qis.portfolio.reports.config import fetch_default_report_kwargs

# select tickers
benchmark = 'SPY'
tickers = [benchmark, 'SSO', 'IEF']

# fetch prices
prices = yf.download(tickers=tickers, start=None, end=None, ignore_tz=True)['Adj Close'][tickers]
prices = prices.asfreq('B', method='ffill').dropna()  # make B frequency

rebalance_freq = 'B'  # each business day
rebalancing_costs = 0.0010  # 10bp for rebalancing

# 50/50 SSO/IEF
unleveraged_portfolio = qis.backtest_model_portfolio(prices=prices[['SSO', 'IEF']],
                                                     weights={'SSO': 0.5, 'IEF': 0.5},
                                                     rebalance_freq=rebalance_freq,
                                                     rebalancing_costs=rebalancing_costs,
                                                     ticker='50/50 SSO/IEF')

# leveraged is funded at 100bp + 3m UST
funding_rate = 0.01 + yf.download('^IRX', start=None, end=None)['Adj Close'].dropna() / 100.0
leveraged_portfolio = qis.backtest_model_portfolio(prices=prices[['SPY', 'IEF']],
                                                   weights={'SPY': 1.0, 'IEF': 0.5},
                                                   rebalance_freq=rebalance_freq,
                                                   rebalancing_costs=rebalancing_costs,
                                                   funding_rate=funding_rate,
                                                   ticker='100/50 SPY/IEF')


prices = pd.concat([prices, unleveraged_portfolio, leveraged_portfolio], axis=1)

# generate report since SSO launch
time_period = qis.TimePeriod('21Jun2006', '01Sep2023')
fig = generate_multi_asset_factsheet(prices=prices,
                                     benchmark=benchmark,
                                     time_period=time_period,
                                     **fetch_default_report_kwargs(time_period=time_period))

qis.save_fig(fig=fig, file_name=f"leveraged_fund_analysis", local_path=qis.local_path.get_output_path())

qis.save_figs_to_pdf(figs=[fig],
                     file_name=f"leveraged_fund_analysis", orientation='landscape',
                     local_path=qis.local_path.get_output_path())

plt.show()

"""
backtest example of leveraged strategies
"""

# packages
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import qis as qis

# select tickers
benchmark = 'SPY'
tickers = [benchmark, 'SSO', 'IEF']

# fetch prices
prices = yf.download(tickers=tickers, start=None, end=None, ignore_tz=True)['Close'][tickers]
prices = prices.asfreq('B', method='ffill').dropna()  # make B frequency

rebalancing_freq = 'B'  # each business day
rebalancing_costs = 0.0010  # 10bp for rebalancing

# 50/50 SSO/IEF
unleveraged_portfolio = qis.backtest_model_portfolio(prices=prices[['SSO', 'IEF']],
                                                     weights={'SSO': 0.5, 'IEF': 0.5},
                                                     rebalancing_freq=rebalancing_freq,
                                                     rebalancing_costs=rebalancing_costs,
                                                     ticker='50/50 SSO/IEF').get_portfolio_nav()

# leveraged is funded at 100bp + 3m UST
funding_rate = 0.01 + yf.download('^IRX', start=None, end=None)['Close'].dropna() / 100.0
leveraged_portfolio = qis.backtest_model_portfolio(prices=prices[['SPY', 'IEF']],
                                                   weights={'SPY': 1.0, 'IEF': 0.5},
                                                   rebalancing_freq=rebalancing_freq,
                                                   rebalancing_costs=rebalancing_costs,
                                                   funding_rate=funding_rate,
                                                   ticker='100/50 SPY/IEF').get_portfolio_nav()


prices = pd.concat([prices, unleveraged_portfolio, leveraged_portfolio], axis=1)

# generate report since SSO launch
time_period = qis.TimePeriod('21Jun2006', '01Sep2023')
fig = qis.generate_multi_asset_factsheet(prices=prices,
                                         benchmark=benchmark,
                                         time_period=time_period,
                                         **qis.fetch_default_report_kwargs(time_period=time_period))

qis.save_fig(fig=fig, file_name=f"leveraged_fund_analysis", local_path=qis.local_path.get_output_path())

qis.save_figs_to_pdf(figs=[fig],
                     file_name=f"leveraged_fund_analysis", orientation='landscape',
                     local_path=qis.local_path.get_output_path())

plt.show()

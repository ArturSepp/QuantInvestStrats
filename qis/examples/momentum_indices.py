"""
run multi asset report for momentum indices: run with open Bloomberg terminal
"""
import qis as qis
from bbg_fetch import fetch_field_timeseries_per_tickers

# define momentum indices for Bloomberg fectch
momentum_indices = {'SPTR Index': 'S&P 500',
                    'MTUM US Equity': 'MTUM Mom Beta',
                    'SP500MUP Index': 'S&P500 Mom Beta',
                    'DJTMNMOT Index': 'DJ Market-Neutral Mom',
                    'PMOMENUS Index': 'BBG Market-Neutral Mom'}
prices = fetch_field_timeseries_per_tickers(tickers=list(momentum_indices.keys())).rename(momentum_indices, axis=1).dropna()

# time period for performance measurement
time_period = qis.TimePeriod('31Dec2019', '18Jul2024')
fig = qis.generate_multi_asset_factsheet(prices=prices,
                                         benchmark='S&P 500',
                                         time_period=time_period,
                                         **qis.fetch_default_report_kwargs(time_period=time_period))
# save report to pdf
qis.save_figs_to_pdf(figs=[fig], file_name=f"momentum_indices_report", orientation='landscape')

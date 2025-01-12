"""
analysis of Oakmark Select Fund Investor shareclass (expense ratio=1%)
"""
import qis as qis
from bbg_fetch import fetch_field_timeseries_per_tickers

# Define benchmark and fetch prices
benchmark = 'SPY'
tickers = {'SPY US Equity': benchmark, 'OAKLX US Equity': 'Oakmark Select Inv'}
prices = fetch_field_timeseries_per_tickers(tickers=tickers, field='PX_LAST', CshAdjNormal=True, freq='B')

# defined time period for analysis and run factcheet
time_period = qis.TimePeriod('01Nov1996', '10Jan2025')
fig = qis.generate_multi_asset_factsheet(prices=prices, benchmark=benchmark,
                                         time_period=time_period,
                                         **qis.fetch_default_report_kwargs(reporting_frequency=qis.ReportingFrequency.MONTHLY))
qis.save_figs_to_pdf(figs=[fig],
                     file_name=f"oakmark_report", orientation='landscape',
                     local_path=qis.local_path.get_output_path())

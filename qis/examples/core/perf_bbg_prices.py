# packages
import matplotlib.pyplot as plt
import qis as qis

from bbg_fetch import fetch_field_timeseries_per_tickers

# tickers = {'TY1 Comdty': '10y', 'UXY1 Comdty': '10y Ultra'}
tickers = {
    'SPTR Index': 'SPTR Index',
    'CIEQVEHG Index': 'Citi SPX 0D Vol Carry',
    'CIEQVRUG Index': 'Citi SX5E 1W Vol Carry',
    'CICXCOSE  Index': 'Citi Brent Vol Carry',
    'GSISXC07 Index': 'GS Multi Asset Carry',
    'GSISXC11 Index': 'GS Macro Carry'}


prices = fetch_field_timeseries_per_tickers(tickers=tickers, freq='B', field='PX_LAST').ffill()
print(prices)

time_period = qis.TimePeriod('31Dec2019', '06Nov2024')
kwargs = qis.fetch_default_report_kwargs(time_period=time_period)

fig = qis.generate_multi_asset_factsheet(prices=prices,
                                         benchmark='SPTR Index',
                                         time_period=time_period,
                                         **kwargs)
qis.save_figs_to_pdf(figs=[fig],
                     file_name=f"bbg_multiasset_report", orientation='landscape',
                     local_path='C://Users//uarts//outputs//')

plt.show()

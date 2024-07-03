"""
performance report for a universe of several assets
with comparison to 1-2 benchmarks
output is one-page figure with key numbers
"""
# packages
import matplotlib.pyplot as plt
import yfinance as yf
import qis as qis
from enum import Enum
from qis.portfolio.reports.multi_assets_factsheet import generate_multi_asset_factsheet
from qis.portfolio.reports.config import fetch_default_report_kwargs


class UnitTests(Enum):
    CORE_ETFS = 1
    BTC_SQQQ = 2
    HEDGED_ETFS = 3
    BBG = 4
    RATES_FUTURES = 5
    HYG_ETFS = 6


def run_unit_test(unit_test: UnitTests):

    prices = None

    if unit_test == UnitTests.CORE_ETFS:
        benchmark = 'SPY'
        tickers = [benchmark, 'QQQ', 'EEM', 'TLT', 'IEF', 'LQD', 'HYG', 'SHY', 'GLD']
        time_period = qis.TimePeriod('31Dec2007', '31May2024')  # time period for reporting

    elif unit_test == UnitTests.BTC_SQQQ:
        benchmark = 'QQQ'
        tickers = [benchmark, 'BTC-USD', 'TQQQ', 'SQQQ']
        time_period = qis.TimePeriod('31Dec2019', '31May2024')

    elif unit_test == UnitTests.HEDGED_ETFS:
        benchmark = 'SPY'
        tickers = [benchmark, 'SHY', 'LQDH', 'HYGH', 'FLOT']
        time_period = qis.TimePeriod('27May2014', '31May2024')

    elif unit_test == UnitTests.BBG:
        benchmark = 'SPTR'
        tickers = {'SPTR Index': benchmark,
                   'SGEPMLU Index': 'SG AI long/short', 'SGEPMLUL Index': 'SG AI long', 'SGEPMLUS Index': 'SG AI short',
                   'SGIXAI Index': 'SG Alpha',
                   'AIPEX Index': 'HSBC Aipex',
                   'MQQFUSAI Index': 'MerQube AI',
                   'CSRPAIS Index': 'CS RavenPack',
                   'CITIMSTR Index': 'Citi Grand'}
        time_period = qis.TimePeriod('31Dec2019', '20Jun2024')
        from bbg_fetch import fetch_field_timeseries_per_tickers
        prices = fetch_field_timeseries_per_tickers(tickers=list(tickers.keys()), field='PX_LAST', CshAdjNormal=True).dropna()
        prices = prices.rename(tickers, axis=1)

    elif unit_test == UnitTests.RATES_FUTURES:
        benchmark = '2y UST'
        tickers = {'TU1 Comdty': benchmark,
                   'ED5 Comdty': 'USD IR',
                   'L 5 Comdty': 'GBP IR',
                   'ER4 Comdty': 'EUR IR',
                   'IR4 Comdty': 'AUD IR',
                   'BA4 Comdty': 'CAD IR'}
        time_period = qis.TimePeriod(start='02Apr1986', end='31Mar2023')
        from bbg_fetch import fetch_field_timeseries_per_tickers
        prices = fetch_field_timeseries_per_tickers(tickers=list(tickers.keys()), field='PX_LAST', CshAdjNormal=True).dropna()
        prices = prices.rename(tickers, axis=1)

    elif unit_test == UnitTests.HYG_ETFS:
        benchmark = 'IBOXHY'
        tickers = {'IBOXHY Index': benchmark,
                   'HYDB US Equity': 'HYDB',
                   'HYG US Equity': 'HYG'
                   }
        from bbg_fetch import fetch_field_timeseries_per_tickers
        prices = fetch_field_timeseries_per_tickers(tickers=list(tickers.keys()), field='PX_LAST', CshAdjNormal=True).dropna()
        prices = prices.rename(tickers, axis=1)
        time_period = qis.get_time_period(prices)


    else:
        raise NotImplementedError

    if prices is None:
        prices = yf.download(tickers=tickers, start=None, end=None, ignore_tz=True)['Adj Close'][tickers]

    prices = prices.asfreq('B', method='ffill')  # make B frequency
    prices = prices.dropna()
    print(prices)

    kwargs = fetch_default_report_kwargs(time_period=time_period)

    fig = generate_multi_asset_factsheet(prices=prices,
                                         benchmark=benchmark,
                                         time_period=time_period,
                                         **kwargs)
    qis.save_figs_to_pdf(figs=[fig],
                         file_name=f"multiasset_report", orientation='landscape',
                         local_path=qis.local_path.get_output_path())
    qis.save_fig(fig=fig, file_name=f"multiassets", local_path=qis.local_path.get_output_path())

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.HYG_ETFS

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

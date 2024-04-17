"""
performance report for a universe of several assets
with comparison to 1-2 benchmarks
output is one-page figure with key numbers
"""
# packages
import matplotlib.pyplot as plt
from enum import Enum
import yfinance as yf
import qis
from qis.portfolio.reports.multi_assets_factsheet import generate_multi_asset_factsheet
from qis.portfolio.reports.config import fetch_default_report_kwargs


class UnitTests(Enum):
    CORE_ETFS = 1
    BTC_SQQQ = 2


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.CORE_ETFS:

        benchmark = 'SPY'
        tickers = [benchmark, 'QQQ', 'EEM', 'TLT', 'IEF', 'LQD', 'HYG', 'SHY', 'GLD']
        time_period = qis.TimePeriod('31Dec2007', '16Apr2024')  # time period for reporting

    elif unit_test == UnitTests.BTC_SQQQ:
        benchmark = 'QQQ'
        tickers = [benchmark, 'BTC-USD', 'TQQQ', 'SQQQ']
        time_period = qis.TimePeriod('31Dec2019', '16Apr2024')

    else:
        raise NotImplementedError

    prices = yf.download(tickers=tickers, start=None, end=None, ignore_tz=True)['Adj Close'][tickers]
    prices = prices.asfreq('B', method='ffill')  # make B frequency
    fig = generate_multi_asset_factsheet(prices=prices,
                                         benchmark=benchmark,
                                         time_period=time_period,
                                         **fetch_default_report_kwargs(time_period=time_period))
    qis.save_figs_to_pdf(figs=[fig],
                         file_name=f"multiasset_report", orientation='landscape',
                         local_path=qis.local_path.get_output_path())
    qis.save_fig(fig=fig, file_name=f"multiassets", local_path=qis.local_path.get_output_path())

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.CORE_ETFS

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

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

from qis.portfolio.reports.config import PERF_PARAMS, REGIME_PARAMS
from qis.portfolio.reports.multi_assets_factsheet import generate_multi_asset_factsheet


class UnitTests(Enum):
    CORE_ETFS = 1


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.CORE_ETFS:

        benchmark = 'SPY'
        tickers = ['SPY', 'QQQ', 'EEM', 'TLT', 'IEF', 'LQD', 'HYG', 'SHY', 'GLD']
        prices = yf.download(tickers=tickers, start=None, end=None, ignore_tz=True)['Adj Close'][tickers]
        time_period = qis.TimePeriod('31Dec2007', '21Jul2023')  # time period for reporting

        fig = generate_multi_asset_factsheet(prices=prices,
                                             benchmark=benchmark,
                                             heatmap_freq='A',
                                             perf_params=PERF_PARAMS,
                                             time_period=time_period,
                                             regime_params=REGIME_PARAMS)
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

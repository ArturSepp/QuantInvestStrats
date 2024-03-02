import matplotlib.pyplot as plt
from enum import Enum
import yfinance as yf
import qis as qis


class UnitTests(Enum):
    CORR1 = 1


def run_unit_test(unit_test: UnitTests):

    tickers = ['SPY', 'QQQ', 'TLT', 'GLD']
    prices = yf.download(tickers, start=None, end=None)['Adj Close'][tickers].dropna()

    if unit_test == UnitTests.CORR1:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7), tight_layout=True)
        qis.plot_returns_ewm_corr_table(prices=prices,
                                        ewm_lambda=0.97,
                                        is_average=False,
                                        ax=ax)
    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.CORR1

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

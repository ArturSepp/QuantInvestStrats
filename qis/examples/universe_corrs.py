import matplotlib.pyplot as plt
from enum import Enum
import yfinance as yf
import qis as qis


class LocalTests(Enum):
    CORR1 = 1


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    tickers = ['SPY', 'QQQ', 'TLT', 'GLD']
    prices = yf.download(tickers, start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True)['Close'][tickers].dropna()

    if local_test == LocalTests.CORR1:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7), tight_layout=True)
        qis.plot_returns_ewm_corr_table(prices=prices,
                                        ewm_lambda=0.97,
                                        ax=ax)
    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.CORR1)

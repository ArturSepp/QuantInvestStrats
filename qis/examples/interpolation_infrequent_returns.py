"""
illustrate interpolation of infrequent returns
"""
# packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import qis as qis


class UnitTests(Enum):
    YF = 1
    BBG = 2


def run_unit_test(unit_test: UnitTests):

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    if unit_test == UnitTests.YF:
        import yfinance as yf
        pivot = 'SPY'
        asset = 'QQQ'
        tickers = [pivot, asset]
        prices = yf.download(tickers=tickers, start=None, end=None)['Close']

    elif unit_test == UnitTests.BBG:
        from bbg_fetch import fetch_field_timeseries_per_tickers
        pivot = 'SPTR Index'
        asset = 'XNDX Index'
        tickers = [pivot, asset]
        prices = fetch_field_timeseries_per_tickers(tickers=tickers, freq='B', field='PX_LAST').ffill()

    else:
        raise NotImplementedError

    is_log_returns = True
    infrequent_returns = qis.to_returns(prices[asset], is_log_returns=is_log_returns, freq='QE')
    pivot_returns = qis.to_returns(prices[pivot], is_log_returns=is_log_returns, freq='ME')
    i_backfill = qis.interpolate_infrequent_returns(infrequent_returns=infrequent_returns.dropna(), pivot_returns=pivot_returns,
                                                    is_to_log_returns=False)

    known_returns = qis.to_returns(prices[asset], is_log_returns=is_log_returns, freq='ME')
    returns = pd.concat([pivot_returns, known_returns.rename(f"{asset} actual"), i_backfill.rename(f"{asset} interpolated")], axis=1).dropna()
    if is_log_returns:
        returns = np.expm1(returns)
    print(f"means={np.nanmean(returns, axis=0)}, stdevs={np.nanstd(returns, axis=0)}")

    navs = qis.returns_to_nav(returns)
    qis.plot_time_series(navs)
    returns = pd.concat([returns.rolling(3).sum(), infrequent_returns.rename('Q')], axis=1)
    print(returns)

    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    gs = fig.add_gridspec(nrows=3, ncols=2, wspace=0.0, hspace=0.0)
    qis.plot_ra_perf_table(prices=navs, perf_params=qis.PerfParams(freq='ME'), title='Monthly Sampling', ax=fig.add_subplot(gs[0, :]))
    qis.plot_ra_perf_table(prices=navs, perf_params=qis.PerfParams(freq='QE'), title='Quarterly Sampling', ax=fig.add_subplot(gs[1, :]))

    qis.plot_returns_corr_table(prices=navs, freq='ME', title='Monthly Sampling', ax=fig.add_subplot(gs[2, 0]))
    qis.plot_returns_corr_table(prices=navs, freq='QE', title='Quarterly Sampling', ax=fig.add_subplot(gs[2, 1]))

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.YF

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

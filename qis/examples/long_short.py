# packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum

# qis
import qis as qis
from qis import PerfParams, TimePeriod

# project
import yfinance as yf


class UnitTests(Enum):
    LONG_IEF_SHORT_LQD = 1


def run_unit_test(unit_test: UnitTests):

    perf_params = PerfParams(freq_drawdown='B', freq='B')
    kwargs = dict(fontsize=12, digits_to_show=1, sharpe_digits=2,
                  alpha_format='{0:+0.0%}',
                  beta_format='{:0.1f}',
                  perf_stats_labels=qis.PerfStatsLabels.TOTAL_DETAILED.value,
                  framealpha=0.9)

    time_period = TimePeriod('31Dec2021', '19Sep2022')

    if unit_test == UnitTests.LONG_IEF_SHORT_LQD:
        prices = yf.download(tickers=['IEF', 'LQD', 'LQDH', 'IGIB'], start=None, end=None)['Adj Close']
        prices = time_period.locate(prices)
        rets = qis.to_returns(prices=prices, is_first_zero=True)
        navs1 = qis.returns_to_nav(returns=4.0*(rets.iloc[:, 0] - rets.iloc[:, 1]).rename('4x(Long 10y ETF / Short IG ETF)'))
        navs2 = qis.returns_to_nav(returns=-4.0*rets.iloc[:, 2].rename('4x(Short LQDH)'))
        navs3 = qis.returns_to_nav(returns=4.0*(0.5*rets.iloc[:, 0] - rets.iloc[:, 3]).rename('4x(Long 10y ETF / Short IGIB)'))
        spy = time_period.locate(yf.download(tickers=['SPY'])['Adj Close'])
        prices2 = pd.concat([navs1, navs2, navs3, spy], axis=1)

        with sns.axes_style('darkgrid'):
            fig1, axs = plt.subplots(2, 1, figsize=(15, 5), constrained_layout=True)
            qis.plot_prices_with_dd(prices=prices, perf_params=perf_params, axs=axs)

            fig2, ax = plt.subplots(1, 1, figsize=(12, 8), constrained_layout=True)
            qis.plot_prices(prices=prices2, perf_params=perf_params,
                            title='2022 YTD total performance of 4x(Long 10y ETF / Short IG ETF) vs SPY ETF',
                            ax=ax, **kwargs)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.LONG_IEF_SHORT_LQD

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

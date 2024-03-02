# packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum

# qis
import qis.utils.dates as da
from qis.perfstats.config import PerfParams
import qis.perfstats.returns as ret
import qis.plots.derived.prices as pdp

# project
import yfinance as yf


class UnitTests(Enum):
    LONG_IEF_SHORT_LQD = 1


def run_unit_test(unit_test: UnitTests):

    perf_params = PerfParams(freq_drawdown='B', freq='B')
    kwargs = dict(fontsize=12, digits_to_show=1, sharpe_digits=2,
                  alpha_format='{0:+0.0%}',
                  beta_format='{:0.1f}',
                  performance_label=pdp.PerformanceLabel.TOTAL_DETAILED,
                  framealpha=0.9)

    time_period = da.TimePeriod('31Dec2021', '19Sep2022')

    if unit_test == UnitTests.LONG_IEF_SHORT_LQD:
        prices = yf.download(tickers=['IEF', 'LQD', 'LQDH', 'IGIB'], start=None, end=None)['Adj Close']
        prices = time_period.locate(prices)
        rets = ret.to_returns(prices=prices, is_first_zero=True)
        navs1 = ret.returns_to_nav(returns=4.0*(rets.iloc[:, 0] - rets.iloc[:, 1]).rename('4x(Long 10y ETF / Short IG ETF)'))
        navs2 = ret.returns_to_nav(returns=-4.0*rets.iloc[:, 2].rename('4x(Short LQDH)'))
        navs3 = ret.returns_to_nav(returns=4.0*(0.5*rets.iloc[:, 0] - rets.iloc[:, 3]).rename('4x(Long 10y ETF / Short IGIB)'))
        spy = time_period.locate(yf.download(tickers=['SPY'])['Adj Close'])
        prices2 = pd.concat([navs1, navs2, navs3, spy], axis=1)

        with sns.axes_style('darkgrid'):
            fig1, axs = plt.subplots(2, 1, figsize=(15, 5), constrained_layout=True)
            pdp.plot_prices_with_dd(prices=prices, perf_params=perf_params, axs=axs)

            fig2, ax = plt.subplots(1, 1, figsize=(12, 8), constrained_layout=True)
            pdp.plot_prices(prices=prices2, perf_params=perf_params,
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

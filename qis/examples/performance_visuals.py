# built in
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from enum import Enum

# qis
import qis


def generate_performances(prices: pd.DataFrame,
                          regime_benchmark_str: str,
                          perf_params: qis.PerfParams = None,
                          performance_label: qis.PerformanceLabel = qis.PerformanceLabel.WITH_DD,
                          ) -> None:

    kwargs = dict(digits_to_show=1, framealpha=0.75, performance_label=performance_label)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4), tight_layout=True)
    qis.plot_ra_perf_table(prices=prices,
                          perf_columns=qis.EXTENDED_TABLE_COLUMNS,
                          perf_params=perf_params,
                          ax=ax)

    fig, ax = plt.subplots(1, 1, figsize=(6, qis.calc_table_height(num_rows=len(prices.columns), scale=0.4)), tight_layout=True)
    qis.plot_periodic_returns_table(prices=prices,
                                                freq='A',
                                                ax=ax,
                                                title=f"Monthly Performance: {qis.get_time_period_label(prices, date_separator='-')}",
                                                total_name='YTD',
                                                **{'square': False, 'x_rotation': 90})

    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        qis.plot_prices(prices=prices,
                        regime_benchmark_str=regime_benchmark_str,
                        perf_params=perf_params,
                        ax=ax,
                        **kwargs)

        fig, axs = plt.subplots(2, 1, figsize=(7, 7))
        qis.plot_prices_with_dd(prices=prices,
                                regime_benchmark_str=regime_benchmark_str,
                                perf_params=perf_params,
                                axs=axs,
                                **kwargs)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        qis.plot_scatter_regression(prices=prices,
                                    regime_benchmark_str=regime_benchmark_str,
                                    perf_params=perf_params,
                                    title='Regime Conditional Regression',
                                    ax=ax,
                                    **kwargs)


class UnitTests(Enum):
    ETF_DATA = 1


def run_unit_test(unit_test: UnitTests):

    tickers = ['SPY', 'QQQ', 'EEM', 'TLT', 'IEF', 'LQD', 'HYG', 'SHY', 'GLD']
    prices = yf.download(tickers, start=None, end=None)['Adj Close'].dropna()
    print(prices)

    ust_3m_rate = yf.download('^IRX', start=None, end=None)['Adj Close'].dropna()
    print(ust_3m_rate)
    perf_params = qis.PerfParams(freq='W-WED', freq_reg='M', freq_drawdown='B', rates_data=ust_3m_rate)

    if unit_test == UnitTests.ETF_DATA:
        generate_performances(prices=prices,
                              perf_params=perf_params,
                              regime_benchmark_str='SPY')

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.ETF_DATA

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

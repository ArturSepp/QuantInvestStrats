# built in
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum

import qis.utils as qu
import qis.plots as qp
import qis.perfstats as qs


# data
from qis.data.yf_data import load_etf_data, fetch_prices
from qis.data.ust_rates import load_ust_3m_rate


def generate_performances(prices: pd.DataFrame,
                          regime_benchmark_str: str,
                          perf_params: qs.PerfParams = None,
                          performance_label: qp.PerformanceLabel = qp.PerformanceLabel.WITH_DD,
                          ) -> None:

    kwargs = dict(digits_to_show=1, framealpha=0.75, performance_label=performance_label)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4), tight_layout=True)
    qp.plot_ra_perf_table(prices=prices,
                          perf_columns=qs.EXTENDED_TABLE_COLUMNS,
                          perf_params=perf_params,
                          ax=ax)

    fig, ax = plt.subplots(1, 1, figsize=(6, qp.calc_table_height(num_rows=len(prices.columns), scale=0.4)), tight_layout=True)
    qp.plot_periodic_returns_table_from_prices(prices=prices,
                                                freq='A',
                                                ax=ax,
                                                title=f"Monthly Performance: {qu.get_time_period_label(prices, date_separator='-')}",
                                                total_name='YTD',
                                                **{'square': False, 'x_rotation': 90})

    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        qp.plot_prices(prices=prices,
                        regime_benchmark_str=regime_benchmark_str,
                        perf_params=perf_params,
                        ax=ax,
                        **kwargs)

        fig, axs = plt.subplots(2, 1, figsize=(7, 7))
        qp.plot_prices_with_dd(prices=prices,
                                regime_benchmark_str=regime_benchmark_str,
                                perf_params=perf_params,
                                axs=axs,
                                **kwargs)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        qp.plot_scatter_regression(prices=prices,
                                    regime_benchmark_str=regime_benchmark_str,
                                    perf_params=perf_params,
                                    title='Regime Conditional Regression',
                                    ax=ax,
                                    **kwargs)


class UnitTests(Enum):
    ETF_DATA = 1


def run_unit_test(unit_test: UnitTests):

    prices = load_etf_data().dropna()
    ust_3m_rate = load_ust_3m_rate()

    perf_params = qs.PerfParams(freq='W-WED', freq_reg='M', freq_drawdown='B', rates_data=ust_3m_rate)

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

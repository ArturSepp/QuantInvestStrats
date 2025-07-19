# packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from enum import Enum
from typing import List
import qis as qis
from qis import PerfStat


# define entries to show in ra perf table

RA_TABLE_COLUMNS = (PerfStat.START_DATE,
                    PerfStat.END_DATE,
                    PerfStat.TOTAL_RETURN,
                    PerfStat.PA_RETURN,
                    PerfStat.VOL,
                    PerfStat.SHARPE_RF0,
                    PerfStat.SHARPE_EXCESS,
                    PerfStat.MAX_DD,
                    PerfStat.MAX_DD_VOL,
                    PerfStat.SKEWNESS,
                    PerfStat.KURTOSIS,
                    PerfStat.ALPHA,
                    PerfStat.BETA,
                    PerfStat.R2)


def generate_performances(prices: pd.DataFrame,
                          regime_benchmark: str,
                          perf_params: qis.PerfParams = None,
                          perf_columns: List[PerfStat] = RA_TABLE_COLUMNS,
                          heatmap_freq: str = 'YE',
                          **kwargs
                          ) -> None:

    local_kwargs = dict(digits_to_show=1,
                        framealpha=0.75,
                        perf_stats_labels=qis.PerfStatsLabels.DETAILED_WITH_DD.value)
    kwargs = qis.update_kwargs(kwargs, local_kwargs)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4), tight_layout=True)

    qis.plot_ra_perf_table_benchmark(prices=prices,
                                     benchmark=regime_benchmark,
                                     perf_params=perf_params,
                                     perf_columns=perf_columns,
                                     title=f"Risk-adjusted performance: {qis.get_time_period_label(prices, date_separator='-')}",
                                     ax=ax,
                                     **kwargs)

    fig, ax = plt.subplots(1, 1, figsize=(7, qis.calc_table_height(num_rows=len(prices.columns)+5, scale=0.5)), tight_layout=True)
    qis.plot_periodic_returns_table(prices=prices,
                                    freq=heatmap_freq,
                                    ax=ax,
                                    title=f"Periodic performance: {qis.get_time_period_label(prices, date_separator='-')}",
                                    total_name='Total',
                                    **qis.update_kwargs(kwargs, dict(square=False, x_rotation=90)))

    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        qis.plot_prices(prices=prices,
                        regime_benchmark=regime_benchmark,
                        perf_params=perf_params,
                        title=f"Time series of $1 invested: {qis.get_time_period_label(prices, date_separator='-')}",
                        ax=ax,
                        **kwargs)

        fig, axs = plt.subplots(2, 1, figsize=(7, 7))
        qis.plot_prices_with_dd(prices=prices,
                                regime_benchmark=regime_benchmark,
                                perf_params=perf_params,
                                title=f"Time series of $1 invested: {qis.get_time_period_label(prices, date_separator='-')}",
                                axs=axs,
                                **kwargs)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        qis.plot_scatter_regression(prices=prices,
                                    regime_benchmark=regime_benchmark,
                                    regime_params=qis.BenchmarkReturnsQuantileRegimeSpecs(freq=perf_params.freq_reg),
                                    perf_params=perf_params,
                                    title=f"Regime Conditional Regression: {qis.get_time_period_label(prices, date_separator='-')}",
                                    ax=ax,
                                    **kwargs)


class UnitTests(Enum):
    ETF_DATA = 1
    CRYPTO_DATA = 2
    TF_ETF = 3
    ETFS = 4
    COMMODITY_ETFS = 5
    VOL_ETFS = 6


def run_unit_test(unit_test: UnitTests):

    ust_3m_rate = yf.download('^IRX', start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True)['Close'].dropna() / 100.0

    if unit_test == UnitTests.ETF_DATA:
        regime_benchmark = 'SPY'
        tickers = [regime_benchmark, 'QQQ', 'EEM', 'TLT', 'IEF', 'LQD', 'HYG', 'SHY', 'GLD']

    elif unit_test == UnitTests.CRYPTO_DATA:
        regime_benchmark = 'BTC-USD'
        tickers = [regime_benchmark, 'ETH-USD', 'SOL-USD']
        regime_benchmark = 'BTC-USD'
        tickers = [regime_benchmark, 'SPY', 'TLT', 'ETH-USD', 'SOL-USD']

    elif unit_test == UnitTests.TF_ETF:
        regime_benchmark = 'SPY'
        tickers = [regime_benchmark, 'DBMF', 'WTMF', 'CTA']

    elif unit_test == UnitTests.ETFS:
        regime_benchmark = 'AOR'
        tickers = [regime_benchmark, 'SPY', 'PEX', 'PSP', 'GSG', 'COMT', 'REET', 'REZ']

    elif unit_test == UnitTests.COMMODITY_ETFS:
        regime_benchmark = 'AOR'
        tickers = [regime_benchmark, 'SPY', 'GLD', 'GSG', 'COMT', 'PDBC']

    elif unit_test == UnitTests.VOL_ETFS:
        regime_benchmark = 'SPY'
        tickers = [regime_benchmark, 'SVOL']

    else:
        raise NotImplementedError

    is_long_period = True
    if is_long_period:
        time_period = None
        time_period = qis.TimePeriod('31Dec2015', None)
        time_period = qis.TimePeriod('16Oct2014', None)
        # time_period = qis.TimePeriod('31Dec2017', '31Mar2023')
        perf_params = qis.PerfParams(freq='W-WED', freq_reg='ME', freq_drawdown='B', rates_data=ust_3m_rate, alpha_an_factor=12)
        kwargs = dict(x_date_freq='YE', heatmap_freq='YE', date_format='%Y', perf_params=perf_params)
    else:
        time_period = qis.TimePeriod('31Dec2022', None)
        perf_params = qis.PerfParams(freq='W-WED', freq_reg='W-WED', freq_drawdown='B', rates_data=ust_3m_rate, alpha_an_factor=52)
        kwargs = dict(x_date_freq='ME', heatmap_freq='ME', date_format='%b-%y', perf_params=perf_params)

    prices = yf.download(tickers, start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True)['Close'][tickers].dropna()

    if time_period is not None:
        prices = time_period.locate(prices)

    generate_performances(prices=prices,
                          regime_benchmark=regime_benchmark,
                          **kwargs)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.VOL_ETFS

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

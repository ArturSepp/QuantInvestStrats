"""
to use pybloqs for pandas > 2.x
locate file "...\Lib\site-packages\pybloqs\jinja\table.html"
change line 44 from:
{% for col_name, cell in row.iteritems() %}
to:
{% for col_name, cell in row.items() %}
"""

# packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum
import pybloqs as p

# qis
import yfinance as yf
import qis
from qis import PerfStat, local_path


# define entries to show in ra perf table
RA_TABLE_COLUMNS = [#PerfStat.START_DATE,
    #PerfStat.END_DATE,
    #PerfStat.START_PRICE,
    #PerfStat.END_PRICE,
    PerfStat.TOTAL_RETURN,
    PerfStat.PA_RETURN,
    PerfStat.VOL,
    PerfStat.SHARPE_RF0,
    PerfStat.SHARPE_EXCESS,
    PerfStat.MAX_DD,
    PerfStat.MAX_DD_VOL,
    PerfStat.SKEWNESS,
    PerfStat.KURTOSIS]


def generate_performances(prices: pd.DataFrame,
                          regime_benchmark_str: str,
                          perf_params: qis.PerfParams = None,
                          heatmap_freq: str = 'YE',
                          **kwargs
                          ) -> None:

    local_kwargs = dict(digits_to_show=1, framealpha=0.75, perf_stats_labels=qis.PerfStatsLabels.DETAILED_WITH_DD.value)
    kwargs = qis.update_kwargs(kwargs, local_kwargs)

    figsize = qis.get_df_table_size(df=pd.DataFrame(),
                                    min_cols=len(RA_TABLE_COLUMNS),
                                    min_rows=len(prices.columns),
                                    scale_rows=0.5,
                                    scale_cols=0.5)
    fig1, ax = plt.subplots(1, 1, figsize=(14, 1.1), tight_layout=True)

    qis.plot_ra_perf_table(prices=prices,
                           perf_columns=RA_TABLE_COLUMNS,
                           perf_params=perf_params,
                           # title=f"Risk-adjusted performance: {qis.get_time_period_label(prices, date_separator='-')}",
                           fontsize=8,
                           ax=ax,
                           **kwargs)

    ra_table = qis.get_ra_perf_columns(prices=prices,
                                       perf_columns=RA_TABLE_COLUMNS,
                                       perf_params=perf_params,
                                       drop_index=True,
                                       **kwargs)

    fig2, ax = plt.subplots(1, 1, figsize=(7, qis.calc_table_height(num_rows=len(prices.columns)+5, scale=0.5)), tight_layout=True)
    qis.plot_periodic_returns_table(prices=prices,
                                    freq=heatmap_freq,
                                    ax=ax,
                                    title=f"Periodic performance: {qis.get_time_period_label(prices, date_separator='-')}",
                                    total_name='Total',
                                    **qis.update_kwargs(kwargs, dict(square=False, x_rotation=90)))

    with sns.axes_style("darkgrid"):
        fig3, ax = plt.subplots(1, 1, figsize=(8, 6))
        qis.plot_prices(prices=prices,
                        regime_benchmark_str=regime_benchmark_str,
                        perf_params=perf_params,
                        title=f"Time series of $1 invested: {qis.get_time_period_label(prices, date_separator='-')}",
                        ax=ax,
                        **kwargs)

        fig4, axs = plt.subplots(2, 1, figsize=(7, 7))
        qis.plot_prices_with_dd(prices=prices,
                                regime_benchmark_str=regime_benchmark_str,
                                perf_params=perf_params,
                                title=f"Time series of $1 invested: {qis.get_time_period_label(prices, date_separator='-')}",
                                axs=axs,
                                **kwargs)

        fig5, ax = plt.subplots(1, 1, figsize=(8, 6))
        qis.plot_scatter_regression(prices=prices,
                                    regime_benchmark_str=regime_benchmark_str,
                                    regime_params=qis.BenchmarkReturnsQuantileRegimeSpecs(freq=perf_params.freq_reg),
                                    perf_params=perf_params,
                                    title=f"Regime Conditional Regression: {qis.get_time_period_label(prices, date_separator='-')}",
                                    ax=ax,
                                    **kwargs)

    bkwargs_title = {'title_wrap': True, 'text-align': 'left', 'color': 'blue', 'font_size': "12px",
                     'title_level': 3, 'line-height': 0.1, 'inherit_cfg': False}
    bkwargs_text = {'title_wrap': True, 'text-align': 'left', 'font_size': "10px",
                    'title_level': 3, 'line-height': 0.1}
    bkwargs_fig = {'title_wrap': True, 'text-align': 'left', 'font_size': "10px",
                   'title_level': 3, 'line-height': 0.1}
    b_fig_ra = p.Block(
        [p.Paragraph(f"Block with Risk-adjusted performance", **bkwargs_title),
         p.Paragraph("Description", **bkwargs_text),
         p.Block(fig1, **bkwargs_fig)],
        **bkwargs_text)

    r = p.VStack([p.Block("Strategy description", title="Backtest strategy"),
                  b_fig_ra,
                  p.Block(fig2, title='New'),
                  p.Block(ra_table),
                  p.Block(fig3), p.Block(fig4), p.Block(fig5)])

    report_file_name = f"{local_path.get_output_path()}PyBloqs_Factsheet_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf"
    print(f"saved to {report_file_name}")
    r.save(report_file_name)


class UnitTests(Enum):
    ETF_DATA = 1
    CRYPTO_DATA = 2


def run_unit_test(unit_test: UnitTests):

    ust_3m_rate = yf.download('^IRX', start=None, end=None)['Adj Close'].dropna() / 100.0

    if unit_test == UnitTests.ETF_DATA:
        regime_benchmark_str = 'SPY'
        tickers = [regime_benchmark_str, 'QQQ', 'EEM', 'TLT', 'IEF', 'LQD', 'HYG', 'SHY', 'GLD']

    elif unit_test == UnitTests.CRYPTO_DATA:
        regime_benchmark_str = 'BTC-USD'
        tickers = [regime_benchmark_str, 'ETH-USD', 'SOL-USD']

    else:
        raise NotImplementedError

    is_long_period = False
    if is_long_period:
        time_period = None
        perf_params = qis.PerfParams(freq='W-WED', freq_reg='QE', freq_drawdown='B', rates_data=ust_3m_rate)
        kwargs = dict(x_date_freq='YE', heatmap_freq='YE', date_format='%Y', perf_params=perf_params)
    else:
        time_period = qis.TimePeriod('31Dec2021', '23Jan2023')
        perf_params = qis.PerfParams(freq='W-WED', freq_reg='W-WED', freq_drawdown='B', rates_data=ust_3m_rate)
        kwargs = dict(x_date_freq='ME', heatmap_freq='ME', date_format='%b-%y', perf_params=perf_params)

    prices = yf.download(tickers, start=None, end=None)['Adj Close'].dropna()

    if time_period is not None:
        prices = time_period.locate(prices)

    generate_performances(prices=prices,
                          regime_benchmark_str=regime_benchmark_str,
                          **kwargs)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.ETF_DATA

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

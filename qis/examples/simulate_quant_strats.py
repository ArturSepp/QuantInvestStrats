
# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union
from enum import Enum

# qis
import qis.file_utils as fu
import qis


# strats
from qis.portfolio.strats.quant_strats_delta1 import simulate_vol_target_strats_range, simulate_trend_starts_range

FIG_SIZE = (8.3, 11.7)  # A4 for portrait

PERF_PARAMS = qis.PerfParams(freq='B')


def create_time_series_report(prices: Union[pd.Series, pd.DataFrame],
                              time_period: qis.TimePeriod,
                              spans: List[int] = (7, 14, 21, 30, 60, 130, 260, 520),
                              vol_span: int = 31,
                              vol_target: float = 0.15,
                              vol_af: float = 260
                              ) -> List[plt.Figure]:

    if isinstance(prices, pd.Series):
        strat_name = prices.name
    else:
        strat_name = ''

    # vol target returns
    vt_nav_weights, vt_navs = simulate_vol_target_strats_range(prices=prices, vol_spans=spans,
                                                               vol_target=vol_target, vol_af=vol_af)

    fig1 = plt.figure(figsize=FIG_SIZE, constrained_layout=True)
    nrows = 5
    gs = fig1.add_gridspec(nrows=nrows, ncols=2, wspace=0.0, hspace=0.0)
    axs1 = [fig1.add_subplot(gs[row, 0]) for row in range(nrows)]
    axs2 = [fig1.add_subplot(gs[row, 1]) for row in range(nrows)]
    fig1.suptitle(f"{strat_name} Volatility-target strats with vol_target={vol_target:0.0%}", fontweight="bold", fontsize=8, color='blue')

    global_kwargs = dict(fontsize=5, linewidth=0.5, first_color_fixed=True, framealpha=0.8,
                         digits_to_show=1, x_date_freq='YE')
    scatter_kwargs = qis.update_kwargs(kwargs=global_kwargs,
                                       new_kwargs=dict(fontsize=5, var_format='{:.0%}', xvar_numticks=7,
                                                       markersize=1))

    plot_strategies_prices(nav_data=vt_navs,
                           nav_weights=vt_nav_weights,
                           time_period=time_period,
                           vol_span=vol_span,
                           vol_af=vol_af,
                           axs=axs1,
                           **global_kwargs)
    plot_strategies_returns_scatter(nav_data=vt_navs,
                                    time_period=time_period,
                                    axs=axs2,
                                    **scatter_kwargs)

    # tf returns
    tf_nav_weights, tf_navs, tf_signals = simulate_trend_starts_range(prices=prices, tf_spans=spans, vol_span=vol_span,
                                                                      vol_target=vol_target/2.0, vol_af=vol_af)

    fig2 = plt.figure(figsize=FIG_SIZE, constrained_layout=True)
    gs = fig2.add_gridspec(nrows=nrows, ncols=2, wspace=0.0, hspace=0.0)
    axs1 = [fig2.add_subplot(gs[row, 0]) for row in range(nrows)]
    axs2 = [fig2.add_subplot(gs[row, 1]) for row in range(nrows)]
    fig2.suptitle(f"{strat_name} Trend-following strats with vol_target={vol_target/2.0:0.0%}", fontweight="bold", fontsize=8, color='blue')

    plot_strategies_prices(nav_data=tf_navs,
                           nav_weights=tf_nav_weights,
                           time_period=time_period,
                           vol_span=vol_span,
                           vol_af=vol_af,
                           axs=axs1,
                           **global_kwargs)

    """
    figure_num += 1
    figure_caption = f"Figure {figure_num}. Pdf plot and qq-plot of returns on Trend-following Strategies by frequency"
    fig = plot_strategies_returns_pdf(nav_data=tf_navs,
                                      time_period=time_period,
                                      **global_kwargs)
    dut.add_fig_par(document, fig, figure_caption=figure_caption, figure_name=f"{file_name}_{figure_num}")
    """

    plot_strategies_returns_scatter(nav_data=tf_navs,
                                    time_period=time_period,
                                    axs=axs2,
                                    **scatter_kwargs)

    return [fig1, fig2]


def plot_strategies_prices(nav_data: pd.DataFrame,
                           nav_weights: pd.DataFrame,
                           time_period: qis.TimePeriod,
                           axs: List[plt.Subplot],
                           vol_span: int = 31,
                           vol_af: float = 260.0,
                           **kwargs
                           ) -> None:

    nav_returns = qis.to_returns(prices=nav_data)
    eod_ewm_vol = qis.compute_ewm_vol(data=nav_returns, span=vol_span, mean_adj_type=qis.MeanAdjType.NONE, af=vol_af)

    # trim plot data
    nav_data = time_period.locate(nav_data)
    nav_weights = time_period.locate(nav_weights)
    nav_returns = time_period.locate(nav_returns)
    eod_ewm_vol = time_period.locate(eod_ewm_vol)

    qis.plot_prices(prices=nav_data,
                    var_format='{:.2f}',
                    title='Logarithm of nav',
                    perf_stats_labels=qis.PerfStatsLabels.DETAILED_WITH_SKEW.value,
                    is_log=True,
                    perf_params=PERF_PARAMS,
                    start_to_one=True,
                    ax=axs[0],
                    **kwargs)

    qis.plot_time_series(df=nav_weights,
                         var_format='{:.0%}',
                         title='Strategy weights',
                         legend_stats=qis.LegendStats.AVG_STD_LAST,
                         ax=axs[1],
                         **kwargs)

    turnover = nav_weights.diff(1).abs().rolling(int(vol_af)).sum()
    qis.plot_time_series(df=turnover,
                        var_format='{:.0%}',
                        title='1y rolling daily Turnover',
                        legend_stats=qis.LegendStats.AVG_STD_LAST,
                        ax=axs[2],
                        **kwargs)

    qis.plot_time_series(df=nav_returns,
                        var_format='{:.2%}',
                        title='Daily returns',
                        legend_stats=qis.LegendStats.AVG_STD_LAST,
                        ax=axs[3],
                        **kwargs)

    qis.plot_time_series(df=eod_ewm_vol,
                        var_format='{:.2%}',
                        title=f"Annualized EWMA-{vol_span} vol of daily returns",
                        legend_stats=qis.LegendStats.AVG_STD_LAST,
                        ax=axs[4],
                        **kwargs)


def plot_strategies_returns_pdf(nav_data: pd.DataFrame,
                                time_period: qis.TimePeriod,
                                **kwargs
                                ) -> plt.Figure:

    # trim plot data
    nav_data = time_period.locate(nav_data)

    freqs = {'Daily': 'B', 'Weekly': 'W-WED', 'Bi-Weekly': '2W-WED', 'Monthly': 'ME'}

    numticks = 7
    major_ticks = np.linspace(-4.5, 4.5, numticks)
    pdf_kwargs = qis.update_kwargs(kwargs=kwargs, new_kwargs={'fontsize': 7, 'var_format': '{:.0%}',
                                                              'xvar_numticks': numticks})

    qq_kwargs = qis.update_kwargs(kwargs=kwargs, new_kwargs={'fontsize': 7, 'var_format': '{:.1f}',
                                                             'yvar_numticks': numticks,
                                                             'xvar_major_ticks': major_ticks})

    with sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(len(freqs.keys()), 2, figsize=FIG_SIZE, tight_layout=True)
        for idx, (title, freq) in enumerate(freqs.items()):
            returns = qis.to_returns(prices=nav_data, freq=freq, drop_first=True)
            qis.plot_histogram(df=returns,
                               x_min_max_quantiles=(0.0001, 0.9999),
                               title=f"{title}",
                               add_data_std_pdf=True,
                               ax=axs[idx][0],
                               **pdf_kwargs)
            qis.plot_qq(df=returns,
                    title=f"{title}",
                    ax=axs[idx][1],
                    **qq_kwargs)
    return fig


def plot_strategies_returns_scatter(nav_data: pd.DataFrame,
                                    time_period: qis.TimePeriod,
                                    axs: List[plt.Subplot],
                                    **kwargs
                                    ) -> None:
    # trim plot data
    nav_data = time_period.locate(nav_data)
    freqs = {'Daily': ('B', 252), 'Weekly': ('W-WED', 52), 'Bi-Weekly': ('2W-WED', 26), 'Monthly': ('ME', 12), 'Quarterly': ('QE', 4)}
    for ax, (title, freq) in zip(axs, freqs.items()):
        new_kwargs = dict(alpha_format='{0:+0.0%}',
                          beta_format='{:+0.1f}',
                          alpha_an_factor=freq[1],
                          framealpha=0.9)
        kwargs = qis.update_kwargs(kwargs, new_kwargs)
        qis.plot_returns_scatter(prices=nav_data,
                                 benchmark=nav_data.columns[0],
                                 ylabel=f"Strategy return",
                                 title=f"{title}",
                                 freq=freq[0],
                                 order=2,
                                 ci=95,
                                 ax=ax,
                                 **kwargs)


class UnitTests(Enum):
    BTC_SIMULATION = 1
    SPY_SIMULATION = 2


def run_unit_test(unit_test: UnitTests):

    import yfinance as yf

    if unit_test == UnitTests.BTC_SIMULATION:
        prices = yf.download(tickers=['BTC-USD'], start=None, end=None)['Adj Close'].rename('BTC').dropna()

        time_period = qis.TimePeriod('31Dec2015', '21Jun2023')
        figs = create_time_series_report(prices=prices, time_period=time_period, vol_target=0.5, vol_af=360)
        fu.save_figs_to_pdf(figs=figs, file_name='btc_analysis', orientation='landscape',
                            add_current_date=True, local_path=None)

    elif unit_test == UnitTests.SPY_SIMULATION:
        prices = yf.download(tickers=['SPY'], start=None, end=None)['Adj Close'].rename('SPY').dropna()
        time_period = qis.TimePeriod('31Dec1999', '29Dec2022')
        figs = create_time_series_report(prices=prices, time_period=time_period, vol_target=0.15, vol_af=260)
        fu.save_figs_to_pdf(figs=figs, file_name='spy_analysis', orientation='landscape',
                            add_current_date=True, local_path=None)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.SPY_SIMULATION

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

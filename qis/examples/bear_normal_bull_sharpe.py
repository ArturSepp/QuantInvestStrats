import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
import qis as qis

# create price data
regime_benchmark = 'SPY'
tickers = [regime_benchmark, 'TLT', 'LQD', 'HYG', 'GLD']
prices = yf.download(tickers=tickers, start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True)['Close'][tickers]
prices = prices.asfreq('B', method='ffill').dropna()  # align

# define performance regime params and regime_classifier
perf_params = qis.PerfParams()
regime_classifier = qis.BenchmarkReturnsQuantilesRegime(freq='QE')

# regime diversification
with sns.axes_style("darkgrid"):
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))

    qis.plot_prices(prices=prices,
                    perf_params=perf_params,
                    is_log=False,
                    ax=axs[0])
    qis.add_bnb_regime_shadows(ax=axs[0],
                               data_df=prices,
                               pivot_prices=prices[regime_benchmark],
                               benchmark=regime_benchmark,
                               regime_classifier=regime_classifier)

    title = f"Sharpe ratio split to {regime_benchmark} Bear/Normal/Bull {regime_classifier.freq}-freq regimes"
    qis.plot_regime_data(regime_classifier=regime_classifier,
                         prices=prices,
                         benchmark=regime_benchmark,
                         is_conditional_sharpe=True,
                         regime_data_to_plot=qis.RegimeData.REGIME_SHARPE,
                         var_format='{:.2f}',
                         perf_params=perf_params,
                         drop_benchmark=False,
                         title=title,
                         ax=axs[1])

# scatter plot
with sns.axes_style("darkgrid"):
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))

    qis.plot_ra_perf_scatter(prices=prices,
                             benchmark=regime_benchmark,
                             x_var=qis.PerfStat.VOL,
                             y_var=qis.PerfStat.PA_RETURN,
                             annotation_labels=prices.columns.to_list(),
                             title=f"Total return vs vol",
                             ci=None,
                             ax=axs[0])
    qis.plot_ra_perf_scatter(prices=prices,
                             benchmark=regime_benchmark,
                             x_var=qis.PerfStat.BEAR_SHARPE,
                             y_var=qis.PerfStat.SHARPE_RF0,
                             annotation_labels=prices.columns.to_list(),
                             title=f"Total Sharpe vs Bear-Sharpe",
                             ci=None,
                             ax=axs[1])


# smart diversification curves
sd_report = qis.SmartDiversificationReport(principal_nav=prices[regime_benchmark],
                                           overlay_navs=prices.drop(regime_benchmark, axis=1),
                                           perf_params=perf_params,
                                           regime_classifier=regime_classifier)
with sns.axes_style('darkgrid'):
    fig, axs = plt.subplots(1, 2, figsize=(16, 7), tight_layout=True)
    sd_report.plot_smart_diversification_curve(x_var=qis.PerfStat.BEAR_SHARPE,
                                               y_var=qis.PerfStat.SHARPE_RF0,
                                               title='Total Sharpe vs Bear-Sharpe for 100% benchmark + x% overlay',
                                               is_principal_weight_fixed=True,
                                               ax=axs[0])

    sd_report.plot_smart_diversification_curve(x_var=qis.PerfStat.BEAR_SHARPE,
                                               y_var=qis.PerfStat.SHARPE_RF0,
                                               title='Total Sharpe vs Bear-Sharpe for x% benchmark + (1-x)% overlay',
                                               is_principal_weight_fixed=False,
                                               ax=axs[1])

plt.show()

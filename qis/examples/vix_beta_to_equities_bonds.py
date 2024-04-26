"""
analyse rolling betas of vix to SPY and TLT ETFs
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
import qis
from qis import PortfolioData

# load VIX ETH
vix = yf.download(tickers=['VXX'], start=None, end=None, ignore_tz=True)['Adj Close'].asfreq('B', method='ffill').rename('Long VIX ETF')

# load becnhmarks benchmarks
benchmark_prices = yf.download(tickers=['SPY', 'TLT'], start=None, end=None, ignore_tz=True)['Adj Close'].asfreq('B', method='ffill')

# create long-only portfolio using vix nav
vix_portfolio = PortfolioData(nav=vix)

# set timeperiod for analysis
time_period = qis.TimePeriod('31Dec2019', None)
perf_params = qis.PerfParams(freq='W-WED', freq_reg='ME', alpha_an_factor=12.0,
                             rates_data=yf.download('^IRX', start=None, end=None)['Adj Close'].dropna() / 100.0)
regime_params = qis.BenchmarkReturnsQuantileRegimeSpecs(freq='ME')

prices = pd.concat([vix, benchmark_prices], axis=1).sort_index().dropna()
prices = time_period.locate(prices)

with sns.axes_style("darkgrid"):
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    kwargs = dict(framealpha=0.9, fontsize=12, legend_loc='lower left')

    # plot performances
    qis.plot_prices(prices=prices,
                    perf_params=perf_params,
                    title='Log Performance',
                    perf_stats_labels=qis.PerfStatsLabels.TOTAL_DETAILED.value,
                    is_log=True,
                    ax=axs[0],
                    **kwargs)
    qis.add_bnb_regime_shadows(ax=axs[0], pivot_prices=prices['SPY'], regime_params=regime_params)

    # plot vix betas to benchmarks
    span = 63  # use 3m for half-live
    vix_benchmark_betas = vix_portfolio.compute_portfolio_benchmark_betas(benchmark_prices=benchmark_prices,
                                                                          factor_beta_span=span)
    qis.plot_time_series(df=vix_benchmark_betas,
                         var_format='{:,.2f}',
                         legend_stats=qis.LegendStats.AVG_LAST,
                         title='VIX ETF benchmark multi-variate betas',
                         ax=axs[1],
                         **kwargs)
    qis.add_bnb_regime_shadows(ax=axs[1], pivot_prices=prices['SPY'], regime_params=regime_params)

    # plot performance attribution to betas
    factor_attribution = vix_portfolio.compute_portfolio_benchmark_attribution(benchmark_prices=benchmark_prices,
                                                                               factor_beta_span=span,
                                                                               time_period=time_period)
    qis.plot_time_series(df=factor_attribution,
                         var_format='{:,.0%}',
                         legend_stats=qis.LegendStats.LAST,
                         title='VIX ETF return attribution to benchmark betas',
                         ax=axs[2],
                         **kwargs)
    pivot_prices = benchmark_prices['SPY'].reindex(index=factor_attribution.index, method='ffill')
    qis.add_bnb_regime_shadows(ax=axs[2], pivot_prices=prices['SPY'], regime_params=regime_params)

    qis.align_x_limits_axs(axs=axs, is_invisible_xs=True)

plt.show()

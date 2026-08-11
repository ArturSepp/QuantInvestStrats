"""
several strategies compared side by side: a ``MultiPortfolioData`` in, a list of A4 pages out.

``generate_multi_portfolio_factsheet`` compares the members against the shared benchmark, and
reports outcomes rather than attributing differences to weights. One page always renders: navs,
drawdowns, rolling performance, turnover, costs and the first portfolio's exposures, beside the
performance bars, the risk-adjusted table, the periodic-returns heatmap, correlations, regime
Sharpes and betas. ``group_data`` only sets ``is_grouped``, which switches the regime panels to
group navs and groups the exposures of an appended strategy factsheet; the exposure and turnover
panels have no group mode. ``add_group_exposures_and_pnl`` and ``add_strategy_factsheets`` each
append further pages, both off by default. Frequencies arrive spread in from
``qis.fetch_default_report_kwargs``.

Attributing the difference between a pair to their weights is ``strategy_benchmark_factsheet.py``;
the same page rendered as HTML is ``multi_strategy_factsheet_pybloqs.py``.
"""
# packages
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List
# qis
import qis as qis
from qis import TimePeriod, PerfParams, PerfStat, BenchmarkReturnsQuantilesRegime
from qis.portfolio.multi_portfolio_data import MultiPortfolioData
from qis.portfolio.reports.config import (PERF_PARAMS, validate_reporting_frequency,
                                          validate_legend_capacity, infer_data_frequency_label)
from qis.portfolio.reports.strategy_factsheet import (
    _use_grouped_summary_tables,
    generate_strategy_factsheet,
)


def generate_multi_portfolio_factsheet(multi_portfolio_data: MultiPortfolioData,
                                       time_period: TimePeriod = None,
                                       perf_params: PerfParams = PERF_PARAMS,
                                       regime_classifier: BenchmarkReturnsQuantilesRegime = BenchmarkReturnsQuantilesRegime(),
                                       regime_benchmark: str = None,
                                       backtest_name: str = None,
                                       heatmap_freq: str = 'YE',
                                       add_benchmarks_to_navs: bool = False,
                                       figsize: Tuple[float, float] = (8.3, 11.7),  # A4 for portrait
                                       group_data: pd.Series = None,
                                       add_group_exposures_and_pnl: bool = False,
                                       add_strategy_factsheets: bool = False,
                                       fontsize: int = 5,
                                       **kwargs
                                       ) -> List[plt.Figure]:
    """
    factsheet comparing several structurally different strategies, as a list of A4 figures.

    For portfolios that are not variants of one another - different universes, different
    mandates - so the report compares outcomes rather than attributing differences to weights.
    Pass ``group_data`` when the universe is large: turnover and exposures are then reported by
    group instead of per instrument, which is the difference between a readable page and a
    hundred-line legend.

    Args:
        multi_portfolio_data: the portfolios to compare
        time_period: reporting window. None uses the full common history
        perf_params: annualisation, frequency and rate conventions for the statistics
        regime_classifier: how benchmark returns are mapped to regimes for the conditional panels
        regime_benchmark: column driving the regime classification. None uses the first benchmark
        backtest_name: title of the report
        heatmap_freq: frequency of the periodic-returns heatmap
        add_benchmarks_to_navs: include the benchmarks as additional lines in the performance panels
        figsize: page size in inches; the default is A4 portrait
        group_data: asset-class label per instrument, indexed by instrument. Given, exposures and
            turnover are reported by group. More than ten groups are omitted from the
            regime-Sharpe panels with a warning
        add_group_exposures_and_pnl: add the per-group exposure and P&L pages
        add_strategy_factsheets: append a full single-strategy factsheet for each portfolio
        fontsize: base font size, small by default because a factsheet page is dense

    Returns:
        the pages, in order, ready for :func:`save_figs_to_pdf`
    """
    if group_data is not None:
        is_grouped = True
    else:
        is_grouped = False
    is_grouped_for_summary_tables = _use_grouped_summary_tables(
        portfolio_data=multi_portfolio_data.portfolio_datas[0],
        is_grouped=is_grouped,
        panel_names='the Sharpe-ratio panels',
    )

    if regime_benchmark is None and multi_portfolio_data.benchmark_prices is not None:
        regime_benchmark = multi_portfolio_data.benchmark_prices.columns[0]

    # guard: the requested reporting frequency must not be finer than the data it is computed on -
    # check every portfolio NAV and the benchmark prices (used for regime / beta / scatter panels)
    nav_datas = [portfolio.get_portfolio_nav() for portfolio in multi_portfolio_data.portfolio_datas]
    for data_series in nav_datas + [multi_portfolio_data.benchmark_prices]:
        if data_series is not None:
            validate_reporting_frequency(data_series, perf_params.freq)

    # guard: the left column carries one legend row per portfolio, and a legend that outgrows its
    # panel collapses the layout of the whole page - see config.estimate_legend_capacity
    n_legend_entries = len(multi_portfolio_data.portfolio_datas)
    if add_benchmarks_to_navs and multi_portfolio_data.benchmark_prices is not None:
        benchmarks = multi_portfolio_data.benchmark_prices
        n_legend_entries += 1 if isinstance(benchmarks, pd.Series) else len(benchmarks.columns)
    validate_legend_capacity(n_legend_entries=n_legend_entries, figsize=figsize, fontsize=fontsize,
                             panel_rows=1, gridspec_rows=7,
                             report_name='multi-portfolio factsheet')

    # native grid of the NAV paths: drawdowns / under-water / cumulative are on this grid (not resampled)
    nav_freq = infer_data_frequency_label(nav_datas[0])
    nav_freq_label = f" ({nav_freq}-freq)" if nav_freq else ""

    plot_kwargs = dict(fontsize=fontsize,
                       linewidth=0.5,
                       digits_to_show=1, sharpe_digits=2,
                       weight='normal',
                       markersize=1,
                       framealpha=0.75)
    kwargs = qis.update_kwargs(kwargs, plot_kwargs)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(nrows=7, ncols=4, wspace=0.0, hspace=0.0)

    if backtest_name is not None:
        fig.suptitle(backtest_name, fontweight="bold", fontsize=8, color='blue')

    if regime_benchmark is not None:
        title = f"Cumulative performance ({perf_params.freq}-freq stats) with " \
                f"bear/normal/bull regimes of {regime_benchmark} {regime_classifier.freq}-returns"
    else:
        title = f"Cumulative performance ({perf_params.freq}-freq stats)"

    multi_portfolio_data.plot_nav(ax=fig.add_subplot(gs[0, :2]),
                                  time_period=time_period,
                                  regime_benchmark=regime_benchmark,
                                  perf_params=perf_params,
                                  regime_classifier=regime_classifier,
                                  add_benchmarks_to_navs=add_benchmarks_to_navs,
                                  title=title,
                                  **kwargs)

    multi_portfolio_data.plot_drawdowns(ax=fig.add_subplot(gs[1, :2]),
                                        time_period=time_period,
                                        regime_benchmark=regime_benchmark,
                                        regime_classifier=regime_classifier,
                                        dd_legend_type=qis.DdLegendType.SIMPLE,
                                        add_benchmarks_to_navs=add_benchmarks_to_navs,
                                        title=f'Running Drawdowns{nav_freq_label}',
                                        **kwargs)

    multi_portfolio_data.plot_rolling_time_under_water(ax=fig.add_subplot(gs[2, :2]),
                                                       time_period=time_period,
                                                       regime_benchmark=regime_benchmark,
                                                       regime_classifier=regime_classifier,
                                                       add_benchmarks_to_navs=add_benchmarks_to_navs,
                                                       title=f'Rolling time under water{nav_freq_label}',
                                                       **kwargs)

    multi_portfolio_data.plot_rolling_perf(ax=fig.add_subplot(gs[3, :2]),
                                           time_period=time_period,
                                           regime_benchmark=regime_benchmark,
                                           regime_classifier=regime_classifier,
                                           add_benchmarks_to_navs=add_benchmarks_to_navs,
                                           **kwargs)

    multi_portfolio_data.plot_exposures(ax=fig.add_subplot(gs[4, :2]),
                                        portfolio_idx=0,
                                        time_period=time_period,
                                        regime_benchmark=regime_benchmark,
                                        regime_classifier=regime_classifier,
                                        **kwargs)

    multi_portfolio_data.plot_turnover(ax=fig.add_subplot(gs[5, :2]),
                                       time_period=time_period,
                                       regime_benchmark=regime_benchmark,
                                       regime_classifier=regime_classifier,
                                       **kwargs)

    multi_portfolio_data.plot_costs(ax=fig.add_subplot(gs[6, :2]),
                                    time_period=time_period,
                                    regime_benchmark=regime_benchmark,
                                    regime_classifier=regime_classifier,
                                    **kwargs)

    multi_portfolio_data.plot_performance_bars(ax=fig.add_subplot(gs[0, 2]),
                                               perf_params=perf_params,
                                               perf_column=PerfStat.SHARPE_RF0,
                                               time_period=time_period,
                                               add_benchmarks_to_navs=add_benchmarks_to_navs,
                                               **qis.update_kwargs(kwargs, dict(fontsize=fontsize)))

    multi_portfolio_data.plot_performance_bars(ax=fig.add_subplot(gs[0, 3]),
                                               perf_params=perf_params,
                                               perf_column=PerfStat.MAX_DD,
                                               time_period=time_period,
                                               add_benchmarks_to_navs=add_benchmarks_to_navs,
                                               **qis.update_kwargs(kwargs, dict(fontsize=fontsize)))

    multi_portfolio_data.plot_ra_perf_table(ax=fig.add_subplot(gs[1, 2:]),
                                            perf_params=perf_params,
                                            time_period=time_period,
                                            **qis.update_kwargs(kwargs, dict(fontsize=fontsize)))

    multi_portfolio_data.plot_periodic_returns(ax=fig.add_subplot(gs[2, 2:]),
                                               heatmap_freq=heatmap_freq,
                                               title=f"{heatmap_freq} returns",
                                               time_period=time_period,
                                               add_benchmarks_to_navs=add_benchmarks_to_navs,
                                               **qis.update_kwargs(kwargs, dict(fontsize=fontsize)))

    """
    multi_portfolio_data.plot_ra_perf_table(ax=fig.add_subplot(gs[3, 2:]),
                                            perf_params=perf_params,
                                            time_period=qis.get_time_period_shifted_by_years(time_period=time_period),
                                            **qis.update_kwargs(kwargs, dict(fontsize=fontsize, freq_reg='W-WED')))
    """
    multi_portfolio_data.plot_corr_table(ax=fig.add_subplot(gs[3, 2:]),
                                         time_period=time_period,
                                         freq=perf_params.freq,
                                         add_benchmarks_to_navs=add_benchmarks_to_navs,
                                         **qis.update_kwargs(kwargs, dict(fontsize=fontsize)))

    if len(multi_portfolio_data.benchmark_prices.columns) > 1:
        multi_portfolio_data.plot_regime_data(ax=fig.add_subplot(gs[4, 2]),
                                              is_grouped=is_grouped_for_summary_tables,
                                              time_period=time_period,
                                              perf_params=perf_params,
                                              regime_classifier=regime_classifier,
                                              benchmark=multi_portfolio_data.benchmark_prices.columns[0],
                                              **kwargs)
        multi_portfolio_data.plot_regime_data(ax=fig.add_subplot(gs[4, 3]),
                                              is_grouped=is_grouped_for_summary_tables,
                                              time_period=time_period,
                                              perf_params=perf_params,
                                              regime_classifier=regime_classifier,
                                              benchmark=multi_portfolio_data.benchmark_prices.columns[1],
                                              **kwargs)
    else:
        multi_portfolio_data.plot_regime_data(ax=fig.add_subplot(gs[4, 2:]),
                                              is_grouped=is_grouped_for_summary_tables,
                                              time_period=time_period,
                                              perf_params=perf_params,
                                              regime_classifier=regime_classifier,
                                              benchmark=multi_portfolio_data.benchmark_prices.columns[0],
                                              **kwargs)
    if len(multi_portfolio_data.benchmark_prices.columns) > 1:
        # take first two benchmarks
        benchmark_prices = multi_portfolio_data.benchmark_prices.iloc[:, :2]
        multi_portfolio_data.plot_factor_betas(axs=[fig.add_subplot(gs[5, 2:]), fig.add_subplot(gs[6, 2:])],
                                               benchmark_prices=benchmark_prices,
                                               time_period=time_period,
                                               regime_benchmark=regime_benchmark,
                                               regime_classifier=regime_classifier,
                                               **kwargs)
    else:
        multi_portfolio_data.plot_returns_scatter(ax=fig.add_subplot(gs[5, 2:]),
                                                  time_period=time_period,
                                                  benchmark=multi_portfolio_data.benchmark_prices.columns[0],
                                                  **qis.update_kwargs(kwargs, dict(freq=perf_params.freq_reg)))

        multi_portfolio_data.plot_factor_betas(axs=[fig.add_subplot(gs[6, 2:])],
                                               time_period=time_period,
                                               benchmark_prices=multi_portfolio_data.benchmark_prices,
                                               regime_benchmark=regime_benchmark,
                                               regime_classifier=regime_classifier,
                                               **kwargs)

    figs = [fig]
    if add_group_exposures_and_pnl:
        figs1 = multi_portfolio_data.plot_group_exposures_and_pnl(time_period=time_period,
                                                                  regime_benchmark=regime_benchmark,
                                                                  regime_classifier=regime_classifier,
                                                                  **kwargs)
        figs.append(figs1)

    if add_strategy_factsheets:
        for portfolio_data in multi_portfolio_data.portfolio_datas:
            figs.append(generate_strategy_factsheet(portfolio_data=portfolio_data,
                                                    benchmark_prices=multi_portfolio_data.benchmark_prices,
                                                    perf_params=perf_params,
                                                    regime_classifier=regime_classifier,
                                                    add_grouped_exposures=is_grouped,
                                                    time_period=time_period,
                                                    **kwargs
                                                    ))
    figs = qis.to_flat_list(figs)
    return figs

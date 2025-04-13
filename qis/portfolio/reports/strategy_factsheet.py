"""
generate portfolio factsheet report using PortfolioData object
PortfolioData object can be generated by a backtester or actucal strategy data
with comparison to 1-2 cash benchmarks
PortfolioData can contain either simulated or actual portfolio data
"""
import numpy as np
# packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, List, Union
import qis as qis
from qis import TimePeriod, PerfParams, BenchmarkReturnsQuantileRegimeSpecs
from qis.portfolio.portfolio_data import PortfolioData
from qis.portfolio.reports.config import PERF_PARAMS, REGIME_PARAMS


def generate_strategy_factsheet(portfolio_data: PortfolioData,
                                benchmark_prices: Union[pd.DataFrame, pd.Series],
                                time_period: TimePeriod = None,
                                ytd_attribution_time_period: TimePeriod = qis.get_ytd_time_period(),
                                weight_report_time_period: TimePeriod = None,
                                perf_params: PerfParams = PERF_PARAMS,
                                regime_params: BenchmarkReturnsQuantileRegimeSpecs = REGIME_PARAMS,
                                regime_benchmark: str = None,  # default is set to benchmark_prices.columns[0]
                                exposures_freq: Optional[str] = 'W-WED',  #'W-WED',
                                turnover_rolling_period: int = 260,
                                freq_turnover: str = 'B',
                                freq_cost: str = 'B',
                                cost_rolling_period: int = 260,
                                factor_beta_span: int = 52,
                                freq_beta: str = 'W-WED',
                                vol_rolling_window: int = 13,
                                freq_sharpe: str = 'B',
                                freq_var: str = 'B',
                                var_span: float = 33,
                                sharpe_rolling_window: int = 260,
                                add_benchmarks_to_navs: bool = False,
                                figsize: Tuple[float, float] = (8.5, 11.7),  # A4 for portrait
                                fontsize: int = 4,
                                weight_change_sample_size: int = 20,
                                add_current_position_var_risk_sheet: bool = False,
                                add_weights_turnover_sheet: bool = False,
                                add_grouped_exposures: bool = False,
                                add_grouped_cum_pnl: bool = False,
                                add_weight_change_report: bool = False,
                                add_current_signal_report: bool = False,
                                add_instrument_history_report: bool = False,
                                y_limits_signal: Tuple[Optional[float], Optional[float]] = (-1.0, 1.0),
                                is_1y_exposures: bool = False,
                                is_grouped: Optional[bool] = None,
                                dd_legend_type: qis.DdLegendType = qis.DdLegendType.SIMPLE,
                                is_unit_based_traded_volume: bool = True,
                                df_to_add: pd.DataFrame = None,
                                factsheet_name: str = None,
                                **kwargs
                                ) -> List[plt.Figure]:
    # align
    if isinstance(benchmark_prices, pd.Series):
        benchmark_prices = benchmark_prices.to_frame()

    benchmark_prices = benchmark_prices.reindex(index=portfolio_data.nav.index, method='ffill')
    if regime_benchmark is None:
        regime_benchmark = benchmark_prices.columns[0]

    if is_grouped is None:
        if len(portfolio_data.get_weights().columns) >= 10:  # otherwise tables look too bad
            is_grouped = True
        else:
            is_grouped = False

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(nrows=14, ncols=4, wspace=0.0, hspace=0.0)

    plot_kwargs = dict(fontsize=fontsize,
                       linewidth=0.5,
                       digits_to_show=1, sharpe_digits=2,
                       weight='normal',
                       markersize=1,
                       framealpha=0.75)
    kwargs = qis.update_kwargs(kwargs, plot_kwargs)
    if factsheet_name is None:
        factsheet_name = f"{portfolio_data.nav.name} factsheet"
    fig.suptitle(factsheet_name, fontweight="bold", fontsize=8, color='blue')

    # prices
    portfolio_nav = portfolio_data.get_portfolio_nav(time_period=time_period)
    # set reporting time period here
    if time_period is None:
        time_period = qis.get_time_period(portfolio_nav)

    if add_benchmarks_to_navs:
        benchmark_prices_ = benchmark_prices
        joint_prices = pd.concat([portfolio_nav, benchmark_prices_], axis=1).dropna()
        pivot_prices = joint_prices[regime_benchmark]
    else:
        benchmark_prices_ = benchmark_prices[regime_benchmark]
        joint_prices = pd.concat([portfolio_nav, benchmark_prices_], axis=1).dropna()
        pivot_prices = joint_prices[regime_benchmark]
        joint_prices = joint_prices[portfolio_nav.name]

    ax = fig.add_subplot(gs[0:2, :2])
    qis.plot_prices(prices=joint_prices,
                    perf_params=perf_params,
                    title=f"Cumulative performance with background colors using bear/normal/bull regimes of {regime_benchmark} {regime_params.freq}-returns",
                    ax=ax,
                    **kwargs)
    qis.add_bnb_regime_shadows(ax=ax, pivot_prices=pivot_prices, regime_params=regime_params)
    qis.set_spines(ax=ax, bottom_spine=False, left_spine=False)

    # dd
    ax = fig.add_subplot(gs[2:4, :2])
    qis.plot_rolling_drawdowns(prices=joint_prices,
                               title='Running Drawdowns',
                               dd_legend_type=dd_legend_type,
                               ax=ax, **kwargs)
    qis.add_bnb_regime_shadows(ax=ax, pivot_prices=pivot_prices, regime_params=regime_params)
    qis.set_spines(ax=ax, bottom_spine=False, left_spine=False)

    # under watre
    ax = fig.add_subplot(gs[4:6, :2])
    qis.plot_rolling_time_under_water(prices=joint_prices,
                                      title='Running Time under Water',
                                      dd_legend_type=dd_legend_type,
                                      ax=ax, **kwargs)
    qis.add_bnb_regime_shadows(ax=ax, pivot_prices=pivot_prices, regime_params=regime_params)
    qis.set_spines(ax=ax, bottom_spine=False, left_spine=False)

    # rolling performance
    ax = fig.add_subplot(gs[6:8, :2])
    portfolio_data.plot_rolling_perf(rolling_perf_stat=qis.RollingPerfStat.TOTAL_RETURNS,
                                     add_benchmarks=add_benchmarks_to_navs,
                                     roll_freq=freq_sharpe,
                                     rolling_window=sharpe_rolling_window,
                                     time_period=time_period,
                                     ax=ax,
                                     **qis.update_kwargs(kwargs, dict(trend_line=qis.TrendLine.ZERO_SHADOWS)))
    qis.add_bnb_regime_shadows(ax=ax, pivot_prices=pivot_prices, regime_params=regime_params)
    qis.set_spines(ax=ax, bottom_spine=False, left_spine=False)

    # exposures
    exposures = portfolio_data.get_weights(is_grouped=is_grouped, time_period=time_period, add_total=False)
    ax = fig.add_subplot(gs[8:10, :2])
    if exposures_freq is not None:
        exposures = exposures.resample(exposures_freq).last()
    qis.plot_stack(df=exposures,
                   use_bar_plot=True,
                   title='Exposures',
                   legend_stats=qis.LegendStats.AVG_NONNAN_LAST,
                   var_format='{:.1%}',
                   ax=ax,
                   **qis.update_kwargs(kwargs, dict(bbox_to_anchor=(0.5, 1.05), ncols=2)))
    qis.set_spines(ax=ax, bottom_spine=False, left_spine=False)

    # turnover
    ax = fig.add_subplot(gs[10:12, :2])
    turnover = portfolio_data.get_turnover(time_period=time_period, roll_period=turnover_rolling_period,
                                           freq=freq_turnover,
                                           is_grouped=is_grouped,
                                           is_unit_based_traded_volume=is_unit_based_traded_volume)
    freq = pd.infer_freq(turnover.index)
    turnover_title = f"{turnover_rolling_period}-period rolling {freq}-freq Turnover"
    qis.plot_time_series(df=turnover,
                         var_format='{:,.2%}',
                         # y_limits=(0.0, None),
                         legend_stats=qis.LegendStats.AVG_NONNAN_LAST,
                         title=turnover_title,
                         ax=ax,
                         **kwargs)
    qis.add_bnb_regime_shadows(ax=ax, pivot_prices=pivot_prices, regime_params=regime_params)
    qis.set_spines(ax=ax, bottom_spine=False, left_spine=False)

    # costs
    ax = fig.add_subplot(gs[12:14, :2])
    costs = portfolio_data.get_costs(time_period=time_period, roll_period=cost_rolling_period,
                                     freq=freq_cost,
                                     is_grouped=is_grouped,
                                     is_unit_based_traded_volume=is_unit_based_traded_volume)
    freq = pd.infer_freq(costs.index)
    costs_title = f"{cost_rolling_period}-period rolling {freq}-freq Costs"
    qis.plot_time_series(df=costs,
                         var_format='{:,.2%}',
                         # y_limits=(0.0, None),
                         legend_stats=qis.LegendStats.AVG_NONNAN_LAST,
                         title=costs_title,
                         ax=ax,
                         **kwargs)
    qis.add_bnb_regime_shadows(ax=ax, pivot_prices=pivot_prices, regime_params=regime_params)
    qis.set_spines(ax=ax, bottom_spine=False, left_spine=False)

    # ra perf table
    if add_benchmarks_to_navs:
        benchmark_price1 = benchmark_prices
    else:
        benchmark_price1 = benchmark_prices[regime_benchmark]
    if is_grouped:
        ax = fig.add_subplot(gs[:2, 2:])
        portfolio_data.plot_ra_perf_table(ax=ax,
                                          benchmark_price=benchmark_price1,
                                          time_period=time_period,
                                          perf_params=perf_params,
                                          is_grouped=is_grouped,
                                          **qis.update_kwargs(kwargs, dict(fontsize=fontsize)))
    else:  # plot two tables
        ax = fig.add_subplot(gs[0, 2:])
        portfolio_data.plot_ra_perf_table(ax=ax,
                                          benchmark_price=benchmark_price1,
                                          time_period=time_period,
                                          perf_params=perf_params,
                                          is_grouped=is_grouped,
                                          **qis.update_kwargs(kwargs, dict(fontsize=fontsize)))
        ax = fig.add_subplot(gs[1, 2:])
        # change regression to weekly
        time_period1 = qis.get_time_period_shifted_by_years(time_period=time_period)
        if pd.infer_freq(benchmark_prices.index) in ['B', 'D']:
            local_kwargs = qis.update_kwargs(kwargs, dict(time_period=time_period1, alpha_an_factor=52, freq_reg='W-WED', fontsize=fontsize))
        else:
            local_kwargs = qis.update_kwargs(kwargs, dict(time_period=time_period1, fontsize=fontsize))
        portfolio_data.plot_ra_perf_table(ax=ax,
                                          benchmark_price=benchmark_price1,
                                          perf_params=perf_params,
                                          is_grouped=is_grouped,
                                          **local_kwargs)

    # heatmap
    ax = fig.add_subplot(gs[2:4, 2:])
    portfolio_data.plot_monthly_returns_heatmap(ax=ax,
                                                time_period=time_period,
                                                title='Monthly Returns',
                                                **qis.update_kwargs(kwargs, dict(fontsize=fontsize, date_format='%Y')))

    # periodic returns
    ax = fig.add_subplot(gs[4:6, 2:])
    local_kwargs = qis.update_kwargs(kwargs=kwargs,
                                     new_kwargs=dict(fontsize=fontsize, square=False, x_rotation=90, transpose=True))
    portfolio_data.plot_periodic_returns(benchmark_prices=benchmark_price1,
                                         is_grouped=is_grouped,
                                         time_period=time_period,
                                         ax=ax,
                                         **local_kwargs)

    # regime data
    portfolio_data.plot_regime_data(is_grouped=is_grouped,
                                    benchmark_price=benchmark_prices[regime_benchmark],
                                    time_period=time_period,
                                    perf_params=perf_params,
                                    regime_params=regime_params,
                                    ax=fig.add_subplot(gs[6:8, 2:]),
                                    **kwargs)

    # perf attribution
    local_kwargs = qis.update_kwargs(kwargs=kwargs, new_kwargs=dict(legend_loc=None))
    with sns.axes_style("whitegrid"):
        portfolio_data.plot_performance_attribution(time_period=time_period,
                                                    attribution_metric=qis.AttributionMetric.PNL,
                                                    ax=fig.add_subplot(gs[8:10, 2:]),
                                                    **local_kwargs)
        portfolio_data.plot_performance_attribution(time_period=time_period,
                                                    attribution_metric=qis.AttributionMetric.PNL_RISK,
                                                    ax=fig.add_subplot(gs[10:12, 2:]),
                                                    **local_kwargs)

    # constituents
    ax = fig.add_subplot(gs[12:, 2:])
    num_investable_instruments = portfolio_data.get_num_investable_instruments(time_period=time_period)
    qis.plot_time_series(df=num_investable_instruments,
                         var_format='{:,.0f}',
                         legend_stats=qis.LegendStats.FIRST_AVG_LAST,
                         title='Number of investable and invested instruments',
                         ax=ax,
                         **kwargs)
    qis.add_bnb_regime_shadows(ax=ax, pivot_prices=pivot_prices, regime_params=regime_params)
    qis.set_spines(ax=ax, bottom_spine=False, left_spine=False)

    figs = [fig]

    if add_current_position_var_risk_sheet:
        # qqq
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        fig.suptitle(f"{portfolio_data.nav.name} 99%-Var and risk profile", fontweight="bold", fontsize=8, color='blue')
        figs.append(fig)
        gs = fig.add_gridspec(nrows=7, ncols=4, wspace=0.0, hspace=0.0)

        # current var grouped
        with sns.axes_style("whitegrid"):
            axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3])]
            portfolio_data.plot_current_var(snapshot_period=qis.SnapshotPeriod.LAST,
                                            is_grouped=True, is_correlated=False, time_period=time_period,
                                            freq=freq_var, vol_span=var_span,
                                            ax=axs[0], **kwargs)
            portfolio_data.plot_current_var(snapshot_period=qis.SnapshotPeriod.MAX,
                                            is_grouped=True, is_correlated=False, time_period=time_period,
                                            freq=freq_var, vol_span=var_span,
                                            ax=axs[1], **kwargs)

            portfolio_data.plot_current_var(snapshot_period=qis.SnapshotPeriod.LAST,
                                            is_grouped=True, is_correlated=True, time_period=time_period,
                                            freq=freq_var, vol_span=var_span,
                                            ax=axs[2], **kwargs)
            portfolio_data.plot_current_var(snapshot_period=qis.SnapshotPeriod.MAX,
                                            is_grouped=True, is_correlated=True, time_period=time_period,
                                            freq=freq_var, vol_span=var_span,
                                            ax=axs[3], **kwargs)
            qis.align_y_limits_axs(axs=axs)

        # last / max var by instrument
        with sns.axes_style("whitegrid"):
            axs = [fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2:4])]
            portfolio_data.plot_current_var(snapshot_period=qis.SnapshotPeriod.LAST,
                                            is_grouped=False, is_correlated=False, time_period=time_period,
                                            freq=freq_var, vol_span=var_span,
                                            ax=axs[0], **kwargs)
            portfolio_data.plot_current_var(snapshot_period=qis.SnapshotPeriod.MAX,
                                            is_grouped=False, is_correlated=False, time_period=time_period,
                                            freq=freq_var, vol_span=var_span,
                                            ax=axs[1], **kwargs)
            qis.align_y_limits_axs(axs=axs)

        # best worst returns
        with sns.axes_style("whitegrid"):
            portfolio_data.plot_best_worst_returns(ax=fig.add_subplot(gs[2, 0:2]),
                                                   num_returns=20,
                                                   time_period=time_period,
                                                   title=f"Portfolio Worst/Best 20 returns for {time_period.to_str()}",
                                                   **qis.update_kwargs(kwargs, dict(date_format='%d%b%Y')))
            num_assets = 10
            num_investable_assets = len(num_investable_instruments.columns)
            if num_investable_assets > 2 * num_assets:
                pre_title = f"-{num_assets}"
            else:
                pre_title = ''
            portfolio_data.plot_contributors(ax=fig.add_subplot(gs[2, 2]),
                                             time_period=time_period,
                                             title=f"Bottom/Top{pre_title} performance contributors {time_period.to_str()}",
                                             **kwargs)
            time_period_1y = qis.get_time_period_shifted_by_years(time_period=time_period)
            portfolio_data.plot_contributors(ax=fig.add_subplot(gs[2, 3]),
                                             time_period=time_period_1y,
                                             title=f"Bottom/Top-{num_assets} performance contributors {time_period_1y.to_str()}",
                                             **kwargs)

        # var attribution
        with sns.axes_style("whitegrid"):
            portfolio_data.plot_var_stack(is_grouped=True, is_correlated=False,
                                          time_period=time_period,
                                          ax=fig.add_subplot(gs[3, :2]),
                                          **kwargs)
            portfolio_data.plot_var_stack(is_grouped=True, is_correlated=True,
                                          time_period=time_period,
                                          ax=fig.add_subplot(gs[3, 2:]),
                                          **kwargs)

        # var time series - independent
        ax = fig.add_subplot(gs[4, :2])
        portfolio_data.plot_portfolio_grouped_var(ax=ax,
                                                  is_correlated=False,
                                                  time_period=time_period,
                                                  **kwargs)
        qis.add_bnb_regime_shadows(ax=ax, pivot_prices=pivot_prices, regime_params=regime_params)
        qis.set_spines(ax=ax, bottom_spine=False, left_spine=False)

        # var time series - correlted
        ax = fig.add_subplot(gs[4, 2:])
        portfolio_data.plot_portfolio_grouped_var(ax=ax,
                                                  is_correlated=True,
                                                  time_period=time_period,
                                                  **kwargs)
        qis.add_bnb_regime_shadows(ax=ax, pivot_prices=pivot_prices, regime_params=regime_params)
        qis.set_spines(ax=ax, bottom_spine=False, left_spine=False)

        # vol time series
        ax = fig.add_subplot(gs[5, 2:])
        portfolio_data.plot_portfolio_vols(freq=perf_params.freq_vol,
                                           span=vol_rolling_window,
                                           time_period=time_period,
                                           ax=ax,
                                           **kwargs)
        qis.add_bnb_regime_shadows(ax=ax, pivot_prices=pivot_prices, regime_params=regime_params)
        qis.set_spines(ax=ax, bottom_spine=False, left_spine=False)

        # benchmark betas
        ax = fig.add_subplot(gs[5, :2])
        factor_exposures = portfolio_data.compute_portfolio_benchmark_betas(benchmark_prices=benchmark_prices,
                                                                            time_period=time_period,
                                                                            freq_beta=freq_beta,
                                                                            factor_beta_span=factor_beta_span)
        factor_beta_title = f"Rolling {factor_beta_span}-span beta of {freq_beta}-freq returns"
        qis.plot_time_series(df=factor_exposures,
                             var_format='{:,.2f}',
                             legend_stats=qis.LegendStats.AVG_NONNAN_LAST,
                             title=factor_beta_title,
                             ax=ax,
                             **kwargs)
        qis.add_bnb_regime_shadows(ax=ax, pivot_prices=pivot_prices, regime_params=regime_params)
        qis.set_spines(ax=ax, bottom_spine=False, left_spine=False)

        # beta attribution
        ax = fig.add_subplot(gs[6, :2])
        factor_attribution = portfolio_data.compute_portfolio_benchmark_attribution(benchmark_prices=benchmark_prices,
                                                                                    freq_beta=freq_beta,
                                                                                    factor_beta_span=factor_beta_span,
                                                                                    time_period=time_period)
        factor_attribution_title = (f"Cumulative return attribution using rolling "
                                    f"{factor_beta_span}-span beta of {freq_beta}-freq returns")
        qis.plot_time_series(df=factor_attribution.cumsum(0),
                             var_format='{:,.0%}',
                             legend_stats=qis.LegendStats.LAST_NONNAN,
                             title=factor_attribution_title,
                             ax=ax,
                             **kwargs)
        qis.add_bnb_regime_shadows(ax=ax, pivot_prices=pivot_prices, regime_params=regime_params)
        qis.set_spines(ax=ax, bottom_spine=False, left_spine=False)
        """
        # returns scatter
        with sns.axes_style("whitegrid"):
            portfolio_data.plot_returns_scatter(ax=fig.add_subplot(gs[6, 2:]),
                                                benchmark_price=benchmark_prices[regime_benchmark],
                                                time_period=time_period,
                                                freq=perf_params.freq_reg,
                                                is_grouped=is_grouped,
                                                **kwargs)

        """
        # vol regime data
        ax = fig.add_subplot(gs[6, 2:])
        portfolio_data.plot_vol_regimes(ax=ax,
                                        benchmark_price=benchmark_prices[regime_benchmark],
                                        is_grouped=is_grouped,
                                        time_period=time_period,
                                        freq=regime_params.freq,
                                        regime_params=regime_params,
                                        **kwargs)
        """
        if len(benchmark_prices.columns) > 1:
            ax = fig.add_subplot(gs[12:14, 2:])
            portfolio_data.plot_vol_regimes(ax=ax,
                                            benchmark_price=benchmark_prices.iloc[:, 1],
                                            is_grouped=is_grouped,
                                            time_period=time_period,
                                            freq=regime_params.freq,
                                            regime_params=regime_params,
                                            **kwargs)
        """
    if add_weights_turnover_sheet:
        # qqq
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        fig.suptitle(f"{portfolio_data.nav.name} current position profile", fontweight="bold", fontsize=8, color='blue')
        figs.append(fig)
        gs = fig.add_gridspec(nrows=7, ncols=4, wspace=0.0, hspace=0.0)

        # current position
        with sns.axes_style("whitegrid"):
            portfolio_data.plot_current_weights(is_grouped=False,
                                                ax=fig.add_subplot(gs[0, :2]), **kwargs)
            portfolio_data.plot_current_weights(is_grouped=True,
                                                ax=fig.add_subplot(gs[0, 2:]), **kwargs)

        # change in position
        with sns.axes_style("whitegrid"):
            portfolio_data.plot_last_weights_change(is_grouped=False,
                                                    ax=fig.add_subplot(gs[1, :2]), **kwargs)
            portfolio_data.plot_last_weights_change(is_grouped=True,
                                                    ax=fig.add_subplot(gs[1, 2:]), **kwargs)

        # weights for ytd performance attribution
        with sns.axes_style("whitegrid"):
            portfolio_data.plot_current_weights(is_grouped=False,
                                                time_period=ytd_attribution_time_period,
                                                ax=fig.add_subplot(gs[2, :2]),
                                                **kwargs)
            portfolio_data.plot_current_weights(is_grouped=True,
                                                time_period=ytd_attribution_time_period,
                                                ax=fig.add_subplot(gs[2, 2:]),
                                                **kwargs)

        # total and ytd performance attribution
        with sns.axes_style("whitegrid"):
            local_kwargs = qis.update_kwargs(kwargs=kwargs, new_kwargs=dict(legend_loc=None))
            portfolio_data.plot_performance_attribution(time_period=time_period,
                                                        attribution_metric=qis.AttributionMetric.PNL,
                                                        ax=fig.add_subplot(gs[3, :2]),
                                                        **local_kwargs)
            portfolio_data.plot_performance_attribution(time_period=ytd_attribution_time_period,
                                                        attribution_metric=qis.AttributionMetric.PNL,
                                                        ax=fig.add_subplot(gs[3, 2:]),
                                                        **local_kwargs)
        # turnover
        with sns.axes_style("whitegrid"):
            local_kwargs = qis.update_kwargs(kwargs=kwargs, new_kwargs=dict(legend_loc=None))
            portfolio_data.plot_performance_attribution(time_period=time_period,
                                                        attribution_metric=qis.AttributionMetric.TURNOVER,
                                                        ax=fig.add_subplot(gs[4, :2]),
                                                        **local_kwargs)
            portfolio_data.plot_performance_attribution(time_period=ytd_attribution_time_period,
                                                        attribution_metric=qis.AttributionMetric.TURNOVER,
                                                        ax=fig.add_subplot(gs[4, 2:]),
                                                        **local_kwargs)

        # vol adjusted turnover
        with sns.axes_style("whitegrid"):
            local_kwargs = qis.update_kwargs(kwargs=kwargs, new_kwargs=dict(legend_loc=None))
            portfolio_data.plot_performance_attribution(time_period=time_period,
                                                        attribution_metric=qis.AttributionMetric.VOL_ADJUSTED_TURNOVER,
                                                        ax=fig.add_subplot(gs[5, :2]),
                                                        **local_kwargs)
            portfolio_data.plot_performance_attribution(time_period=ytd_attribution_time_period,
                                                        attribution_metric=qis.AttributionMetric.VOL_ADJUSTED_TURNOVER,
                                                        ax=fig.add_subplot(gs[5, 2:]),
                                                        **local_kwargs)

        # costs
        with sns.axes_style("whitegrid"):
            local_kwargs = qis.update_kwargs(kwargs=kwargs, new_kwargs=dict(legend_loc=None, is_unit_based_traded_volume=is_unit_based_traded_volume))
            portfolio_data.plot_performance_attribution(time_period=time_period,
                                                        attribution_metric=qis.AttributionMetric.COSTS,
                                                        ax=fig.add_subplot(gs[6, :2]),
                                                        **local_kwargs)
            portfolio_data.plot_performance_attribution(time_period=ytd_attribution_time_period,
                                                        attribution_metric=qis.AttributionMetric.COSTS,
                                                        ax=fig.add_subplot(gs[6, 2:]),
                                                        **local_kwargs)

    if add_current_signal_report and portfolio_data.strategy_signal_data is not None:
        fig = qis.generate_current_signal_report(portfolio_data=portfolio_data,
                                                 **qis.update_kwargs(kwargs, dict(fontsize=5, figsize=figsize,
                                                                                  y_limits=y_limits_signal)))
        figs.append(fig)

    if add_weight_change_report and portfolio_data.strategy_signal_data is not None:
        fig = qis.generate_weight_change_report(portfolio_data=portfolio_data,
                                                time_period=weight_report_time_period or time_period,
                                                sample_size=weight_change_sample_size,
                                                is_grouped=True,
                                                **qis.update_kwargs(kwargs, dict(fontsize=5, figsize=figsize)))
        figs.append(fig)

    # set 1y time period for exposures
    if is_1y_exposures:
        time_period1 = qis.get_time_period_shifted_by_years(time_period=time_period)
        regime_params1 = BenchmarkReturnsQuantileRegimeSpecs(freq='ME')
    else:
        time_period1 = weight_report_time_period or time_period
        regime_params1 = regime_params

    if add_grouped_exposures:
        grouped_exposures_agg, grouped_exposures_by_inst = portfolio_data.get_grouped_long_short_exposures(time_period=time_period1)
        nrows = len(grouped_exposures_agg.keys())
        fig1 = plt.figure(figsize=figsize, constrained_layout=True)
        figs.append(fig1)
        fig1.suptitle(f"{portfolio_data.nav.name} Exposures by groups for period {time_period1.to_str()}",
                     fontweight="bold", fontsize=8, color='blue')
        gs = fig1.add_gridspec(nrows=nrows, ncols=2, wspace=0.0, hspace=0.0)
        local_kwargs = qis.update_kwargs(kwargs=kwargs, new_kwargs=dict(framealpha=0.9))
        for idx, (group, exposures_agg) in enumerate(grouped_exposures_agg.items()):
            datas = {f"{group} aggregated": grouped_exposures_agg[group],
                     f"{group} by instrument": grouped_exposures_by_inst[group]}
            for idx_, (key, df) in enumerate(datas.items()):
                ax = fig1.add_subplot(gs[idx, idx_])
                qis.plot_time_series(df=df,
                                     var_format='{:,.0%}',
                                     legend_stats=qis.LegendStats.AVG_MIN_MAX_LAST,
                                     title=f"{key}",
                                     ax=ax,
                                     **local_kwargs)
                qis.add_bnb_regime_shadows(ax=ax, pivot_prices=time_period1.locate(pivot_prices),
                                           regime_params=regime_params1)
                qis.set_spines(ax=ax, bottom_spine=False, left_spine=False)
                ax.axhline(0, color='black', linewidth=0.5)

    if add_grouped_cum_pnl:
        grouped_pnls_agg, grouped_pnls_by_inst = portfolio_data.get_grouped_cum_pnls(time_period=time_period1)
        nrows = len(grouped_pnls_agg.keys())
        fig1 = plt.figure(figsize=figsize, constrained_layout=True)
        figs.append(fig1)
        fig1.suptitle(f"{portfolio_data.nav.name} P&L by groups for period {time_period1.to_str()}",
                     fontweight="bold", fontsize=8, color='blue')
        gs = fig1.add_gridspec(nrows=nrows, ncols=2, wspace=0.0, hspace=0.0)
        local_kwargs = qis.update_kwargs(kwargs=kwargs, new_kwargs=dict(framealpha=0.9))
        for idx, (group, pnls_agg) in enumerate(grouped_pnls_agg.items()):
            datas = {f"{group} aggregated": grouped_pnls_agg[group],
                     f"{group} by instrument": grouped_pnls_by_inst[group]}
            for idx_, (key, df) in enumerate(datas.items()):
                ax = fig1.add_subplot(gs[idx, idx_])
                qis.plot_time_series(df=df,
                                     var_format='{:,.0%}',
                                     legend_stats=qis.LegendStats.LAST_NONNAN,
                                     title=f"{key}",
                                     ax=ax,
                                     **local_kwargs)
                qis.add_bnb_regime_shadows(ax=ax,
                                           pivot_prices=time_period1.locate(pivot_prices),
                                           regime_params=regime_params1)
                qis.set_spines(ax=ax, bottom_spine=False, left_spine=False)

    if add_instrument_history_report:
        if df_to_add is None:
            ac_data = portfolio_data.group_data.to_frame(name='AC')
            if portfolio_data.instrument_names is not None:
                df_to_add = pd.concat([portfolio_data.instrument_names.rename('Name'),
                                       ac_data], axis=1)
            else:
                df_to_add = ac_data

        perf_columns = (qis.PerfStat.START_DATE, qis.PerfStat.END_DATE, qis.PerfStat.PA_RETURN,
                        qis.PerfStat.VOL, qis.PerfStat.SHARPE_RF0,
                        qis.PerfStat.MAX_DD, qis.PerfStat.MAX_DD_VOL, qis.PerfStat.SKEWNESS)

        fig = qis.generate_price_history_report(prices=portfolio_data.prices,
                                                **qis.update_kwargs(kwargs, dict(fontsize=4, figsize=figsize,
                                                                                 perf_columns=perf_columns)))
        fig.suptitle('Program Instrument Universe', fontweight="bold", fontsize=8, color='blue')
        figs.append(fig)

    return figs

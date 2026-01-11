"""
report for strategy benchmark with tre
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict

import qis as qis
from qis import TimePeriod, PerfParams, BenchmarkReturnsQuantilesRegime
from qis.portfolio.multi_portfolio_data import MultiPortfolioData
from qis.portfolio.risk.factor_model import LinearModel
from qis.portfolio.reports.config import PERF_PARAMS, regime_classifier


def weights_tracking_error_report_by_ac_subac(multi_portfolio_data: MultiPortfolioData,
                                              strategy_idx: int = 0,
                                              benchmark_idx: int = 1,
                                              ac_group_data: pd.Series = None,
                                              ac_group_order: List[str] = None,
                                              sub_ac_group_data: pd.Series = None,
                                              sub_ac_group_order: List[str] = None,
                                              turnover_groups: pd.Series = None,
                                              turnover_order: List[str] = None,
                                              risk_model: LinearModel = None,
                                              time_period: TimePeriod = None,
                                              perf_params: PerfParams = PERF_PARAMS,
                                              regime_classifier: BenchmarkReturnsQuantilesRegime = BenchmarkReturnsQuantilesRegime(),
                                              add_benchmarks_to_navs: bool = True,
                                              tre_max_clip: Optional[float] = None,
                                              figsize: Tuple[float, float] = (11.7, 8.3),
                                              var_format: str = '{:.1%}',
                                              add_titles: bool = True,
                                              **kwargs
                                              ) -> Tuple[Dict[str, plt.Figure], Dict[str, pd.DataFrame]]:
    """
    weights and tracking error report relative to benchnmark using ac and sub_as groups
    """
    regime_benchmark = multi_portfolio_data.benchmark_prices.columns[0]
    benchmark_price = multi_portfolio_data.benchmark_prices[regime_benchmark]

    figs: Dict[str, plt.Figure] = {}
    dfs: Dict[str, pd.DataFrame] = {}

    with (sns.axes_style('darkgrid')):
        # navs + ra table
        fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
        figs['navs'] = fig
        if add_titles:
            title = f"Cumulative performance with background colors using bear/normal/bull "
            f"regimes of {regime_benchmark} {regime_classifier.freq}-returns"
        else:
            title = None
        multi_portfolio_data.plot_nav(regime_benchmark=regime_benchmark,
                                      time_period=time_period,
                                      perf_params=perf_params,
                                      regime_classifier=regime_classifier,
                                      add_benchmarks_to_navs=add_benchmarks_to_navs,
                                      title=title,
                                      ax=ax,
                                      **kwargs)

        fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
        figs['ra_table'] = fig
        dfs['nav'] = multi_portfolio_data.get_navs(time_period=time_period, add_benchmarks_to_navs=add_benchmarks_to_navs)
        multi_portfolio_data.plot_ra_perf_table(benchmark_price=benchmark_price,
                                                                add_benchmarks_to_navs=add_benchmarks_to_navs,
                                                                perf_params=perf_params,
                                                                time_period=time_period,
                                                                add_turnover=True,
                                                                ax=ax,
                                                                **kwargs)
        # ra perf table without strings
        ra_perf_table = multi_portfolio_data.plot_ra_perf_table(benchmark_price=benchmark_price,
                                                                add_benchmarks_to_navs=add_benchmarks_to_navs,
                                                                perf_params=perf_params,
                                                                time_period=time_period,
                                                                add_turnover=True,
                                                                is_convert_to_str=False,
                                                                ax=ax,
                                                                **kwargs)
        dfs['ra_perf_table'] = ra_perf_table

        # strategy weights
        strategy_data = multi_portfolio_data.portfolio_datas[strategy_idx]
        strategy_ticker = strategy_data.ticker
        weight_kwargs = dict(is_grouped=True, time_period=time_period, add_total=False, is_input_weights=True)
        strategy_exposures_ac = strategy_data.get_weights(group_data=ac_group_data, group_order=ac_group_order,
                                                          **weight_kwargs)
        strategy_exposures_subac = strategy_data.get_weights(group_data=sub_ac_group_data, group_order=sub_ac_group_order,
                                                             **weight_kwargs)

        # benchmark weights
        benchmark_data = multi_portfolio_data.portfolio_datas[benchmark_idx]
        benchmark_ticker = benchmark_data.ticker
        benchmark_exposures_ac = benchmark_data.get_weights(group_data=ac_group_data, group_order=ac_group_order,
                                                            **weight_kwargs)
        benchmark_exposures_subac = benchmark_data.get_weights(group_data=sub_ac_group_data, group_order=sub_ac_group_order,
                                                               **weight_kwargs)

        # plot strategy and benchmark weights by ac
        kwargs = qis.update_kwargs(kwargs, dict(strategy_ticker=f"(B) {strategy_ticker}",
                                                benchmark_ticker=f"(A) {benchmark_ticker}"))
        fig, axs = plt.subplots(1, 2, figsize=figsize, tight_layout=True)
        if add_titles:
            qis.set_suptitle(fig, title=f"Time series of weights by asset classes")
        figs['strategy_benchmark_weights_stack'] = fig
        plot_exposures_strategy_vs_benchmark_stack(strategy_exposures=strategy_exposures_ac,
                                                   benchmark_exposures=benchmark_exposures_ac,
                                                   axs=axs,
                                                   var_format=var_format,
                                                   **kwargs)

        # boxplot by subac
        fig, axs = plt.subplots(1, 2, figsize=figsize, tight_layout=True)
        if add_titles:
            qis.set_suptitle(fig, title=f"Boxplot of weights")
        figs['strategy_benchmark_weights_box'] = fig
        plot_exposures_strategy_vs_benchmark_boxplot(strategy_exposures=strategy_exposures_ac,
                                                     benchmark_exposures=benchmark_exposures_ac,
                                                     ax=axs[0],
                                                     ylabel='Weights',
                                                     title='(A) Weights by asset classes',
                                                     hue_var_name='Asset Class',
                                                     var_format=var_format,
                                                     allow_negative=True,
                                                     **kwargs)
        plot_exposures_strategy_vs_benchmark_boxplot(strategy_exposures=strategy_exposures_subac,
                                                     benchmark_exposures=benchmark_exposures_subac,
                                                     ax=axs[1],
                                                     ylabel='Weights',
                                                     title='(B) Weights by sub-asset classes',
                                                     hue_var_name='Sub-Asset Class',
                                                     var_format=var_format,
                                                     **kwargs)
        # portfolio vol
        strategy_ex_anti_vol = strategy_data.compute_ex_anti_portfolio_vol_implied_by_covar(
            covar_dict=multi_portfolio_data.covar_dict)
        benchmark_ex_anti_vol = benchmark_data.compute_ex_anti_portfolio_vol_implied_by_covar(
            covar_dict=multi_portfolio_data.covar_dict)

        ex_anti_vols = pd.concat([strategy_ex_anti_vol, benchmark_ex_anti_vol], axis=1)
        dfs['ex_anti_vols'] = ex_anti_vols
        fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
        if add_titles:
            qis.set_suptitle(fig, title=f"Ex-anti portfolio volatility")
        figs['ex_anti_vols'] = fig
        qis.plot_time_series(df=ex_anti_vols,
                             var_format='{:.2%}',
                             ax=ax,
                             **kwargs)
        if regime_benchmark is not None:
            multi_portfolio_data.add_regime_shadows(ax=ax, regime_benchmark=regime_benchmark,
                                                    index=ex_anti_vols.index, regime_classifier=regime_classifier)

        # risk contributions
        rc_kwargs = dict(covar_dict=multi_portfolio_data.covar_dict, freq='QE', normalise=True, time_period=time_period)
        strategy_risk_contributions_ac = strategy_data.compute_risk_contributions_implied_by_covar(
            group_data=ac_group_data,
            group_order=ac_group_order,
            **rc_kwargs)
        strategy_risk_contributions_subac = strategy_data.compute_risk_contributions_implied_by_covar(
            group_data=sub_ac_group_data,
            group_order=sub_ac_group_order,
            **rc_kwargs)

        benchmark_risk_contributions_ac = benchmark_data.compute_risk_contributions_implied_by_covar(
            group_data=ac_group_data,
            group_order=ac_group_order,
            **rc_kwargs)
        benchmark_risk_contributions_subac = benchmark_data.compute_risk_contributions_implied_by_covar(
            group_data=sub_ac_group_data,
            group_order=sub_ac_group_order,
            **rc_kwargs)

        # stack for ac
        fig, axs = plt.subplots(1, 2, figsize=figsize, tight_layout=True)
        if add_titles:
            qis.set_suptitle(fig, title=f"Time Series of risk contributions by asset classes")
        figs['time_series_risk_contrib'] = fig
        plot_exposures_strategy_vs_benchmark_stack(strategy_exposures=strategy_risk_contributions_ac,
                                                   benchmark_exposures=benchmark_risk_contributions_ac,
                                                   axs=axs,
                                                   var_format=var_format,
                                                   **kwargs)

        # box plots for subac
        fig, axs = plt.subplots(1, 2, figsize=figsize, tight_layout=True)
        if add_titles:
            qis.set_suptitle(fig, title=f"Boxplot of risk contributions")
        figs['risk_contributions_boxplot'] = fig
        plot_exposures_strategy_vs_benchmark_boxplot(
            strategy_exposures=strategy_risk_contributions_ac,
            benchmark_exposures=benchmark_risk_contributions_ac,
            ax=axs[0],
            title='(A) Risk contributions by asset classes',
            hue_var_name='Asset Class',
            ylabel='Risk contributions',
            var_format=var_format,
            allow_negative=True,
            **kwargs)
        plot_exposures_strategy_vs_benchmark_boxplot(
            strategy_exposures=strategy_risk_contributions_subac,
            benchmark_exposures=benchmark_risk_contributions_subac,
            ax=axs[1],
            title='(B) Risk contributions by sub-asset classes',
            hue_var_name='Sub-Asset Class',
            ylabel='Risk contributions',
            var_format=var_format,
            allow_negative=True,
            **kwargs)

        # brinson by asset class
        totals_table, active_total, grouped_allocation_return, grouped_selection_return, grouped_interaction_return = \
            multi_portfolio_data.compute_brinson_attribution(strategy_idx=strategy_idx,
                                                             benchmark_idx=benchmark_idx,
                                                             time_period=time_period,
                                                             group_data=ac_group_data,
                                                             group_order=ac_group_order,
                                                             freq=None,
                                                             total_column='Total Sum',
                                                             is_exclude_interaction_term=True)
        figs['brinson_table_ac'] = qis.plot_brinson_totals_table(totals_table=totals_table, **kwargs)
        dfs['brinson_table_ac'] = totals_table

        fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
        figs['brinson_total_time_series'] = fig
        if add_titles:
            title = 'Active total return'
        else:
            title = None
        qis.plot_time_series(df=active_total.cumsum(axis=0),
                             title=title,
                             legend_stats=qis.LegendStats.LAST,
                             var_format='{:.0%}',
                             ax=ax, **kwargs)
        if regime_benchmark is not None:
            multi_portfolio_data.add_regime_shadows(ax=ax, regime_benchmark=regime_benchmark,
                                                    index=active_total.index, regime_classifier=regime_classifier)
        dfs['brinson_active_total'] = active_total.cumsum(axis=0)

        fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
        figs['brinson_grouped_allocation_return'] = fig
        qis.plot_time_series(df=grouped_allocation_return.cumsum(axis=0),
                             title='Grouped allocation return',
                             legend_stats=qis.LegendStats.LAST,
                             var_format='{:.0%}',
                             ax=ax, **kwargs)
        if regime_benchmark is not None:
            multi_portfolio_data.add_regime_shadows(ax=ax, regime_benchmark=regime_benchmark,
                                                    index=grouped_allocation_return.index, regime_classifier=regime_classifier)

        fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
        figs['brinson_grouped_selection_return'] = fig
        qis.plot_time_series(df=grouped_selection_return.cumsum(axis=0),
                             title='Grouped selection return',
                             legend_stats=qis.LegendStats.LAST,
                             var_format='{:.0%}',
                             ax=ax, **kwargs)
        if regime_benchmark is not None:
            multi_portfolio_data.add_regime_shadows(ax=ax, regime_benchmark=regime_benchmark,
                                                    index=grouped_selection_return.index, regime_classifier=regime_classifier)

        # brinson by sub-asset class
        totals_table, active_total, grouped_allocation_return, grouped_selection_return, grouped_interaction_return = \
            multi_portfolio_data.compute_brinson_attribution(strategy_idx=strategy_idx,
                                                             benchmark_idx=benchmark_idx,
                                                             time_period=time_period,
                                                             group_data=sub_ac_group_data,
                                                             group_order=sub_ac_group_order,
                                                             freq=None,
                                                             total_column='Total Sum',
                                                             is_exclude_interaction_term=True)
        figs['brinson_table_subac'] = qis.plot_brinson_totals_table(totals_table=totals_table, **kwargs)
        dfs['brinson_table_subac'] = totals_table

        # tracking error
        fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
        figs['tre_time_series'] = fig
        if add_titles:
            title = 'Tracking Error'
        else:
            title = None
        multi_portfolio_data.plot_tre_time_series(strategy_idx=strategy_idx,
                                                  benchmark_idx=benchmark_idx,
                                                  regime_benchmark=regime_benchmark,
                                                  regime_classifier=regime_classifier,
                                                  title=title,
                                                  ax=ax,
                                                  time_period=time_period,
                                                  **kwargs)

        # group tracking error
        fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
        figs['tre_group_time_series'] = fig
        if add_titles:
            title = 'Asset Class Tracking Error'
        else:
            title = None
        multi_portfolio_data.plot_tre_time_series(strategy_idx=strategy_idx,
                                                  benchmark_idx=benchmark_idx,
                                                  is_grouped=True,
                                                  group_data=ac_group_data,
                                                  group_order=ac_group_order,
                                                  regime_benchmark=regime_benchmark,
                                                  regime_classifier=regime_classifier,
                                                  tre_max_clip=tre_max_clip,
                                                  title=title,
                                                  ax=ax,
                                                  time_period=time_period,
                                                  **kwargs)
        dfs['ac_tracking_error'] = multi_portfolio_data.compute_tracking_error_implied_by_covar(strategy_idx=strategy_idx,
                                                                                                benchmark_idx=benchmark_idx,
                                                                                                is_grouped=True,
                                                                                                group_data=ac_group_data,
                                                                                                group_order=ac_group_order,
                                                                                                total_column='Total')

        # turnover
        fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
        figs['joint_turnover'] = fig
        multi_portfolio_data.plot_turnover(ax=ax,
                                           time_period=time_period,
                                           regime_benchmark=regime_benchmark,
                                           regime_classifier=regime_classifier,
                                           **kwargs)
        if not add_titles:
            ax.title.set_visible(False)

        # group turnover
        fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
        figs['group_turnover'] = fig
        multi_portfolio_data.portfolio_datas[strategy_idx].plot_turnover(ax=ax,
                                                                         time_period=time_period,
                                                                         regime_benchmark=regime_benchmark,
                                                                         regime_classifier=regime_classifier,
                                                                         is_grouped=True,
                                                                         group_data=turnover_groups,
                                                                         group_order=turnover_order,
                                                                         add_total=False,
                                                                         **kwargs)
        if regime_benchmark is not None:
            multi_portfolio_data.add_regime_shadows(ax=ax, regime_benchmark=regime_benchmark,
                                                    index=grouped_selection_return.index, regime_classifier=regime_classifier)
        if not add_titles:
            ax.title.set_visible(False)

        dfs['ac_turnover'] = multi_portfolio_data.portfolio_datas[strategy_idx].get_turnover(is_agg=False,
                                                                                             is_grouped=True,
                                                                                             group_data=turnover_groups,
                                                                                             group_order=turnover_order,
                                                                                             time_period=time_period,
                                                                                             add_total=False,
                                                                                             **kwargs)

        # pdf of returns
        freqs = dict(Monthly='ME', Quarterly='QE', Annual='YE')
        fig, axs = plt.subplots(1, len(freqs.keys()), figsize=figsize, tight_layout=True)
        figs['returns_pdfs'] = fig
        navs = multi_portfolio_data.get_navs(time_period=time_period, add_benchmarks_to_navs=False)
        for idx, (key, freq) in enumerate(freqs.items()):
            returns = qis.to_returns(prices=navs, freq=freq, drop_first=True)
            if len(returns.index) > 3:
                qis.plot_histogram(df=returns,
                                   xvar_format='{:.0%}',
                                   add_bar_at_peak=True,
                                   desc_table_type=qis.DescTableType.NONE,
                                   title=f"({qis.idx_to_alphabet(idx+1)}) {key} Returns",
                                   xlabel='return',
                                   ax=axs[idx])
            #if not add_titles:
            #    ax.title.set_visible(False)

        # outputs with risk model
        if risk_model is not None:

            # factor - level
            out_dict = risk_model.compute_active_factor_risk(portfolio_weights=multi_portfolio_data.portfolio_datas[strategy_idx].get_weights(),
                                                             benchmark_weights=multi_portfolio_data.portfolio_datas[benchmark_idx].get_weights())

            # strategy factor betas
            strategy_factor_betas = out_dict['portfolio_exposures']
            # strategy_factor_betas = risk_model.compute_agg_factor_exposures(weights=multi_portfolio_data.portfolio_datas[strategy_idx].get_weights())
            fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
            figs['strategy_factor_betas'] = fig
            qis.plot_time_series(df=strategy_factor_betas,
                                 title=f"{strategy_ticker} Factor Beta Exposures",
                                 var_format='{:,.2f}',
                                 ax=ax)
            if regime_benchmark is not None:
                multi_portfolio_data.add_regime_shadows(ax=ax, regime_benchmark=regime_benchmark,
                                                        index=strategy_factor_betas.index, regime_classifier=regime_classifier)
            # benchmark factor betas
            benchmark_factor_betas = out_dict['benchmark_exposures']
            # benchmark_factor_betas = risk_model.compute_agg_factor_exposures(weights=multi_portfolio_data.portfolio_datas[benchmark_idx].get_weights())
            fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
            figs['benchmark_factor_betas'] = fig
            qis.plot_time_series(df=benchmark_factor_betas,
                                 title=f"{benchmark_ticker} Factor Beta Exposures",
                                 var_format='{:,.2f}',
                                 ax=ax)
            if regime_benchmark is not None:
                multi_portfolio_data.add_regime_shadows(ax=ax, regime_benchmark=regime_benchmark,
                                                        index=benchmark_factor_betas.index, regime_classifier=regime_classifier)

            # active exposure
            fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
            figs['active_exposure'] = fig
            qis.plot_stack(df=out_dict['active_exposures'],
                           legend_stats=qis.LegendStats.AVG_NONNAN_LAST,
                           title=f"{strategy_ticker} vs {benchmark_ticker} active exposure",
                           var_format='{:,.2f}',
                           ax=ax)
            # active factor risk
            fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
            figs['factor_risk_contributions_rc'] = fig
            qis.plot_stack(df=out_dict['factor_risk_contributions_rc'],
                           legend_stats=qis.LegendStats.AVG_NONNAN_LAST,
                           title=f"{strategy_ticker} vs {benchmark_ticker} active risk contribution %",
                           var_format='{:,.2%}',
                           ax=ax)

            # strategy attribution
            portfolio_returns = qis.to_returns(prices=multi_portfolio_data.portfolio_datas[strategy_idx].get_portfolio_nav().reindex(
                index=strategy_factor_betas.index).ffill(), is_first_zero=True)
            attributions = qis.compute_benchmarks_beta_attribution_from_returns(portfolio_returns=portfolio_returns,
                                                                                benchmark_returns=risk_model.x,
                                                                                portfolio_benchmark_betas=strategy_factor_betas,
                                                                                total_name='Total')
            fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
            figs['strategy_factor_attributions'] = fig
            qis.plot_time_series(df=attributions.cumsum(axis=0),
                                 title=f"{strategy_ticker} Factor Attribution",
                                 legend_stats=qis.LegendStats.LAST,
                                 var_format='{:,.1%}',
                                 ax=ax)
            if regime_benchmark is not None:
                multi_portfolio_data.add_regime_shadows(ax=ax, regime_benchmark=regime_benchmark,
                                                        index=attributions.index, regime_classifier=regime_classifier)

            # benchmark attribution
            portfolio_returns = qis.to_returns(prices=multi_portfolio_data.portfolio_datas[benchmark_idx].get_portfolio_nav().reindex(
                index=benchmark_factor_betas.index).ffill(), is_first_zero=True)
            attributions = qis.compute_benchmarks_beta_attribution_from_returns(portfolio_returns=portfolio_returns,
                                                                                benchmark_returns=risk_model.x,
                                                                                portfolio_benchmark_betas=benchmark_factor_betas,
                                                                                total_name='Total')
            fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
            figs['benchmark_factor_attributions'] = fig
            qis.plot_time_series(df=attributions.cumsum(axis=0),
                                 title=f"{benchmark_ticker} Factor Attribution",
                                 legend_stats=qis.LegendStats.LAST,
                                 var_format='{:,.1%}',
                                 ax=ax)
            if regime_benchmark is not None:
                multi_portfolio_data.add_regime_shadows(ax=ax, regime_benchmark=regime_benchmark,
                                                        index=attributions.index, regime_classifier=regime_classifier)

            # strategy risk attribution
            factor_rcs_ratios, strategy_factor_risk_contrib_idio, factor_risk_contrib, strategy_portfolio_var = \
                risk_model.compute_factor_risk_contribution(weights=multi_portfolio_data.portfolio_datas[strategy_idx].get_weights())
            fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
            figs['strategy_factor_risk_cb'] = fig
            qis.plot_stack(df=strategy_factor_risk_contrib_idio,
                           use_bar_plot=True,
                           title=f"{strategy_ticker} relative factor risk contribution",
                           var_format='{:,.2%}',
                           ax=ax)

            # benchmark attribution
            factor_rcs_ratios, benchmark_factor_risk_contrib_idio, factor_risk_contrib, benchmark_portfolio_var = \
                risk_model.compute_factor_risk_contribution(weights=multi_portfolio_data.portfolio_datas[benchmark_idx].get_weights())
            fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
            figs['benchmark_factor_risk_cb'] = fig
            qis.plot_stack(df=benchmark_factor_risk_contrib_idio,
                           use_bar_plot=True,
                           title=f"{benchmark_ticker} relative factor risk contribution",
                           var_format='{:,.2%}',
                           ax=ax)

            # joint risk contribs
            kwargs = qis.update_kwargs(kwargs, dict(strategy_ticker=f"(B) {strategy_ticker}",
                                                    benchmark_ticker=f"(A) {benchmark_ticker}"))
            fig, axs = plt.subplots(1, 2, figsize=figsize, tight_layout=True)
            if add_titles:
                qis.set_suptitle(fig, title=f"Relative risk contributions")
            figs['strategy_benchmark_risk_contributions'] = fig
            plot_exposures_strategy_vs_benchmark_stack(strategy_exposures=strategy_factor_risk_contrib_idio,
                                                       benchmark_exposures=benchmark_factor_risk_contrib_idio,
                                                       axs=axs,
                                                       var_format=var_format,
                                                       **kwargs)

            # portfolio vars
            fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
            figs['strategy_portfolio_vars'] = fig
            qis.plot_time_series(df=np.sqrt(strategy_portfolio_var),
                                 title=f"{strategy_ticker} Portfolio sqrt(Vars)",
                                 var_format='{:,.2%}',
                                 ax=ax)
            if regime_benchmark is not None:
                multi_portfolio_data.add_regime_shadows(ax=ax, regime_benchmark=regime_benchmark,
                                                        index=strategy_portfolio_var.index, regime_classifier=regime_classifier)

            fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
            figs['benchmark_portfolio_vars'] = fig
            qis.plot_time_series(df=np.sqrt(benchmark_portfolio_var),
                                 title=f"{benchmark_ticker} Portfolio sqrt(Vars)",
                                 var_format='{:,.2%}',
                                 ax=ax)
            if regime_benchmark is not None:
                multi_portfolio_data.add_regime_shadows(ax=ax, regime_benchmark=regime_benchmark,
                                                        index=benchmark_portfolio_var.index, regime_classifier=regime_classifier)

    return figs, dfs


def plot_exposures_long_short_groups(exposures_short: pd.DataFrame,
                                     exposures_long: pd.DataFrame,
                                     axs: List[plt.Subplot],
                                     ylabel: str = 'weights',
                                     var_format: str = '{:.1%}',
                                     hue_var_name: str = 'asset class',
                                     **kwargs
                                     ) -> None:
    qis.plot_stack(df=exposures_short,
                   use_bar_plot=True,
                   legend_stats=qis.LegendStats.AVG_NONNAN_LAST,
                   var_format=var_format,
                   colors=qis.get_n_sns_colors(n=len(exposures_short.columns)),
                   ax=axs[0],
                   **qis.update_kwargs(kwargs, dict(bbox_to_anchor=(0.5, 1.01), ncols=1,
                                                    framealpha=0.9)))

    qis.df_boxplot_by_columns(df=exposures_long,
                              hue_var_name=hue_var_name,
                              y_var_name=ylabel,
                              ylabel=ylabel,
                              showmedians=True,
                              add_y_median_labels=False,
                              yvar_format=var_format,
                              x_rotation=90,
                              colors=qis.get_n_sns_colors(n=len(exposures_long.columns)),
                              y_limits=(0.0, None),
                              ax=axs[1],
                              **kwargs)


def plot_exposures_strategy_vs_benchmark_stack(strategy_exposures: pd.DataFrame,
                                               benchmark_exposures: pd.DataFrame,
                                               axs: List[plt.Subplot],
                                               var_format: str = '{:.1%}',
                                               strategy_ticker: str = 'TAA',
                                               benchmark_ticker: str = 'SAA',
                                               **kwargs
                                               ) -> None:
    qis.plot_stack(df=benchmark_exposures,
                   use_bar_plot=True,
                   legend_stats=qis.LegendStats.AVG_NONNAN_LAST,
                   var_format=var_format,
                   colors=qis.get_n_sns_colors(n=len(benchmark_exposures.columns)),
                   title=benchmark_ticker,
                   ax=axs[0],
                   **qis.update_kwargs(kwargs, dict(bbox_to_anchor=(0.5, 1.01), ncols=1,
                                                    framealpha=0.9)))
    qis.plot_stack(df=strategy_exposures,
                   use_bar_plot=True,
                   legend_stats=qis.LegendStats.AVG_NONNAN_LAST,
                   var_format=var_format,
                   colors=qis.get_n_sns_colors(n=len(strategy_exposures.columns)),
                   title=strategy_ticker,
                   ax=axs[1],
                   **qis.update_kwargs(kwargs, dict(bbox_to_anchor=(0.5, 1.01), ncols=1,
                                                    framealpha=0.9)))


def plot_exposures_strategy_vs_benchmark_boxplot(strategy_exposures: pd.DataFrame,
                                                 benchmark_exposures: pd.DataFrame,
                                                 ax: plt.Subplot,
                                                 ylabel: str = 'weights',
                                                 var_format: str = '{:.1%}',
                                                 hue_var_name: str = 'asset class',
                                                 strategy_ticker: str = 'TAA',
                                                 benchmark_ticker: str = 'SAA',
                                                 allow_negative: bool = False,
                                                 title: str = '',
                                                 **kwargs
                                                 ) -> None:
    dfs = {benchmark_ticker: benchmark_exposures, strategy_ticker: strategy_exposures}
    if allow_negative:
        y_limits = None
    else:
        y_limits = (0.0, None)
    qis.df_dict_boxplot_by_columns(dfs=dfs,
                                   hue_var_name=hue_var_name,
                                   y_var_name=ylabel,
                                   ylabel=ylabel,
                                   showmedians=True,
                                   add_y_median_labels=False,
                                   yvar_format=var_format,
                                   x_rotation=90,
                                   title=title,
                                   # colors=qis.get_n_sns_colors(n=len(exposures_long.columns)),
                                   y_limits=y_limits,
                                   ax=ax,
                                   **kwargs)

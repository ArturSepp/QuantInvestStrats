"""
strategy against benchmark by asset class: named landscape figures keyed by panel, not pages.

``weights_tracking_error_report_by_ac_subac`` takes a ``MultiPortfolioData``, the pair
``strategy_idx`` and ``benchmark_idx``, and three independent groupings - ``ac_group_data``,
``sub_ac_group_data`` and ``turnover_groups``, the last driving the turnover panels alone. It
returns a dict of figures and a dict of frames, keyed by panel name; the two key sets overlap
but are not the same, so read each on its own keys rather than assuming a figure has a frame.

Weights are read as input weights and risk contributions are normalised. Tracking error needs
``multi_portfolio_data.covar_dict`` and raises without it; the ex-ante volatility and
risk-contribution panels fall back to the covariance each member portfolio carries itself.
``risk_model`` adds the factor exposure, attribution and risk-contribution figures.

The same pair as a paginated A4 factsheet is ``strategy_benchmark_factsheet.py``.
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
from qis.portfolio.risk.risk_model import RiskModel
from qis.portfolio.reports.config import PERF_PARAMS
from qis.plots.derived.perf_table import get_ra_perf_benchmark_columns
from qis.plots.utils import get_n_sns_colors
from qis.utils.df_str import idx_to_alphabet


def _compute_ex_post_benchmark_series(
        strategy_nav: pd.Series,
        benchmark_nav: pd.Series,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute the 36-month realised TRE, benchmark beta, and annualised alpha series."""
    realised_tre = qis.compute_ewma_realised_tracking_error(
        portfolio_nav=strategy_nav,
        benchmark_nav=benchmark_nav,
    ).rename('Realised TRE (EWMA 36m)')
    navs = pd.concat(
        [strategy_nav.rename('strategy'), benchmark_nav.rename('benchmark')],
        axis=1,
        sort=True,
    ).dropna(how='all').ffill()
    monthly_returns = qis.to_returns(
        prices=navs,
        freq='ME',
        is_log_returns=False,
        drop_first=True,
    )
    beta, alpha, _, _, _, _ = qis.compute_ewm_beta_alpha_forecast(
        x_data=monthly_returns['benchmark'],
        y_data=monthly_returns[['strategy']],
        span=36,
        init_type=qis.InitType.X0,
        beta_init_value=1.0,
    )
    ex_post_beta = beta.iloc[:, 0].rename('Ex-post beta (EWMA 36m)')
    ex_post_alpha = (
        alpha.iloc[:, 0] * qis.get_annualization_factor('ME')
    ).rename('Ex-post alpha (EWMA 36m, annualised)')
    return realised_tre, ex_post_beta, ex_post_alpha


def _compute_ex_ante_benchmark_beta(
        risk_model: RiskModel,
        benchmark_weights: pd.DataFrame,
        portfolio_weights: pd.DataFrame,
        time_period: Optional[TimePeriod],
) -> pd.Series:
    """Compute beta only on covariance dates included in the report period."""
    if time_period is not None:
        date_marker = pd.Series(index=risk_model.dates, dtype=float)
        selected_dates = time_period.locate(date_marker).index
        if len(selected_dates) == 0:
            return pd.Series(index=selected_dates, name='Benchmark beta', dtype=float)
        risk_model = RiskModel(
            covar={date: risk_model.covar[date] for date in selected_dates},
        )
    return risk_model.compute_benchmark_beta_history(
        benchmark_weights=benchmark_weights,
        portfolio_weights=portfolio_weights,
        strict=False,
    )


def _add_ex_post_benchmark_panels(
        figs: Dict[str, plt.Figure],
        dfs: Dict[str, pd.DataFrame],
        multi_portfolio_data: MultiPortfolioData,
        regime_benchmark: str,
        regime_classifier: BenchmarkReturnsQuantilesRegime,
        realised_tre: pd.Series,
        ex_post_beta: pd.Series,
        ex_post_alpha: pd.Series,
        ex_ante_tre: Optional[pd.Series],
        ex_ante_beta: Optional[pd.Series],
        figsize: Tuple[float, float],
        add_titles: bool,
        **kwargs,
) -> None:
    """Append the additive ex-post report panels and their numeric frames."""
    tre_series = [realised_tre]
    if ex_ante_tre is not None:
        tre_series.insert(0, ex_ante_tre.rename('Ex-ante TRE'))
    tre_frame = pd.concat(tre_series, axis=1, sort=True)
    dfs['tre_ex_ante_vs_ex_post'] = tre_frame
    fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
    figs['tre_ex_ante_vs_ex_post'] = fig
    qis.plot_time_series(
        df=tre_frame,
        title='Ex-ante vs realised tracking error' if add_titles else None,
        var_format='{:.2%}',
        y_limits=(0.0, None),
        ax=ax,
        **kwargs,
    )
    multi_portfolio_data.add_regime_shadows(
        ax=ax,
        regime_benchmark=regime_benchmark,
        index=tre_frame.index,
        regime_classifier=regime_classifier,
    )

    if ex_ante_beta is not None:
        beta_frame = pd.concat(
            [ex_ante_beta.rename('Ex-ante beta'), ex_post_beta],
            axis=1,
            sort=True,
        )
        dfs['benchmark_beta_time_series'] = beta_frame
        fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
        figs['benchmark_beta_time_series'] = fig
        qis.plot_time_series(
            df=beta_frame,
            title='Ex-ante vs ex-post benchmark beta' if add_titles else None,
            var_format='{:,.2f}',
            ax=ax,
            **kwargs,
        )
        multi_portfolio_data.add_regime_shadows(
            ax=ax,
            regime_benchmark=regime_benchmark,
            index=beta_frame.index,
            regime_classifier=regime_classifier,
        )

    alpha_frame = ex_post_alpha.to_frame()
    dfs['ex_post_alpha_time_series'] = alpha_frame
    fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
    figs['ex_post_alpha_time_series'] = fig
    qis.plot_time_series(
        df=alpha_frame,
        title='Annualised ex-post alpha (EWMA 36m)' if add_titles else None,
        var_format='{:.2%}',
        ax=ax,
        **kwargs,
    )
    multi_portfolio_data.add_regime_shadows(
        ax=ax,
        regime_benchmark=regime_benchmark,
        index=alpha_frame.index,
        regime_classifier=regime_classifier,
    )


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
                                              covar_risk_model: Optional[RiskModel] = None,
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
    """Build weights and tracking-error panels for a strategy and benchmark.

    ``risk_model`` and ``covar_risk_model`` are deliberately distinct. The former is the
    returns-based ``LinearModel`` used by the legacy factor-attribution panels. The latter is
    the weights-and-covariance ``RiskModel`` used for ex-ante tracking error, factor exposures,
    and systematic/residual tracking-error decomposition.

    Args:
        multi_portfolio_data: Strategy and benchmark portfolios with shared report data.
        strategy_idx: Index of the strategy portfolio.
        benchmark_idx: Index of the benchmark portfolio.
        ac_group_data: Asset-class label per instrument.
        ac_group_order: Asset-class display order.
        sub_ac_group_data: Sub-asset-class label per instrument.
        sub_ac_group_order: Sub-asset-class display order.
        turnover_groups: Group labels used only by the turnover panels.
        turnover_order: Turnover-group display order.
        risk_model: Optional returns-based ``LinearModel`` for legacy attribution panels.
        covar_risk_model: Optional weights-and-covariance ``RiskModel``. Its covariance is used
            when ``multi_portfolio_data.covar_dict`` is absent; a complete factor block adds
            tracking-error decomposition and strategy factor-exposure panels.
        time_period: Optional reporting period.
        perf_params: Performance-statistic configuration.
        regime_classifier: Regime definition used for plot backgrounds.
        add_benchmarks_to_navs: Whether to include benchmark NAVs in the performance panel.
        tre_max_clip: Optional upper clip for grouped tracking-error lines.
        figsize: Figure size for each panel.
        var_format: Weight and contribution formatting string.
        add_titles: Whether to draw panel titles.
        **kwargs: Additional plotting arguments.

    Returns:
        Figure and DataFrame dictionaries keyed by panel name.
    """
    regime_benchmark = multi_portfolio_data.benchmark_prices.columns[0]
    benchmark_price = multi_portfolio_data.benchmark_prices[regime_benchmark]

    figs: Dict[str, plt.Figure] = {}
    dfs: Dict[str, pd.DataFrame] = {}
    report_covar_dict = multi_portfolio_data.covar_dict
    if report_covar_dict is None and covar_risk_model is not None:
        report_covar_dict = covar_risk_model.covar

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
        # Keep a full-precision frame for callers without drawing it over the formatted table.
        prices = multi_portfolio_data.get_navs(
            benchmark=regime_benchmark,
            add_benchmarks_to_navs=add_benchmarks_to_navs,
            time_period=time_period,
        )
        turnover = multi_portfolio_data.get_turnover(
            time_period=time_period,
            **kwargs,
        ).mean(axis=0).to_frame('Turnover')
        ra_perf_table = get_ra_perf_benchmark_columns(
            prices=prices,
            benchmark=regime_benchmark,
            benchmark_price=benchmark_price,
            perf_params=perf_params,
            is_convert_to_str=False,
            df_to_add=qis.df_to_str(turnover, var_format='{:,.0%}'),
            **kwargs,
        )
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

        strategy_nav = strategy_data.get_portfolio_nav(time_period=time_period)
        benchmark_nav = benchmark_data.get_portfolio_nav(time_period=time_period)
        realised_tre, ex_post_beta, ex_post_alpha = _compute_ex_post_benchmark_series(
            strategy_nav=strategy_nav,
            benchmark_nav=benchmark_nav,
        )
        if report_covar_dict is None:
            _add_ex_post_benchmark_panels(
                figs=figs,
                dfs=dfs,
                multi_portfolio_data=multi_portfolio_data,
                regime_benchmark=regime_benchmark,
                regime_classifier=regime_classifier,
                realised_tre=realised_tre,
                ex_post_beta=ex_post_beta,
                ex_post_alpha=ex_post_alpha,
                ex_ante_tre=None,
                ex_ante_beta=None,
                figsize=figsize,
                add_titles=add_titles,
                **kwargs,
            )
            return figs, dfs

        # portfolio vol
        strategy_ex_anti_vol = strategy_data.compute_ex_anti_portfolio_vol_implied_by_covar(
            covar_dict=report_covar_dict)
        benchmark_ex_anti_vol = benchmark_data.compute_ex_anti_portfolio_vol_implied_by_covar(
            covar_dict=report_covar_dict)

        ex_anti_vols = pd.concat([strategy_ex_anti_vol, benchmark_ex_anti_vol], axis=1, sort=True)
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
        rc_kwargs = dict(covar_dict=report_covar_dict, freq='QE', normalise=True,
                         time_period=time_period)
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
        use_covar_risk_model = (
            multi_portfolio_data.covar_dict is None and covar_risk_model is not None)
        if use_covar_risk_model:
            strategy_input_weights = strategy_data.get_weights(freq=None, is_input_weights=True)
            benchmark_input_weights = benchmark_data.get_weights(freq=None, is_input_weights=True)
            total_tre = covar_risk_model.compute_tre_history(
                benchmark_weights=benchmark_input_weights,
                portfolio_weights=strategy_input_weights,
                strict=False)
            if time_period is not None:
                total_tre = time_period.locate(total_tre)
            qis.plot_time_series(df=total_tre,
                                 var_format='{:.2%}',
                                 legend_stats=qis.LegendStats.AVG_NONNAN_LAST,
                                 title=title,
                                 y_limits=(0.0, None),
                                 ax=ax,
                                 **kwargs)
            if regime_benchmark is not None:
                multi_portfolio_data.add_regime_shadows(
                    ax=ax,
                    regime_benchmark=regime_benchmark,
                    index=total_tre.index,
                    regime_classifier=regime_classifier)
        else:
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
        if use_covar_risk_model:
            grouped_tre = covar_risk_model.compute_tre_history(
                benchmark_weights=benchmark_input_weights,
                portfolio_weights=strategy_input_weights,
                group_data=ac_group_data,
                total_column='Total',
                strict=False)
            if tre_max_clip is not None:
                group_columns = grouped_tre.columns.drop('Total', errors='ignore')
                grouped_tre[group_columns] = grouped_tre[group_columns].clip(upper=tre_max_clip)
            grouped_tre_to_plot = grouped_tre
            if time_period is not None:
                grouped_tre_to_plot = time_period.locate(grouped_tre_to_plot)
            qis.plot_time_series(df=grouped_tre_to_plot,
                                 var_format='{:.2%}',
                                 legend_stats=qis.LegendStats.AVG_NONNAN_LAST,
                                 title=title,
                                 y_limits=(0.0, None),
                                 ax=ax,
                                 **kwargs)
            if regime_benchmark is not None:
                multi_portfolio_data.add_regime_shadows(
                    ax=ax,
                    regime_benchmark=regime_benchmark,
                    index=grouped_tre_to_plot.index,
                    regime_classifier=regime_classifier)
            dfs['ac_tracking_error'] = grouped_tre
        else:
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
            dfs['ac_tracking_error'] = (
                multi_portfolio_data.compute_tracking_error_implied_by_covar(
                    strategy_idx=strategy_idx,
                    benchmark_idx=benchmark_idx,
                    is_grouped=True,
                    group_data=ac_group_data,
                    group_order=ac_group_order,
                    total_column='Total'))

        ex_ante_tre = dfs['ac_tracking_error']['Total']
        if time_period is not None:
            ex_ante_tre = time_period.locate(ex_ante_tre)
        benchmark_beta_risk_model = (
            covar_risk_model if use_covar_risk_model else RiskModel(covar=report_covar_dict)
        )
        strategy_input_weights = strategy_data.get_weights(freq=None, is_input_weights=True)
        benchmark_input_weights = benchmark_data.get_weights(freq=None, is_input_weights=True)
        ex_ante_beta = _compute_ex_ante_benchmark_beta(
            risk_model=benchmark_beta_risk_model,
            benchmark_weights=benchmark_input_weights,
            portfolio_weights=strategy_input_weights,
            time_period=time_period,
        )
        _add_ex_post_benchmark_panels(
            figs=figs,
            dfs=dfs,
            multi_portfolio_data=multi_portfolio_data,
            regime_benchmark=regime_benchmark,
            regime_classifier=regime_classifier,
            realised_tre=realised_tre,
            ex_post_beta=ex_post_beta,
            ex_post_alpha=ex_post_alpha,
            ex_ante_tre=ex_ante_tre,
            ex_ante_beta=ex_ante_beta,
            figsize=figsize,
            add_titles=add_titles,
            **kwargs,
        )

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
                                   title=f"({idx_to_alphabet(idx+1)}) {key} Returns",
                                   xlabel='return',
                                   ax=axs[idx])
            #if not add_titles:
            #    ax.title.set_visible(False)

        # outputs with the weights-and-covariance risk model
        has_complete_factor_block = (
            covar_risk_model is not None
            and covar_risk_model.factor_loadings is not None
            and covar_risk_model.factor_covar is not None
            and covar_risk_model.residual_vars is not None)
        if has_complete_factor_block:
            strategy_input_weights = strategy_data.get_weights(freq=None, is_input_weights=True)
            benchmark_input_weights = benchmark_data.get_weights(freq=None, is_input_weights=True)
            tre_decomposition = covar_risk_model.compute_tre_decomposition_history(
                benchmark_weights=benchmark_input_weights,
                portfolio_weights=strategy_input_weights,
                strict=False)
            factor_exposures = covar_risk_model.compute_exposures_history(
                portfolio_weights=strategy_input_weights,
                strict=False)

            dfs['tre_decomposition'] = tre_decomposition
            fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
            figs['tre_decomposition'] = fig
            qis.plot_time_series(df=tre_decomposition,
                                 title='Tracking Error Decomposition' if add_titles else None,
                                 var_format='{:.2%}',
                                 ax=ax)

            dfs['factor_exposures'] = factor_exposures
            fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
            figs['factor_exposures'] = fig
            qis.plot_time_series(df=factor_exposures,
                                 title=(f'{strategy_ticker} Factor Exposures'
                                        if add_titles else None),
                                 var_format='{:,.2f}',
                                 ax=ax)

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


def plot_exposures_strategy_vs_benchmark_stack(strategy_exposures: pd.DataFrame,
                                               benchmark_exposures: pd.DataFrame,
                                               axs: List[plt.Subplot],
                                               var_format: str = '{:.1%}',
                                               strategy_ticker: str = 'TAA',
                                               benchmark_ticker: str = 'SAA',
                                               **kwargs
                                               ) -> None:
    """
    draw strategy and benchmark exposures as two stacked-area panels on a shared scale.

    Side by side rather than as a difference, because the active bet is easier to read against
    the allocation it departs from than as a signed residual. The tracking-error factsheet uses
    this as its allocation page.

    Args:
        strategy_exposures: strategy weights over time, one column per asset or group
        benchmark_exposures: benchmark weights on the same index and columns
        axs: the two axes to draw on, benchmark first
        var_format: format for the weights, a percentage by convention
        strategy_ticker: label for the strategy panel
        benchmark_ticker: label for the benchmark panel

    Returns:
        None; the supplied axes are drawn on
    """
    qis.plot_stack(df=benchmark_exposures,
                   use_bar_plot=True,
                   legend_stats=qis.LegendStats.AVG_NONNAN_LAST,
                   var_format=var_format,
                   colors=get_n_sns_colors(n=len(benchmark_exposures.columns)),
                   title=benchmark_ticker,
                   ax=axs[0],
                   **qis.update_kwargs(kwargs, dict(bbox_to_anchor=(0.5, 1.01), ncols=1,
                                                    framealpha=0.9)))
    qis.plot_stack(df=strategy_exposures,
                   use_bar_plot=True,
                   legend_stats=qis.LegendStats.AVG_NONNAN_LAST,
                   var_format=var_format,
                   colors=get_n_sns_colors(n=len(strategy_exposures.columns)),
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
                                   # colors=get_n_sns_colors(n=len(exposures_long.columns)),
                                   y_limits=y_limits,
                                   ax=ax,
                                   **kwargs)

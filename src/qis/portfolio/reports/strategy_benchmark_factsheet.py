"""
one strategy against one benchmark: a ``MultiPortfolioData`` in, a list of A4 pages out.

``generate_strategy_benchmark_factsheet_plt`` picks the pair by position, ``strategy_idx``
against ``benchmark_idx``, and returns one ``plt.Figure`` per page. Page one always renders the
joint performance, exposure and cost panels beside the risk-adjusted tables, regime Sharpes and
rolling annualised alpha computed from the adjacent panel's lagged EWMA betas.
``add_performance_inference_sheet`` and ``add_brinson_attribution`` are on by default:
paired Sharpe / marginal holdings-alpha inference precedes the allocation / selection page.
The Brinson call assigns the interaction term to instrument
selection. The other page flags are off; ``add_benchmarks_to_navs`` adds benchmark
lines to the page-one panels rather than a page, and ``add_grouped_exposures`` /
``add_grouped_cum_pnl``
apply only to the factsheet appended by ``add_strategy_factsheet``. Frequencies are per panel and
are meant to arrive spread in from ``qis.fetch_default_report_kwargs``.

Several structurally different strategies is ``multi_strategy_factsheet.py``; tracking error and
risk contributions on the same pair is ``strategy_benchmark_tre_factsheet.py``.
"""
import numpy as np
# packages
import pandas as pd

from qis.portfolio.attribution.group_exposures import resolve_reporting_inputs
from qis.utils._pandas_compat import infer_freq as _infer_freq
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import qis as qis
from qis import TimePeriod, PerfParams, BenchmarkReturnsQuantilesRegime

# portfolio
from qis.portfolio.portfolio_data import AttributionMetric
from qis.portfolio.multi_portfolio_data import MultiPortfolioData
from qis.perfstats.significance import InterceptInference
from qis.portfolio.reports._paired_performance_inference import compute_paired_performance_inference
from qis.portfolio.reports._performance_inference import plot_performance_inference
from qis.portfolio._asset_display import (
    factsheet_kwargs, price_tails, reduce_rows, selection_title, tail_index,
)
from qis.portfolio.reports.strategy_factsheet import (
    _use_grouped_summary_tables,
    generate_strategy_factsheet,
)
from qis.portfolio.reports.config import (
    DEFAULT_RECENT_RA_PERF_TABLE_START_DATE,
    PERF_PARAMS,
    _get_recent_ra_perf_table_time_period,
    infer_data_frequency_label,
    validate_legend_capacity,
    validate_reporting_frequency,
)
from qis.plots.utils import set_spines, set_y_limits

_RecentStart = Optional[pd.Timestamp]


def generate_strategy_benchmark_factsheet_plt(
    multi_portfolio_data: MultiPortfolioData,
    strategy_idx: int = 0,  # strategy is multi_portfolio_data[strategy_idx]
    benchmark_idx: int = 1,  # benchmark is multi_portfolio_data[benchmark_idx]
    time_period: TimePeriod = None,
    perf_params: PerfParams = PERF_PARAMS,
    regime_classifier: BenchmarkReturnsQuantilesRegime = BenchmarkReturnsQuantilesRegime(),
    backtest_name: str = None,
    add_benchmarks_to_navs: bool = False,
    add_brinson_attribution: bool = True,
    add_exposures_pnl_attribution: bool = False,
    add_strategy_factsheet: bool = False,
    add_grouped_exposures: bool = False,  # for strategy factsheet
    add_grouped_cum_pnl: bool = False,  # for strategy factsheet
    add_tracking_error_table: bool = False,
    add_exposures_comp: bool = False,
    is_grouped: Optional[bool] = True,
    figsize: Tuple[float, float] = (8.3, 11.7),  # A4 for portrait
    fontsize: int = 5,
    add_joint_instrument_history_report: bool = False,
    recent_ra_perf_table_start_date: _RecentStart = (DEFAULT_RECENT_RA_PERF_TABLE_START_DATE),
    brinson_time_period: Optional[TimePeriod] = None,
    brinson_group_data: Optional[pd.Series] = None,
    *,
    add_performance_inference_sheet: bool = True,
    alpha_inference: InterceptInference = InterceptInference.BARTLETT_HAC,
    group_data: Optional[pd.Series] = None,
    lookthrough_matrix: Optional[pd.DataFrame] = None,
    **kwargs,
) -> List[plt.Figure]:
    """
    factsheet comparing one strategy against one benchmark, as a list of A4 figures.

    The two-portfolio case, where the difference between them is the subject: the first page is
    the joint summary, followed by paired inference and Brinson attribution decomposing the
    active return into allocation and selection (including interaction). This is why this is a
    separate report from :func:`generate_multi_portfolio_factsheet`, which compares strategies that
    share no common weights.

    The Brinson page labels its realised-trading-cost basis: gross by default, or net when
    ``is_net=True`` is passed through ``kwargs``. This option does not attribute management
    fees, funding or other cash flows absent from instrument P&L.

    Args:
        alpha_inference: Primary marginal-alpha uncertainty method; defaults to Bartlett HAC.
            PREWHITENED_AR1 is opt-in; the alternate is shown on the same sample as unadjusted
            sensitivity only. Forwarded to embedded strategy pages; Sharpe stays unchanged.
        add_performance_inference_sheet: Default-on paired arithmetic-Sharpe and marginal
            holdings-alpha appendix before Brinson. Uses strategy_idx minus benchmark_idx;
            alpha uses only the first market benchmark, with shared freq_beta/factor_beta_span
            from kwargs (defaults B/260). Sharpe uses perf_params.freq_vol. Short/missing samples
            warn and remain unavailable. False omits inference here and in any appended
            standalone strategy factsheets, preserving the earlier kwargs forwarding behaviour.
            Page one is unchanged.
        lookthrough_matrix: Optional static instrument-by-sleeve fractions. Supplied rows
            are finite, nonnegative and sum to one; omitted rows use group_data.
            None preserves categorical reporting. This is an accounting split.
        group_data: Ordinary instrument labels as a Series; defaults to portfolio metadata
            where available. Fractional memberships belong in lookthrough_matrix.
        multi_portfolio_data: the portfolios, strategy and benchmark among them
        strategy_idx: position of the strategy within ``multi_portfolio_data``
        benchmark_idx: position of the benchmark within ``multi_portfolio_data``
        time_period: reporting window. None uses the full common history
        recent_ra_perf_table_start_date: start of the second risk-adjusted performance table.
            Defaults to 31 December 2020; None uses the trailing year
        perf_params: annualisation, frequency and rate conventions for the statistics
        regime_classifier: how benchmark returns are mapped to regimes
        backtest_name: title of the report
        add_benchmarks_to_navs: include the benchmarks as additional lines in the performance panels
        brinson_time_period: Independent attribution window. None uses time_period.
            The first shared NAV is its baseline; earlier returns are excluded.
        brinson_group_data: Legacy alias for report-wide group_data Series labels.
            Use lookthrough_matrix for fractional memberships.
        add_brinson_attribution: add the allocation / selection decomposition, with interaction
            assigned entirely to instrument selection
        add_exposures_pnl_attribution: add the exposure and P&L attribution pages
        add_strategy_factsheet: append the full single-strategy factsheet
        add_grouped_exposures: report exposures by group in that appended factsheet
        add_grouped_cum_pnl: report cumulative P&L by group in that appended factsheet
        add_tracking_error_table: add the per-instrument attribution and IR table; the keyword
            keeps its historical name for compatibility
        add_exposures_comp: add the strategy-versus-benchmark exposure comparison
        is_grouped: report annual returns by asset class rather than by instrument. None decides by
            the universe size, grouping once there are more than ten instruments. More than ten
            groups are omitted from the RA summary tables with a warning
        figsize: page size in inches; the default is A4 portrait
        fontsize: base font size
        add_joint_instrument_history_report: add the per-instrument history page
        **kwargs: plotting options, including ``asset_tail_count``: 10 (default) or 20
            assets per tail. Selection affects display only; totals retain every holding

    Returns:
        Pages in order, ready for :func:`save_figs_to_pdf`. Completed per-instrument exposure
        comparison pages are closed in pyplot but remain drawable and saveable. Other pages
        remain open for the caller to close after use; unrelated figures are left open.
    """
    if brinson_group_data is not None:
        if not isinstance(brinson_group_data, pd.Series):
            raise TypeError("brinson_group_data must be a Series; use lookthrough_matrix")
        if group_data is not None and not group_data.equals(brinson_group_data):
            raise ValueError("Pass group_data or legacy brinson_group_data, not conflicting labels")
        group_data = brinson_group_data
    group_data, lookthrough_matrix = resolve_reporting_inputs(
        multi_portfolio_data.portfolio_datas, group_data, lookthrough_matrix)
    brinson_group_data = group_data
    kwargs = factsheet_kwargs(kwargs)
    if 'heatmap_fontsize' in kwargs:
        raise TypeError("heatmap_fontsize was removed; use fontsize")
    if len(multi_portfolio_data.portfolio_datas) == 1:
        raise ValueError("must be at least two strategies")

    if is_grouped is None:
        if (
            len(
                multi_portfolio_data.portfolio_datas[0]
                .get_weights(lookthrough_matrix=lookthrough_matrix, group_data=group_data)
                .columns
            )
            >= 10
        ):  # otherwise tables look too bad
            is_grouped = True
        else:
            is_grouped = False
    else:
        if is_grouped is False:
            if (
                len(
                    multi_portfolio_data.portfolio_datas[0]
                    .get_weights(lookthrough_matrix=lookthrough_matrix, group_data=group_data)
                    .columns
                )
                >= 10
            ):  # tables look too bad
                is_grouped = True
    is_grouped_for_summary_tables = _use_grouped_summary_tables(
        portfolio_data=multi_portfolio_data.portfolio_datas[strategy_idx],
        is_grouped=is_grouped and not add_benchmarks_to_navs,
        panel_names="the RA performance panels",
        lookthrough_matrix=lookthrough_matrix,
        group_data=group_data,
    )

    # set reporting time period here
    if time_period is None:
        time_period = qis.get_time_period(
            multi_portfolio_data.portfolio_datas[0].get_portfolio_nav())

    # guard: the requested reporting frequency must not be finer than the data it is computed on -
    # check both portfolio NAVs and benchmark prices (used for regime / beta / scatter panels)
    strategy_nav = multi_portfolio_data.portfolio_datas[strategy_idx].get_portfolio_nav()
    for data_series in (strategy_nav,
                        multi_portfolio_data.portfolio_datas[benchmark_idx].get_portfolio_nav(),
                        multi_portfolio_data.benchmark_prices):
        validate_reporting_frequency(data_series, perf_params.freq)

    # guard: the left column carries one legend row per portfolio, and a legend that outgrows its
    # panel collapses the layout of the whole page - see config.estimate_legend_capacity
    n_legend_entries = len(multi_portfolio_data.portfolio_datas)
    if add_benchmarks_to_navs and multi_portfolio_data.benchmark_prices is not None:
        benchmarks = multi_portfolio_data.benchmark_prices
        n_legend_entries += 1 if isinstance(benchmarks, pd.Series) else len(benchmarks.columns)
    validate_legend_capacity(n_legend_entries=n_legend_entries, figsize=figsize, fontsize=fontsize,
                             panel_rows=2, gridspec_rows=14,
                             report_name='strategy-benchmark factsheet')

    # native grid of the NAV path: drawdowns / under-water / cumulative are computed on this grid
    # (not resampled to the reporting frequency), so their titles carry this frequency
    nav_freq = infer_data_frequency_label(strategy_nav)
    nav_freq_label = f" ({nav_freq}-freq)" if nav_freq else ""

    # set report specific kqargs
    plot_kwargs = dict(fontsize=fontsize,
                       linewidth=0.5,
                       digits_to_show=1, sharpe_digits=2,
                       weight='normal',
                       markersize=1,
                       framealpha=0.75,
                       time_period=time_period)
    kwargs = qis.update_kwargs(kwargs, plot_kwargs)

    figs = []
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    figs.append(fig)
    gs = fig.add_gridspec(nrows=14, ncols=4, wspace=0.0, hspace=0.0)

    if backtest_name is not None:
        fig.suptitle(backtest_name, fontweight="bold", fontsize=8, color='blue')

    regime_benchmark = multi_portfolio_data.benchmark_prices.columns[0]
    benchmark_price = multi_portfolio_data.benchmark_prices[regime_benchmark]

    multi_portfolio_data.plot_nav(ax=fig.add_subplot(gs[:2, :2]),
                                  add_benchmarks_to_navs=add_benchmarks_to_navs,
                                  regime_benchmark=regime_benchmark,
                                  perf_params=perf_params,
                                  regime_classifier=regime_classifier,
                                  title=(f"Cumulative performance ({perf_params.freq}-freq stats) "
                                         f"with bear/normal/bull regimes of {regime_benchmark} "
                                         f"{regime_classifier.freq}-returns"),
                                  **kwargs)

    multi_portfolio_data.plot_drawdowns(ax=fig.add_subplot(gs[2:4, :2]),
                                        add_benchmarks_to_navs=add_benchmarks_to_navs,
                                        regime_benchmark=regime_benchmark,
                                        regime_classifier=regime_classifier,
                                        title=f'Running Drawdowns{nav_freq_label}',
                                        **kwargs)

    multi_portfolio_data.plot_rolling_time_under_water(ax=fig.add_subplot(gs[4:6, :2]),
                                                       add_benchmarks_to_navs=add_benchmarks_to_navs,
                                                       regime_benchmark=regime_benchmark,
                                                       regime_classifier=regime_classifier,
                                                       title=(f'Rolling time under water'
                                                              f'{nav_freq_label}'),
                                                       **kwargs)

    multi_portfolio_data.plot_rolling_perf(ax=fig.add_subplot(gs[6:8, :2]),
                                           add_benchmarks_to_navs=add_benchmarks_to_navs,
                                           regime_benchmark=regime_benchmark,
                                           regime_classifier=regime_classifier,
                                           **kwargs)

    exposure_kwargs = kwargs
    if perf_params.freq == 'QE':
        exposure_kwargs = qis.update_kwargs(kwargs, dict(weights_freq='ME'))
    multi_portfolio_data.plot_exposures(ax=fig.add_subplot(gs[8:10, :2]),
                                        regime_benchmark=regime_benchmark,
                                        regime_classifier=regime_classifier,
                                        **exposure_kwargs)

    multi_portfolio_data.plot_turnover(ax=fig.add_subplot(gs[10:12, :2]),
                                       regime_benchmark=regime_benchmark,
                                       regime_classifier=regime_classifier,
                                       **kwargs)

    multi_portfolio_data.plot_costs(ax=fig.add_subplot(gs[12:14, :2]),
                                    regime_benchmark=regime_benchmark,
                                    regime_classifier=regime_classifier,
                                    **kwargs)

    multi_portfolio_data.plot_ac_ra_perf_table(
        ax=fig.add_subplot(gs[0:2, 2:]),
        benchmark_price=benchmark_price,
        add_benchmarks_to_navs=add_benchmarks_to_navs,
        perf_params=perf_params,
        is_grouped=is_grouped_for_summary_tables,
        **qis.update_kwargs(kwargs, dict(fontsize=fontsize)),
        lookthrough_matrix=lookthrough_matrix,
        group_data=group_data,
    )

    time_period1 = _get_recent_ra_perf_table_time_period(
        time_period=time_period,
        recent_ra_perf_table_start_date=recent_ra_perf_table_start_date,
    )
    # change regression to weekly
    if _infer_freq(benchmark_price.index) in ['B', 'D']:
        local_kwargs = qis.update_kwargs(
            kwargs, dict(time_period=time_period1, freq_reg='W-WED', fontsize=fontsize))
    else:
        local_kwargs = qis.update_kwargs(kwargs, dict(time_period=time_period1, fontsize=fontsize))

    multi_portfolio_data.plot_ac_ra_perf_table(
        ax=fig.add_subplot(gs[2:4, 2:]),
        benchmark_price=benchmark_price,
        add_benchmarks_to_navs=add_benchmarks_to_navs,
        perf_params=perf_params,
        is_grouped=is_grouped_for_summary_tables,
        **local_kwargs,
        lookthrough_matrix=lookthrough_matrix,
        group_data=group_data,
    )

    # periodic returns
    local_kwargs = qis.update_kwargs(kwargs=kwargs,
                                     new_kwargs=dict(fontsize=fontsize, square=False,
                                                     x_rotation=90, transpose=True))

    # periodic returns
    multi_portfolio_data.plot_periodic_returns(add_benchmarks_to_navs=add_benchmarks_to_navs,
                                                                  ax=fig.add_subplot(gs[4:6, 2:4]),
                                                                  **local_kwargs)

    # regime data for 1 or two benchmarks
    if len(multi_portfolio_data.benchmark_prices.columns) == 1:
        benchmark_prices_ = {benchmark_price.name: fig.add_subplot(gs[6:8, 2:4])}
    else:
        benchmark_prices_ = {
            multi_portfolio_data.benchmark_prices.columns[0]: fig.add_subplot(gs[6:8, 2]),
            multi_portfolio_data.benchmark_prices.columns[1]: fig.add_subplot(gs[6:8, 3])}
    for benchmark_, ax_ in benchmark_prices_.items():
        post_title = (f"Sharpe in {benchmark_} Bear/Normal/Bull "
                      f"{regime_classifier.freq}-freq regimes")
        multi_portfolio_data.plot_regime_data(
            benchmark=benchmark_,
            add_benchmarks_to_navs=add_benchmarks_to_navs,
            is_grouped=False,
            title=f"{post_title}",
            perf_params=perf_params,
            regime_classifier=regime_classifier,
            ax=ax_,
            **qis.update_kwargs(kwargs, dict(fontsize=fontsize, x_rotation=90)),
            lookthrough_matrix=lookthrough_matrix,
            group_data=group_data,
        )

    # vol regimes
    """
    multi_portfolio_data.portfolio_datas[0].plot_vol_regimes(ax=fig.add_subplot(gs[8:10, 2]),
                                                             benchmark_price=benchmark_price,
                                                             perf_params=perf_params,
                                                             freq=regime_classifier.freq,
                        **qis.update_kwargs(kwargs, dict(fontsize=fontsize, x_rotation=90)))
    multi_portfolio_data.portfolio_datas[1].plot_vol_regimes(ax=fig.add_subplot(gs[8:10, 3]),
                                                             benchmark_price=benchmark_price,
                                                             freq=perf_params.freq,
                                                             regime_classifier=regime_classifier,
                        **qis.update_kwargs(kwargs, dict(fontsize=fontsize, x_rotation=90)))
    """
    multi_portfolio_data.plot_rolling_benchmark_alpha(
        ax=fig.add_subplot(gs[8:10, 2:]),
        benchmark_price=benchmark_price,
        regime_benchmark=regime_benchmark,
        regime_classifier=regime_classifier,
        **kwargs,
    )

    # plot beta to the regime_benchmark
    multi_portfolio_data.plot_factor_betas(axs=[fig.add_subplot(gs[10:12, 2:])],
                                           benchmark_prices=multi_portfolio_data.benchmark_prices[regime_benchmark].to_frame(),
                                           regime_benchmark=regime_benchmark,
                                           regime_classifier=regime_classifier,
                                           **kwargs)
    with sns.axes_style('whitegrid'):
        multi_portfolio_data.plot_returns_scatter(ax=fig.add_subplot(gs[12:, 2:]),
                                                  benchmark=regime_benchmark,
                    **qis.update_kwargs(kwargs, dict(freq=perf_params.freq_reg)))

    if add_performance_inference_sheet:
        inference = compute_paired_performance_inference(
            strategy=multi_portfolio_data.portfolio_datas[strategy_idx],
            comparison=multi_portfolio_data.portfolio_datas[benchmark_idx],
            benchmark_prices=multi_portfolio_data.benchmark_prices,
            time_period=time_period, freq=perf_params.freq_vol, rates_data=perf_params.rates_data,
            freq_beta=kwargs.get('freq_beta', 'B'),
            factor_beta_span=kwargs.get('factor_beta_span', 260),
            alpha_inference=alpha_inference,
        )
        figs.append(plot_performance_inference(
            inference, f"{inference.metadata['strategy']} vs {inference.metadata['comparison']}",
            figsize=figsize))

    if add_brinson_attribution:
        brinson_period = time_period if brinson_time_period is None else brinson_time_period
        brinson_kwargs = qis.update_kwargs(kwargs, dict(time_period=brinson_period))
        if lookthrough_matrix is not None:
            brinson_kwargs.setdefault('col_widths', [4.5] + [2.0] * 7)
            brinson_kwargs.setdefault('var_format', '{:.1%}')
        shared = pd.concat([
            multi_portfolio_data.portfolio_datas[strategy_idx].nav,
            multi_portfolio_data.portfolio_datas[benchmark_idx].nav,
        ], axis=1, join='inner', sort=True).dropna()
        shared = brinson_period.locate(shared)
        if len(shared) < 2:
            raise ValueError('Brinson requires at least two shared NAV observations')
        period_label = f'{shared.index[0]:%d%b%Y} - {shared.index[-1]:%d%b%Y}'
        with sns.axes_style("darkgrid"):
            fig1 = plt.figure(figsize=figsize, constrained_layout=True)
            this_name = backtest_name if backtest_name is not None else ''
            cost_basis = 'Net' if kwargs.get('is_net', False) else 'Gross'
            fig1.suptitle(
                f'{this_name} Brinson performance attribution report: {period_label}\n'
                f'{cost_basis} of realised trading costs. '
                'Interaction returns added 100% to instrument selection',
                fontweight='bold', fontsize=8, color='blue')
            if lookthrough_matrix is not None:
                # Reserve a stable heading band across both PDF and raster draws.
                fig1.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.94))
                fig1._suptitle.set_position((0.5, 0.99))
                fig1._suptitle.set_in_layout(False)
            figs.append(fig1)
            gs = fig1.add_gridspec(nrows=3, ncols=2, wspace=0.0, hspace=0.0)
            large_partition = (lookthrough_matrix is not None
                               and len(lookthrough_matrix.columns) > 6)
            table_spec = gs[:2, 0] if large_partition else gs[0, 0]
            active_spec = gs[2, 0] if large_partition else gs[1, 0]
            axs = [fig1.add_subplot(table_spec), fig1.add_subplot(active_spec),
                   fig1.add_subplot(gs[0, 1]), fig1.add_subplot(gs[1, 1]),
                   fig1.add_subplot(gs[2, 1])]
            multi_portfolio_data.plot_brinson_attribution(
                strategy_idx=strategy_idx,
                benchmark_idx=benchmark_idx,
                freq=None,
                axs=axs,
                total_column="Total Sum",
                is_exclude_interaction_term=True,
                group_data=brinson_group_data,
                **brinson_kwargs,
                lookthrough_matrix=lookthrough_matrix,
            )
            brinson_index = shared.index
            for ax in axs[1:]:
                multi_portfolio_data.add_regime_shadows(
                    ax=ax,
                    regime_benchmark=regime_benchmark,
                    index=brinson_index,
                    regime_classifier=regime_classifier)
        # Exposure comparison uses the same attribution period and classification.
        exposure_ax = None if large_partition else fig1.add_subplot(gs[2, 0])
        if large_partition:
            pass  # The separate look-through factsheet carries the exposure bars.
        elif brinson_group_data is not None or lookthrough_matrix is not None:
            from qis.portfolio.reports.lookthrough_exposures import _exposure_tables
            pair = MultiPortfolioData([
                multi_portfolio_data.portfolio_datas[strategy_idx],
                multi_portfolio_data.portfolio_datas[benchmark_idx]])
            average, _, _ = _exposure_tables(
                pair,
                time_period=brinson_period,
                group_data=group_data,
                lookthrough_matrix=lookthrough_matrix,
            )
            difference = average.iloc[:, 0] - average.iloc[:, 1]
            qis.plot_bars(difference, stacked=False, is_horizontal=True,
                          annotate_totals=False, is_sns=False, fontsize=fontsize,
                          xvar_format='{:.0%}', yvar_format='{:.0%}',
                          title='Average look-through active weights',
                          legend_loc=None, ax=exposure_ax)
        else:
            multi_portfolio_data.plot_exposures_diff(
                ax=exposure_ax,
                regime_benchmark=regime_benchmark,
                regime_classifier=regime_classifier,
                **brinson_kwargs,
                lookthrough_matrix=lookthrough_matrix,
                group_data=group_data,
            )

    if add_tracking_error_table:
        tre_table = multi_portfolio_data.compute_tracking_error_table(strategy_idx=strategy_idx,
                                                                      benchmark_idx=benchmark_idx,
                                                                      **kwargs)
        tre_table, tre_title = reduce_rows(
            tre_table, tre_table.iloc[:, 0], kwargs['asset_tail_count'],
            f'{backtest_name} Per-instrument attribution and IR', metric='active contribution')
        fig1, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
        fig1.suptitle(tre_title,
                      fontweight="bold", fontsize=8, color='blue')
        figs.append(fig1)
        qis.plot_df_table(df=tre_table,
                          first_row_height=0.075,
                          var_format='{:.2%}',
                          rotation_for_columns_headers=90,
                          heatmap_columns=[x+1 for x in np.arange(len(tre_table.columns))],
                          ax=ax,
                          cmap='Blues',
                          **kwargs)

    if add_exposures_comp:
        strategy_name = multi_portfolio_data.portfolio_datas[strategy_idx].ticker
        benchmark_name = multi_portfolio_data.portfolio_datas[benchmark_idx].ticker
        strategy_weights, benchmark_weights = multi_portfolio_data.get_aligned_weights(
            strategy_idx=strategy_idx,
            benchmark_idx=benchmark_idx,
            is_grouped=is_grouped,
            **kwargs,
            lookthrough_matrix=lookthrough_matrix,
            group_data=group_data,
        )

        strategy_turnover, benchmark_turnover = multi_portfolio_data.get_aligned_turnover(
            strategy_idx=strategy_idx,
            benchmark_idx=benchmark_idx,
            is_grouped=is_grouped,
            **kwargs,
            lookthrough_matrix=lookthrough_matrix,
            group_data=group_data,
        )
        exposure_scores = strategy_weights.subtract(benchmark_weights).ffill().iloc[-1]
        exposure_names = tail_index(exposure_scores, kwargs['asset_tail_count'])
        exposure_title = selection_title(
            'Exposures and Two-sided Turnover', len(exposure_scores), len(exposure_names),
            kwargs['asset_tail_count'], 'last active exposure')
        for inst in exposure_names:
            df1 = pd.concat([strategy_weights[inst].rename(strategy_name),
                            benchmark_weights[inst].rename(benchmark_name)], axis=1, sort=True)
            df2 = pd.concat([strategy_turnover[inst].rename(strategy_name),
                            benchmark_turnover[inst].rename(benchmark_name)], axis=1, sort=True)
            with sns.axes_style("darkgrid"):
                fig1, axs = plt.subplots(2, 1, figsize=figsize, constrained_layout=True)
                fig1.suptitle(
                    f'{inst} {exposure_title}',
                    fontweight="bold",
                    fontsize=8,
                    color='blue')
                figs.append(fig1)
                qis.plot_time_series(df=df1, title='Exposures',
                                     legend_stats=qis.LegendStats.AVG_MIN_MAX_LAST,
                                     var_format='{:,.2%}',
                                     ax=axs[0], **kwargs)
                qis.plot_time_series(df=df2, title='Two-sided Turnover',
                                     legend_stats=qis.LegendStats.AVG_MIN_MAX_LAST,
                                     var_format='{:,.2%}',
                                     ax=axs[1], **kwargs)
            # Close only the completed exposure page: keep earlier report and caller figures
            # registered, while the returned Figure remains usable by save_figs_to_pdf.
            plt.close(fig1)

    if add_exposures_pnl_attribution:
        strategy_name = multi_portfolio_data.portfolio_datas[strategy_idx].ticker
        benchmark_name = multi_portfolio_data.portfolio_datas[benchmark_idx].ticker
        strategy_grouped_exposures_agg, strategy_grouped_exposures_by_inst = (
            multi_portfolio_data.portfolio_datas[strategy_idx].get_grouped_long_short_weights(
                time_period=time_period,
                lookthrough_matrix=lookthrough_matrix,
                group_data=group_data,
            )
        )

        benchmark_grouped_exposures_agg, benchmark_grouped_exposures_by_inst = (
            multi_portfolio_data.portfolio_datas[benchmark_idx].get_grouped_long_short_weights(
                time_period=time_period,
                lookthrough_matrix=lookthrough_matrix,
                group_data=group_data,
            )
        )

        strategy_grouped_pnls_agg, strategy_grouped_pnls_by_inst = (
            multi_portfolio_data.portfolio_datas[strategy_idx].get_grouped_cum_pnls(
                time_period=time_period,
                lookthrough_matrix=lookthrough_matrix,
                group_data=group_data,
            )
        )

        benchmark_grouped_pnls_agg, benchmark_grouped_pnls_by_inst = (
            multi_portfolio_data.portfolio_datas[benchmark_idx].get_grouped_cum_pnls(
                time_period=time_period,
                lookthrough_matrix=lookthrough_matrix,
                group_data=group_data,
            )
        )

        scores = pd.Series({
            group: pnl['Total'].iloc[-1] - benchmark_grouped_pnls_agg[group]['Total'].iloc[-1]
            for group, pnl in strategy_grouped_pnls_agg.items()})
        selected_groups = tail_index(scores, kwargs['asset_tail_count'])
        page_title = selection_title(
            f'{backtest_name}: Strategy vs Benchmark exposures and P&L {time_period.to_str()}',
            len(scores), len(selected_groups), kwargs['asset_tail_count'],
            'active cumulative P&L')
        local_kwargs = qis.update_kwargs(kwargs=kwargs, new_kwargs=dict(framealpha=0.9))
        for position, group in enumerate(selected_groups):
            idx = position % 4
            if idx == 0:
                fig2 = plt.figure(figsize=figsize, constrained_layout=True)
                figs.append(fig2)
                fig2.suptitle(page_title, fontweight='bold', fontsize=8, color='blue')
                nrows = min(4, len(selected_groups) - position)
                gs = fig2.add_gridspec(nrows=nrows, ncols=2, wspace=0.0, hspace=0.0)
            df1 = pd.concat([strategy_grouped_exposures_agg[group]['Total'].rename(strategy_name),
                             benchmark_grouped_exposures_agg[group]['Total'].rename(benchmark_name)],
                            axis=1, sort=True)
            df2 = pd.concat([strategy_grouped_pnls_agg[group]['Total'].rename(strategy_name),
                             benchmark_grouped_pnls_agg[group]['Total'].rename(benchmark_name)],
                            axis=1, sort=True)

            datas = {f"{group} aggregated net exposures": df1,
                     f"{group} cumulative P&L attribution": df2}
            for idx_, (key, df) in enumerate(datas.items()):
                ax = fig2.add_subplot(gs[idx, idx_])
                qis.plot_time_series(df=df,
                                     var_format='{:,.2%}',
                                     legend_stats=qis.LegendStats.AVG_MIN_MAX_LAST,
                                     title=f"{key}",
                                     ax=ax,
                                     **local_kwargs)
                multi_portfolio_data.add_regime_shadows(
                    ax=ax, regime_benchmark=regime_benchmark, index=df.index,
                    regime_classifier=regime_classifier)
                set_spines(ax=ax, bottom_spine=False, left_spine=False)

    if add_joint_instrument_history_report:
        perf_columns = (qis.PerfStat.START_DATE, qis.PerfStat.END_DATE, qis.PerfStat.PA_RETURN,
                        qis.PerfStat.VOL, qis.PerfStat.SHARPE_RF0,
                        qis.PerfStat.MAX_DD, qis.PerfStat.MAX_DD_VOL, qis.PerfStat.SKEWNESS)

        joint_prices = pd.concat([multi_portfolio_data.portfolio_datas[strategy_idx].prices,
                                  multi_portfolio_data.portfolio_datas[benchmark_idx].prices],
                                 axis=1, sort=True)
        joint_prices = joint_prices.loc[:, ~joint_prices.columns.duplicated(keep='first')]
        joint_prices, history_title = price_tails(
            time_period.locate(joint_prices), kwargs['asset_tail_count'],
            'Instruments in Joint Universe')
        fig = qis.generate_price_history_report(
            prices=joint_prices,
            **qis.update_kwargs(
                kwargs, dict(fontsize=4, figsize=figsize, perf_columns=perf_columns)))
        fig.suptitle(history_title, fontweight="bold", fontsize=8, color='blue')
        figs.append(fig)

    if add_strategy_factsheet:
        for portfolio_data in multi_portfolio_data.portfolio_datas:
            figs.append(
                generate_strategy_factsheet(
                    portfolio_data=portfolio_data,
                    benchmark_prices=multi_portfolio_data.benchmark_prices,
                    perf_params=perf_params,
                    regime_classifier=regime_classifier,
                    add_grouped_exposures=add_grouped_exposures,
                    add_grouped_cum_pnl=add_grouped_cum_pnl,
                    is_grouped=is_grouped,
                    recent_ra_perf_table_start_date=(recent_ra_perf_table_start_date),
                    add_performance_inference_sheet=add_performance_inference_sheet,
                    alpha_inference=alpha_inference,
                    **kwargs,  # time period will be in kwargs
                    lookthrough_matrix=lookthrough_matrix,
                    group_data=group_data,
                )
            )
        figs = qis.to_flat_list(figs)

    return figs


def generate_strategy_benchmark_active_perf_plt(
    multi_portfolio_data: MultiPortfolioData,
    strategy_idx: int = 0,  # strategy is multi_portfolio_data[strategy_idx]
    benchmark_idx: int = 1,  # benchmark is multi_portfolio_data[benchmark_idx]
    time_period: TimePeriod = None,
    perf_params: PerfParams = PERF_PARAMS,
    regime_classifier: BenchmarkReturnsQuantilesRegime = BenchmarkReturnsQuantilesRegime(),
    backtest_name: str = None,
    weight_freq: Optional[str] = "ME",
    add_strategy_factsheet: bool = False,
    add_grouped_exposures: bool = False,  # for strategy factsheet
    add_grouped_cum_pnl: bool = False,  # for strategy factsheet
    figsize: Tuple[float, float] = (8.3, 11.7),  # A4 for portrait
    is_long_only: bool = False,
    fontsize: int = 8,
    *,
    group_data: Optional[pd.Series] = None,
    lookthrough_matrix: Optional[pd.DataFrame] = None,
    **kwargs,
) -> List[plt.Figure]:
    """
    display 2*2 plot with nav and Brinson attribution


    Args:
        lookthrough_matrix: Optional static instrument-by-sleeve fractions. Supplied rows
            are finite, nonnegative and sum to one; omitted rows use group_data.
            None preserves categorical reporting. This is an accounting split.
        group_data: Ordinary instrument labels as a Series; defaults to portfolio metadata
            where available. Fractional memberships belong in lookthrough_matrix.
    """
    group_data, lookthrough_matrix = resolve_reporting_inputs(
        multi_portfolio_data.portfolio_datas, group_data, lookthrough_matrix)
    if len(multi_portfolio_data.portfolio_datas) == 1:
        raise ValueError("must be at least two strategies")

    # set report specific kqargs
    plot_kwargs = dict(fontsize=fontsize,
                       linewidth=1.0,
                       digits_to_show=1, sharpe_digits=2,
                       weight='normal',
                       markersize=1,
                       framealpha=0.75,
                       time_period=time_period,
                       perf_stats_labels=qis.PerfStatsLabels.DETAILED_WITH_DD.value)
    kwargs = qis.update_kwargs(kwargs, plot_kwargs)

    figs = []
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    figs.append(fig)
    gs = fig.add_gridspec(nrows=3, ncols=2, wspace=0.0, hspace=0.0)

    if backtest_name is not None:
        fig.suptitle(backtest_name, fontweight="bold", fontsize=8, color='blue')

    regime_benchmark = multi_portfolio_data.benchmark_prices.columns[0]
    # benchmark_price = multi_portfolio_data.benchmark_prices[regime_benchmark]

    multi_portfolio_data.plot_nav(ax=fig.add_subplot(gs[0, 0]),
                                  regime_benchmark=regime_benchmark,
                                  perf_params=perf_params,
                                  regime_classifier=regime_classifier,
                                  title='Cumulative performance',
                                  **kwargs)

    (
        totals_table,
        active_total,
        grouped_allocation_return,
        grouped_selection_return,
        grouped_interaction_return,
    ) = multi_portfolio_data.compute_brinson_attribution(
        strategy_idx=strategy_idx,
        benchmark_idx=benchmark_idx,
        freq=None,
        time_period=time_period,
        is_exclude_interaction_term=True,
        lookthrough_matrix=lookthrough_matrix,
        group_data=group_data,
    )

    datas = {
        'Active total return (Brinson attribution)':
            (active_total.cumsum(axis=0), fig.add_subplot(gs[0, 1])),
        'Asset class allocation return':
            (grouped_allocation_return.cumsum(axis=0), fig.add_subplot(gs[1, 1])),
        'Instrument selection return':
            (grouped_selection_return.cumsum(axis=0), fig.add_subplot(gs[2, 1]))}
    for key, (df, ax) in datas.items():
        legend_labels = [column + ', sum=' + '{:.1%}'.format(df[column].iloc[-1])
                         for column in df.columns]
        qis.plot_time_series(df=df,
                             title=key,
                             var_format='{:.1%}',
                             legend_labels=legend_labels,
                             ax=ax,
                             **kwargs)

        if regime_benchmark is not None:
            multi_portfolio_data.add_regime_shadows(
                ax=ax, regime_benchmark=regime_benchmark, index=df.index,
                regime_classifier=regime_classifier)

    # weights box plot
    # make horizontal is too many instruments
    if len(multi_portfolio_data.portfolio_datas[0].prices.columns) > 9:
        local_kwargs = qis.update_kwargs(kwargs, dict(x_rotation=90, add_hue_to_legend_title=False))
    else:
        local_kwargs = qis.update_kwargs(kwargs, dict(add_hue_to_legend_title=False))
    ax = fig.add_subplot(gs[1, 0])
    multi_portfolio_data.plot_weights_boxplot(
        strategy_idx=strategy_idx,
        benchmark_idx=benchmark_idx,
        freq=weight_freq,
        is_grouped=True,
        title=f"Boxplot of weights by groups at {weight_freq}-freq",
        ax=ax,
        **local_kwargs,
        lookthrough_matrix=lookthrough_matrix,
        group_data=group_data,
    )
    if is_long_only:
        set_y_limits(ax=ax, y_limits=(0, None))
    ax = fig.add_subplot(gs[2, 0])
    multi_portfolio_data.plot_weights_boxplot(
        strategy_idx=strategy_idx,
        benchmark_idx=benchmark_idx,
        freq=weight_freq,
        is_grouped=False,
        title=f"Boxplot of weights by instruments at {weight_freq}-freq",
        ax=ax,
        **local_kwargs,
        lookthrough_matrix=lookthrough_matrix,
        group_data=group_data,
    )
    if is_long_only:
        set_y_limits(ax=ax, y_limits=(0, None))

    if add_strategy_factsheet:
        for portfolio_data in multi_portfolio_data.portfolio_datas:
            figs.append(
                generate_strategy_factsheet(
                    portfolio_data=portfolio_data,
                    benchmark_prices=multi_portfolio_data.benchmark_prices,
                    perf_params=perf_params,
                    regime_classifier=regime_classifier,
                    add_grouped_exposures=add_grouped_exposures,
                    add_grouped_cum_pnl=add_grouped_cum_pnl,
                    **kwargs,  # time period will be in kwargs
                    lookthrough_matrix=lookthrough_matrix,
                    group_data=group_data,
                )
            )
        figs = qis.to_flat_list(figs)
    return figs


def generate_performance_attribution_report(multi_portfolio_data: MultiPortfolioData,
                                            time_period: TimePeriod = None,
                                            figsize: Tuple[float, float] = (8.3, 11.7),
                                            **kwargs
                                            ) -> List[plt.Figure]:
    figs = []
    with sns.axes_style('darkgrid'):
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        figs.append(fig)
        gs = fig.add_gridspec(nrows=3, ncols=2, wspace=0.0, hspace=0.0)

        multi_portfolio_data.plot_nav(ax=fig.add_subplot(gs[0, :]),
                                      time_period=time_period,
                                      **kwargs)

        multi_portfolio_data.plot_performance_attribution(portfolio_ids=[0],
                                                          time_period=time_period,
                                                          attribution_metric=AttributionMetric.PNL,
                                                          ax=fig.add_subplot(gs[1, :]),
                                                          **kwargs)

        multi_portfolio_data.plot_performance_attribution(portfolio_ids=[1],
                                                          time_period=time_period,
                                                          attribution_metric=AttributionMetric.PNL,
                                                          ax=fig.add_subplot(gs[2, :]),
                                                          **kwargs)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        figs.append(fig)
        multi_portfolio_data.plot_performance_periodic_table(portfolio_ids=[0],
                                                             time_period=time_period,
                                                             attribution_metric=AttributionMetric.INST_PNL,
                                                             freq='ME',
                                                             ax=ax,
                                                             **kwargs)

    return figs

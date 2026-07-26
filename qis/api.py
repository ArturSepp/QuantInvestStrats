"""
the documented core of the qis public API.

Public means exported from ``qis/__init__.py``; 386 symbols are. That set is what may be
imported, and it does not change here. This module records a smaller set - the core - which is
what the documentation promises: every name in ``CORE_API`` carries an ``Args`` or
``Attributes`` block, is demonstrated in the cookbook, and is what the paper describes. The rest
stay exported and usable, with no prose promise and no stability guarantee beyond the CHANGELOG.

The boundary is measured, not chosen. Every consumer in the stack was parsed for attribute
references and ``from qis import`` targets, and a symbol is core when a published package
(``optimalportfolios``, ``privateassets``) or qis's own examples, README or docs call it. On the
2026-07-26 measurement that is 98 of 386; the other 288 are called only by private research
repositories (109) or by nothing at all (179).

Five bootstrap symbols are promoted by intent rather than by measurement: today only ``rosaa``
calls them, but resampling is one of the eight capabilities the cookbook and the paper describe,
and the FAJ replication code migrates onto them. They are marked below.

Two consequences worth knowing:

  * ``market_data`` has no core symbol. No published package imports it at the top level, which
    is the same finding as its 5 exports against 34 deep imports - the capability is real and
    the namespace does not publish it. Widening those exports is roadmap item T5.
  * grouping here is by capability, not by defining module, so moving a symbol between
    subpackages costs no documentation change. ``docs/conf.py`` renders these groups directly.

``qis/tests/test_core_api.py`` enforces the promise: a core symbol without documented arguments
fails the suite.
"""
# packages
from typing import Dict, Tuple

CORE_API: Dict[str, Tuple[str, ...]] = {
    'Performance statistics': (
        'PerfParams', 'PerfStat', 'compute_ra_perf_table', 'get_ra_perf_columns',
        'compute_asset_returns_dict', 'to_returns', 'returns_to_nav', 'delever_returns',
        'implied_leverage', 'interpolate_infrequent_returns', 'SignalDiagnosticsResult',
        'estimate_ic_ir', 'estimate_signal_diagnostics',
    ),
    'Portfolio and backtesting': (
        'backtest_model_portfolio', 'PortfolioData', 'MultiPortfolioData',
        'compute_portfolio_risk_contributions', 'EwmLinearModel',
    ),
    'Factsheets and reporting': (
        'factsheet', 'generate_strategy_factsheet',
        'generate_strategy_benchmark_factsheet_plt', 'generate_multi_portfolio_factsheet',
        'plot_exposures_strategy_vs_benchmark_stack', 'ReportingFrequency',
        'fetch_default_report_kwargs', 'FACTSHEET_CONFIG_DAILY_DATA_SHORT_PERIOD',
    ),
    'EWM estimation': (
        'MeanAdjType', 'NanBackfill', 'compute_ewm', 'compute_ewm_vol',
        'compute_ewm_covar_tensor', 'compute_ewm_covar_tensor_vol_norm_returns',
        'compute_ewm_long_short_filtered_ra_returns', 'compute_masked_covar_corr',
        'estimate_rolling_ewma_covar', 'estimate_hf_ohlc_vol', 'plot_corr_matrix_from_covar',
    ),
    # rosaa is the consumer, and it reached eleven symbols through six deep imports because only
    # five were exported. The six are exported now, and the capability is core: a private
    # production consumer is still a consumer, and the cookbook has a page for it.
    'Market data and FX': (
        'FxRatesData', 'FactorsData', 'load_fx_rates_data', 'get_aligned_fx_spots',
        'compute_local_and_fx_return', 'compute_performance_of_local_ccy_asset_in_reference_ccy',
        'compute_fx_vol_beta', 'compute_fx_optimal_hedge', 'compute_futures_fx_adjusted_returns',
        'compute_cash_fx_adjusted_returns', 'compute_multi_asset_fx_hedging',
        'run_asset_fx_hedging_report', 'plot_multi_asset_fx_hedging_report',
    ),
    'Regime reporting': (
        'BenchmarkReturnsQuantilesRegime',
    ),
    'Bootstrap': (
        'BootstrapType', 'BootstrapOutput', 'generate_bootstrapped_indices', 'bootstrap_data',
        'bootstrap_price_data',
    ),
    'Unsmoothing': (
        'compute_ar_unsmoothed_prices', 'unsmooth_returns_ar1_ewma', 'unsmooth_returns_glm',
    ),
    'Plots': (
        'plot_time_series', 'plot_prices', 'plot_prices_with_dd', 'plot_bars', 'plot_scatter',
        'plot_classification_scatter', 'plot_heatmap', 'plot_qq', 'plot_df_table',
        'df_boxplot_by_classification_var', 'df_boxplot_by_hue_var', 'LegendStats',
        'set_suptitle',
    ),
    'Dates, schedules and annualisation': (
        'TimePeriod', 'get_time_period', 'get_time_period_label', 'generate_dates_schedule',
        'generate_rebalancing_indicators', 'generate_fixed_maturity_rolls',
        'find_upto_date_from_datetime_index', 'truncate_prior_to_start',
        'get_annualisation_conversion_factor', 'get_annualization_factor',
        'infer_annualisation_factor_from_df',
    ),
    'DataFrame utilities': (
        'df_abssum', 'df_abssum_positive', 'df_abssum_negative', 'df_nansum',
        'df_nansum_positive', 'df_nansum_negative', 'df_nansum_clip', 'df_nanmean',
        'df_nanmean_positive', 'df_nanmean_clip', 'df_nanmedian', 'df_last_row',
        'series_nansum_weighted', 'get_group_dict', 'set_group_loadings', 'split_df_by_groups',
        'df_to_cross_sectional_score', 'df_to_equal_weight_allocation',
        'df_to_long_only_allocation_sum1', 'df_to_weight_allocation_sum1', 'covar_to_corr',
        'np_array_to_df_columns', 'fit_multivariate_ols', 'update_kwargs',
    ),
    'File and figure output': (
        'load_df_from_csv', 'load_df_dict_from_csv', 'load_df_from_excel', 'save_df_to_csv',
        'save_df_dict_to_csv', 'save_fig', 'save_figs_to_pdf', 'timer', 'get_resource_path',
    ),
}


def core_api_names() -> Tuple[str, ...]:
    """
    every name in the documented core, flattened.

    Returns:
        the core symbol names, in capability order
    """
    return tuple(name for names in CORE_API.values() for name in names)

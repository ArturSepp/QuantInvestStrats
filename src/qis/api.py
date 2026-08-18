"""
the documented core of the qis public API.

Public means exported from ``src/qis/__init__.py``. That set is what may be imported, and it does
not change here. This module records a smaller set - the core - which is what the documentation
promises: every name in ``CORE_API`` carries an ``Args`` or ``Attributes`` block, is
demonstrated in the cookbook, and is what the paper describes. The rest stay exported and
usable, with no prose promise and no stability guarantee beyond the CHANGELOG.

The boundary is measured, not chosen. Every consumer in the stack was parsed for attribute
references and ``from qis import`` targets, and a symbol is core when a published package
(``optimalportfolios``, ``trendfollowing``, ``privateassets``) or qis's own examples, README or
docs call it. ``tools/audit_consumers.py`` reproduces the public half of that measurement at the
revisions pinned in ``docs/audit/consumers.json``.

Counts are deliberately absent from this docstring. The export count, the core count and the
number of capability groups all move when the namespace moves, and a count written into prose
goes stale without anything failing. They are generated into ``docs/audit/paper_numbers.json``
by ``tools/paper_audit.py``, and ``src/qis/tests/test_paper_audit.py`` fails when that record and
the repository disagree. The version of this docstring written on 2026-07-26 carried five such
counts and four of them were wrong within a day.

Five bootstrap symbols are promoted by intent rather than by measurement: today only ``rosaa``
calls them, but resampling is one of the capabilities the cookbook and the paper describe, and
the FAJ replication code migrates onto them.

Grouping is by capability rather than by defining module, so moving a symbol between subpackages
costs no documentation change. ``docs/conf.py`` renders these groups directly.

``src/qis/tests/test_core_api.py`` enforces the promise: a core symbol without documented arguments
fails the suite.
"""
# packages
from typing import Dict, Tuple

# every name the package exports, enumerated rather than inferred. ``src/qis/__init__.py`` sets
# ``__all__`` from its own namespace, so this tuple does not decide what is public; it records
# what is, in a form a diff can show. Adding an export without adding it here fails
# ``src/qis/tests/test_core_api.py``. Regenerate with ``python tools/sync_public_api.py``.
PUBLIC_API: Tuple[str, ...] = (
    'AttributionMetric', 'BENCHMARK_TABLE_COLUMNS', 'BENCHMARK_TABLE_COLUMNS2',
    'BenchmarkReturnsPositiveNegativeRegime', 'BenchmarkReturnsQuantilesRegime',
    'BenchmarkVolsQuantilesRegime', 'BootstrapOutput', 'BootstrapType', 'COMPACT_TABLE_COLUMNS',
    'ColVar', 'ColumnData', 'ConvolutionType', 'CorrMatrixOutput', 'CrossXyType', 'DdLegendType',
    'DescTableType', 'EXTENDED_TABLE_COLUMNS', 'EnumMap', 'EwmLinearModel',
    'FACTSHEET_CONFIG_DAILY_DATA_LONG_PERIOD', 'FACTSHEET_CONFIG_DAILY_DATA_SHORT_PERIOD',
    'FACTSHEET_CONFIG_MONTHLY_DATA_LONG_PERIOD', 'FACTSHEET_CONFIG_MONTHLY_DATA_SHORT_PERIOD',
    'FACTSHEET_CONFIG_QUARTERLY_DATA_LONG_PERIOD', 'FULL_TABLE_COLUMNS', 'FactorsData',
    'FactsheetConfig', 'FileTypes', 'FxRatesData', 'InitType', 'LN_BENCHMARK_TABLE_COLUMNS',
    'LN_BENCHMARK_TABLE_COLUMNS_SHORT', 'LN_TABLE_COLUMNS', 'LastLabel', 'LegendStats',
    'LinearModel', 'MeanAdjType', 'MultiAssetsReport', 'MultiPortfolioData', 'NanBackfill',
    'OhlcEstimatorType', 'OutlierPolicy', 'PdfType', 'PerfParams', 'PerfStat', 'PerfStatsLabels',
    'PortfolioData', 'PortfolioInput', 'RA_TABLE_COLUMNS', 'RA_TABLE_COMPACT_COLUMNS',
    'RegimeClassifier', 'RegimeData', 'ReplacementType', 'ReportingFrequency', 'ReturnTypes',
    'ReturnsTransform', 'RiskModel', 'RollingPerfStat', 'SD_PERF_COLUMNS',
    'STANDARD_TABLE_COLUMNS', 'SharpeConvention', 'SignalAggType', 'SignalDiagnosticsColumns',
    'SignalDiagnosticsResult', 'SignalMapType', 'SmartDiversificationReport', 'SnapshotPeriod',
    'StrategySignalData', 'TimePeriod', 'TrainLivePeriod', 'TrainLiveSamples', 'TrendLine',
    'ValueType', 'WEIGHT_TOL', 'add_bnb_regime_shadows', 'adjust_component_navs_to_portfolio',
    'adjust_returns_with_ar', 'adjust_returns_with_factor_lag',
    'adjust_returns_with_joint_unsmoothing', 'align_weights_to_columns', 'append_df_to_feather',
    'append_time_series', 'apply_pca', 'assert_list_subset', 'backtest_model_portfolio',
    'backtest_rebalanced_portfolio', 'bfill_timeseries', 'bootstrap_ar_process', 'bootstrap_data',
    'bootstrap_price_data', 'bootstrap_price_fundamental_data',
    'check_df_for_duplicated_columns_index', 'compute_ar1_unsmoothed_prices',
    'compute_ar_residuals', 'compute_ar_unsmoothed_prices', 'compute_asset_returns_dict',
    'compute_autocorr_df', 'compute_autocorrelation_at_int_periods', 'compute_avg_max_dd',
    'compute_benchmark_portfolio_risk_contributions',
    'compute_benchmarks_beta_attribution_from_prices',
    'compute_benchmarks_beta_attribution_from_returns', 'compute_bnb_regimes_pa_perf_table',
    'compute_brinson_attribution_table', 'compute_cash_fx_adjusted_returns', 'compute_data_pca_r2',
    'compute_desc_freq_table', 'compute_desc_table', 'compute_drawdowns_stats_table',
    'compute_eigen_portfolio_weights', 'compute_ewm', 'compute_ewm_alpha_r2_given_prediction',
    'compute_ewm_beta_alpha_forecast', 'compute_ewm_corr_df', 'compute_ewm_corr_single',
    'compute_ewm_covar', 'compute_ewm_covar_newey_west', 'compute_ewm_covar_tensor',
    'compute_ewm_covar_tensor_vol_norm_returns', 'compute_ewm_cross_xy', 'compute_ewm_long_short',
    'compute_ewm_long_short_filter', 'compute_ewm_long_short_filtered_ra_returns',
    'compute_ewm_matrix_autocorr', 'compute_ewm_matrix_autocorr_df', 'compute_ewm_newey_west_vol',
    'compute_ewm_ra_returns_momentum', 'compute_ewm_score', 'compute_ewm_sharpe',
    'compute_ewm_sharpe_from_prices', 'compute_ewm_std1_norm', 'compute_ewm_vector_autocorr',
    'compute_ewm_vector_autocorr_df', 'compute_ewm_vol', 'compute_ewm_xy_beta_tensor',
    'compute_ewma_realised_tracking_error', 'compute_excess_return_navs', 'compute_excess_returns',
    'compute_futures_fx_adjusted_returns', 'compute_fx_optimal_hedge', 'compute_fx_vol_beta',
    'compute_group_portfolio_risk_contribution_ratios', 'compute_ic_timeseries',
    'compute_info_ratio_table', 'compute_local_and_fx_return', 'compute_masked_covar_corr',
    'compute_max_current_drawdown', 'compute_mean_freq_regimes', 'compute_multi_asset_fx_hedging',
    'compute_net_navs_ex_perf_man_fees', 'compute_net_return_ex_perf_man_fees',
    'compute_num_years', 'compute_one_factor_ewm_betas', 'compute_pa_excess_compounded_returns',
    'compute_pa_return', 'compute_path_autocorr', 'compute_path_autocorr_given_lags',
    'compute_path_corr', 'compute_path_lagged_corr', 'compute_path_lagged_corr_given_lags',
    'compute_pca_r2', 'compute_per_asset_betas',
    'compute_performance_of_local_ccy_asset_in_reference_ccy', 'compute_performance_table',
    'compute_periodic_returns', 'compute_periodic_returns_table',
    'compute_portfolio_benchmark_ewm_beta_alpha_attribution',
    'compute_portfolio_correlated_var_by_groups', 'compute_portfolio_ewm_benchmark_betas',
    'compute_portfolio_independent_var_by_ac', 'compute_portfolio_risk_contribution_ratios',
    'compute_portfolio_risk_contributions', 'compute_portfolio_var_np', 'compute_portfolio_vol',
    'compute_ra_perf_table', 'compute_ra_perf_table_with_benchmark', 'compute_ra_returns',
    'compute_regime_avg', 'compute_regimes_pa_perf_table_from_sampled_returns',
    'compute_returns_dict', 'compute_returns_transform', 'compute_risk_table', 'compute_roll_mean',
    'compute_rolling_drawdown_time_under_water', 'compute_rolling_drawdowns',
    'compute_rolling_mean_adj', 'compute_rolling_perf_stat', 'compute_rolling_ra_returns',
    'compute_sampled_vols', 'compute_sum_freq_ra_returns', 'compute_sum_rolling_ra_returns',
    'compute_te_ir_errors', 'compute_total_return', 'corr_to_pivot_row', 'covar_to_corr',
    'create_overlay_portfolio_curve', 'create_rebalancing_indicators_from_freqs',
    'delever_returns', 'df_abssum', 'df_abssum_negative', 'df_abssum_positive', 'df_asfreq',
    'df_boxplot_by_classification_var', 'df_boxplot_by_columns', 'df_boxplot_by_hue_var',
    'df_boxplot_by_index', 'df_dict_boxplot_by_classification_var', 'df_dict_boxplot_by_columns',
    'df_last_row', 'df_nanmean', 'df_nanmean_clip', 'df_nanmean_negative', 'df_nanmean_positive',
    'df_nanmean_weighted', 'df_nanmedian', 'df_nansum', 'df_nansum_clip', 'df_nansum_negative',
    'df_nansum_positive', 'df_to_cross_sectional_score', 'df_to_equal_weight_allocation',
    'df_to_long_only_allocation_sum1', 'df_to_str', 'df_to_weight_allocation_sum1',
    'estimate_acf_from_path', 'estimate_acf_from_paths', 'estimate_dimson_beta',
    'estimate_ewm_factor_model', 'estimate_hf_ohlc_vol', 'estimate_ic_ir', 'estimate_ohlc_var',
    'estimate_rolling_ewma_covar', 'estimate_signal_diagnostics', 'estimate_vol',
    'ewm_insample_winsorising', 'ewm_recursion', 'ewm_xy_convolution', 'factsheet',
    'fetch_default_perf_params', 'fetch_default_report_kwargs', 'fetch_factsheet_config_kwargs',
    'file_utils', 'filter_outliers', 'find_upto_date_from_datetime_index', 'fit_multivariate_ols',
    'flatten_group_attribution', 'generate_bootstrapped_indices', 'generate_current_signal_report',
    'generate_dates_schedule', 'generate_fixed_maturity_rolls', 'generate_multi_asset_factsheet',
    'generate_multi_portfolio_factsheet', 'generate_price_history_report',
    'generate_rebalancing_indicators', 'generate_static_weights_schedule',
    'generate_strategy_benchmark_active_perf_plt', 'generate_strategy_benchmark_factsheet_plt',
    'generate_strategy_factsheet', 'generate_strategy_signal_factsheet_by_instrument',
    'generate_weight_change_report', 'get_aligned_fx_spots', 'get_all_folder_files',
    'get_annualisation_conversion_factor', 'get_annualization_factor', 'get_excess_returns_nav',
    'get_group_dict', 'get_local_file_path', 'get_nonnan_index', 'get_output_path',
    'get_paired_rareturns_signals', 'get_paths', 'get_pdf_path', 'get_ra_perf_columns',
    'get_resource_path', 'get_time_period', 'get_time_period_label',
    'get_time_period_shifted_by_years', 'get_time_to_maturity', 'get_ytd_time_period',
    'idx_to_alphabet', 'implied_leverage', 'infer_annualisation_factor_from_df',
    'interpolate_infrequent_returns', 'join_file_name_parts', 'lever_returns',
    'limit_weights_to_max_var_limit', 'load_df_dict_from_csv', 'load_df_dict_from_excel',
    'load_df_dict_from_feather', 'load_df_dict_from_parquet', 'load_df_from_csv',
    'load_df_from_excel', 'load_df_from_feather', 'load_df_from_parquet', 'load_fx_rates_data',
    'local_path', 'log_returns_to_nav', 'long_short_to_relative_nav', 'map_signal_to_weight',
    'market_data', 'matrix_regularization', 'melt_scatter_data_with_xvar', 'models',
    'mult_df_columns_with_vector', 'np_array_to_df_columns', 'np_array_to_df_index', 'perfstats',
    'plot_bars', 'plot_best_worst_returns', 'plot_box', 'plot_brinson_attribution_table',
    'plot_brinson_totals_table', 'plot_classification_scatter', 'plot_contour',
    'plot_corr_matrix_from_covar', 'plot_data_timeseries', 'plot_desc_freq_table', 'plot_df_table',
    'plot_df_table_with_ci', 'plot_errorbar', 'plot_exposures_strategy_vs_benchmark_stack',
    'plot_heatmap', 'plot_histogram', 'plot_histplot2d', 'plot_line', 'plot_lines_list',
    'plot_multi_asset_fx_hedging_report', 'plot_multivariate_scatter_with_prediction',
    'plot_periodic_returns_table', 'plot_pie', 'plot_price_history', 'plot_prices',
    'plot_prices_2ax', 'plot_prices_with_dd', 'plot_prices_with_fundamentals', 'plot_qq',
    'plot_quantile_class_table', 'plot_ra_perf_annual_matrix', 'plot_ra_perf_bars',
    'plot_ra_perf_by_dates', 'plot_ra_perf_scatter', 'plot_ra_perf_table',
    'plot_ra_perf_table_benchmark', 'plot_regime_boxplot', 'plot_regime_data', 'plot_regime_pdf',
    'plot_returns_corr_matrix_time_series', 'plot_returns_corr_table',
    'plot_returns_ewm_corr_table', 'plot_returns_heatmap', 'plot_returns_scatter',
    'plot_returns_table', 'plot_rolling_drawdowns', 'plot_rolling_perf_stat',
    'plot_rolling_time_under_water', 'plot_scatter', 'plot_scatter_regression',
    'plot_signal_diagnostics', 'plot_signal_diagnostics_beta_boxplot',
    'plot_signal_diagnostics_boxplot', 'plot_signal_diagnostics_for_returns',
    'plot_signal_diagnostics_group_boxplot', 'plot_sorted_periodic_returns', 'plot_stack',
    'plot_time_series', 'plot_time_series_2ax', 'plot_top_bottom_performers',
    'plot_top_drawdowns_paths', 'plot_vbars', 'plot_xy_qq', 'plots', 'portfolio',
    'portfolio_navs_to_additive', 'portfolio_returns_to_nav', 'prices_at_freq',
    'prices_to_scaled_nav', 'qis', 'returns_to_nav', 'run_asset_fx_hedging_report',
    'save_df_dict_to_csv', 'save_df_dict_to_excel', 'save_df_dict_to_feather',
    'save_df_dict_to_parquet', 'save_df_to_csv', 'save_df_to_excel', 'save_df_to_feather',
    'save_df_to_parquet', 'save_fig', 'save_figs', 'save_figs_to_pdf', 'series_nansum_weighted',
    'set_group_loadings', 'set_suptitle', 'split_df_by_groups', 'timer', 'to_finite_reciprocal',
    'to_flat_list', 'to_portfolio_returns', 'to_quarterly_returns', 'to_returns',
    'to_total_returns', 'to_zero_first_nonnan_returns', 'truncate_prior_to_start',
    'unsmooth_returns_ar1_ewma', 'unsmooth_returns_glm', 'update_df_in_csv', 'update_kwargs',
    'utils', 'weights_tracking_error_report_by_ac_subac',
)


CORE_API: Dict[str, Tuple[str, ...]] = {
    'Performance statistics': (
        'PerfParams', 'PerfStat', 'compute_ra_perf_table', 'get_ra_perf_columns',
        'compute_asset_returns_dict', 'to_returns', 'returns_to_nav', 'delever_returns',
        'implied_leverage', 'interpolate_infrequent_returns', 'SignalDiagnosticsResult',
        'estimate_ic_ir', 'estimate_signal_diagnostics',
    ),
    'Portfolio and backtesting': (
        'backtest_model_portfolio', 'generate_static_weights_schedule', 'PortfolioData',
        'MultiPortfolioData', 'RiskModel', 'compute_portfolio_risk_contributions',
        'compute_portfolio_risk_contribution_ratios',
        'compute_group_portfolio_risk_contribution_ratios', 'EwmLinearModel',
        'compute_ewma_realised_tracking_error',
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

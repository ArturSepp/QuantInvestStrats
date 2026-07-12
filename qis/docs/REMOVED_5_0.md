# qis 5.0 — removed from the public namespace

**568 -> 373 public symbols.** Nothing is deleted unless listed under *Deleted* below:
the top-level `qis` namespace is reduced, and every other symbol is still importable
from its defining module.

```python
qis.set_spines(ax)                        # 4.x  -> AttributeError in 5.0

from qis.plots.utils import set_spines    # 5.0
set_spines(ax)
```

To locate any symbol not listed here:

```bash
python -c "import qis, pkgutil, importlib; [print(m.name) for m in pkgutil.walk_packages(qis.__path__, 'qis.') if hasattr(importlib.import_module(m.name), 'YOUR_SYMBOL')]"
```

---

## 1. Renamed (breaking, no shim)

`qis.nanmean` / `nanmedian` / `nansum` shadowed the numpy names while carrying different
semantics: DataFrame in, Series out, non-finite entries excluded, and `axis=1` by default
(the opposite of pandas). The whole module is renamed for consistency.

| 4.x | 5.0 |
| --- | --- |
| `qis.nanmean` | `qis.df_nanmean` |
| `qis.nanmedian` | `qis.df_nanmedian` |
| `qis.nansum` | `qis.df_nansum` |
| `qis.nanmean_positive` | `qis.df_nanmean_positive` |
| `qis.nansum_positive` | `qis.df_nansum_positive` |
| `qis.nansum_negative` | `qis.df_nansum_negative` |
| `qis.nanmean_clip` | `qis.df_nanmean_clip` |
| `qis.nansum_clip` | `qis.df_nansum_clip` |
| `qis.nanmean_weighted` | `qis.df_nanmean_weighted` |
| `qis.abssum` | `qis.df_abssum` |
| `qis.abssum_positive` | `qis.df_abssum_positive` |
| `qis.abssum_negative` | `qis.df_abssum_negative` |
| `qis.last_row` | `qis.df_last_row` |
| `qis.sum_weighted` | `qis.series_nansum_weighted` |
| `qis.get_signed_np_data` | `qis.utils.df_agg._get_signed_np_data (private)` |

`sum_weighted` takes two `pd.Series` and returns a `float`, so it is not prefixed `df_`.
Its first parameter is renamed from `df` to `data`.

---

## 2. Deleted (no replacement in the namespace)

| symbol | note |
| --- | --- |
| `qis.ReportType` | module `qis.plots.reports.utils` deleted |
| `qis.df_price_fill_first_nan_by_cross_median` | deleted; use `qis.utils.df_ops.df_fill_first_nan_by_cross_median` |
| `qis.econ_data_report` | module `qis.plots.reports.econ_data_single` deleted |
| `qis.replace_nan_by_median` | deleted; no replacement |

`qis/plots/reports/` is removed. `price_history.py` and `gantt_data_history.py` moved to
`qis/plots/derived/`; `econ_data_single.py` and `reports/utils.py` are deleted.
`qis.plots.derived.gantt_data_history` is not imported by `qis/plots/__init__.py`
because it requires plotly, which is not a qis dependency. Import it by full path.

---

## 3. Example code, never API

| symbol | file |
| --- | --- |
| `qis.DEFAULT_RA_TABLE_COLUMNS` | `qis/examples/_helpers/reporting_helpers.py` |
| `qis.generate_performance_report` | `qis/examples/_helpers/reporting_helpers.py` |
| `qis.load_usd_assets` | `qis/examples/market_data/fx_hedging_example.py` |

Do not import from `qis/examples/`. It is documentation and is restructured without notice.

---

## 4. Moved out of the namespace (207 symbols)

Grouped by the module to import from.

### `qis.plots.utils` (45)

```python
from qis.plots.utils import (
    add_scatter_points,
    align_x_limits_ax12,
    align_x_limits_axs,
    align_xy_limits,
    align_y_limits_ax12,
    align_y_limits_axs,
    autolabel,
    calc_df_table_size,
    calc_table_height,
    calc_table_width,
    compute_heatmap_colors,
    create_dummy_line,
    get_cmap_colors,
    get_data_group_colors,
    get_df_table_size,
    get_legend_lines,
    get_n_cmap_colors,
    get_n_colors,
    get_n_fixed_colors,
    get_n_hatch,
    get_n_markers,
    get_n_mlt_colors,
    get_n_sns_colors,
    get_table_lines_for_group_data,
    map_dates_index_to_str,
    rand_cmap,
    remove_spines,
    reset_xticks,
    scale_ax_bar_width,
    set_ax_tick_labels,
    set_ax_tick_params,
    set_ax_ticks_format,
    set_ax_xy_labels,
    set_date_on_axis,
    set_labels_frequency,
    set_legend,
    set_legend_colors,
    set_legend_with_stats_table,
    set_linestyles,
    set_spines,
    set_title,
    set_x_limits,
    set_y_limits,
    subplot_border,
    validate_returns_plot
)
```

### `qis.utils.df_ops` (26)

```python
from qis.utils.df_ops import (
    align_df1_to_df2,
    align_dfs_dict_with_df,
    compute_last_score,
    compute_nans_zeros_ratio_after_first_non_nan,
    df12_merge_with_tz,
    df_align_to_common_index,
    df_ffill_negatives,
    df_fill_first_nan_by_cross_median,
    df_indicator_like,
    df_joint_indicator,
    df_ones_like,
    df_price_ffill_between_nans,
    df_time_dict_to_pd,
    df_zero_like,
    dfs_indicators,
    dfs_to_upper_lower_diag,
    drop_first_nan_data,
    factor_dict_to_asset_dict,
    get_first_nonnan_values,
    get_last_nonnan,
    get_last_nonnan_values,
    merge_dfs_on_column,
    multiply_df_by_dt,
    norm_df_by_ax_mean,
    np_txy_tensor_to_pd_dict,
    reindex_upto_last_nonnan
)
```

### `qis.utils.np_ops` (24)

```python
from qis.utils.np_ops import (
    compute_expanding_power,
    compute_histogram_data,
    compute_paired_signs,
    find_nearest,
    np_array_to_matrix,
    np_array_to_n_column_array,
    np_array_to_t_rows_array,
    np_cumsum,
    np_get_sorted_idx,
    np_matrix_add_array,
    np_nanmean,
    np_nanstd,
    np_nansum,
    np_nanvar,
    np_nonan_weighted_avg,
    np_shift,
    repeat_by_columns,
    repeat_by_rows,
    running_mean,
    select_non_nan_x_y,
    set_nans_for_warmup_period,
    to_finite_np,
    to_finite_ratio,
    to_nearest_values
)
```

### `qis.utils.dates` (15)

```python
from qis.utils.dates import (
    generate_sample_dates,
    get_current_time_with_tz,
    get_month_days,
    get_sample_dates_idx,
    get_weekday,
    get_year_quarter,
    is_leap_year,
    min_timestamp,
    months_between,
    set_rebalancing_timeindex_on_given_timeindex,
    shift_date_by_day,
    shift_dates_by_n_years,
    shift_dates_by_year,
    shift_time_period_by_days,
    split_df_by_freq
)
```

### `qis.utils.df_str` (14)

```python
from qis.utils.df_str import (
    date_to_str,
    df_all_to_str,
    df_index_to_str,
    df_to_numeric,
    df_with_ci_to_str,
    float_to_str,
    get_fmt_str,
    join_str_series,
    series_to_date_str,
    series_to_numeric,
    series_to_str,
    series_values_to_str,
    str_to_float,
    timeseries_df_to_str
)
```

### `qis.utils.struct_ops` (10)

```python
from qis.utils.struct_ops import (
    assert_list_unique,
    flatten,
    flatten_dict_tuples,
    list_diff,
    list_intersection,
    list_to_unique_and_dub,
    merge_lists_unique,
    move_item_to_first,
    separate_number_from_string,
    split_dict
)
```

### `qis.perfstats.perf_stats` (8)

```python
from qis.perfstats.perf_stats import (
    BENCHMARK_TABLE_COLUMNS,  # const
    BENCHMARK_TABLE_COLUMNS2,  # const
    COMPACT_TABLE_COLUMNS,  # const
    EXTENDED_TABLE_COLUMNS,  # const
    LN_BENCHMARK_TABLE_COLUMNS,  # const
    LN_BENCHMARK_TABLE_COLUMNS_SHORT,  # const
    LN_TABLE_COLUMNS,  # const
    STANDARD_TABLE_COLUMNS  # const
)
```

### `qis.utils.df_to_weights` (7)

```python
from qis.utils.df_to_weights import (
    compute_long_only_portfolio_weights,
    compute_long_short_ind,
    compute_long_short_ind_by_row,
    df_nans_to_one_zero,
    df_to_top_bottom_n_indicators,
    fill_long_short_signal,
    mult_df_columns_with_vector_group
)
```

### `qis.plots.table` (6)

```python
from qis.plots.table import (
    set_align_for_column,
    set_cells_facecolor,
    set_column_edge_color,
    set_data_colors,
    set_diag_cells_facecolor,
    set_row_edge_color
)
```

### `qis.utils.df_cut` (6)

```python
from qis.utils.df_cut import (
    add_classification,
    add_hue_fixed_years,
    add_hue_years,
    add_quantile_classification,
    sort_index_by_hue,
    x_bins_cut
)
```

### `qis.utils.df_groups` (6)

```python
from qis.utils.df_groups import (
    agg_df_by_group_with_avg,
    agg_df_by_groups,
    agg_df_by_groups_ax1,
    convert_df_column_to_df_by_groups,
    fill_df_with_group_avg,
    sort_df_by_index_group
)
```

### `qis.perfstats.config` (5)

```python
from qis.perfstats.config import (
    FULL_TABLE_COLUMNS,  # const
    RA_TABLE_COLUMNS,  # const
    RA_TABLE_COMPACT_COLUMNS,  # const
    SD_PERF_COLUMNS,  # const
    TRE_TABLE_COLUMNS  # const
)
```

### `qis.utils.regression` (5)

```python
from qis.utils.regression import (
    estimate_alpha_beta_paired_dfs,
    estimate_ols_alpha_beta,
    fit_ols,
    get_ols_x,
    reg_model_params_to_str
)
```

### `qis.utils.df_agg` (4)

```python
from qis.utils.df_agg import (
    agg_data_by_axis,
    agg_dfs,
    agg_median_mad,
    compute_df_desc_data
)
```

### `qis.utils.df_freq` (4)

```python
from qis.utils.df_freq import (
    agg_remained_data_on_right,
    df_resample_at_freq,
    df_resample_at_int_index,
    df_resample_at_other_index
)
```

### `qis.utils.df_melt` (4)

```python
from qis.utils.df_melt import (
    melt_df_by_columns,
    melt_paired_df,
    melt_scatter_data_with_xdata,
    melt_signed_paired_df
)
```

### `qis.market_data.reports.fx_hedging_report` (3)

```python
from qis.market_data.reports.fx_hedging_report import (
    compute_multi_asset_fx_hedging,
    plot_multi_asset_fx_hedging_report,
    run_asset_fx_hedging_report
)
```

### `qis.plots.derived.regime_scatter` (3)

```python
from qis.plots.derived.regime_scatter import (
    ConditionalRegressionColumns,  # enum
    estimate_cond_regression,
    get_regime_regression_params
)
```

### `qis.utils.df_to_scores` (3)

```python
from qis.utils.df_to_scores import (
    compute_aggregate_scores,
    df_to_max_score,
    select_top_integrated_scores
)
```

### `qis.utils.sampling` (3)

```python
from qis.utils.sampling import (
    get_data_samples_df,
    split_to_samples,
    split_to_train_live_samples
)
```

### `qis.utils.generic` (2)

```python
from qis.utils.generic import (
    DotDict,  # class
    column_datas_to_df
)
```

### `qis.plots.derived.perf_table` (1)

```python
from qis.plots.derived.perf_table import (
    get_ra_perf_benchmark_columns
)
```

### `qis.plots.derived.prices` (1)

```python
from qis.plots.derived.prices import (
    get_performance_labels_for_stats
)
```

### `qis.plots.derived.regime_class_table` (1)

```python
from qis.plots.derived.regime_class_table import (
    get_quantile_class_table
)
```

### `qis.plots.derived.returns_heatmap` (1)

```python
from qis.plots.derived.returns_heatmap import (
    compute_periodic_returns_by_row_table
)
```

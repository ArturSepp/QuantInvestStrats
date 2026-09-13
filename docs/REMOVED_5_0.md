---
myst:
  html_meta:
    description: >-
      Historical qis API removals and renames, practical upgrade checks, and
      a current module-import reference for legacy helper calls.
---

# API migration history

*[author / affiliation / date — placeholder]*

This guide accompanies [qis — Quantitative Investment Strategies](https://github.com/ArturSepp/QuantInvestStrats).
Software citation: [CITATION.cff](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

The version sections record historical changes; the module reference below lists imports that
resolve in the current source. Later releases may restore a public export or remove an internal
helper. Use the [changelog](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CHANGELOG.md)
for the transition you are making and `qis.__all__` for the current public surface.

## Upgrade checks

Record the version and import path as shown in the [installation guide](install.md).
Then check a specific public name without importing every submodule:

~~~python
import qis

name = 'df_nanmean'
assert name in qis.__all__
print(name, 'is exported by the imported qis package')
~~~

The [API reference](api/index.rst) and
[public API record](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/api.py)
locate supported operations. For an internal helper, search its definition in a source checkout,
then import the identified module. For example:

~~~console
rg -n "^def set_spines" src/qis
~~~

Avoid a recursive import scan: it executes module-level code and can reach optional backends
or development diagnostics. A name's presence in a module does not give it the compatibility
status of a top-level public export.

## qis 5.9.2 — removed from the public API

| Historical symbol | Replacement or reason |
|---|---|
| `qis.PerfStat.TE` | `compute_te_ir_errors` returns a Series named `'TE'`; the former enum member was removed. |
| `qis.PerfStat.IR` | `compute_te_ir_errors` returns a Series named `'IR'`; the former enum member was removed. |
| `qis.TRE_TABLE_COLUMNS` | Removed without a replacement; the requested statistics were not filled by the table builder. |

These removals are recorded in the
[5.9.2 changelog](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CHANGELOG.md).
For the current calculation contract, see [tracking error and information ratio](tracking_error_and_risk.md).

## qis 5.0 — removed from the public namespace

The 5.0 changelog records a reduction from **568 to 373 public symbols**. Those are historical
counts, not the size of the current API. Many helper implementations remained importable from
their defining module, while renamed and deleted operations required separate changes.

For example, legacy code using `qis.set_spines(ax)` can import the plotting helper explicitly.
This complete example creates its own axes and closes the figure:

~~~python
import matplotlib.pyplot as plt

from qis.plots.utils import set_spines

figure, ax = plt.subplots()
ax.plot([0, 1], [0, 1])
set_spines(ax)
figure.canvas.draw()
plt.close(figure)
~~~

### 1. Renamed (breaking, no shim)

`qis.nanmean` / `nanmedian` / `nansum` shadowed the numpy names while carrying different
semantics: DataFrame in, Series out, non-finite entries excluded, and `axis=1` by default
(the opposite of pandas). The aggregation functions were renamed for consistency; the module remains `qis.utils.df_agg`.

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

`sum_weighted` takes two `pd.Series` and returns a `float`, so its replacement is not
prefixed `df_`. Its first parameter changed from `df` to `data`. `df_last_row` is also an
exception to the Series-returning aggregation pattern: it returns an array and defaults to
`axis=0`, taking each column's last nonmissing value with `is_nonan=True`. Check each operation's
signature rather than applying one rule to the entire table.

---

### 2. Deleted

| Removed symbol | Module and migration note |
| --- | --- |
| `ReportType` | `qis` export; module `qis.plots.reports.utils` deleted |
| `df_price_fill_first_nan_by_cross_median` | `qis` export; deleted; use `qis.utils.df_ops.df_fill_first_nan_by_cross_median` |
| `econ_data_report` | `qis` export; module `qis.plots.reports.econ_data_single` deleted |
| `replace_nan_by_median` | `qis` export; deleted; no replacement |
| `norm_df_by_ax_mean` | `qis.utils.df_ops`; deleted after 5.19; unused |
| `get_data_samples_df` | `qis.utils.sampling`; deleted after 5.19; no direct replacement |
| `split_to_train_live_samples` | `qis.utils.sampling`; deleted after 5.19; no direct replacement |
| `select_non_nan_x_y` | `qis.utils.np_ops`; deleted after 5.19; OLS paths use `qis.utils.regression.filter_x_y` |
| `contour_multi` | `qis.plots.contour`; deleted after 5.19; unused and its defaults were invalid |
| `validate_returns_plot` | `qis.plots.utils`; deleted after 5.19; unused |
| `align_x_limits_ax12` | `qis.plots.utils`; deleted after 5.19; use `align_x_limits_axs` |
| `get_n_mlt_colors` | `qis.plots.utils`; deleted after 5.19; use the active palette helpers |
| `set_column_edge_color` | `qis.plots.table`; deleted after 5.19; unused |
| `plot_exposures_long_short_groups` | `qis.portfolio.reports.strategy_benchmark_tre_factsheet`; deleted after 5.19; unused |
| `dfs_indicators` | `qis.utils.df_ops`; deleted after 5.19; unused |
| `factor_dict_to_asset_dict` | `qis.utils.df_ops`; deleted after 5.19; unused |
| `df_time_dict_to_pd` | `qis.utils.df_ops`; deleted after 5.19; unused |
| `dfs_to_upper_lower_diag` | `qis.utils.df_ops`; deleted after 5.19; unused |
| `df12_merge_with_tz` | `qis.utils.df_ops`; deleted after 5.19; unused |
| `np_matrix_add_array` | `qis.utils.np_ops`; deleted after 5.19; unused and dimensionally defective |
| `to_nearest_values` | `qis.utils.np_ops`; deleted after 5.19; unused |
| `split_dict` | `qis.utils.struct_ops`; deleted after 5.19; unused |
| `shift_time_period_by_days` | `qis.utils.dates`; deleted after 5.19; unused |
| `get_month_days` | `qis.utils.dates`; deleted after 5.19; use `calendar.monthrange` |
| `months_between` | `qis.utils.dates`; deleted after 5.19; unused and incomplete |
| `min_timestamp` | `qis.utils.dates`; deleted after 5.19; unused |
| `WeightMethod` | `qis.utils.df_to_weights`; deleted after 5.19 with its sole consumer |
| `compute_long_only_portfolio_weights` | `qis.utils.df_to_weights`; deleted after 5.19; use the supported allocation helpers |
| `fill_long_short_signal` | `qis.utils.df_to_weights`; deleted after 5.19 with its defective callers |
| `compute_long_short_ind_by_row` | `qis.utils.df_to_weights`; deleted after 5.19; use `df_to_top_bottom_n_indicators` |
| `compute_long_short_ind` | `qis.utils.df_to_weights`; deleted after 5.19; use `df_to_top_bottom_n_indicators` |

This table also records later removals after 5.19. `split_to_samples` remains a calendar-period
slicer; it is not an equivalent train/live splitter.

`qis/plots/reports/` is removed. `price_history.py` and `gantt_data_history.py` moved to
`qis/plots/derived/`; `econ_data_single.py` and `reports/utils.py` are deleted.
`qis.plots.derived.gantt_data_history` is not imported by `qis/plots/__init__.py`
because Plotly is optional. That module requires the `visualization` extra; import it by full path.

---

### 3. Example code, never API

| symbol | file |
| --- | --- |
| `qis.DEFAULT_RA_TABLE_COLUMNS` | [examples/_helpers/reporting_helpers.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/_helpers/reporting_helpers.py) |
| `qis.generate_performance_report` | Same reporting example helper. |
| `qis.load_usd_assets` | [examples/market_data/fx_hedging_example.py](https://github.com/ArturSepp/QuantInvestStrats/blob/main/examples/market_data/fx_hedging_example.py) |

These helpers now live under the repository's `examples/` tree, outside the installed
package. They demonstrate workflows and may be reorganised; do not use them as production APIs.

---

<a id="4-moved-out-of-the-namespace-206-symbols"></a>

## Current module import reference

These grouped imports resolve in the current source. Some names have since returned to
`qis.__all__`; their module paths remain usable. Other entries are internal helpers whose
compatibility is governed by the changelog rather than a promise of a stable top-level API.
The list is a migration aid, not a count of the present public namespace.

### `qis.plots.utils` (42)

```python
from qis.plots.utils import (
    add_scatter_points,
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
    subplot_border
)
```

### `qis.utils.df_ops` (20)

```python
from qis.utils.df_ops import (
    align_df1_to_df2,
    align_dfs_dict_with_df,
    compute_last_score,
    compute_nans_zeros_ratio_after_first_non_nan,
    df_align_to_common_index,
    df_ffill_negatives,
    df_fill_first_nan_by_cross_median,
    df_indicator_like,
    df_joint_indicator,
    df_ones_like,
    df_price_ffill_between_nans,
    df_zero_like,
    drop_first_nan_data,
    get_first_nonnan_values,
    get_last_nonnan,
    get_last_nonnan_values,
    merge_dfs_on_column,
    multiply_df_by_dt,
    np_txy_tensor_to_pd_dict,
    reindex_upto_last_nonnan
)
```

### `qis.utils.np_ops` (21)

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
    np_nanmean,
    np_nanstd,
    np_nansum,
    np_nanvar,
    np_nonan_weighted_avg,
    np_shift,
    repeat_by_columns,
    repeat_by_rows,
    running_mean,
    set_nans_for_warmup_period,
    to_finite_np,
    to_finite_ratio
)
```

### `qis.utils.dates` (11)

```python
from qis.utils.dates import (
    generate_sample_dates,
    get_current_time_with_tz,
    get_sample_dates_idx,
    get_weekday,
    get_year_quarter,
    is_leap_year,
    set_rebalancing_timeindex_on_given_timeindex,
    shift_date_by_day,
    shift_dates_by_n_years,
    shift_dates_by_year,
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

### `qis.utils.struct_ops` (9)

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
    separate_number_from_string
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

### `qis.utils.df_to_weights` (3)

```python
from qis.utils.df_to_weights import (
    df_nans_to_one_zero,
    df_to_top_bottom_n_indicators,
    mult_df_columns_with_vector_group
)
```

### `qis.plots.table` (5)

```python
from qis.plots.table import (
    set_align_for_column,
    set_cells_facecolor,
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

<a id="qisperfstatsconfig-5"></a>

### `qis.perfstats.config` (4)

```python
from qis.perfstats.config import (
    FULL_TABLE_COLUMNS,  # const
    RA_TABLE_COLUMNS,  # const
    RA_TABLE_COMPACT_COLUMNS,  # const
    SD_PERF_COLUMNS  # const
)
```

### `qis.utils.regression` (4)

```python
from qis.utils.regression import (
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

<a id="qisutilssampling-3"></a>

### `qis.utils.sampling` (1)

```python
from qis.utils.sampling import (
    split_to_samples
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

## References

- qis contributors. [Release changelog](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CHANGELOG.md),
  [public API record](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/api.py),
  and [current source tree](https://github.com/ArturSepp/QuantInvestStrats/tree/main/src/qis).
- Sepp, A. qis: Performance analytics, portfolio backtesting, risk analysis, and factsheet
  reporting in Python. [Software citation metadata](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).

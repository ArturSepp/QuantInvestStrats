
from qis.plots.utils import (
    TrendLine,
    LastLabel,
    LegendStats,
    add_scatter_points,
    align_x_limits_ax12,
    align_x_limits_axs,
    align_xy_limits,
    align_y_limits_ax12,
    align_y_limits_axs,
    autolabel,
    compute_heatmap_colors,
    create_dummy_line,
    get_cmap_colors,
    get_data_group_colors,
    get_legend_lines,
    get_n_cmap_colors,
    get_n_colors,
    get_n_fixed_colors,
    get_n_hatch,
    get_n_markers,
    get_n_mlt_colors,
    get_n_sns_colors,
    map_dates_index_to_str,
    rand_cmap,
    remove_spines,
    set_ax_tick_labels,
    set_ax_tick_params,
    set_ax_ticks_format,
    set_ax_xy_labels,
    set_date_on_axis,
    set_legend,
    set_legend_colors,
    set_legend_with_stats_table,
    set_linestyles,
    set_spines,
    set_suptitle,
    set_title,
    set_x_limits,
    set_y_limits,
    subplot_border,
    validate_returns_plot,
    calc_table_height,
    calc_table_width,
    calc_df_table_size,
    get_df_table_size,
    reset_xticks
)

from qis.plots.bars import plot_bars, plot_vbars

from qis.plots.boxplot import (
    plot_box,
    df_boxplot_by_classification_var,
    df_boxplot_by_hue_var,
    df_boxplot_by_index,
    df_boxplot_by_columns,
    df_dict_boxplot_by_columns,
    df_dict_boxplot_by_classification_var
)

from qis.plots.contour import plot_contour

from qis.plots.errorbar import plot_errorbar

from qis.plots.heatmap import plot_heatmap

from qis.plots.histogram import plot_histogram, PdfType

from qis.plots.histplot2d import plot_histplot2d

from qis.plots.lineplot import plot_line

from qis.plots.pie import plot_pie

from qis.plots.qqplot import plot_qq, plot_xy_qq

from qis.plots.scatter import plot_scatter, plot_classification_scatter

from qis.plots.stackplot import plot_stack

from qis.plots.table import (
    plot_df_table,
    plot_df_table_with_ci,
    set_align_for_column,
    set_cells_facecolor,
    set_column_edge_color,
    set_data_colors,
    set_diag_cells_facecolor,
    set_row_edge_color
)

from qis.plots.time_series import (
    plot_lines_list,
    plot_time_series,
    plot_time_series_2ax
)


from qis.plots.derived.prices import (
    add_bnb_regime_shadows,
    get_performance_labels_for_stats,
    get_performance_labels_for_stats,
    PerfStatsLabels,
    plot_prices,
    plot_prices_2ax,
    plot_prices_with_dd,
    plot_prices_with_fundamentals,
    plot_rolling_perf_stat
)

from qis.plots.derived.data_timeseries import plot_data_timeseries


from qis.plots.derived.perf_table import (
    plot_desc_freq_table,
    plot_ra_perf_annual_matrix,
    plot_ra_perf_bars,
    plot_ra_perf_by_dates,
    plot_ra_perf_scatter,
    get_ra_perf_columns,
    plot_ra_perf_table,
    plot_ra_perf_table_benchmark,
    get_ra_perf_benchmark_columns,
    plot_top_bottom_performers
)

from qis.plots.derived.regime_class_table import get_quantile_class_table, plot_quantile_class_table

from qis.plots.derived.regime_pdf import plot_regime_pdf

from qis.plots.derived.regime_scatter import plot_scatter_regression

from qis.plots.derived.returns_heatmap import (
    compute_periodic_returns_by_row_table,
    compute_periodic_returns_table,
    compute_periodic_returns,
    plot_periodic_returns_table,
    plot_returns_heatmap,
    plot_returns_table,
    plot_sorted_periodic_returns
)

from qis.plots.derived.returns_scatter import plot_returns_scatter


from qis.plots.derived.drawdowns import (
    DdLegendType,
    plot_rolling_drawdowns,
    plot_rolling_time_under_water,
    plot_top_drawdowns_paths
)

from qis.plots.derived.regime_data import (
    plot_regime_data,
    plot_regime_boxplot,
    add_bnb_regime_shadows
)

from qis.plots.derived.desc_table import plot_desc_table

from qis.plots.reports.utils import ReportType
from qis.plots.reports.econ_data_single import econ_data_report

"""
public plotting API of qis

axis, legend, colour and table-styling helpers are internal machinery: they are not
exported here and are imported by full path, e.g. `from qis.plots.utils import set_spines`
"""

# note: qis.plots.derived.gantt_data_history is NOT imported here: it requires plotly,
# which is not a qis dependency. import it by full path if plotly is installed.

from qis.plots.utils import (
    TrendLine,
    LastLabel,
    LegendStats,
    set_suptitle
)

from qis.plots.bars import (
    plot_bars,
    plot_vbars
)

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

from qis.plots.histogram import (
    plot_histogram,
    PdfType
)

from qis.plots.histplot2d import plot_histplot2d

from qis.plots.lineplot import (
    plot_line,
    plot_lines_list
)

from qis.plots.pie import plot_pie

from qis.plots.qqplot import (
    plot_qq,
    plot_xy_qq
)

from qis.plots.scatter import (
    plot_scatter,
    plot_classification_scatter,
    plot_multivariate_scatter_with_prediction
)

from qis.plots.stackplot import plot_stack

from qis.plots.table import (
    plot_df_table,
    plot_df_table_with_ci
)

from qis.plots.time_series import (
    plot_time_series,
    plot_time_series_2ax
)

from qis.plots.derived.prices import (
    add_bnb_regime_shadows,
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
    plot_top_bottom_performers,
    plot_best_worst_returns
)

from qis.plots.derived.regime_class_table import plot_quantile_class_table

from qis.plots.derived.regime_pdf import plot_regime_pdf

from qis.plots.derived.regime_scatter import plot_scatter_regression

from qis.plots.derived.returns_heatmap import (
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
    plot_regime_boxplot
)

from qis.plots.derived.price_history import (
    plot_price_history,
    generate_price_history_report
)

from qis.plots.derived.signal_diagnostics_plot import (
    plot_signal_diagnostics,
    plot_signal_diagnostics_boxplot,
    plot_signal_diagnostics_group_boxplot,
    plot_signal_diagnostics_for_returns,
    plot_signal_diagnostics_beta_boxplot
)
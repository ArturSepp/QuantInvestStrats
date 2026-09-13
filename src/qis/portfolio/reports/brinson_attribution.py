"""
Brinson Performance Attribution Analysis

This module implements the Brinson performance attribution model for decomposing
active returns between portfolios and benchmarks into allocation and selection effects.

References:
    Brinson, G.P., Hood, L.R., & Beebower, G.L. (1986). Determinants of portfolio performance.
    https://en.wikipedia.org/wiki/Performance_attribution
"""
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

import qis as qis
from qis.portfolio.attribution.brinson import (
    compute_brinson_attribution_table as compute_brinson_attribution_table,
)
import qis.plots.time_series as pts
from qis.plots.table import plot_df_table


def plot_brinson_attribution_table(
    totals_table: pd.DataFrame,
    active_total: pd.DataFrame,
    grouped_allocation_return: pd.DataFrame,
    grouped_selection_return: pd.DataFrame,
    grouped_interaction_return: pd.DataFrame,
    var_format: str = '{:.0%}',
    total_column: str = 'Total Sum',
    is_exclude_interaction_term: bool = True,
    axs: List[plt.Subplot] = (None, None, None, None, None),
    **kwargs
) -> Tuple[plt.Figure, plt.Figure, plt.Figure, plt.Figure, plt.Figure]:
    """Create comprehensive visualization of Brinson attribution results.

    Generates a multi-panel visualization including:
    1. Summary table with attribution statistics
    2. Time series of cumulative total attribution effects
    3. Time series of cumulative active effects by asset class when interaction is excluded
    4. Time series of cumulative allocation effects by asset class
    5. Time series of cumulative selection effects by asset class

    When interaction is reported separately, panel 3 is omitted and panel 5 shows cumulative
    interaction effects instead.

    Args:
        totals_table: Summary statistics table from compute_brinson_attribution_table.
        active_total: Time series of total attribution effects.
        grouped_allocation_return: Allocation effects by asset class over time.
        grouped_selection_return: Selection effects by asset class over time.
        grouped_interaction_return: Interaction effects by asset class over time.
        var_format: Format string for displaying numeric values (default: percentage).
        total_column: Name of the total column for portfolio-level aggregation.
        is_exclude_interaction_term: Whether interaction terms are assigned entirely to
            instrument selection instead of reported separately.
        axs: Optional list of matplotlib axes for plotting (if None, creates new figures).
        **kwargs: Additional arguments passed to plotting functions.

    Returns:
        Tuple of matplotlib figures: (table_fig, active_fig, allocation_fig, selection_fig,
            final_fig). ``final_fig`` is grouped active effects when interaction is excluded and
            interaction effects otherwise.

    Example:
        >>> # After running compute_brinson_attribution_table
        >>> figs = plot_brinson_attribution_table(
        ...     totals_table, active_total, allocation_return,
        ...     selection_return, interaction_return
        ... )
        >>> table_fig, active_fig, alloc_fig, select_fig, final_fig = figs
        >>> plt.show()
    """
    # Generate formatted summary table
    fig_table = plot_brinson_totals_table(
        totals_table=totals_table,
        var_format=var_format,
        ax=axs[0],
        **kwargs
    )

    # Linked inputs are already Frongello-adjusted increments; cumsum links only once.
    active_total_cumsum = active_total.cumsum(axis=0)
    fig_active_total = pts.plot_time_series(
        df=active_total_cumsum,
        var_format='{:.0%}',
        title='Cumulative Active Attribution Effects',
        legend_stats=qis.LegendStats.LAST_NONNAN,
        ax=axs[1],
        **kwargs
    )

    if is_exclude_interaction_term:
        grouped_active_return = (
            grouped_allocation_return + grouped_selection_return
        ).drop(columns=total_column, errors='ignore')
        fig_ts_final = pts.plot_time_series(
            df=grouped_active_return.cumsum(axis=0),
            var_format='{:.0%}',
            title='Total Cumulative Active Effects by Groups',
            legend_stats=qis.LegendStats.LAST_NONNAN,
            ax=axs[2],
            **kwargs
        )
        allocation_ax = axs[3]
        selection_ax = axs[4]
    else:
        allocation_ax = axs[2]
        selection_ax = axs[3]

    # Plot cumulative allocation effects by asset class
    cum_allocation_return = grouped_allocation_return.cumsum(axis=0)
    fig_ts_alloc = pts.plot_time_series(
        df=cum_allocation_return,
        var_format='{:.0%}',
        title='Cumulative Asset Class Allocation Effects',
        legend_stats=qis.LegendStats.LAST_NONNAN,
        ax=allocation_ax,
        **kwargs
    )

    # Plot cumulative selection effects by asset class
    cum_selection_return = grouped_selection_return.cumsum(axis=0)
    fig_ts_sel = pts.plot_time_series(
        df=cum_selection_return,
        var_format='{:.0%}',
        title='Cumulative Instrument Selection Effects',
        legend_stats=qis.LegendStats.LAST_NONNAN,
        ax=selection_ax,
        **kwargs
    )

    if not is_exclude_interaction_term:
        cum_interaction_return = grouped_interaction_return.cumsum(axis=0)
        fig_ts_final = pts.plot_time_series(
            df=cum_interaction_return,
            trend_line=pts.TrendLine.TREND_LINE,
            var_format='{:.0%}',
            title='Cumulative Asset Class Interaction Effects',
            ax=axs[4],
            **kwargs
        )

    return fig_table, fig_active_total, fig_ts_alloc, fig_ts_sel, fig_ts_final


def plot_brinson_totals_table(
    totals_table: pd.DataFrame,
    var_format: str = '{:.0%}',
    ax: Optional[plt.Subplot] = None,
    **kwargs
) -> Optional[plt.Figure]:
    """Create formatted table visualization of Brinson attribution summary.

    Generates a professional-looking table with:
    - Color-coded sections for different data types
    - Edge lines separating logical groups
    - Formatted numeric values
    - Highlighted totals row

    Args:
        totals_table: Summary statistics DataFrame with attribution results.
        var_format: String format for numeric values (default: percentage format).
        ax: Optional matplotlib axis for plotting. If None, creates new figure.
        **kwargs: Additional arguments passed to plot_df_table function.

    Returns:
        matplotlib Figure object containing the formatted table.

    Example:
        >>> fig = plot_brinson_totals_table(totals_table, var_format='{:.2%}')
        >>> plt.show()
    """
    # Define visual formatting for the table
    # Highlight the totals row at the bottom
    special_rows_colors = [(len(totals_table.index), 'steelblue')]
    rows_edge_lines = [len(totals_table.index) - 1]  # Separator line before totals

    # Color-code different column groups
    special_columns_colors = [
        (0, 'lightblue'),  # First column (asset class names)
        (len(totals_table.columns), 'steelblue')  # Last column
    ]

    # Add vertical separator lines between logical column groups
    columns_edge_lines = [
        (1, 'black'),  # After asset class names
        (3, 'black'),  # After weight columns
        (5, 'black'),  # After return columns
        (8, 'black')   # After attribution columns
    ]

    # Format all numeric values according to specified format
    totals_table_formatted = qis.df_to_str(df=totals_table, var_format=var_format)

    # Generate the formatted table plot
    fig_table = plot_df_table(
        df=totals_table_formatted,
        column_width=2.0,
        first_column_width=2.0,
        special_rows_colors=special_rows_colors,
        rows_edge_lines=rows_edge_lines,
        special_columns_colors=special_columns_colors,
        columns_edge_lines=columns_edge_lines,
        ax=ax,
        **kwargs
    )
    return fig_table

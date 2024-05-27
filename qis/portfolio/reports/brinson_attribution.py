"""
compute brinson attribution table
see https://en.wikipedia.org/wiki/Performance_attribution
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

# qis
import qis.utils.df_groups as dfg
import qis.utils.df_agg as dfa
import qis.plots.time_series as pts
from qis.plots.table import plot_df_table


def compute_brinson_attribution_table(benchmark_pnl: pd.DataFrame,
                                      strategy_pnl: pd.DataFrame,
                                      strategy_weights: pd.DataFrame,
                                      benchmark_weights: pd.DataFrame,
                                      asset_class_data: pd.Series,
                                      group_order: List[str] = None,
                                      total_column: str = 'Total Sum',
                                      is_exclude_interaction_term: bool = True
                                      ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    brinson attribution
    """
    # 1 get grouped pnl
    grouped_strategy_pnl = dfg.agg_df_by_groups(df=strategy_pnl,
                                                group_data=asset_class_data,
                                                group_order=group_order,
                                                agg_func=dfa.nansum)

    grouped_benchmark_pnl = dfg.agg_df_by_groups(df=benchmark_pnl,
                                                 group_data=asset_class_data,
                                                 group_order=group_order,
                                                 agg_func=dfa.nansum)

    # 2. get grouped weights
    grouped_strategy_weights = dfg.agg_df_by_groups(df=strategy_weights,
                                                    group_data=asset_class_data,
                                                    group_order=group_order,
                                                    agg_func=dfa.nansum)

    grouped_benchmark_weights = dfg.agg_df_by_groups(df=benchmark_weights,
                                                     group_data=asset_class_data,
                                                     group_order=group_order,
                                                     agg_func=dfa.nansum)

    # active return by instrument
    active_return = strategy_pnl - benchmark_pnl
    # active return by group
    is_revaluate_active_return = False
    if is_revaluate_active_return:
        grouped_active_return = dfg.agg_df_by_groups(df=active_return,
                                                     group_data=asset_class_data,
                                                     group_order=group_order,
                                                     agg_func=dfa.nansum)
    else:
        grouped_active_return = grouped_strategy_pnl - grouped_benchmark_pnl

    # allocation return by group
    grouped_allocation_return = (grouped_strategy_weights - grouped_benchmark_weights) * grouped_benchmark_pnl

    # selection return by group
    grouped_selection_return = grouped_benchmark_weights * (grouped_strategy_pnl - grouped_benchmark_pnl)

    # interaction by group
    grouped_interaction_return = grouped_active_return - (grouped_allocation_return + grouped_selection_return)

    if is_exclude_interaction_term:
        # exlude by allocation half to allocation and selection
        grouped_allocation_return = grouped_allocation_return + 0.5 * grouped_interaction_return
        grouped_selection_return = grouped_selection_return + 0.5 * grouped_interaction_return
        grouped_interaction_return = 0.0 * grouped_interaction_return

    # create mean table: indexed by group
    totals_table = pd.DataFrame(index=grouped_strategy_pnl.columns.to_list() + [total_column])

    totals_table['Strategy\nWeight Ave'] = dfa.agg_data_by_axis(df=grouped_strategy_weights,
                                                                agg_func=np.nanmean, total_column=total_column)
    totals_table['Benchmark\nWeight Ave'] = dfa.agg_data_by_axis(df=grouped_benchmark_weights,
                                                                 agg_func=np.nanmean, total_column=total_column)
    totals_table['Strategy\nReturn Sum'] = dfa.agg_data_by_axis(df=grouped_strategy_pnl,
                                                                agg_func=np.nansum, total_column=total_column)
    totals_table['Benchmark\nReturn Sum'] = dfa.agg_data_by_axis(df=grouped_benchmark_pnl,
                                                                 agg_func=np.nansum, total_column=total_column)
    totals_table['Asset\nAllocation'] = dfa.agg_data_by_axis(df=grouped_allocation_return,
                                                             agg_func=np.nansum, total_column=total_column)
    totals_table['Instrument\nSelection'] = dfa.agg_data_by_axis(df=grouped_selection_return,
                                                                 agg_func=np.nansum, total_column=total_column)

    if not is_exclude_interaction_term:
        totals_table['Interaction'] = dfa.agg_data_by_axis(df=grouped_interaction_return,
                                                           agg_func=np.nansum, total_column=total_column)

    totals_table['Total\nActive'] = dfa.agg_data_by_axis(df=grouped_active_return,
                                                         agg_func=np.nansum, total_column=total_column)

    # now enter totals for grouped pnls
    grouped_allocation_return[total_column] = np.sum(grouped_allocation_return, axis=1)
    grouped_selection_return[total_column] = np.sum(grouped_selection_return, axis=1)
    grouped_interaction_return[total_column] = np.sum(grouped_interaction_return, axis=1)

    # get total of allocation and selection:
    allocation_total = grouped_allocation_return[total_column].to_frame(name='Allocation Total')
    selection_total = grouped_selection_return[total_column].to_frame(name='Selection Total')
    active_total = pd.concat([allocation_total, selection_total], axis=1)
    if not is_exclude_interaction_term:
        interaction_total = grouped_interaction_return[total_column].to_frame(name='Interaction Total')
        active_total = pd.concat([active_total, interaction_total], axis=1)

    return totals_table, active_total, grouped_allocation_return, grouped_selection_return, grouped_interaction_return


def plot_brinson_attribution_table(totals_table: pd.DataFrame,
                                   active_total: pd.DataFrame,
                                   grouped_allocation_return: pd.DataFrame,
                                   grouped_selection_return: pd.DataFrame,
                                   grouped_interaction_return: pd.DataFrame,
                                   var_format: str = '{:.0%}',
                                   total_column: str = 'Total Sum',
                                   is_exclude_interaction_term: bool = True,
                                   axs: List[plt.Subplot] = (None, None, None, None, None),
                                   **kwargs):
    for column in totals_table:
        totals_table[column] = totals_table[column].apply(lambda x: var_format.format(x))

    special_rows_colors = [(len(totals_table.index), 'steelblue')]
    rows_edge_lines = [len(totals_table.columns)]  # line before totals

    special_columns_colors = [(0, 'lightblue'),
                              (len(totals_table.columns), 'steelblue')]
    columns_edge_lines = [(1, 'black'),
                          (3, 'black'),
                          (5, 'black'),
                          (8, 'black')]

    fig_table = plot_df_table(df=totals_table,
                              column_width=2.0,
                              first_column_width=2.0,
                              special_rows_colors=special_rows_colors,
                              rows_edge_lines=rows_edge_lines,
                              special_columns_colors=special_columns_colors,
                              columns_edge_lines=columns_edge_lines,
                              ax=axs[0],
                              **kwargs)

    # get total of allocation and selection:
    active_total = active_total.cumsum(axis=0)
    legend_labels = [column + ', sum=' + '{:.0%}'.format(active_total[column].iloc[-1]) for column in active_total.columns]
    fig_active_total = pts.plot_time_series(df=active_total,
                                            var_format='{:.0%}',
                                            title='Active total',
                                            legend_labels=legend_labels,
                                            ax=axs[1],
                                            **kwargs)

    cum_grouped_ac_pnl_diff = grouped_allocation_return.cumsum(axis=0)
    fig_ts_alloc = pts.plot_time_series(df=cum_grouped_ac_pnl_diff,
                                        var_format='{:.0%}',
                                        title='Asset class allocation return',
                                        ax=axs[2],
                                        **kwargs)

    cum_grouped_ac_pnl_diff = grouped_selection_return.cumsum(axis=0)
    fig_ts_sel = pts.plot_time_series(df=cum_grouped_ac_pnl_diff,
                                      var_format='{:.0%}',
                                      title='Asset class selection return',
                                      ax=axs[3],
                                      **kwargs)

    cum_grouped_ac_pnl_diff = grouped_interaction_return.cumsum(axis=0)
    fig_ts_inter = pts.plot_time_series(df=cum_grouped_ac_pnl_diff,
                                        trend_line=pts.TrendLine.TREND_LINE,
                                        var_format='{:.0%}',
                                        title='Asset class interaction return',
                                        ax=axs[4],
                                        **kwargs)

    return fig_table, fig_active_total, fig_ts_alloc, fig_ts_sel, fig_ts_inter

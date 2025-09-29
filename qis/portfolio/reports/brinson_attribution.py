"""
Brinson Performance Attribution Analysis

This module implements the Brinson performance attribution model for decomposing
active returns between portfolios and benchmarks into allocation and selection effects.

References:
    Brinson, G.P., Hood, L.R., & Beebower, G.L. (1986). Determinants of portfolio performance.
    https://en.wikipedia.org/wiki/Performance_attribution
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from enum import Enum

import qis as qis
from qis.utils.df_groups import agg_df_by_groups
from qis.utils.struct_ops import merge_lists_unique
import qis.utils.df_agg as dfa
import qis.plots.time_series as pts
from qis.plots.table import plot_df_table


def compute_brinson_attribution_table(
    benchmark_pnl: pd.DataFrame,
    strategy_pnl: pd.DataFrame,
    strategy_weights: pd.DataFrame,
    benchmark_weights: pd.DataFrame,
    asset_class_data: pd.Series,
    group_order: List[str] = None,
    total_column: str = 'Total Sum',
    is_exclude_interaction_term: bool = True,
    strategy_name: str = 'Strategy',
    benchmark_name: str = 'Benchmark'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute Brinson performance attribution decomposition.

    Decomposes active returns into allocation and selection effects using the Brinson model:
    - Allocation Effect: (w_p - w_b) × r_b (overweight/underweight better/worse performers)
    - Selection Effect: w_b × (r_p - r_b) (security selection within asset classes)
    - Interaction Effect: (w_p - w_b) × (r_p - r_b) (combined allocation & selection)

    Where:
        w_p, w_b = portfolio and benchmark weights
        r_p, r_b = portfolio and benchmark returns

    Args:
        benchmark_pnl: Benchmark P&L by asset over time. Shape: (T, N_assets).
        strategy_pnl: Strategy P&L by asset over time. Shape: (T, N_assets).
        strategy_weights: Strategy weights by asset over time. Shape: (T, N_assets).
        benchmark_weights: Benchmark weights by asset over time. Shape: (T, N_assets).
        asset_class_data: Mapping of assets to asset classes. Index: assets, Values: classes.
        group_order: Optional ordering for asset classes in output tables.
        total_column: Name for the total/summary column in outputs.
        is_exclude_interaction_term: If True, allocates interaction effect to selection.
        strategy_name: Display name for strategy in output tables.
        benchmark_name: Display name for benchmark in output tables.

    Returns:
        Tuple containing:
            - totals_table: Summary statistics by asset class (weights, returns, attribution)
            - active_total: Time series of total allocation and selection effects
            - grouped_allocation_return: Allocation effect by asset class over time
            - grouped_selection_return: Selection effect by asset class over time
            - grouped_interaction_return: Interaction effect by asset class over time

    Raises:
        ValueError: If input DataFrames have mismatched dimensions or invalid data.

    Example:
        >>> benchmark_pnl = pd.DataFrame(...)  # Daily benchmark returns by asset
        >>> strategy_pnl = pd.DataFrame(...)   # Daily strategy returns by asset
        >>> strategy_weights = pd.DataFrame(...)  # Daily strategy weights by asset
        >>> benchmark_weights = pd.DataFrame(...) # Daily benchmark weights by asset
        >>> asset_classes = pd.Series({'AAPL': 'Equities', 'TLT': 'Bonds', ...})
        >>>
        >>> results = compute_brinson_attribution_table(
        ...     benchmark_pnl, strategy_pnl, strategy_weights,
        ...     benchmark_weights, asset_classes
        ... )
        >>> totals, active_ts, allocation, selection, interaction = results
    """
    # Data alignment and preprocessing
    # Create unified asset universe by merging strategy and benchmark assets
    joint_assets = merge_lists_unique(
        benchmark_pnl.columns.to_list(),
        strategy_pnl.columns.to_list()
    )

    # Ensure all datasets have consistent asset coverage
    # Fill missing assets with appropriate defaults (0 for weights/PnL, 'Unclassified' for classes)
    asset_class_data = asset_class_data.reindex(index=joint_assets).fillna('Unclassified')
    strategy_pnl = strategy_pnl.reindex(columns=joint_assets).fillna(0.0)
    benchmark_pnl = benchmark_pnl.reindex(columns=joint_assets).fillna(0.0)
    strategy_weights = strategy_weights.reindex(columns=joint_assets).fillna(0.0)
    benchmark_weights = benchmark_weights.reindex(columns=joint_assets).fillna(0.0)

    # Asset class aggregation
    # Group individual assets into asset classes and sum P&L within each class
    grouped_strategy_pnl = agg_df_by_groups(
        df=strategy_pnl,
        group_data=asset_class_data,
        group_order=group_order,
        agg_func=dfa.nansum
    )

    grouped_benchmark_pnl = agg_df_by_groups(
        df=benchmark_pnl,
        group_data=asset_class_data,
        group_order=group_order,
        agg_func=dfa.nansum
    )

    # Aggregate weights by asset class (sum individual asset weights within class)
    grouped_strategy_weights = agg_df_by_groups(
        df=strategy_weights,
        group_data=asset_class_data,
        group_order=group_order,
        agg_func=dfa.nansum
    )

    grouped_benchmark_weights = agg_df_by_groups(
        df=benchmark_weights,
        group_data=asset_class_data,
        group_order=group_order,
        agg_func=dfa.nansum
    )

    # Core Brinson attribution calculations
    # Active return computation (two equivalent approaches)
    active_return = strategy_pnl - benchmark_pnl

    # Flag to choose between direct aggregation vs grouped calculation
    # Both should yield identical results - using grouped for consistency
    is_revaluate_active_return = False
    if is_revaluate_active_return:
        # Approach 1: Aggregate individual asset active returns
        grouped_active_return = agg_df_by_groups(
            df=active_return,
            group_data=asset_class_data,
            group_order=group_order,
            agg_func=dfa.nansum
        )
    else:
        # Approach 2: Calculate active return from grouped P&L (preferred)
        grouped_active_return = grouped_strategy_pnl - grouped_benchmark_pnl

    # Brinson allocation effect: (w_strategy - w_benchmark) × r_benchmark
    # Measures value added/lost from asset allocation decisions
    grouped_allocation_return = (
        (grouped_strategy_weights - grouped_benchmark_weights) * grouped_benchmark_pnl
    )

    # Brinson selection effect: w_benchmark × (r_strategy - r_benchmark)
    # Measures value added/lost from security selection within asset classes
    grouped_selection_return = (
        grouped_benchmark_weights * (grouped_strategy_pnl - grouped_benchmark_pnl)
    )

    # Interaction effect: Residual not captured by allocation + selection
    # Represents the interaction between allocation and selection decisions
    grouped_interaction_return = (
        grouped_active_return - (grouped_allocation_return + grouped_selection_return)
    )

    # Interaction term handling
    if is_exclude_interaction_term:
        # Common practice: Allocate interaction effect to selection component
        # Alternative approaches: split 50/50 between allocation and selection
        grouped_selection_return = grouped_selection_return + grouped_interaction_return
        grouped_interaction_return = 0.0 * grouped_interaction_return

    # Summary statistics table creation
    # Create comprehensive summary with time-aggregated statistics by asset class
    totals_table = pd.DataFrame(index=grouped_strategy_pnl.columns.to_list() + [total_column])

    # Average weights over the analysis period
    totals_table[f"{strategy_name}\nWeight Ave"] = dfa.agg_data_by_axis(
        df=grouped_strategy_weights,
        agg_func=np.nanmean,
        total_column=total_column
    )
    totals_table[f"{benchmark_name}\nWeight Ave"] = dfa.agg_data_by_axis(
        df=grouped_benchmark_weights,
        agg_func=np.nanmean,
        total_column=total_column
    )

    # Cumulative returns over the analysis period
    totals_table[f"{strategy_name}\nReturn Sum"] = dfa.agg_data_by_axis(
        df=grouped_strategy_pnl,
        agg_func=np.nansum,
        total_column=total_column
    )
    totals_table[f"{benchmark_name}\nReturn Sum"] = dfa.agg_data_by_axis(
        df=grouped_benchmark_pnl,
        agg_func=np.nansum,
        total_column=total_column
    )

    # Attribution effects summed over the analysis period
    totals_table['Asset\nAllocation'] = dfa.agg_data_by_axis(
        df=grouped_allocation_return,
        agg_func=np.nansum,
        total_column=total_column
    )
    totals_table['Instrument\nSelection'] = dfa.agg_data_by_axis(
        df=grouped_selection_return,
        agg_func=np.nansum,
        total_column=total_column
    )

    # Include interaction term in summary if not excluded
    if not is_exclude_interaction_term:
        totals_table['Interaction'] = dfa.agg_data_by_axis(
            df=grouped_interaction_return,
            agg_func=np.nansum,
            total_column=total_column
        )

    # Total active return for verification (should equal allocation + selection + interaction)
    totals_table['Total\nActive'] = dfa.agg_data_by_axis(
        df=grouped_active_return,
        agg_func=np.nansum,
        total_column=total_column
    )

    # Time series preparation
    # Add portfolio-level totals (sum across all asset classes) for time series analysis
    grouped_allocation_return[total_column] = np.sum(grouped_allocation_return, axis=1)
    grouped_selection_return[total_column] = np.sum(grouped_selection_return, axis=1)
    grouped_interaction_return[total_column] = np.sum(grouped_interaction_return, axis=1)

    # Create time series of total attribution effects for plotting
    allocation_total = grouped_allocation_return[total_column].to_frame(name='Allocation Total')
    selection_total = grouped_selection_return[total_column].to_frame(name='Selection Total')
    active_total = pd.concat([allocation_total, selection_total], axis=1)

    if not is_exclude_interaction_term:
        interaction_total = grouped_interaction_return[total_column].to_frame(name='Interaction Total')
        active_total = pd.concat([active_total, interaction_total], axis=1)

    return (totals_table, active_total, grouped_allocation_return,
            grouped_selection_return, grouped_interaction_return)


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
    3. Time series of cumulative allocation effects by asset class
    4. Time series of cumulative selection effects by asset class
    5. Time series of cumulative interaction effects by asset class

    Args:
        totals_table: Summary statistics table from compute_brinson_attribution_table.
        active_total: Time series of total attribution effects.
        grouped_allocation_return: Allocation effects by asset class over time.
        grouped_selection_return: Selection effects by asset class over time.
        grouped_interaction_return: Interaction effects by asset class over time.
        var_format: Format string for displaying numeric values (default: percentage).
        total_column: Name of the total column for portfolio-level aggregation.
        is_exclude_interaction_term: Whether interaction terms are excluded from analysis.
        axs: Optional list of matplotlib axes for plotting (if None, creates new figures).
        **kwargs: Additional arguments passed to plotting functions.

    Returns:
        Tuple of matplotlib figures: (table_fig, active_fig, allocation_fig,
                                    selection_fig, interaction_fig)

    Example:
        >>> # After running compute_brinson_attribution_table
        >>> figs = plot_brinson_attribution_table(
        ...     totals_table, active_total, allocation_return,
        ...     selection_return, interaction_return
        ... )
        >>> table_fig, active_fig, alloc_fig, select_fig, interact_fig = figs
        >>> plt.show()
    """
    # Generate formatted summary table
    fig_table = plot_brinson_totals_table(
        totals_table=totals_table,
        var_format=var_format,
        ax=axs[0],
        **kwargs
    )

    # Plot cumulative total attribution effects over time
    # Convert to cumulative for better visualization of performance evolution
    active_total_cumsum = active_total.cumsum(axis=0)
    fig_active_total = pts.plot_time_series(
        df=active_total_cumsum,
        var_format='{:.0%}',
        title='Cumulative Active Attribution Effects',
        legend_stats=qis.LegendStats.LAST_NONNAN,
        ax=axs[1],
        **kwargs
    )

    # Plot cumulative allocation effects by asset class
    cum_allocation_return = grouped_allocation_return.cumsum(axis=0)
    fig_ts_alloc = pts.plot_time_series(
        df=cum_allocation_return,
        var_format='{:.0%}',
        title='Cumulative Asset Class Allocation Effects',
        legend_stats=qis.LegendStats.LAST_NONNAN,
        ax=axs[2],
        **kwargs
    )

    # Plot cumulative selection effects by asset class
    cum_selection_return = grouped_selection_return.cumsum(axis=0)
    fig_ts_sel = pts.plot_time_series(
        df=cum_selection_return,
        var_format='{:.0%}',
        title='Cumulative Asset Class Selection Effects',
        legend_stats=qis.LegendStats.LAST_NONNAN,
        ax=axs[3],
        **kwargs
    )

    # Plot cumulative interaction effects by asset class (with trend line)
    cum_interaction_return = grouped_interaction_return.cumsum(axis=0)
    fig_ts_inter = pts.plot_time_series(
        df=cum_interaction_return,
        trend_line=pts.TrendLine.TREND_LINE,
        var_format='{:.0%}',
        title='Cumulative Asset Class Interaction Effects',
        ax=axs[4],
        **kwargs
    )

    return fig_table, fig_active_total, fig_ts_alloc, fig_ts_sel, fig_ts_inter


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


class LocalTests(Enum):
    """Enumeration of available local test scenarios."""
    ATTRIBUTION = 1


def run_local_test(local_test: LocalTests) -> None:
    """Execute local integration tests for development and validation.

    Provides comprehensive test scenarios with realistic data to validate
    the Brinson attribution implementation. These tests include edge cases
    such as assets present in only one portfolio (strategy vs benchmark).

    Args:
        local_test: Test scenario to execute (currently only ATTRIBUTION available).

    Example:
        >>> run_local_test(LocalTests.ATTRIBUTION)
        # Executes attribution test with mixed asset universe and displays results
    """
    if local_test == LocalTests.ATTRIBUTION:
        # Test data setup with realistic scenarios
        dates = pd.date_range('2024-01-01', periods=3, freq='D')

        # Define asset class ordering for consistent output formatting
        group_order = ['Equities', 'Bonds', 'ALT_IN_STRATEGY', 'COMMODITIES_IN_BENCHMARK']

        # Asset class mapping including assets unique to each portfolio
        asset_classes = pd.Series({
            'STOCK_A': 'Equities',
            'STOCK_B': 'Equities',
            'BOND_A': 'Bonds',
            'BOND_B': 'Bonds',
            'ALT_IN_STRATEGY': 'ALT_IN_STRATEGY',  # Only in strategy portfolio
            'COMMODITIES_IN_BENCHMARK': 'COMMODITIES_IN_BENCHMARK'  # Only in benchmark
        })

        # Strategy P&L data (equities outperform, alternatives included)
        strategy_pnl = pd.DataFrame({
            'STOCK_A': [0.02, 0.015, 0.01],     # Strong equity performance
            'STOCK_B': [0.018, 0.012, 0.008],   # Strong equity performance
            'BOND_A': [0.005, 0.003, 0.002],    # Modest bond performance
            'BOND_B': [0.004, 0.002, 0.001],    # Modest bond performance
            'ALT_IN_STRATEGY': [0.04, 0.02, 0.01]  # High alternative returns
        }, index=dates)

        # Benchmark P&L data (similar patterns, commodities included)
        benchmark_pnl = pd.DataFrame({
            'STOCK_A': [0.018, 0.013, 0.009],
            'STOCK_B': [0.016, 0.010, 0.007],
            'BOND_A': [0.004, 0.002, 0.001],
            'BOND_B': [0.003, 0.001, 0.0005],
            'COMMODITIES_IN_BENCHMARK': [0.03, 0.01, 0.005]  # Commodity exposure
        }, index=dates)

        # Strategy weights (overweight equities, include alternatives)
        strategy_weights = pd.DataFrame({
            'STOCK_A': [0.35, 0.35, 0.35],                    # Overweight equities
            'STOCK_B': [0.25, 0.25, 0.25],                    # Overweight equities
            'BOND_A': [0.25, 0.25, 0.25],                     # Standard bond allocation
            'BOND_B': [0.15 / 2.0, 0.15 / 2.0, 0.15 / 2.0],   # Split remaining weight
            'ALT_IN_STRATEGY': [0.15 / 2.0, 0.15 / 2.0, 0.15 / 2.0]  # Alternative allocation
        }, index=dates)

        # Benchmark weights (balanced allocation, include commodities)
        benchmark_weights = pd.DataFrame({
            'STOCK_A': [0.25, 0.25, 0.25],                    # Equal equity weight
            'STOCK_B': [0.25, 0.25, 0.25],                    # Equal equity weight
            'BOND_A': [0.25, 0.25, 0.25],                     # Equal bond weight
            'BOND_B': [0.25 / 2.0, 0.25 / 2.0, 0.25 / 2.0],   # Split remaining weight
            'COMMODITIES_IN_BENCHMARK': [0.25 / 2.0, 0.25 / 2.0, 0.25 / 2.0]  # Commodity allocation
        }, index=dates)

        # Execute Brinson attribution analysis
        (totals_table, active_total, grouped_allocation_return,
         grouped_selection_return, grouped_interaction_return) = compute_brinson_attribution_table(
            benchmark_pnl=benchmark_pnl,
            strategy_pnl=strategy_pnl,
            strategy_weights=strategy_weights,
            benchmark_weights=benchmark_weights,
            asset_class_data=asset_classes,
            group_order=group_order
        )

        # Display results
        print("=== BRINSON ATTRIBUTION ANALYSIS RESULTS ===")
        print(totals_table)

        # Generate and display summary table visualization
        plot_brinson_totals_table(totals_table=totals_table)
        plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.ATTRIBUTION)

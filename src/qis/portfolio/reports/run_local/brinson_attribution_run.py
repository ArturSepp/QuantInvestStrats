"""Development runner extracted from ``qis.portfolio.reports.brinson_attribution``."""

import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum

from qis.portfolio.reports.brinson_attribution import (
    compute_brinson_attribution_table,
    plot_brinson_totals_table,
)

class Locals(Enum):
    """Enumeration of available local test scenarios."""
    ATTRIBUTION = 1

def run_local(local: Locals) -> None:
    """Execute local integration tests for development and validation.

    Provides comprehensive test scenarios with realistic data to validate
    the Brinson attribution implementation. These tests include edge cases
    such as assets present in only one portfolio (strategy vs benchmark).

    Args:
        local: Test scenario to execute (currently only ATTRIBUTION available).

    Example:
        >>> run_local(Locals.ATTRIBUTION)
        # Executes attribution test with mixed asset universe and displays results
    """
    if local == Locals.ATTRIBUTION:
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

if __name__ == "__main__":
    run_local(local=Locals.ATTRIBUTION)

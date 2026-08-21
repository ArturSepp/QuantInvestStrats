"""Development runner extracted from ``qis.portfolio.risk.contributions``."""

import numpy as np
import pandas as pd
from enum import Enum

from qis.portfolio.risk.contributions import (
    calculate_active_risk_squared,
    calculate_marginal_active_risk,
)

class Locals(Enum):
    """Enumeration of available local diagnostic cases."""

    MARGINAL_ACTIVE_RISK = 1

def run_local(local: Locals) -> None:
    """Run one print-driven local diagnostic.

    Args:
        local: Diagnostic case to run.
    """
    if local != Locals.MARGINAL_ACTIVE_RISK:
        return

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    print("=== Marginal Active Risk Demo ===\n")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Portfolio setup
    n_assets = 6
    n_factors = 3

    asset_names = [f'Stock_{chr(65 + i)}' for i in range(n_assets)]  # Stock_A, Stock_B, etc.
    factor_names = ['Market', 'Value', 'Size']

    print(f"Portfolio: {n_assets} assets, {n_factors} factors")
    print(f"Assets: {asset_names}")
    print(f"Factors: {factor_names}\n")

    # Generate factor covariance matrix (annual)
    factor_corr = np.array([
        [1.00, 0.10, -0.15],
        [0.10, 1.00, 0.20],
        [-0.15, 0.20, 1.00]
    ])
    factor_vols = np.array([0.16, 0.12, 0.14])  # 16%, 12%, 14% annual vol
    factor_covar = np.outer(factor_vols, factor_vols) * factor_corr

    # Scale to daily (assuming 260 business days)
    dt = 1.0 / 260.0
    factor_covar = factor_covar * dt

    print("Factor Covariance Matrix (daily):")
    factor_covar_df = pd.DataFrame(factor_covar, index=factor_names, columns=factor_names)
    print(factor_covar_df.round(6))
    print()

    # Generate asset factor loadings (betas)
    asset_betas = np.array([
        [1.2, 0.8, 1.1, 0.9, 1.0, 1.3],  # Market beta
        [0.5, -0.2, 0.8, -0.1, 0.3, -0.4],  # Value factor
        [-0.3, 0.6, -0.1, 0.4, 0.2, -0.2]  # Size factor
    ])

    print("Asset Factor Loadings (Betas):")
    betas_df = pd.DataFrame(asset_betas, index=factor_names, columns=asset_names)
    print(betas_df.round(2))
    print()

    # Generate idiosyncratic variances (daily)
    annual_idio_vols = np.array([0.25, 0.30, 0.20, 0.35, 0.28, 0.22])  # Annual idio vols
    idiosyncratic_var = (annual_idio_vols * np.sqrt(dt)) ** 2  # Convert to daily variance

    print("Idiosyncratic Volatilities (daily):")
    idio_df = pd.DataFrame({
        'Annual_Vol': annual_idio_vols,
        'Daily_Vol': np.sqrt(idiosyncratic_var),
        'Daily_Variance': idiosyncratic_var
    }, index=asset_names)
    print(idio_df.round(4))
    print()

    # Define benchmark weights (market cap weighted)
    benchmark_weights = np.array([0.25, 0.20, 0.18, 0.15, 0.12, 0.10])

    # Define portfolio weights (active positions)
    portfolio_weights = np.array([0.30, 0.15, 0.20, 0.10, 0.15, 0.10])

    # Show portfolio vs benchmark
    weights_df = pd.DataFrame({
        'Benchmark': benchmark_weights,
        'Portfolio': portfolio_weights,
        'Active_Weight': portfolio_weights - benchmark_weights,
        'Active_Weight_pct': (portfolio_weights - benchmark_weights) * 100
    }, index=asset_names)

    print("Portfolio vs Benchmark Weights:")
    print(weights_df.round(4))
    print()

    # Calculate factor exposures
    benchmark_exposures = asset_betas @ benchmark_weights
    portfolio_exposures = asset_betas @ portfolio_weights
    active_exposures = portfolio_exposures - benchmark_exposures

    exposures_df = pd.DataFrame({
        'Benchmark': benchmark_exposures,
        'Portfolio': portfolio_exposures,
        'Active': active_exposures
    }, index=factor_names)

    print("Factor Exposures:")
    print(exposures_df.round(4))
    print()

    # Calculate total active risk
    total_active_risk_sq = calculate_active_risk_squared(
        portfolio_weights, benchmark_weights, asset_betas, factor_covar, idiosyncratic_var
    )
    total_active_risk = np.sqrt(total_active_risk_sq)
    annual_tracking_error = total_active_risk * np.sqrt(260) * 100  # Convert to annual %

    print(f"Total Active Risk (daily): {total_active_risk:.6f}")
    print(f"Annualized Tracking Error: {annual_tracking_error:.2f}%\n")

    # Calculate marginal active risk
    marginal_risk, systematic_marginal, idiosyncratic_marginal = calculate_marginal_active_risk(
        portfolio_weights, benchmark_weights, asset_betas, factor_covar, idiosyncratic_var
    )

    # Calculate risk contributions
    weight_diff = portfolio_weights - benchmark_weights
    risk_contributions = marginal_risk * weight_diff

    # Create results DataFrame
    results_df = pd.DataFrame({
        'Active_Weight': weight_diff,
        'Marginal_Risk_Total': marginal_risk,
        'Marginal_Risk_Systematic': systematic_marginal,
        'Marginal_Risk_Idiosyncratic': idiosyncratic_marginal,
        'Risk_Contribution': risk_contributions,
        'Risk_Contribution_pct': risk_contributions / total_active_risk_sq * 100
    }, index=asset_names)

    print("Marginal Active Risk Analysis:")
    print(results_df.round(6))
    print()

    # Verify risk decomposition
    total_risk_contrib = np.sum(risk_contributions)
    print("Verification:")
    print(f"Sum of Risk Contributions: {total_risk_contrib:.8f}")
    print(f"Total Active Risk Squared: {total_active_risk_sq:.8f}")
    print(f"Difference: {abs(total_risk_contrib - total_active_risk_sq):.2e}")
    print(f"Risk contributions sum correctly: {np.isclose(total_risk_contrib, total_active_risk_sq)}\n")

    # Risk/Return Analysis
    print("=== Risk-Adjusted Analysis ===")

    # Simulate expected alpha (for demo purposes)
    np.random.seed(123)
    expected_alpha = np.random.normal(0, 0.001, n_assets)  # Small daily alphas

    # Calculate risk-adjusted scores
    risk_adj_scores = np.where(marginal_risk > 0, expected_alpha / marginal_risk, np.inf)

    risk_return_df = pd.DataFrame({
        'Expected_Alpha': expected_alpha,
        'Marginal_Risk': marginal_risk,
        'Risk_Adj_Score': risk_adj_scores,
        'Current_Active_Weight': weight_diff
    }, index=asset_names)

    # Sort by risk-adjusted score
    risk_return_df = risk_return_df.sort_values('Risk_Adj_Score', ascending=False)

    print("Risk-Adjusted Analysis (sorted by score):")
    print(risk_return_df.round(6))
    print()

    # Factor contribution analysis
    print("=== Factor Risk Contribution Analysis ===")

    # Calculate each factor's contribution to active risk
    factor_marginal_risks = 2 * factor_covar @ active_exposures
    factor_risk_contributions = factor_marginal_risks * active_exposures

    factor_analysis_df = pd.DataFrame({
        'Active_Exposure': active_exposures,
        'Marginal_Risk': factor_marginal_risks,
        'Risk_Contribution': factor_risk_contributions,
        'Risk_Contribution_pct': factor_risk_contributions / total_active_risk_sq * 100
    }, index=factor_names)

    print("Factor-Level Risk Analysis:")
    print(factor_analysis_df.round(6))
    print()

    # Risk budget allocation example
    print("=== Risk Budget Allocation Example ===")

    target_tracking_error = 0.02  # 2% annual
    target_daily_risk_sq = (target_tracking_error / np.sqrt(260)) ** 2

    # Calculate position limits based on marginal risk
    position_limits = np.where(marginal_risk > 0,
                               target_daily_risk_sq / marginal_risk,
                               np.inf)

    budget_df = pd.DataFrame({
        'Current_Active_Weight': weight_diff,
        'Marginal_Risk': marginal_risk,
        'Position_Limit': position_limits,
        'Utilization_pct': np.abs(weight_diff) / position_limits * 100
    }, index=asset_names)

    print(f"Risk Budget Analysis (Target TE: {target_tracking_error * 100:.1f}%):")
    print(budget_df.round(4))

    print("\n=== Summary ===")
    print(f"• Total tracking error: {annual_tracking_error:.2f}% annually")
    print(f"• Largest risk contributor: {asset_names[np.argmax(np.abs(risk_contributions))]}")
    print(f"• Highest marginal risk: {asset_names[np.argmax(marginal_risk)]}")
    print(f"• Risk decomposition verified: {np.isclose(total_risk_contrib, total_active_risk_sq)}")

if __name__ == "__main__":
    run_local(local=Locals.MARGINAL_ACTIVE_RISK)

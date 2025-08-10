"""
computation of risk contributions
"""
import numpy as np
import pandas as pd
from typing import Union, Tuple



def compute_portfolio_risk_contributions(w: Union[np.ndarray, pd.Series],
                                         covar: Union[np.ndarray, pd.DataFrame]
                                         ) -> Union[np.ndarray, pd.Series]:
    """Computes the risk contribution of each asset to the portfolio's total risk.

    Args:
        w: Portfolio weights as array or Series.
        covar: Covariance matrix as array or DataFrame.

    Returns:
        Risk contributions for each asset.

    Raises:
        ValueError: If input types are not compatible.
        AssertionError: If dimensions don't match for numpy arrays.
    """
    if isinstance(covar, pd.DataFrame) and isinstance(w, pd.Series):  # make sure weights are alined
        w = w.reindex(index=covar.index).fillna(0.0)
    elif isinstance(covar, np.ndarray) and isinstance(w, np.ndarray):
        assert covar.shape[0] == covar.shape[1] == w.shape[0]
    else:
        raise ValueError(f"unnsuported types {type(w)} and {type(covar)}")
    portfolio_vol = np.sqrt(w.T @ covar @ w)
    marginal_risk_contribution = covar @ w.T
    rc = np.multiply(marginal_risk_contribution, w) / portfolio_vol
    return rc


def compute_benchmark_portfolio_risk_contributions(w_portfolio: Union[np.ndarray, pd.Series],
                                                    w_benchmark: Union[np.ndarray, pd.Series],
                                                    covar: Union[np.ndarray, pd.DataFrame],
                                                    is_independent_risk: bool = False
                                                    ) -> Union[np.ndarray, pd.Series]:
    """Computes risk contributions of active positions relative to benchmark.

    Args:
        w_portfolio: Portfolio weights as array or Series.
        w_benchmark: Benchmark weights as array or Series.
        covar: Covariance matrix as array or DataFrame.
        is_independent_risk: If True, assumes positions are independent (diagonal risk only).

    Returns:
        Risk contributions of active positions (portfolio - benchmark).

    Raises:
        ValueError: If input types are not compatible.
        AssertionError: If dimensions don't match for numpy arrays.
    """
    if isinstance(covar, pd.DataFrame) and isinstance(w_portfolio, pd.Series):  # make sure weights are alined
        w_portfolio = w_portfolio.reindex(index=covar.index).fillna(0.0)
    elif isinstance(covar, pd.DataFrame) and isinstance(w_benchmark, pd.Series):  # make sure weights are alined
        w_benchmark = w_benchmark.reindex(index=covar.index).fillna(0.0)
    elif isinstance(covar, np.ndarray) and isinstance(w_portfolio, np.ndarray) and isinstance(w_benchmark, np.ndarray):
        assert covar.shape[0] == covar.shape[1] == w_portfolio.shape[0] == w_benchmark.shape[0]
    else:
        raise ValueError(f"unnsuported types {type(w_portfolio)}, {type(w_benchmark)} and {type(covar)}")
    if is_independent_risk:
        rc = np.sqrt(np.multiply(np.square(w_portfolio-w_benchmark),  np.diag(covar)))
    else:
        portfolio_vol = np.sqrt(w_benchmark.T @ covar @ w_benchmark)
        marginal_risk_contribution = covar @ (w_portfolio-w_benchmark).T
        rc = np.multiply(marginal_risk_contribution, (w_portfolio-w_benchmark)) / portfolio_vol
    return rc


def calculate_marginal_active_risk(portfolio_weights: np.ndarray,
                                   benchmark_weights: np.ndarray,
                                   asset_betas: np.ndarray,
                                   factor_covar: np.ndarray,
                                   idiosyncratic_var: np.ndarray
                                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate marginal active risk for each asset.

    Args:
        portfolio_weights: Portfolio weights (N,)
        benchmark_weights: Benchmark weights (N,)
        asset_betas: Factor loadings for all assets (K x N)
        factor_covar: Factor covariance matrix (K x K)
        idiosyncratic_var: Asset idiosyncratic variances (N,)

    Returns:
        marginal_risk: Total marginal active risk for each asset
        systematic_marginal: Systematic component only
        idiosyncratic_marginal: Idiosyncratic component only
    """
    # Calculate benchmark factor exposures from its asset weights
    benchmark_factor_exposures = asset_betas @ benchmark_weights  # Shape: (K,)

    # Current portfolio factor exposures
    portfolio_factor_exposures = asset_betas @ portfolio_weights  # Shape: (K,)

    # Active factor exposures
    active_exposures = portfolio_factor_exposures - benchmark_factor_exposures  # Shape: (K,)

    # Weight differences
    weight_diff = portfolio_weights - benchmark_weights  # Shape: (N,)

    # Marginal contributions for each asset
    marginal_risk = np.zeros(len(portfolio_weights))
    systematic_marginal = np.zeros(len(portfolio_weights))
    idiosyncratic_marginal = np.zeros(len(portfolio_weights))

    for i in range(len(portfolio_weights)):
        # Asset i's factor loadings
        asset_i_betas = asset_betas[:, i]  # Shape: (K,)

        # Systematic marginal risk
        systematic_marginal[i] = 2.0 * asset_i_betas.T @ factor_covar @ active_exposures

        # Idiosyncratic marginal risk
        idiosyncratic_marginal[i] = 2.0 * idiosyncratic_var[i] * weight_diff[i]

        # Total marginal risk
        marginal_risk[i] = systematic_marginal[i] + idiosyncratic_marginal[i]

    return marginal_risk, systematic_marginal, idiosyncratic_marginal


def calculate_active_risk_squared(portfolio_weights: np.ndarray,
                                  benchmark_weights: np.ndarray,
                                  asset_betas: np.ndarray,
                                  factor_covar: np.ndarray,
                                  idiosyncratic_var: np.ndarray
                                  ) -> float:
    """Calculate total active risk squared."""
    # Active factor exposures
    portfolio_exposures = asset_betas @ portfolio_weights
    benchmark_exposures = asset_betas @ benchmark_weights
    active_exposures = portfolio_exposures - benchmark_exposures

    # Systematic active risk
    systematic_risk_sq = active_exposures.T @ factor_covar @ active_exposures

    # Idiosyncratic active risk
    weight_diff = portfolio_weights - benchmark_weights
    idiosyncratic_risk_sq = weight_diff.T @ np.diag(idiosyncratic_var) @ weight_diff

    return systematic_risk_sq + idiosyncratic_risk_sq


def demo_marginal_active_risk():
    """Comprehensive demo of marginal active risk calculation."""

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
    print(f"Verification:")
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

    demo_marginal_active_risk()

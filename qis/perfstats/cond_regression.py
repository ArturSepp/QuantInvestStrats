"""
Regime-conditional regression analysis for portfolio factor exposure estimation.

This module implements regime-conditional regression where asset returns are regressed
on regime-specific benchmark returns. The key innovation is partitioning the sample
by market regimes (e.g., bear/normal/bull) and estimating separate factor loadings
(betas) for each regime.

Regression specification (without alpha):
    R_i,t = β_i,bear * R_b,t * I_bear + β_i,normal * R_b,t * I_normal + β_i,bull * R_b,t * I_bull + ε_i,t

where:
    - R_i,t: asset i returns at time t
    - R_b,t: benchmark returns at time t
    - I_regime: indicator function (1 if regime active, 0 otherwise)
    - β_i,regime: asset i's beta exposure in each regime
    - ε_i,t: residual (idiosyncratic) returns

Alpha term consideration (is_add_alpha parameter):
    - is_add_alpha=False (RECOMMENDED): Regime-specific betas capture all systematic variation.
      No constant term needed as regime dummies span the full sample space.
    - is_add_alpha=True: Adds global constant α_i, but creates interpretation issues:
      * What does constant mean when you have regime-specific exposures?
      * May cause multicollinearity as regime dummies already partition sample
      * Alpha becomes unidentified or captures numerical artifacts

Standard practice in regime-switching models excludes the global alpha term, letting
the regime-conditional betas fully characterize the asset's behavior across market states.

Alternative: For regime-specific alphas, extend model to:
    R_i,t = α_i,bear * I_bear + β_i,bear * R_b,t * I_bear + ... + ε_i,t
"""

# built-in
import numpy as np
import pandas as pd
import statsmodels.api as sm
from enum import Enum
from typing import Dict, Optional

# qis
from qis.perfstats.regime_classifier import (BenchmarkReturnsQuantilesRegime,
                                             RegimeClassifier)

class ConditionalRegressionColumns(str, Enum):
    # Column name constants
    RESID_VAR_COLUMN = 'An var resid'
    ALPHA_COLUMN = 'Alpha'
    COLOR_COLUMN = 'Color'
    PREDICTION = 'Prediction'
    VAR = 'Var'
    R2 = 'R2'


def estimate_cond_regression(prices: pd.DataFrame,
                             benchmark: str,
                             regime_classifier: Optional[RegimeClassifier] = None,
                             is_print_summary: bool = False,
                             drop_benchmark: bool = True,
                             min_period: int = 0,
                             is_add_alpha: bool = False,
                             regmodel_out_dict: Optional[Dict[str, pd.DataFrame]] = None
                             ) -> pd.DataFrame:
    """
    Estimate regime-conditional regression parameters (betas) for multiple assets.

    Fits separate betas for each market regime (e.g., bear, normal, bull) by regressing
    asset returns on regime-specific benchmark returns.

    Args:
        prices: DataFrame of asset prices with benchmark as first column
        benchmark: Name of benchmark column used for regime classification
        regime_classifier: Specifications for regime classification (quantiles, frequency)
        is_print_summary: If True, print OLS regression summary for each asset
        drop_benchmark: If True, exclude benchmark from regression outputs
        min_period: Minimum number of valid returns required to estimate parameters
        is_add_alpha: If True, include intercept term in regression
        regmodel_out_dict: Optional dict to store detailed regression outputs per asset

    Returns:
        DataFrame with regime betas (and alpha if requested) as columns, assets as rows

    Raises:
        ValueError: If benchmark has all NaN returns (invalid benchmark specification)
    """
    # Initialize regime classifier with default params if not provided

    if regime_classifier is None:
        regime_classifier = BenchmarkReturnsQuantilesRegime()

    # Classify returns into market regimes based on benchmark quantiles
    sampled_returns_with_regime_id = regime_classifier.compute_sampled_returns_with_regime_id(
        prices=prices,
        benchmark=benchmark,
        **regime_classifier.to_dict()
    )

    # Remove rows where all assets have missing returns
    sampled_returns_with_regime_id = sampled_returns_with_regime_id.dropna(how='all', axis=0)

    # Group returns by regime for regime-conditional analysis
    regime_groups = sampled_returns_with_regime_id.groupby([regime_classifier.REGIME_COLUMN], observed=False)

    # Build regression matrix with regime-specific benchmark returns as columns
    regression_datas = []
    for regime in regime_classifier.get_regime_ids():
        # Extract benchmark returns for this regime and rename column by regime name
        benchmark_data = regime_groups.get_group((regime,))[benchmark].rename(regime)
        regression_datas.append(benchmark_data)

    # Concatenate regime columns and fill missing values with zeros (non-regime periods)
    regression_matrix = pd.concat(regression_datas, axis=1).fillna(0)

    # Add constant term (alpha) to regression matrix if requested
    if is_add_alpha:
        #regression_matrix_with_const = sm.add_constant(regression_matrix).rename(
        #    columns={'const': ConditionalRegressionColumns.ALPHA_COLUMN.value}
        #)
        # Add separate alpha for each regime
        for regime in regime_classifier.get_regime_ids():
            regime_dummy = (sampled_returns_with_regime_id[regime_classifier.REGIME_COLUMN] == regime).astype(float)
            regression_matrix[f'{ConditionalRegressionColumns.ALPHA_COLUMN.value}_{regime}'] = regime_dummy
        regression_matrix_with_const = regression_matrix

    else:
        regression_matrix_with_const = regression_matrix

    # Prepare asset returns for regression (drop regime column and optionally benchmark)
    columns_to_drop = [regime_classifier.REGIME_COLUMN]
    if drop_benchmark:
        columns_to_drop.append(benchmark)
    asset_returns = sampled_returns_with_regime_id.drop(columns=columns_to_drop)

    # Create regime color mapping for visualization outputs
    regime_colors = regime_classifier.class_data_to_colors(
        sampled_returns_with_regime_id[regime_classifier.REGIME_COLUMN]
    ).rename(ConditionalRegressionColumns.COLOR_COLUMN.value)

    # Fit regression model for each asset
    model_params = []
    for asset in asset_returns.columns:
        y = asset_returns[asset]
        nan_mask = y.isna()

        # Check if all returns are missing
        if nan_mask.all():
            if len(model_params) == 0:
                # First asset must be benchmark - should never have all NaNs
                raise ValueError(f"Benchmark '{benchmark}' has all NaN returns - invalid specification")
            else:
                # For other assets, record NaN parameters matching structure of previous assets
                estimated_model_params = pd.Series(np.nan, index=model_params[0].index, name=asset)
        # Check if insufficient valid observations for regression
        elif (~nan_mask).sum() < min_period:
            estimated_model_params = pd.Series(np.nan, index=model_params[0].index, name=asset)
        else:
            # Filter to valid observations only
            if nan_mask.any():
                y_clean = y[~nan_mask]
            else:
                y_clean = y

            # Align regression matrix with valid y observations
            X = regression_matrix_with_const.loc[y_clean.index, :]

            # Fit OLS regression model
            model = sm.OLS(y_clean, X)
            estimated_model = model.fit()

            # Print regression summary if requested
            if is_print_summary:
                print(f"\n{'=' * 80}")
                print(f"Regression Summary for {asset}")
                print('=' * 80)
                print(estimated_model.summary())

            # Store detailed regression outputs if dict provided
            if regmodel_out_dict is not None:
                prediction = estimated_model.predict(regression_matrix_with_const)
                pandas_out = pd.concat([
                    sampled_returns_with_regime_id[regime_classifier.REGIME_COLUMN],
                    sampled_returns_with_regime_id[benchmark],
                    regression_matrix,
                    y,
                    prediction.rename(ConditionalRegressionColumns.PREDICTION.value),
                    regime_colors
                ], axis=1)
                regmodel_out_dict[asset] = pandas_out

            # Extract parameter estimates and residual variance
            estimated_model_params = estimated_model.params.copy()
            estimated_model_params[ConditionalRegressionColumns.RESID_VAR_COLUMN.value] = estimated_model.mse_resid
            estimated_model_params[ConditionalRegressionColumns.R2.value] = estimated_model.rsquared_adj
            estimated_model_params.name = asset

        # Append parameters for this asset
        model_params.append(estimated_model_params)

    # Combine all asset parameters into single DataFrame (assets as rows)
    model_params_df = pd.concat(model_params, axis=1).T

    return model_params_df


def get_regime_regression_params(prices: pd.DataFrame,
                                 regime_classifier: RegimeClassifier,
                                 benchmark: str,
                                 is_print_summary: bool = True,
                                 drop_benchmark: bool = False,
                                 min_period: int = 0,
                                 is_add_alpha: bool = False
                                 ) -> pd.DataFrame:
    """
    Convenience wrapper for estimate_cond_regression with common parameters.

    Args:
        prices: DataFrame of asset prices
        regime_classifier: Regime classification specifications
        benchmark: Benchmark asset name
        is_print_summary: Whether to print regression summaries
        drop_benchmark: Whether to exclude benchmark from results
        min_period: Minimum observations required per asset
        is_add_alpha: Whether to include regression intercept

    Returns:
        DataFrame of estimated regime betas per asset
    """
    estimated_params = estimate_cond_regression(
        prices=prices,
        benchmark=benchmark,
        regime_classifier=regime_classifier,
        is_print_summary=is_print_summary,
        drop_benchmark=drop_benchmark,
        min_period=min_period,
        is_add_alpha=is_add_alpha
    )
    return estimated_params


class LocalTests(Enum):
    """Enumeration of available local test cases."""
    REGRESSION = 1


def run_local_test(local_test: LocalTests):
    """
    Run local tests for development and debugging purposes.

    Integration tests that download real data and generate reports.
    Use for quick verification during development.

    Args:
        local_test: Which test case to run
    """
    from qis.test_data import load_etf_data

    # Load sample ETF price data
    prices = load_etf_data().dropna()
    print(prices)

    if local_test == LocalTests.REGRESSION:
        # Test regime-conditional regression with SPY as benchmark
        estimated_params = get_regime_regression_params(
            prices=prices,
            regime_classifier=BenchmarkReturnsQuantilesRegime(),
            benchmark='SPY',
            drop_benchmark=True,
            is_print_summary=True,
            is_add_alpha=True
        )
        print("\nEstimated Parameters:")
        print(estimated_params)


if __name__ == '__main__':
    run_local_test(local_test=LocalTests.REGRESSION)

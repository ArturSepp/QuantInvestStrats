"""
imlementation of basic linear factor model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Union, Dict, Literal, Tuple

from qis import TimePeriod
from qis.models.linear import ewm as ewm
from qis.perfstats import returns as ret
from qis.plots import time_series as pts


@dataclass
class LinearModel:
    """A linear factor model for analyzing asset returns using multiple factors.

    This class implements a linear factor model of the form y = loadings^T @ x + alpha,
    where factors (x) explain asset returns (y) through time-varying factor loadings.
    The model supports factor attribution analysis, alpha calculation, model diagnostics,
    and performance evaluation.

    The linear relationship assumes:
    - y_t = β_t^T * x_t + α_t
    - Where β_t are the factor loadings (betas) at time t
    - x_t are the factor returns at time t
    - α_t is the unexplained return (alpha) at time t

    Attributes:
        x (pd.DataFrame or pd.Series): Factor data of shape (T, N) where T is time periods
            and N is number of factors. Each column represents a different factor's
            time series returns.
        y (pd.DataFrame or pd.Series): Asset data of shape (T, M) where T is time periods
            and M is number of assets. Each column represents a different asset's
            time series returns.
        loadings (Dict[str, pd.DataFrame], optional): Estimated factor loadings for each
            factor. Keys are factor names (matching x.columns), values are DataFrames
            with shape (T, M) containing time-varying loadings for each asset.

    Methods:
        Core Analysis:
            - get_factor_loadings(): Retrieve loadings for specific factors
            - get_asset_factor_betas(): Get factor exposures for specific assets
            - get_asset_factor_attribution(): Calculate factor attribution for returns
            - get_factor_alpha(): Calculate unexplained returns (alpha) and explained returns

        Portfolio Analysis:
            - compute_agg_factor_exposures(): Aggregate factor exposures using portfolio weights

        Model Diagnostics:
            - get_model_ewm_r2(): Calculate exponentially weighted R-squared
            - get_model_residuals_corrs(): Analyze residual correlations

        Visualization:
            - plot_factor_loadings(): Plot factor loading time series
            - print(): Debug output of model components

    Example:
        ```python
        # Create factor and asset return data
        factors = pd.DataFrame({
            'Market': market_returns,
            'Size': size_returns,
            'Value': value_returns
        }, index=dates)

        assets = pd.DataFrame({
            'Stock_A': stock_a_returns,
            'Stock_B': stock_b_returns
        }, index=dates)

        # Assume loadings have been estimated elsewhere
        loadings = {
            'Market': market_betas_df,  # Shape: (T, 2) for 2 assets
            'Size': size_betas_df,
            'Value': value_betas_df
        }

        # Create model
        model = LinearModel(x=factors, y=assets, loadings=loadings)

        # Analyze factor attribution for Stock_A
        attribution = model.get_asset_factor_attribution(asset='Stock_A')

        # Calculate model alpha (unexplained returns)
        alpha, explained = model.get_factor_alpha(lag=1)

        # Check model fit
        r2 = model.get_model_ewm_r2(span=52)
        ```
        See example implementation with EwmLinearModel
    Notes:
        - Factor loadings are assumed to be pre-estimated (e.g., via rolling regression)
        - Time series must be aligned with matching indices
        - The model supports both in-sample (lag=0) and out-of-sample (lag=1) analysis
        - All DataFrames should have datetime indices for proper time series handling
        - Missing values in loadings or returns may affect calculations

    Raises:
        AssertionError: If loadings keys don't match factor column names in x
    """

    x: Union[pd.DataFrame, pd.Series] # t, x_n factors
    y: Union[pd.DataFrame, pd.Series]  # t, y_m factors
    loadings: Dict[str, pd.DataFrame] = None  # estimated factor loadings

    def __post_init__(self):

        if isinstance(self.x, pd.Series):
            self.x = self.x.to_frame()

        if isinstance(self.y, pd.Series):
            self.y = self.y.to_frame()

        if self.loadings is not None:
            str_factors = list(self.loadings.keys())
            assert str_factors == self.x.columns.to_list()

    def print(self):
        """Print model components for debugging."""
        print(f"x:\n{self.x}")
        print(f"y:\n{self.y}")
        for factor, loading in self.loadings.items():
            print(f"{factor}:\n{loading}")

    def get_factor1_loadings(self) -> pd.DataFrame:
        """Get loadings for the first factor."""
        return self.loadings[list(self.loadings.keys())[0]]

    def get_factor_loadings(self, factor: str) -> pd.DataFrame:
        """Get loadings for a specific factor."""
        return self.loadings[factor]

    def compute_agg_factor_exposures(self,
                                     weights: pd.DataFrame
                                     ) -> pd.DataFrame:
        """Compute aggregate factor exposures by multiplying loadings with weights."""
        factor_exposures = {}
        for factor, loading in self.loadings.items():
            factor_exposures[factor] = loading.multiply(weights).sum(axis=1)
        factor_exposures = pd.DataFrame.from_dict(factor_exposures)
        return factor_exposures

    def get_asset_factor_betas(self,
                               time_period: TimePeriod = None,
                               asset: str = None
                               ) -> pd.DataFrame:
        """Get asset exposures to factors over specified time period."""
        if asset is None:
            asset = self.y.columns[0]
        exps = {}
        for factor, factor_exp in self.loadings.items():
            exps[factor] = factor_exp[asset]
        exps = pd.DataFrame.from_dict(exps)
        if time_period is not None:
            exps = time_period.locate(exps)
        return exps

    def get_asset_factor_attribution(self, asset: str = None, add_total: bool = True) -> pd.DataFrame:
        """Calculate factor attribution for an asset's returns."""
        factor_betas = self.get_asset_factor_betas(asset=asset)
        exposures = self.x
        attribution = exposures.multiply(factor_betas.shift(1))
        if add_total:
            total = attribution.sum(1).rename('Total')
            attribution = pd.concat([total, attribution], axis=1)
        return attribution

    def get_factor_alpha(self, lag: Literal[0, 1] = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Calculate factor alpha and explained returns.

        Args:
            lag: 1 for step-ahead forecast, 0 for in-sample.

        Returns:
            Tuple of (factor_alpha, explained_returns).
        """
        explained_returns = pd.DataFrame(0.0, index=self.y.index, columns=self.y.columns)
        for factor in self.x.columns:
            factor_betas = self.loadings[factor]
            explained_return = (factor_betas.shift(lag)).multiply(self.x[factor].to_numpy(), axis=0)
            explained_returns = explained_returns.add(explained_return)
        factor_alpha = self.y.subtract(explained_returns)
        return factor_alpha, explained_returns

    def get_model_ewm_r2(self, span: int = 52, lag: Literal[0, 1] = 0) -> pd.DataFrame:
        """Calculate exponentially weighted R-squared using residuals."""
        residuals, explained_returns = self.get_factor_alpha(lag=lag)
        ewm_residuals_2 = ewm.compute_ewm(data=np.square(residuals), span=span)
        y_demean = self.y  #- ewm.compute_ewm(data=self.y, span=span)
        ewm_variance = ewm.compute_ewm(data=np.square(y_demean), span=span)
        r_2 = 1.0 - ewm_residuals_2.divide(ewm_variance)
        return r_2

    def get_model_residuals_corrs(self, span: int = 52) -> Tuple[pd.DataFrame, pd.Series]:
        """Calculate EWM correlation matrix of residuals and average correlation."""
        residuals, explained_returns = self.get_factor_alpha(lag=0)
        corr = ewm.compute_ewm_covar(residuals.to_numpy(), span=span, is_corr=True)
        avg_corr = pd.Series(0.5*np.nanmean(corr - np.eye(corr.shape[0]), axis=1), index=self.y.columns)
        corr_pd = pd.DataFrame(corr, index=self.y.columns, columns=self.y.columns)
        return corr_pd, avg_corr

    def plot_factor_loadings(self,
                             factor: str,
                             var_format: str = '{:,.2f}',
                             time_period: TimePeriod = None,
                             ax: plt.Subplot = None,
                             **kwargs):
        """Plot factor loadings time series."""
        df = self.loadings[factor]
        if time_period is not None:
            df = time_period.locate(df)
        pts.plot_time_series(df=df,
                             var_format=var_format,
                             ax=ax, **kwargs)


def compute_benchmarks_beta_attribution_from_prices(portfolio_nav: pd.Series,
                                                    benchmark_prices: pd.DataFrame,
                                                    portfolio_benchmark_betas: pd.DataFrame,
                                                    residual_name: str = 'Alpha',
                                                    time_period: TimePeriod = None
                                                    ) -> pd.DataFrame:
    """Compute benchmark attribution from price data using pre-calculated betas."""
    benchmark_prices = benchmark_prices.reindex(index=portfolio_benchmark_betas.index, method='ffill')
    portfolio_nav = portfolio_nav.reindex(index=portfolio_benchmark_betas.index, method='ffill')
    x = ret.to_returns(prices=benchmark_prices, freq=None)
    x_attribution = (portfolio_benchmark_betas.shift(1)).multiply(x)
    total_attrib = x_attribution.sum(axis=1)
    total = portfolio_nav.pct_change()
    residual = np.subtract(total, total_attrib)
    joint_attrib = pd.concat([x_attribution, residual.rename(residual_name)], axis=1)
    if time_period is not None:
        joint_attrib = time_period.locate(joint_attrib)
    return joint_attrib


def compute_benchmarks_beta_attribution_from_returns(portfolio_returns: pd.Series,
                                                    benchmark_returns: pd.DataFrame,
                                                    portfolio_benchmark_betas: pd.DataFrame,
                                                    residual_name: str = 'Alpha',
                                                    time_period: TimePeriod = None
                                                    ) -> pd.DataFrame:
    """Compute benchmark attribution from return data using pre-calculated betas."""
    if isinstance(benchmark_returns, pd.Series):
        benchmark_returns = benchmark_returns.to_frame()
    # to be replaced with qis
    benchmark_returns = benchmark_returns.reindex(index=portfolio_returns.index)
    x_attribution = (portfolio_benchmark_betas.shift(1)).multiply(benchmark_returns)
    total_attrib = x_attribution.sum(axis=1)
    residual = np.subtract(portfolio_returns, total_attrib)
    joint_attrib = pd.concat([x_attribution, residual.rename(residual_name)], axis=1)
    if time_period is not None:
        joint_attrib = time_period.locate(joint_attrib)
        joint_attrib[0, :] = 0.0
    return joint_attrib

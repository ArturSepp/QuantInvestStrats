"""
imlementation of basic linear factor model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Union, Dict, Literal, Tuple, Optional
# qis
from qis.utils.dates import TimePeriod, find_upto_date_from_datetime_index
from qis.utils.struct_ops import merge_lists_unique
import qis.models.linear.ewm as ewm
import qis.perfstats.returns as ret
import qis.plots.time_series as pts
from qis.portfolio.risk.contributions import calculate_marginal_active_risk

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
    y: Union[pd.DataFrame, pd.Series] # t, y_m factors
    loadings: Dict[str, pd.DataFrame] = None  # estimated factor loadings
    x_covars: Optional[Dict[pd.Timestamp, pd.DataFrame]] = None  # variances of x factors
    residual_vars: Optional[pd.DataFrame] = None  # residual variance of y

    def __post_init__(self):

        if isinstance(self.x, pd.Series):
            self.x = self.x.to_frame()

        if self.y is not None and isinstance(self.y, pd.Series):
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

    def get_loadings_at_date(self, date: pd.Timestamp) -> pd.DataFrame:
        if self.loadings is None:
            raise ValueError
        betas = {}
        for factor, df in self.loadings.items():
            last_update_date = find_upto_date_from_datetime_index(index=df.index, date=date)
            betas[factor] = df.loc[last_update_date, :]
        betas = pd.DataFrame.from_dict(betas, orient='index')  # index by factor
        return betas

    def compute_agg_factor_exposures(self,
                                     weights: pd.DataFrame
                                     ) -> pd.DataFrame:
        """Compute aggregate factor exposures by multiplying loadings with weights."""
        factor_exposures = {}
        for factor, loading in self.loadings.items():
            weights1 = weights.reindex(index=loading.index).ffill()
            factor_exposures[factor] = loading.multiply(weights1).sum(axis=1)
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

    def compute_factor_risk_contribution(self, weights: pd.DataFrame,
                                         factor_var_name: str = 'Systematic',
                                         idiosyncratic_var_name: str = 'Idiosyncratic'
                                         ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Compute factor risk contributions and their ratios for portfolio weights over time.
        Calculates the contribution of each factor to portfolio risk using factor loadings,
        covariance matrices, and residual variances. This method performs risk decomposition
        by computing how much each systematic factor and idiosyncratic risk contribute to
        the total portfolio variance.
        The calculation follows the standard risk attribution framework:
        - Portfolio variance = w'Bfactor_covB'w + w'Dresidual_varw
        - Factor risk contribution = (factor_exposure * marginal_contribution) / total_variance
        - Marginal contribution = factor_covariance @ factor_exposure
        Args:
            weights: Portfolio weights DataFrame with dates as index and assets as columns.
                    Each row represents portfolio weights at a specific date, with columns
                    corresponding to individual assets/securities.
            factor_var_name: Name for the factor risk column in output. Defaults to 'Systematic'.
            idiosyncratic_var_name: Name for the idiosyncratic risk column in output. Defaults to 'Idiosyncratic'.
        Returns:
            A tuple containing five elements:
                factor_rcs_ratios: DataFrame of normalized risk contribution ratios where each
                                 row sums to 1.0, including both systematic factors and
                                 idiosyncratic risk.
                factor_risk_contrib_idio: DataFrame of absolute factor risk contributions
                                        normalized by total portfolio variance (systematic + idiosyncratic).
                factor_risk_contrib: DataFrame of factor risk contributions normalized by
                                   systematic variance only (excludes idiosyncratic risk).
                portfolio_var: DataFrame containing systematic (factor-based) variance and idiosyncratic variance for each date.
        Raises:
            ValueError: If x_covars (factor covariance matrices) is None.
            ValueError: If residual_vars (asset-specific residual variances) is None.
        """
        if self.x_covars is None:
            raise ValueError(f"self.x_covars must be provided")
        if self.residual_vars is None:
            raise ValueError(f"self.residual_vars must be provided")
        # Initialize dictionaries to store results for each date
        factor_risk_contrib = {}  # Factor contributions normalized by systematic variance
        factor_risk_contrib_idio = {}  # Factor contributions normalized by total variance
        portfolio_factor_vars = {}  # Systematic variance for each date
        idio_vars = {}  # Idiosyncratic variance for each date
        idio_vars_contrib = {}  # Idiosyncratic risk contribution ratios
        # Iterate through each date in the factor covariance matrix timeline
        for date, factor_covar in self.x_covars.items():
            # Find the most recent portfolio weights available up to the current date
            weight_last_update_date = find_upto_date_from_datetime_index(index=weights.index, date=date)
            last_weight = weights.loc[weight_last_update_date, :]
            # Align factor loadings with portfolio weights (same asset universe)
            last_betas = self.get_loadings_at_date(date=date).loc[:, last_weight.index]
            last_residual_var = self.residual_vars.loc[date, last_weight.index]
            # Calculate portfolio's exposure to each factor: factor_exposure = Beta' * weights
            factor_exposures = last_betas @ last_weight
            # Ensure factor exposures align with covariance matrix factors, fill missing with 0
            factor_exposures = factor_exposures.reindex(index=factor_covar.index).fillna(0.0)
            # Calculate systematic portfolio variance: w'BFB'w where F is factor covariance
            portfolio_var_factors = factor_exposures.T @ (factor_covar @ factor_exposures)
            # Calculate idiosyncratic portfolio variance: w'Dw where D is diagonal residual variance matrix
            portfolio_var_idio = last_weight.T @ np.diag(last_residual_var) @ last_weight
            # Calculate total portfolio variance (systematic + idiosyncratic)
            portfolio_total_var = portfolio_var_factors + portfolio_var_idio
            # Calculate marginal contribution of each factor: dVar/dExposure = F * exposure
            marginal_factor_contrib = factor_covar @ factor_exposures
            # Calculate each factor's contribution to variance: exposure * marginal_contribution
            factor_vars = np.multiply(factor_exposures, marginal_factor_contrib)
            # Normalize factor contributions by systematic variance only
            factor_risk_contrib[date] = factor_vars / portfolio_var_factors if portfolio_var_factors > 0.0 \
                else pd.Series(np.nan, index=factor_covar.index)
            portfolio_factor_vars[date] = portfolio_var_factors
            idio_vars[date] = portfolio_var_idio
            # Calculate idiosyncratic risk contribution ratio
            factor_risk_contrib_idio[date] = factor_vars / portfolio_total_var if portfolio_total_var > 0.0 \
                else pd.Series(np.nan, index=factor_covar.index)
            idio_vars_contrib[date] = portfolio_var_idio / portfolio_total_var if portfolio_total_var > 0.0 else np.nan
        # Convert dictionaries to DataFrames with dates as index
        factor_risk_contrib = pd.DataFrame.from_dict(factor_risk_contrib, orient='index')
        factor_risk_contrib_idio = pd.DataFrame.from_dict(factor_risk_contrib_idio, orient='index')
        # Convert variance dictionaries to Series
        portfolio_factor_vars = pd.Series(portfolio_factor_vars, name=factor_var_name)
        idio_vars = pd.Series(idio_vars, name=idiosyncratic_var_name)
        portfolio_var = pd.concat([portfolio_factor_vars, idio_vars], axis=1)
        idio_vars_contrib = pd.Series(idio_vars_contrib)
        # Add idiosyncratic risk contribution as a separate column
        factor_risk_contrib_idio[idiosyncratic_var_name] = idio_vars_contrib.to_numpy()
        # Normalize all risk contributions to sum to 1.0 across factors for each date
        factor_rcs_ratios = factor_risk_contrib_idio.divide(np.nansum(factor_risk_contrib_idio, axis=1, keepdims=True)).fillna(0.0)
        # Return all computed metrics as a tuple
        return factor_rcs_ratios, factor_risk_contrib_idio, factor_risk_contrib, portfolio_var

    def compute_active_factor_risk(self,
                                   portfolio_weights: pd.DataFrame,
                                   benchmark_weights: pd.DataFrame
                                   ) -> Dict[str, pd.DataFrame]:
        """
        qqq
        """
        if self.x_covars is None:
            raise ValueError(f"self.x_covars must be provided")
        if self.residual_vars is None:
            raise ValueError(f"self.residual_vars must be provided")

        # find joint index
        joint_assets = merge_lists_unique(list1=portfolio_weights.columns.to_list(),
                                          list2=benchmark_weights.columns.to_list())
        portfolio_weights = portfolio_weights.reindex(columns=joint_assets).fillna(0.0)
        benchmark_weights = benchmark_weights.reindex(columns=joint_assets).fillna(0.0)
        residual_vars = self.residual_vars.reindex(columns=joint_assets).fillna(0.0)

        # Initialize dictionaries to store results for each date
        portfolio_exposures_ts = {}
        benchmark_exposures_ts = {}
        active_exposures_ts = {}
        factor_marginal_risks_ts = {}
        factor_risk_contributions_ts = {}
        # Iterate through each date in the factor covariance matrix timeline
        for date, factor_covar in self.x_covars.items():
            # Find the most recent portfolio weights available up to the current date
            weight_last_update_date = find_upto_date_from_datetime_index(index=portfolio_weights.index, date=date)

            if weight_last_update_date in portfolio_weights.index and weight_last_update_date in benchmark_weights.index:
                portfolio_weights_t = portfolio_weights.loc[weight_last_update_date, :]
                benchmark_weights_t = benchmark_weights.loc[weight_last_update_date, :]
                asset_betas_t = self.get_loadings_at_date(date=date).reindex(columns=joint_assets).fillna(0.0)
                idiosyncratic_var_t = residual_vars.loc[date, :]

                """
                # todo: marginal risk by position and groupped risks
                marginal_risk, systematic_marginal, idiosyncratic_marginal = calculate_marginal_active_risk(
                    portfolio_weights=portfolio_weights_t.to_numpy(),
                    benchmark_weights=benchmark_weights_t.to_numpy(),
                    asset_betas=asset_betas_t.to_numpy(),
                    factor_covar=factor_covar.to_numpy(),
                    idiosyncratic_var=idiosyncratic_var_t.to_numpy())
                """
                # Calculate factor exposures
                portfolio_exposures = asset_betas_t @ portfolio_weights_t
                benchmark_exposures = asset_betas_t @ benchmark_weights_t
                active_exposures = portfolio_exposures - benchmark_exposures
                factor_marginal_risks = 2.0 * factor_covar @ active_exposures
                factor_risk_contributions = factor_marginal_risks * active_exposures

                portfolio_exposures_ts[date] = portfolio_exposures
                benchmark_exposures_ts[date] = benchmark_exposures
                active_exposures_ts[date] = active_exposures
                factor_marginal_risks_ts[date] = factor_marginal_risks
                factor_risk_contributions_ts[date] = factor_risk_contributions

        # Convert dictionaries to DataFrames with dates as index
        portfolio_exposures = pd.DataFrame.from_dict(portfolio_exposures_ts, orient='index')
        benchmark_exposures = pd.DataFrame.from_dict(benchmark_exposures_ts, orient='index')
        active_exposures = pd.DataFrame.from_dict(active_exposures_ts, orient='index')
        factor_marginal_risks = pd.DataFrame.from_dict(factor_marginal_risks_ts, orient='index')
        factor_risk_contributions = pd.DataFrame.from_dict(factor_risk_contributions_ts, orient='index')
        factor_risk_contributions_rc = factor_risk_contributions.divide(np.nansum(factor_risk_contributions, axis=1, keepdims=True)).fillna(0.0)

        out_dict = dict(portfolio_exposures=portfolio_exposures,
                        benchmark_exposures=benchmark_exposures,
                        active_exposures=active_exposures,
                        factor_marginal_risks=factor_marginal_risks,
                        factor_risk_contributions=factor_risk_contributions,
                        factor_risk_contributions_rc=factor_risk_contributions_rc)

        return out_dict

    def plot_factor_loadings(self,
                             factor: str,
                             var_format: str = '{:,.2f}',
                             time_period: TimePeriod = None,
                             ax: plt.Subplot = None,
                             **kwargs
                             ) -> None:
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
                                                     time_period: TimePeriod = None,
                                                     total_name: Optional[str] = None
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
    if total_name is not None:
        joint_attrib = pd.concat([portfolio_returns.rename(total_name), joint_attrib], axis=1)
    if time_period is not None:
        joint_attrib = time_period.locate(joint_attrib)
    joint_attrib.iloc[0, :] = 0.0
    return joint_attrib

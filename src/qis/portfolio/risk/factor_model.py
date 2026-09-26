"""
the linear factor model container: y_t = B_t x_t + α_t, with loadings supplied, not estimated.

Orientation. In this module ``B_t`` is the assets-by-factors loading matrix of the handbook, but
it is stored per factor: ``loadings[factor]`` is a dates-by-assets frame, so row ``t`` holds one
column of ``B_t``. ``get_loadings_at_date`` returns the factors-by-assets transpose ``B_t'``, and
the exposures of weights ``w`` are ``B_t' w``, one value per factor. ``qis.RiskModel`` stores the
same loadings assets by factors; transpose a ``get_loadings_at_date`` snapshot before passing it
there. ``residual_vars`` is a dates-by-assets frame of variances, where ``RiskModel`` takes one
Series per date.

``LinearModel`` is a dataclass of a factor panel ``x``, an asset panel ``y`` and a ``loadings``
dict whose keys ``__post_init__`` asserts against the columns of ``x``; estimation belongs to a
subclass, and ``ewm_factor_model.py`` supplies the EWM one. ``get_factor_alpha`` shifts the
loadings by ``lag``, so lag 1 is the point-in-time reading and lag 0 is the in-sample fit, while
``get_asset_factor_attribution`` always shifts by one period.

Given ``x_covars`` and ``residual_vars``, ``compute_factor_risk_contribution`` splits portfolio
variance as (B' w)' F (B' w) + w' D w, under both the systematic-only and the total-variance
normalisation. ``compute_active_factor_risk`` is factor-only and does not carry the idiosyncratic
term: it returns exposures B' Δw per date, marginal risks 2 F B' Δw and their products,
normalised to sum to one across factors. The nav-level version of the same question is
``compute_benchmarks_beta_attribution_from_prices``.

Undefined values are NaN: a portfolio exposure, contribution or attribution is missing when an
asset the portfolio holds has no loading (for example during the estimator warm-up), never zero.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from dataclasses import dataclass
from typing import Union, Dict, Literal, Tuple, Optional
# qis
from qis.utils.dates import TimePeriod, find_upto_date_from_datetime_index
from qis.utils.struct_ops import merge_lists_unique
import qis.models.linear.ewm as ewm
import qis.perfstats.returns as ret
import qis.plots.time_series as pts
from qis.portfolio.risk.contributions import calculate_marginal_active_risk


def _held_weighted_sum(values: Union[pd.DataFrame, pd.Series],
                       weights: Union[pd.DataFrame, pd.Series]
                       ) -> Union[pd.Series, float]:
    """Sum ``values * weights`` over the assets held with a non-zero weight.

    Assets are the columns of a frame and the index of a Series. A missing weight means the asset
    is not held, so its value is ignored even when missing; a missing value of a held asset makes
    the sum NaN instead of silently dropping the asset.

    Args:
        values: Dates-by-assets frame with ``weights`` a frame on the same dates; or a
            factors-by-assets frame, or an asset Series, with ``weights`` an asset Series.
        weights: Weights by asset, as a dated frame or a single Series.

    Returns:
        One sum per date (NaN on a date whose weights row is entirely missing), one sum per row
        of ``values`` for Series weights, or a scalar for Series values and weights.
    """
    if isinstance(weights, pd.DataFrame):
        assets = weights.columns.union(values.columns, sort=False)
        weights = weights.reindex(columns=assets)
        no_portfolio = weights.isna().all(axis=1)
        weights = weights.fillna(0.0)
        held_values = values.reindex(columns=assets).where(weights != 0.0, 0.0)
        total = held_values.multiply(weights).sum(axis=1, skipna=False)
        return total.where(~no_portfolio)
    asset_index = values.columns if isinstance(values, pd.DataFrame) else values.index
    assets = weights.index.union(asset_index, sort=False)
    weights = weights.reindex(index=assets).astype(float).fillna(0.0)
    held = weights != 0.0
    if isinstance(values, pd.DataFrame):
        held_values = values.reindex(columns=assets).loc[:, held]
        return held_values.multiply(weights[held], axis=1).sum(axis=1, skipna=False)
    held_values = values.reindex(index=assets)[held]
    return float(held_values.multiply(weights[held]).sum(skipna=False))


@dataclass
class LinearModel:
    """A linear factor model for analyzing asset returns using multiple factors.

    This class implements a linear factor model in which factors explain asset returns through
    time-varying loadings and an unexplained alpha component.
    The model supports factor attribution analysis, alpha calculation, model diagnostics,
    and performance evaluation.

    Attributes:
        x: Factor returns with dates as rows and factors as columns.
        y: Asset returns with dates as rows and assets as columns.
        loadings: Time-varying factor loadings keyed by factor name, in the order of the
            columns of ``x``. Each value is a dates-by-assets frame: ``loadings[q].loc[t, i]`` is
            the loading of asset ``i`` on factor ``q`` estimated at ``t``.
        x_covars: Factor covariance matrices, factors by factors, keyed by estimation date.
        residual_vars: Asset residual variances (not volatilities), dates by assets.

    Note:
        Factor loadings are assumed to be pre-estimated. Inputs must use aligned datetime
        indices. Portfolio-level outputs are NaN where an asset with a non-zero weight has a
        missing loading or residual variance.

    Raises:
        AssertionError: If loading keys do not match the factor columns in ``x``.
    """

    x: Union[pd.DataFrame, pd.Series]  # t, x_n factors
    y: Union[pd.DataFrame, pd.Series]  # t, y_m assets
    loadings: Dict[str, pd.DataFrame] = None  # factor -> dates by assets
    x_covars: Optional[Dict[pd.Timestamp, pd.DataFrame]] = None  # date -> factors by factors
    residual_vars: Optional[pd.DataFrame] = None  # dates by assets, variances

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
        """Get the dates-by-assets loadings of the first factor."""
        return self.loadings[list(self.loadings.keys())[0]]

    def get_factor_loadings(self, factor: str) -> pd.DataFrame:
        """Get the dates-by-assets loadings of a specific factor."""
        return self.loadings[factor]

    def get_loadings_at_date(self, date: pd.Timestamp) -> pd.DataFrame:
        """Return the loadings in force at ``date``, factors by assets.

        Each factor's loadings are read as of ``date``: the last row dated at or before it.

        Args:
            date: Evaluation date.

        Returns:
            Loadings with factors as rows and assets as columns, the transpose ``B'`` of the
            assets-by-factors matrix that ``qis.RiskModel.factor_loadings`` holds.

        Raises:
            ValueError: If the model has no loadings.
        """
        if self.loadings is None:
            raise ValueError("loadings are not available; fit the model or supply them")
        betas = {}
        for factor, df in self.loadings.items():
            last_update_date = find_upto_date_from_datetime_index(index=df.index, date=date)
            betas[factor] = df.loc[last_update_date, :]
        betas = pd.DataFrame.from_dict(betas, orient='index')  # index by factor
        return betas

    def compute_agg_factor_exposures(self,
                                     weights: pd.DataFrame
                                     ) -> pd.DataFrame:
        """Compute portfolio factor exposures ``X[t, q] = sum_i B[t, i, q] * w[t, i]``.

        Loadings and weights are taken on the same date ``t`` of each factor's loading index.
        Weights are selected as of each loading date: the last weight row dated at or before it,
        so weight rows dated off the loading grid are not dropped. A missing weight inside a
        weight row means the asset is not held.

        Args:
            weights: Portfolio weights, dates by assets.

        Returns:
            Exposures, loading dates by factors. A date is NaN before the first weight row and
            wherever an asset with a non-zero weight has no loading, including the estimator
            warm-up rows; a portfolio of zero weights has exposure zero.
        """
        weights = weights.sort_index()
        factor_exposures = {}
        for factor, loading in self.loadings.items():
            factor_exposures[factor] = _held_weighted_sum(
                values=loading,
                weights=weights.reindex(index=loading.index, method='ffill'))
        factor_exposures = pd.DataFrame.from_dict(factor_exposures)
        return factor_exposures

    def get_asset_factor_betas(self,
                               time_period: TimePeriod = None,
                               asset: str = None
                               ) -> pd.DataFrame:
        """Get the loadings of one asset, dates by factors, over an optional time period."""
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
        """Attribute an asset's return to factors with the loadings of the previous date.

        The contribution of factor ``q`` at ``t`` is ``B[t-1, asset, q] * x[t, q]``.

        Args:
            asset: Asset column; the first asset of ``y`` when None.
            add_total: Prepend a ``Total`` column with the sum of the factor contributions.

        Returns:
            Contributions, dates by factors. ``Total`` is NaN on any date where a lagged loading
            or a factor return is missing, such as the warm-up rows.
        """
        factor_betas = self.get_asset_factor_betas(asset=asset)
        exposures = self.x
        attribution = exposures.multiply(factor_betas.shift(1))
        if add_total:
            total = attribution.sum(axis=1, skipna=False).rename('Total')
            attribution = pd.concat([total, attribution], axis=1, sort=True)
        return attribution

    def get_factor_alpha(self,
                         x: Optional[pd.DataFrame] = None,
                         y: Optional[pd.DataFrame] = None,
                         lag: Literal[0, 1] = 1,
                         span: Optional[int] = None
                         ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compute the model residual ``y[t] - sum_q B[t - lag, q] x[t, q]`` and the explained part.

        Loadings are forward-filled onto the index of ``x`` before the shift. The residual is
        called factor alpha, but it is the residual of a regression without an intercept, so it
        holds any intercept plus noise.

        Args:
            x: Factor returns, dates by factors; ``self.x`` when None.
            y: Asset returns, dates by assets, on the same index as ``x``; ``self.y`` when None.
            lag: 1 applies the loadings estimated on the previous date, a point-in-time
                step-ahead residual; 0 applies the same-date loadings, an in-sample fit.
            span: If given, the residual is smoothed by ``qis.compute_ewm`` with this span.

        Returns:
            Tuple of (factor_alpha, explained_returns), both dates by assets.

        Raises:
            AssertionError: If the indexes of ``x`` and ``y`` differ.
        """
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        assert x.index.equals(y.index)

        explained_returns = pd.DataFrame(0.0, index=y.index, columns=y.columns)
        for factor in x.columns:
            factor_betas = self.loadings[factor].reindex(index=x.index).ffill()
            explained_return = (factor_betas.shift(lag)).multiply(x[factor].to_numpy(), axis=0)
            explained_returns = explained_returns.add(explained_return)
        factor_alpha = y.subtract(explained_returns)
        if span is not None:
            factor_alpha = ewm.compute_ewm(data=factor_alpha, span=span)
        return factor_alpha, explained_returns

    def get_model_ewm_r2(self, span: int = 52, lag: Literal[0, 1] = 0) -> pd.DataFrame:
        """Compute the exponentially weighted, uncentred R² of each asset.

        ``R2[t] = 1 - sum_m lambda^(t-m) e[m]^2 / sum_m lambda^(t-m) y[m]^2`` with
        ``lambda = 1 - 2 / (span + 1)``, where ``e`` is the residual of ``get_factor_alpha(lag)``
        and both sums run over the same dates, those on which the residual and the return are
        finite, with decay counted over those dates only. The two moments therefore start
        together, after the warm-up, and no seed biases their ratio. The result is clipped to
        [0, 1].

        The R² is uncentred: both moments are about zero, not about a mean, which matches the
        regression through the origin that ``EwmLinearModel.fit`` runs by default. With the
        default ``lag=0`` the loadings dated ``t`` explain the return dated ``t``, which they
        were estimated from, so it is an in-sample fit; ``lag=1`` gives the point-in-time,
        step-ahead R².

        Args:
            span: EWM span of both second moments.
            lag: Loading lag passed to ``get_factor_alpha``: 0 in sample, 1 point in time.

        Returns:
            R², dates by assets, NaN before the first finite residual.
        """
        residuals, explained_returns = self.get_factor_alpha(lag=lag)
        y = self.y.reindex(index=residuals.index)
        is_joint = residuals.notna() & y.notna()
        ewm_residuals_2 = np.square(residuals).where(is_joint).ewm(
            span=span, adjust=True, ignore_na=True).mean()
        ewm_returns_2 = np.square(y).where(is_joint).ewm(
            span=span, adjust=True, ignore_na=True).mean()
        r_2 = 1.0 - ewm_residuals_2.divide(ewm_returns_2)
        r_2 = r_2.clip(0.0, 1.0)
        return r_2

    def get_model_residuals_corrs(self, span: int = 52) -> Tuple[pd.DataFrame, pd.Series]:
        """Compute the EWM correlation matrix of the in-sample residuals at the last date.

        The correlation comes from EWM second moments about zero of the lag-0 residuals of
        ``get_factor_alpha``, seeded at zero, evaluated at the last date only.

        Args:
            span: EWM span of the second moments.

        Returns:
            Tuple of (corr, avg_corr): the assets-by-assets correlation matrix and, per asset,
            the mean of its correlations with the other assets. ``avg_corr`` is NaN for a single
            asset.
        """
        residuals, explained_returns = self.get_factor_alpha(lag=0)
        corr = ewm.compute_ewm_covar(residuals.to_numpy(), span=span, is_corr=True)
        off_diagonal = np.where(np.eye(corr.shape[0], dtype=bool), np.nan, corr)
        with warnings.catch_warnings():  # a single asset has no off-diagonal correlation
            warnings.simplefilter('ignore', RuntimeWarning)
            avg_corr = pd.Series(np.nanmean(off_diagonal, axis=1), index=self.y.columns)
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

        For each date of ``x_covars``, the weights, the loadings and the residual variances are
        each read as of that date: the last row dated at or before it. With exposures
        ``x = B' w`` (``B`` assets by factors), the systematic variance is ``x' F x`` and the
        residual variance ``sum_i psi_i w_i^2``. A missing weight means the asset is not held.

        Args:
            weights: Portfolio weights DataFrame with dates as index and assets as columns.
                Each row represents portfolio weights at a specific date.
            factor_var_name: Name for the factor risk column in output. Defaults to 'Systematic'.
            idiosyncratic_var_name: Name for the idiosyncratic risk column in output. Defaults
                to 'Idiosyncratic'.

        Returns:
            Tuple containing total-risk contribution ratios, factor contributions normalized by
            total variance, factor contributions normalized by systematic variance, and the
            systematic and idiosyncratic portfolio variance panel. A date is NaN when an asset
            with a non-zero weight has a missing loading or residual variance, or when no
            residual variances are dated at or before it. A factor of ``x_covars`` that no asset
            loads on has exposure zero.

        Raises:
            ValueError: If factor covariance matrices or residual variances are unavailable.
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
            last_weight = weights.loc[weight_last_update_date, :].astype(float).fillna(0.0)
            # Loadings as of the date, factors by assets, on the weight universe
            last_betas = self.get_loadings_at_date(date=date).reindex(columns=last_weight.index)
            # Residual variances as of the date, like the weights and the loadings
            residual_last_update_date = find_upto_date_from_datetime_index(
                index=self.residual_vars.index, date=date)
            if residual_last_update_date is None:
                last_residual_var = pd.Series(np.nan, index=last_weight.index)
            else:
                last_residual_var = self.residual_vars.loc[residual_last_update_date, :].reindex(
                    index=last_weight.index)
            # Portfolio exposure to each factor, x = B' w, NaN if a held asset has no loading
            factor_exposures = _held_weighted_sum(values=last_betas, weights=last_weight)
            # A factor of the covariance that no asset loads on has zero exposure
            factor_exposures = factor_exposures.reindex(index=factor_covar.index, fill_value=0.0)
            # Calculate systematic portfolio variance: w'BFB'w where F is factor covariance
            portfolio_var_factors = factor_exposures.T @ (factor_covar @ factor_exposures)
            # Idiosyncratic portfolio variance w'Dw over the held assets, D the residual variances
            portfolio_var_idio = _held_weighted_sum(values=last_residual_var,
                                                    weights=np.square(last_weight))
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
        portfolio_var = pd.concat([portfolio_factor_vars, idio_vars], axis=1, sort=True)
        idio_vars_contrib = pd.Series(idio_vars_contrib)
        # Add idiosyncratic risk contribution as a separate column
        factor_risk_contrib_idio[idiosyncratic_var_name] = idio_vars_contrib.to_numpy()
        # Normalize all risk contributions to sum to 1.0 across factors for each date;
        # an undefined date stays NaN rather than becoming a zero share
        factor_rcs_ratios = factor_risk_contrib_idio.divide(
            factor_risk_contrib_idio.sum(axis=1, min_count=1), axis=0)
        # Return all computed metrics as a tuple
        return factor_rcs_ratios, factor_risk_contrib_idio, factor_risk_contrib, portfolio_var

    def compute_active_factor_risk(self,
                                   portfolio_weights: pd.DataFrame,
                                   benchmark_weights: pd.DataFrame
                                   ) -> Dict[str, pd.DataFrame]:
        """Compute factor-only active risk, excluding idiosyncratic risk.

        This legacy method forms portfolio, benchmark, and active factor exposures, then
        multiplies active exposures by their factor-covariance marginal risks. It does not
        include an idiosyncratic term. Use
        ``RiskModel.compute_tre_decomposition_at_date`` for systematic/residual tracking error
        or ``RiskModel.compute_marginal_tre_at_date`` for Euler tracking-error contributions.

        Args:
            portfolio_weights: Dated portfolio weights by asset.
            benchmark_weights: Dated benchmark weights by asset.

        Returns:
            Dictionary of portfolio, benchmark, and active factor exposures, marginal factor
            risks, absolute factor risk contributions, and normalized factor contributions.

        Warns:
            DeprecationWarning: On every direct call; the method remains available in this
                release with its body unchanged.
        """
        warnings.warn(
            "LinearModel.compute_active_factor_risk is deprecated; use "
            "qis.RiskModel.compute_tre_decomposition_at_date or "
            "qis.RiskModel.compute_marginal_tre_at_date",
            DeprecationWarning,
            stacklevel=2)
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
    """Split each period's portfolio return into benchmark contributions and a residual.

    On the date grid of ``portfolio_benchmark_betas``, with simple returns ``r_p`` of the NAV and
    ``r_q`` of each benchmark, the contribution of benchmark ``q`` at ``t`` is
    ``beta[t-1, q] * r_q[t]`` and the residual is ``r_p[t] - sum_q beta[t-1, q] * r_q[t]``. The
    identity is exact in simple returns for whatever betas are supplied. Betas from
    ``compute_portfolio_ewm_benchmark_betas`` are estimated on log returns; applying them to
    simple returns is a second-order approximation, because a log-return and a simple-return
    regression slope differ by terms of the order of the squared per-period return.

    Args:
        portfolio_nav: Portfolio NAV; read as of each beta date.
        benchmark_prices: Benchmark prices, one column per benchmark; read as of each beta date.
        portfolio_benchmark_betas: Portfolio betas, dates by benchmarks, applied one date later.
        residual_name: Name of the residual column.
        time_period: Optional period to which the output is restricted.

    Returns:
        Benchmark contributions and the residual, dates by columns. A row is NaN while any
        lagged beta is missing (the estimator warm-up), so the residual never absorbs the whole
        portfolio return.
    """
    benchmark_prices = benchmark_prices.reindex(index=portfolio_benchmark_betas.index, method='ffill')
    portfolio_nav = portfolio_nav.reindex(index=portfolio_benchmark_betas.index, method='ffill')
    x = ret.to_returns(prices=benchmark_prices, freq=None)
    x_attribution = (portfolio_benchmark_betas.shift(1)).multiply(x)
    total_attrib = x_attribution.sum(axis=1, skipna=False)
    total = portfolio_nav.pct_change()
    residual = np.subtract(total, total_attrib)
    joint_attrib = pd.concat([x_attribution, residual.rename(residual_name)], axis=1, sort=True)
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
    """Split each period's portfolio return into benchmark contributions and a residual.

    The contribution of benchmark ``q`` at ``t`` is ``beta[t-1, q] * r_q[t]`` and the residual
    is ``r_p[t] - sum_q beta[t-1, q] * r_q[t]``, on the return basis of the inputs. The row
    dated ``t`` attributes the return over ``(t-1, t]``; cumulate from the row after a base date
    to measure performance from that date.

    Args:
        portfolio_returns: Portfolio returns.
        benchmark_returns: Benchmark returns, one column per benchmark (or a Series).
        portfolio_benchmark_betas: Portfolio betas, dates by benchmarks, applied one date later.
        residual_name: Name of the residual column.
        time_period: Optional period to which the output is restricted.
        total_name: If given, prepend the portfolio return under this name.

    Returns:
        Benchmark contributions and the residual, plus the total when requested. Contributions
        and residual are NaN while any lagged beta is missing, including the first row; the
        total column always holds the portfolio return.
    """
    if isinstance(benchmark_returns, pd.Series):
        benchmark_returns = benchmark_returns.to_frame()
    # to be replaced with qis
    benchmark_returns = benchmark_returns.reindex(index=portfolio_returns.index)
    x_attribution = (portfolio_benchmark_betas.shift(1)).multiply(benchmark_returns)
    total_attrib = x_attribution.sum(axis=1, skipna=False)
    residual = np.subtract(portfolio_returns, total_attrib)
    joint_attrib = pd.concat([x_attribution, residual.rename(residual_name)], axis=1, sort=True)
    if total_name is not None:
        joint_attrib = pd.concat([portfolio_returns.rename(total_name), joint_attrib],
                                 axis=1, sort=True)
    if time_period is not None:
        joint_attrib = time_period.locate(joint_attrib)
    return joint_attrib

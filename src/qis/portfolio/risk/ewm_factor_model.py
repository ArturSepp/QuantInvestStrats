"""
the EWM estimator for ``LinearModel``: factor betas re-estimated at every date, not once.

``EwmLinearModel.fit`` runs ``compute_ewm_xy_beta_tensor`` over the factor panel ``x`` and the
asset panel ``y``, producing a (t, factors, assets) tensor unpacked into one loadings frame per
factor. Decay is set by ``span`` through λ = 1 - 2/(span + 1), overriding ``ewm_lambda``, and
``warmup_period`` blanks the opening observations, where a covariance built from a handful of
points gives betas that are large and meaningless.

``estimate_ewm_factor_model`` is the price-level entry point and works on log returns at
``freq``. ``compute_portfolio_ewm_benchmark_betas`` aggregates asset betas by weights, and
``compute_portfolio_benchmark_ewm_beta_alpha_attribution`` decomposes the nav into benchmark
betas and a residual. The base class is in ``qis/portfolio/risk/factor_model.py``.
"""
# packages
import numpy as np
import pandas as pd
from typing import Optional, Union

# qis
import qis as qis
from qis import TimePeriod
import qis.utils.df_ops as dfo
import qis.perfstats.returns as ret
import qis.models.linear.ewm as ewm
from qis.models.linear.ewm import MeanAdjType, InitType
from qis.portfolio.risk.factor_model import LinearModel, compute_benchmarks_beta_attribution_from_prices


class EwmLinearModel(LinearModel):
    """
    linear factor model with exponentially weighted time-varying loadings.

    Implements :class:`LinearModel` with EWM estimation: the betas are re-estimated at every date
    from an exponentially weighted covariance, so an exposure that changes is tracked rather than
    averaged away over the sample. Construct with the factor and asset panels, call ``fit``, then
    read the loadings through the LinearModel interface.

    Attributes:
        x: factor returns, shape (T, N), one column per factor
        y: asset returns, shape (T, M), one column per asset
        loadings: factor name to a (T, M) frame of time-varying betas, populated by ``fit``
    """

    def fit(self,
            span: Optional[int] = 31,
            ewm_lambda: float = 0.94,
            is_x_correlated: bool = True,
            mean_adj_type: MeanAdjType = MeanAdjType.NONE,
            init_type: InitType = InitType.MEAN,
            warmup_period: int = 20  # to avoid excessive betas at start,
            ) -> None:
        """Estimate time series EWM betas using exponential weighting.

        Args:
            span: Span for EWM calculation.
            ewm_lambda: Decay parameter for EWM.
            is_x_correlated: Whether to use diagonal (True) or full covariance matrix.
            mean_adj_type: Type of mean adjustment to apply.
            init_type: Initialization method for EWM.
        """
        x = self.x
        y = self.y
        if span is not None:
            ewm_lambda = 1.0 - 2.0 / (span + 1.0)
        if mean_adj_type != MeanAdjType.NONE:
            x = ewm.compute_rolling_mean_adj(data=x,
                                             mean_adj_type=mean_adj_type,
                                             ewm_lambda=ewm_lambda,
                                             init_type=init_type)

            y = ewm.compute_rolling_mean_adj(data=y,
                                             mean_adj_type=mean_adj_type,
                                             ewm_lambda=ewm_lambda,
                                             init_type=init_type)

        # compute list of betas using ewm numba recursion for cross product of x y and covariance of x
        # output is tensor of betas per date = [t, factors, assets]
        betas_ts = ewm.compute_ewm_xy_beta_tensor(x=x.to_numpy(),
                                                  y=y.to_numpy(),
                                                  ewm_lambda=ewm_lambda,
                                                  is_x_correlated=is_x_correlated,
                                                  warmup_period=warmup_period)
        # factor_loadings = {factor_id: pd.DataFrame(factor loadings)}
        loadings = dfo.np_txy_tensor_to_pd_dict(np_tensor_txy=betas_ts,
                                                dateindex=x.index,
                                                factor_names=x.columns.to_list(),
                                                asset_names=y.columns.to_list())
        self.x = x
        self.y = y
        self.loadings = loadings


def compute_portfolio_ewm_benchmark_betas(instrument_prices: pd.DataFrame,
                                          weights: pd.DataFrame,
                                          benchmark_prices: pd.DataFrame,
                                          time_period: TimePeriod = None,
                                          freq_beta: str = None,
                                          factor_beta_span: int = 63,  # quarter
                                          mean_adj_type: MeanAdjType = MeanAdjType.EWMA
                                          ) -> pd.DataFrame:
    """Compute portfolio benchmark betas using instrument exposures.

    Args:
        instrument_prices: Individual instrument price data.
        weights: Portfolio exposures to instruments.
        benchmark_prices: Benchmark price data.
        time_period: Optional time period filter.
        freq_beta: Frequency for return calculation.
        factor_beta_span: Span for EWM beta estimation.
        mean_adj_type: Mean adjustment method.

    Returns:
        Portfolio benchmark betas over time.
    """
    benchmark_prices = benchmark_prices.reindex(index=instrument_prices.index, method='ffill')
    ewm_linear_model = EwmLinearModel(x=ret.to_returns(prices=benchmark_prices, freq=freq_beta, is_log_returns=True),
                                      y=ret.to_returns(prices=instrument_prices, freq=freq_beta, is_log_returns=True))
    ewm_linear_model.fit(span=factor_beta_span, is_x_correlated=True, mean_adj_type=mean_adj_type)
    weights = weights.reindex(index=instrument_prices.index, method='ffill')
    benchmark_betas = ewm_linear_model.compute_agg_factor_exposures(weights=weights)
    benchmark_betas = benchmark_betas.replace({0.0: np.nan}).ffill()  # fillholidays
    if time_period is not None:
        benchmark_betas = time_period.locate(benchmark_betas)
    return benchmark_betas


def compute_portfolio_benchmark_ewm_beta_alpha_attribution(instrument_prices: pd.DataFrame,
                                                           weights: pd.DataFrame,
                                                           benchmark_prices: pd.DataFrame,
                                                           portfolio_nav: pd.Series,
                                                           time_period: TimePeriod = None,
                                                           freq_beta: str = None,
                                                           factor_beta_span: int = 63,  # quarter
                                                           residual_name: str = 'Alpha'
                                                           ) -> pd.DataFrame:
    """Compute portfolio beta-alpha attribution using benchmark decomposition.

    Args:
        instrument_prices: Individual instrument prices.
        weights: Portfolio weights to instruments.
        benchmark_prices: Benchmark prices for attribution.
        portfolio_nav: Portfolio NAV time series.
        time_period: Optional time period filter.
        freq_beta: Frequency for beta estimation.
        factor_beta_span: EWM span for beta calculation.
        residual_name: Name for alpha/residual component.

    Returns:
        Attribution breakdown including alpha component.
    """
    portfolio_benchmark_betas = compute_portfolio_ewm_benchmark_betas(instrument_prices=instrument_prices,
                                                                      weights=weights,
                                                                      benchmark_prices=benchmark_prices,
                                                                      time_period=None,
                                                                      freq_beta=freq_beta,
                                                                      factor_beta_span=factor_beta_span)
    joint_attrib = compute_benchmarks_beta_attribution_from_prices(portfolio_nav=portfolio_nav,
                                                                   benchmark_prices=benchmark_prices,
                                                                   portfolio_benchmark_betas=portfolio_benchmark_betas,
                                                                   residual_name=residual_name,
                                                                   time_period=time_period)
    return joint_attrib


def estimate_ewm_factor_model(asset_prices: Union[pd.Series, pd.DataFrame],
                              factor_prices: Union[pd.Series, pd.DataFrame],
                              freq: str = 'W-WED',
                              span: int = 26,
                              mean_adj_type: MeanAdjType = MeanAdjType.NONE
                              ) -> EwmLinearModel:
    """Estimate linear factor model from price data.

    Args:
        asset_prices: Asset price time series.
        factor_prices: Factor price time series.
        freq: Frequency for return calculation.
        span: EWM span for model estimation.
        mean_adj_type: Mean adjustment method.

    Returns:
        Fitted EWM linear model.
    """
    y = qis.to_returns(asset_prices, freq=freq, is_log_returns=True, drop_first=True)
    x = qis.to_returns(factor_prices, freq=freq, is_log_returns=True, drop_first=True)
    ewm_linear_model = EwmLinearModel(x=x.reindex(index=y.index), y=y)
    ewm_linear_model.fit(span=span, is_x_correlated=True, mean_adj_type=mean_adj_type)
    return ewm_linear_model

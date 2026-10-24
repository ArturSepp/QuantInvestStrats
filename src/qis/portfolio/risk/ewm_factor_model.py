"""
the EWM estimator for ``LinearModel``: factor betas re-estimated at every date, not once.

``EwmLinearModel.fit`` runs ``compute_ewm_xy_beta_tensor`` over the factor panel ``x`` and the
asset panel ``y``, producing a (t, factors, assets) tensor unpacked into one dates-by-assets
loadings frame per factor. Decay is set by ``span`` through λ = 1 - 2/(span + 1), overriding
``ewm_lambda``, and ``warmup_period`` blanks the opening observations, positions 0 to
``warmup_period``, where a covariance built from a handful of points gives betas that are large
and meaningless. Every estimate is point in time: the loadings dated t use returns up to t.

``estimate_ewm_factor_model`` is the price-level entry point and works on log returns at
``freq``. ``compute_portfolio_ewm_benchmark_betas`` aggregates asset betas by weights, and
``compute_portfolio_benchmark_ewm_beta_alpha_attribution`` decomposes the nav into benchmark
betas and a residual. The base class is in ``qis/portfolio/risk/factor_model.py``.
"""
# packages
import warnings

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
from qis.portfolio.risk.factor_model import (
    LinearModel, compute_benchmarks_beta_attribution_from_prices,
)


class EwmLinearModel(LinearModel):
    """
    linear factor model with exponentially weighted time-varying loadings.

    Implements :class:`LinearModel` with EWM estimation: the betas are re-estimated at every date
    from an exponentially weighted covariance, so an exposure that changes is tracked rather than
    averaged away over the sample. Construct with the factor and asset panels, call ``fit``, then
    read the loadings through the LinearModel interface.

    Attributes:
        x: factor returns, shape (T, K), one column per factor
        y: asset returns, shape (T, n), one column per asset
        loadings: factor name to a (T, n) dates-by-assets frame of time-varying betas, populated
            by ``fit``; row t of the frames of all factors is the assets-by-factors matrix B_t
    """

    def fit(self,
            span: Optional[int] = 31,
            ewm_lambda: float = 0.94,
            is_x_correlated: bool = True,
            mean_adj_type: MeanAdjType = MeanAdjType.NONE,
            init_type: InitType = InitType.X0,
            warmup_period: int = 20  # to avoid excessive betas at start,
            ) -> None:
        """Estimate time series EWM betas using exponential weighting.

        The loadings dated t solve the EWM least-squares problem on the returns up to and
        including t. ``x`` and ``y`` are left unchanged: any mean adjustment is applied to
        copies used for estimation only, so ``get_factor_alpha`` and ``get_model_ewm_r2`` work on
        the returns as supplied.

        Args:
            span: Span for EWM calculation; overrides ``ewm_lambda`` when given.
            ewm_lambda: Decay parameter for EWM, used when ``span`` is None.
            is_x_correlated: If True, invert the full factor cross-moment matrix; if False, use
                its diagonal only, which treats the factors as uncorrelated.
            mean_adj_type: Mean removed from both panels before the moments are formed. ``NONE``
                regresses through the origin; ``EWMA`` and ``EXPANDING`` are point in time;
                ``INSAMPLE`` subtracts the full-sample mean and is forward-looking.
            init_type: Seed of the EWMA mean when ``mean_adj_type`` is ``EWMA``; no effect
                otherwise. The default ``InitType.X0`` seeds with the first observation, which
                is point in time. ``InitType.MEAN`` seeds with the full-sample mean, a
                look-ahead whose weight in the mean at position t is lambda^(t+1) (about 0.24 at
                the first reported beta for span 31); it was the default before this release.
            warmup_period: Last position whose betas are left missing: positions 0 to
                ``warmup_period``, that is ``warmup_period + 1`` rows (21 by default), are NaN.

        Raises:
            ValueError: If the factor and asset return index labels or order do not match exactly.
        """
        if not self.x.index.equals(self.y.index):
            raise ValueError("x and y pandas index labels and order must match exactly")

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

        # Compute betas from the EWM cross product of x and y and the covariance of x.
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
        # self.x and self.y keep the supplied returns; the demeaned panels were used for the
        # moments only
        self.loadings = loadings


def _get_instrument_beta_estimation_start(instrument_prices: pd.DataFrame,
                                          benchmark_prices: pd.DataFrame,
                                          freq: str) -> pd.Timestamp:
    """Return the first paired instrument/factor return date used for estimation."""
    benchmark_prices = benchmark_prices.reindex(index=instrument_prices.index, method='ffill')
    instrument_returns = ret.to_returns(instrument_prices, freq=freq, is_log_returns=True)
    benchmark_returns = ret.to_returns(benchmark_prices, freq=freq, is_log_returns=True)
    paired = instrument_returns.notna().any(axis=1) & benchmark_returns.notna().all(axis=1)
    if not paired.any():
        raise ValueError('no paired instrument and benchmark returns for beta estimation')
    return pd.Timestamp(paired.index[paired][0])


def _aggregate_available_benchmark_betas(loadings: dict[str, pd.DataFrame],
                                         weights: pd.DataFrame) -> pd.DataFrame:
    """Sum available benchmark loadings at actual holdings, warning on partial coverage.

    This reporting policy is deliberately separate from LinearModel's strict aggregation
    for risk calculations. Missing instrument betas are not estimated or filled, and the
    surviving weights are not renormalised. No usable held loading means an unknown beta,
    not zero; genuinely all-cash weight rows have beta zero.
    """
    weights = weights.sort_index()
    exposures = {}
    coverage_messages = []
    for factor, loading in loadings.items():
        assets = weights.columns.union(loading.columns, sort=False)
        aligned = weights.reindex(index=loading.index, method='ffill').reindex(columns=assets)
        no_portfolio = aligned.isna().all(axis=1)
        aligned = aligned.fillna(0.0)
        held = aligned.ne(0.0)
        loading = loading.reindex(columns=assets)
        available = held & np.isfinite(loading)
        contributions = loading.where(available).multiply(aligned)
        total = contributions.sum(axis=1, min_count=1)
        total = total.mask(~held.any(axis=1), 0.0).where(~no_portfolio)
        exposures[factor] = total

        missing = held & ~available
        partial = available.any(axis=1) & missing.any(axis=1) & ~no_portfolio
        if partial.any():
            excluded = aligned.abs().where(missing, 0.0).sum(axis=1).loc[partial]
            affected = missing.loc[partial].any(axis=0)
            names = [str(asset) for asset in assets[affected]]
            shown = ', '.join(names[:5])
            if len(names) > 5:
                shown += f', ... ({len(names)} instruments)'
            coverage_messages.append(
                f'{factor}: {int(partial.sum())} dates '
                f'({excluded.index[0]:%d%b%Y}-{excluded.index[-1]:%d%b%Y}), '
                f'maximum excluded gross weight {100 * excluded.max():.3g}% of NAV; '
                f'instruments: {shown}'
            )
    if coverage_messages:
        warnings.warn(
            'Partial benchmark beta coverage: summing available instrument betas at actual '
            'weights; weights are not renormalised. Unavailable instruments remain unestimated '
            'and their returns remain in residual alpha. ' + '; '.join(coverage_messages),
            UserWarning,
            stacklevel=2,
        )
    return pd.DataFrame(exposures)


def compute_portfolio_ewm_benchmark_betas(instrument_prices: pd.DataFrame,
                                          weights: pd.DataFrame,
                                          benchmark_prices: pd.DataFrame,
                                          time_period: TimePeriod = None,
                                          freq_beta: str = None,
                                          factor_beta_span: int = 63,  # quarter
                                          mean_adj_type: MeanAdjType = MeanAdjType.EWMA
                                          ) -> pd.DataFrame:
    """Compute portfolio benchmark betas as weight-aggregated instrument betas.

    Instrument betas to all benchmarks jointly are EWM regressions of instrument log returns on
    benchmark log returns at ``freq_beta``, with a point-in-time mean adjustment seeded by the
    first observation. The portfolio beta to benchmark ``q`` on each beta date is
    ``sum_i w_i * beta_iq`` over available instrument estimates, with the weights in force on
    that date, selected as of it. A young instrument does not suppress other holdings' betas.
    Partial coverage emits a warning: weights are not renormalised, and the unestimated
    holdings' returns remain in residual alpha rather than receiving an invented beta.

    Args:
        instrument_prices: Individual instrument price data.
        weights: Portfolio exposures to instruments, dates by instruments.
        benchmark_prices: Benchmark price data, one column per benchmark.
        time_period: Optional time period filter.
        freq_beta: Frequency for return calculation; the price index when None.
        factor_beta_span: EWM span and number of jointly finite return observations withheld
            for each instrument's warm-up. The next observation can publish a beta.
        mean_adj_type: Mean adjustment method.

    Returns:
        Portfolio benchmark betas, beta dates by benchmarks. Rows are NaN before the first
        weight row or when no held instrument has a finite, warmed-up beta. A portfolio with
        zero weights (all cash) has beta zero. Missing estimates are never replaced by a prior.
    """
    benchmark_prices = benchmark_prices.reindex(index=instrument_prices.index, method='ffill')
    ewm_linear_model = EwmLinearModel(
        x=ret.to_returns(prices=benchmark_prices, freq=freq_beta, is_log_returns=True),
        y=ret.to_returns(prices=instrument_prices, freq=freq_beta, is_log_returns=True),
    )
    ewm_linear_model.fit(
        span=factor_beta_span,
        is_x_correlated=True,
        mean_adj_type=mean_adj_type,
        init_type=InitType.X0,
        warmup_period=factor_beta_span,
    )
    # Global panel age cannot warm up an instrument whose own return history starts later.
    paired_returns = ewm_linear_model.y.notna().mul(
        ewm_linear_model.x.notna().all(axis=1), axis=0)
    warmed_up = paired_returns.cumsum().gt(factor_beta_span)
    ewm_linear_model.loadings = {
        factor: loadings.where(warmed_up)
        for factor, loadings in ewm_linear_model.loadings.items()
    }
    weights = weights.reindex(index=instrument_prices.index, method='ffill')
    # weights are aligned as of each beta date, so no off-grid date needs filling; a genuine zero
    # beta (for example a portfolio fully in cash) is kept rather than replaced by the last one
    benchmark_betas = _aggregate_available_benchmark_betas(
        loadings=ewm_linear_model.loadings, weights=weights)
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

    The betas of ``compute_portfolio_ewm_benchmark_betas`` are estimated on log returns and
    applied, one beta date later, to the simple returns of the benchmarks and of the NAV by
    ``compute_benchmarks_beta_attribution_from_prices``. The attribution is an exact identity in
    simple returns; using log-return betas in it is a second-order approximation.

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
        Attribution breakdown including alpha component; rows are NaN while the lagged betas
        are missing.
    """
    portfolio_benchmark_betas = compute_portfolio_ewm_benchmark_betas(
        instrument_prices=instrument_prices, weights=weights, benchmark_prices=benchmark_prices,
        time_period=None, freq_beta=freq_beta, factor_beta_span=factor_beta_span,
    )
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

    Log returns at ``freq`` (first period dropped) are fitted with ``EwmLinearModel.fit`` using
    the full factor inversion and the default warm-up and mean seed.

    Args:
        asset_prices: Asset price time series.
        factor_prices: Factor price time series.
        freq: Frequency for return calculation.
        span: EWM span for model estimation.
        mean_adj_type: Mean adjustment method.

    Returns:
        Fitted EWM linear model whose ``x`` and ``y`` hold the log returns.
    """
    y = qis.to_returns(asset_prices, freq=freq, is_log_returns=True, drop_first=True)
    x = qis.to_returns(factor_prices, freq=freq, is_log_returns=True, drop_first=True)
    ewm_linear_model = EwmLinearModel(x=x.reindex(index=y.index), y=y)
    ewm_linear_model.fit(span=span, is_x_correlated=True, mean_adj_type=mean_adj_type)
    return ewm_linear_model

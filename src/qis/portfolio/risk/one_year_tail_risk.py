"""Compare annual Gaussian linear loss and compounded monthly bootstrap loss."""

from numbers import Integral

import numpy as np
import pandas as pd
from scipy.stats import norm

from qis.models.bootstrap.bootstrap_numba import bootstrap_indices_fixed_block
from qis.perfstats.tail_risk import compute_cvar
from qis.portfolio.risk.gaussian_var import (
    GaussianMonteCarlo, GaussianVarSpec, compute_gaussian_portfolio_risk,
)


def _matched_returns(returns, weights, as_of):
    """Find the latest uninterrupted run of complete months for all held assets."""
    if returns is None:
        return pd.DataFrame(), 'Monthly simple asset returns were not supplied.'
    if not isinstance(returns.index, pd.DatetimeIndex) or returns.index.hasnans:
        raise ValueError('Monthly returns require observed dates on a DatetimeIndex')
    sample = returns.loc[returns.index <= as_of].sort_index()
    months = sample.index.tz_localize(None).to_period('M')
    if not months.is_unique or not sample.columns.is_unique:
        raise ValueError('Supply one row per month and unique asset columns')
    held = weights.index[weights.ne(0).any(axis=1)]
    missing = held.difference(sample.columns)
    if len(missing):
        return pd.DataFrame(), 'Missing held assets: ' + ', '.join(map(str, missing))
    sample = sample.loc[:, held].astype(float)
    if np.isinf(sample.to_numpy()).any():
        raise ValueError('Monthly simple returns must be finite or missing')
    sample = sample.dropna()
    periods = sample.index.tz_localize(None).to_period('M').asi8
    gaps = np.flatnonzero(np.diff(periods) != 1)
    if gaps.size:
        sample = sample.iloc[gaps[-1] + 1:]
    portfolios = sample @ weights.loc[held]
    if len(sample) < 12:
        return portfolios, 'Fewer than 12 consecutive matched months.'
    if (portfolios < -1).any().any():
        return portfolios, 'A monthly portfolio loss exceeds capital in the supplied history.'
    return portfolios, None


def _annual_bootstrap_returns(monthly, indices):
    """Compound each twelve-month draw, then centre each portfolio across paths."""
    annual = np.prod(1.0 + monthly.to_numpy()[indices], axis=0) - 1.0
    return pd.DataFrame(annual - annual.mean(axis=0), columns=monthly.columns)


def compute_one_year_tail_risk(
        covariance: pd.DataFrame | None, portfolio_weights: pd.Series,
        benchmark_weights: pd.Series, as_of: pd.Timestamp, currency: str,
        monthly_returns: pd.DataFrame | None = None, *, covariance_date: pd.Timestamp | None = None,
        block_months: int = 3, simulations: int = 100_000, seed: int = 0) -> pd.DataFrame:
    """Compare 99% one-year VaR and ES as fractional losses in a supplied reference currency.

    Args:
        covariance: Annual asset covariance known at as_of, or None for unavailable Gaussian risk.
        portfolio_weights: Fixed strategy target exposures, reapplied each historical month.
        benchmark_weights: Fixed benchmark targets with the same return and currency convention.
        as_of: Inclusive cut-off for the inputs. Each monthly row denotes a completed month.
        currency: Reference currency of all inputs. The function performs no conversion.
        monthly_returns: Completed simple asset returns, one dated row per month. Use NaN
            for missing marks. Unallocated cash earns zero unless supplied as a held asset.
        covariance_date: Vintage of the supplied covariance. Defaults to as_of.
        block_months: Fixed circular block length in months, a divisor of twelve.
        simulations: Number of paths per simulated method. Gaussian draws are antithetic.
        seed: Common seed passed separately to the Gaussian and circular bootstrap engines.

    Returns:
        Rows for the analytical Gaussian, Gaussian simulation and fixed-block bootstrap.
        Risk cells are full-precision decimal fractions. Coverage and vintage columns travel
        with each row when it is exported. Monte Carlo ES is unavailable.
    """
    if not isinstance(currency, str) or not currency.strip():
        raise ValueError('Specify the reference currency of the covariance and returns')
    if (isinstance(block_months, bool) or not isinstance(block_months, Integral)
            or block_months < 1 or 12 % block_months):
        raise ValueError('block_months must be a positive divisor of 12')
    as_of = pd.Timestamp(as_of)
    if pd.isna(as_of):
        raise ValueError('Specify an observed as_of date')
    if not isinstance(seed, Integral) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer between 0 and 2**32 - 1')
    spec = GaussianVarSpec(simulations=simulations, random_seed=seed)
    for exposure in (portfolio_weights, benchmark_weights):
        if not exposure.index.is_unique or not np.isfinite(exposure.to_numpy(dtype=float)).all():
            raise ValueError('Target weights must be finite and uniquely labelled')
    weights = pd.concat([portfolio_weights, benchmark_weights], axis=1).fillna(0.0)
    weights.columns = ['strategy', 'benchmark']
    methods = ['Gaussian analytic', 'Gaussian Monte Carlo', 'Historical bootstrap']
    columns = ['strategy_var', 'strategy_es', 'benchmark_var', 'benchmark_es']
    table = pd.DataFrame(np.nan, index=pd.Index(methods, name='method'), columns=columns)
    table['as_of'] = as_of
    table['currency'] = currency.strip().upper()
    table['confidence'] = spec.confidence
    table['horizon_months'] = 12
    table['block_months'] = pd.array([None, None, block_months], dtype='Int64')
    vintage = pd.Timestamp(covariance_date) if covariance_date is not None else as_of
    if covariance is not None and (pd.isna(vintage) or vintage > as_of):
        raise ValueError('Covariance must be dated on or before as_of')
    table['covariance_date'] = [vintage if covariance is not None else pd.NaT] * 2 + [pd.NaT]
    gaussian_coverage = 'No annual covariance available at the as-of date.'
    if covariance is not None:
        risk = compute_gaussian_portfolio_risk(
            covariance, portfolio_weights, benchmark_weights, vintage, GaussianMonteCarlo(spec))
        z = norm.ppf(spec.confidence)
        es_multiple = norm.pdf(z) / (1 - spec.confidence)
        pvol, bvol = risk['portfolio_volatility'], risk['benchmark_volatility']
        table.loc[methods[0], columns] = [risk['analytical_var'], es_multiple * pvol,
                                         z * bvol, es_multiple * bvol]
        table.loc[methods[1], ['strategy_var', 'benchmark_var']] = [
            risk['portfolio_var'], risk['benchmark_var']]
        gaussian_coverage = 'Annual covariance, zero-drift linear loss.'
    monthly, reason = _matched_returns(monthly_returns, weights, as_of)
    table['coverage'] = [gaussian_coverage, gaussian_coverage,
                         reason or 'Latest consecutive matched monthly returns.']
    table['paths'] = pd.array([None, simulations if covariance is not None else 0,
                               simulations if reason is None else 0], dtype='Int64')
    table['seed'] = pd.array([None, seed if covariance is not None else None,
                              seed if reason is None else None], dtype='Int64')
    table['matched_months'] = pd.array([None, None, len(monthly)], dtype='Int64')
    table['sample_start'] = [pd.NaT, pd.NaT, monthly.index.min() if len(monthly) else pd.NaT]
    table['sample_end'] = [pd.NaT, pd.NaT, monthly.index.max() if len(monthly) else pd.NaT]
    if reason is None:
        indices = bootstrap_indices_fixed_block(
            len(monthly), num_samples=simulations, num_bootstrap_index=12,
            block_size=block_months, seed=seed)
        centered = _annual_bootstrap_returns(monthly, indices)
        var = (-centered).quantile(spec.confidence, interpolation='linear')
        es = compute_cvar(centered, confidence_level=spec.confidence)
        table.loc[methods[2], columns] = [var.strategy, es.strategy, var.benchmark, es.benchmark]
    return table

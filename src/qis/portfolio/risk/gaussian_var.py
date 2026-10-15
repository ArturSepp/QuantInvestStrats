"""Gaussian VaR for supplied covariance, with benchmark-relative risk diagnostics.

Covariance is per unit of horizon (normally annualised); horizon_years then uses
years. Fixed exposures, zero drift and linear P&L are assumed. No covariance
estimation, return compounding or nonlinear repricing occurs here. The existing
price-based EWM VaR functions remain unchanged.
"""
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd
from scipy.stats import norm

from qis.portfolio.risk.contributions import fill_unavailable_unheld
from qis.portfolio.risk.risk_model import RiskModel


@dataclass(frozen=True)
class GaussianVarSpec:
    """Settings for zero-drift linear Gaussian Monte Carlo VaR.

    Attributes:
        confidence: Loss probability quantile, strictly between 0.5 and 1.
        horizon_years: Positive horizon in units of the supplied covariance.
        simulations: Even number of antithetic scenarios, at least 1,000.
        random_seed: Seed for a local NumPy generator; global RNG state is untouched.
    """
    confidence: float = 0.99
    horizon_years: float = 1.0
    simulations: int = 500_000
    random_seed: int = 0

    def __post_init__(self):
        if not 0.5 < self.confidence < 1.0:
            raise ValueError('confidence must lie strictly between 0.5 and 1')
        if not np.isfinite(self.horizon_years) or self.horizon_years <= 0:
            raise ValueError('horizon_years must be finite and positive')
        if (isinstance(self.simulations, bool) or not isinstance(self.simulations, Integral)
                or self.simulations < 1000 or self.simulations % 2):
            raise ValueError('simulations must be an even integer of at least 1000')


class GaussianMonteCarlo:
    """Reusable antithetic normal draws for portfolio and benchmark marginal VaR.

    Projecting an asset Gaussian onto two linear portfolios has the same
    distribution as simulating all asset shocks and applying their weights.
    The first shock represents the benchmark; the second the portfolio residual.
    Identical portfolios have exactly unit VaR ratio. Draws can be reused across
    reporting dates for reproducible common-random-number comparisons.

    Attributes:
        spec: Confidence, horizon, scenario count and random seed.
    """

    def __init__(self, spec: GaussianVarSpec):
        self.spec = spec
        draws = np.random.default_rng(spec.random_seed).standard_normal(
            (spec.simulations // 2, 2))
        self.draws = np.concatenate([draws, -draws], axis=0)
        self.benchmark_quantile = float(np.quantile(-self.draws[:, 0], spec.confidence))

    def var(self, portfolio_variance: float, benchmark_variance: float,
            cross_covariance: float) -> tuple[float, float]:
        """Compute loss-positive portfolio and benchmark VaR.

        Args:
            portfolio_variance: Portfolio variance per covariance time unit.
            benchmark_variance: Benchmark variance in the same units.
            cross_covariance: Portfolio/benchmark covariance in the same units.

        Returns:
            Portfolio and benchmark loss quantiles over the configured horizon.

        Raises:
            ValueError: If joint moments are nonfinite or not positive semidefinite.
        """
        moments = np.array([[portfolio_variance, cross_covariance],
                            [cross_covariance, benchmark_variance]], dtype=float)
        if not np.isfinite(moments).all():
            raise ValueError('Monte Carlo requires finite portfolio/benchmark covariance')
        scale = max(float(np.max(np.abs(moments))), np.finfo(float).tiny)
        if np.linalg.eigvalsh(moments).min() < -1e-10 * scale:
            raise ValueError('portfolio/benchmark covariance is not positive semidefinite')
        pvar, bvar = max(portfolio_variance, 0.0), max(benchmark_variance, 0.0)
        root_horizon = np.sqrt(self.spec.horizon_years)
        bvol = np.sqrt(bvar)
        if bvar > 0:
            loading = cross_covariance / bvol
            residual_variance = pvar - loading * loading
            if residual_variance < -1e-10 * scale:
                raise ValueError('negative conditional portfolio variance')
            if abs(residual_variance) <= 1e-12 * scale:
                residual_variance = 0.0
            losses = -(loading * self.draws[:, 0]
                       + np.sqrt(max(residual_variance, 0.0)) * self.draws[:, 1])
        else:
            losses = -np.sqrt(pvar) * self.draws[:, 1]
        return (float(np.quantile(losses, self.spec.confidence) * root_horizon),
                float(self.benchmark_quantile * bvol * root_horizon))


def compute_gaussian_portfolio_risk(
        covariance: pd.DataFrame, portfolio_weights: pd.Series,
        benchmark_weights: pd.Series, date: pd.Timestamp,
        simulation: GaussianMonteCarlo,
) -> dict[str, float]:
    """Compute Monte Carlo VaR, benchmark beta and TE from supplied covariance.

    Args:
        covariance: Labelled asset covariance per unit of simulation horizon.
        portfolio_weights: Finite, uniquely labelled fractional exposures.
        benchmark_weights: Benchmark exposures in the same asset universe.
        date: Date of the supplied covariance; no as-of selection is performed here.
        simulation: Reusable QIS Gaussian Monte Carlo engine.

    Returns:
        Monte Carlo VaRs, their ratio, ex-ante beta, tracking error, portfolio and
        benchmark volatility, closed-form portfolio VaR and relative MC error.
        Risk measures inherit covariance units; VaR applies the configured horizon.
        A zero-variance benchmark makes beta and VaR ratio NaN.

    Raises:
        ValueError: For missing held risk, invalid weights or invalid covariance.
    """
    if not covariance.index.equals(covariance.columns):
        raise ValueError('covariance row and column labels must match')
    for label, weights in [('portfolio', portfolio_weights), ('benchmark', benchmark_weights)]:
        if not weights.index.is_unique or not np.isfinite(weights.to_numpy(dtype=float)).all():
            raise ValueError(f'{label} weights must be uniquely labelled and finite')
        outside = weights.loc[~weights.index.isin(covariance.index)]
        if (outside.abs() > 1e-10).any():
            missing = outside[outside.abs() > 1e-10].index.tolist()
            raise ValueError(f'{label} assets missing covariance: {missing}')
    p = portfolio_weights.reindex(covariance.index, fill_value=0.0)
    b = benchmark_weights.reindex(covariance.index, fill_value=0.0)
    covariance = fill_unavailable_unheld(covariance, p, b)
    risk_model = RiskModel(covar={pd.Timestamp(date): covariance})
    zero = pd.Series(0.0, index=covariance.index)
    pvol = risk_model.compute_tre_at_date(zero, p, date, strict=True)
    bvol = risk_model.compute_tre_at_date(zero, b, date, strict=True)
    cross = float(p @ covariance @ b)
    pvar, bvar = simulation.var(pvol ** 2, bvol ** 2, cross)
    analytical_var = float(norm.ppf(simulation.spec.confidence) * pvol
                           * np.sqrt(simulation.spec.horizon_years))
    return {
        'portfolio_var': pvar,
        'benchmark_var': bvar,
        'var_ratio': pvar / bvar if bvar > 0 else np.nan,
        'ex_ante_beta': risk_model.compute_benchmark_beta_at_date(b, p, date, strict=True)
        if bvol > 0 else np.nan,
        'tracking_error': risk_model.compute_tre_at_date(b, p, date, strict=True),
        'portfolio_volatility': pvol,
        'benchmark_volatility': bvol,
        'analytical_var': analytical_var,
        'mc_relative_error': pvar / analytical_var - 1.0 if analytical_var > 0 else 0.0,
    }

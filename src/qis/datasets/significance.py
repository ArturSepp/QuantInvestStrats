"""Known-target calibration fixtures; not the frozen synthetic market generator.

Residuals obey e[t] = phi*e[t-1] + u[t], with Q = sigma²((1-rho)I+rho11').
Stationary covariance is Q/(1-phi²), long-run covariance Q/(1-phi)². Independent
zero-mean IID factors give responses with population intercept zero. Units are
fractional simple returns per observation on a monthly calibration grid.
"""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import pandas as pd
from scipy.signal import lfilter


@dataclass(frozen=True)
class SignificanceDgp:
    """Parameters for a known-target stationary calibration sample.

    Attributes:
        nobs: Retained observations, at least 24.
        num_series: Positive number of responses.
        correlation: Innovation cross-series correlation in [0, 1).
        ar_coefficient: Common persistence in [-0.9, 0.9].
        innovation_std: Positive per-observation innovation volatility.
        factor_count: Independent exogenous factor count, one through six.
        tail_df: None for Gaussian; otherwise Student degrees of freedom > 4.
    """

    nobs: int
    num_series: int = 12
    correlation: float = 0.0
    ar_coefficient: float = 0.0
    innovation_std: float = 0.02
    factor_count: int = 3
    tail_df: float | None = None

    def __post_init__(self) -> None:
        for name, value, minimum in [('nobs', self.nobs, 24),
                                     ('num_series', self.num_series, 1),
                                     ('factor_count', self.factor_count, 1)]:
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
                    or value < minimum):
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if self.factor_count > 6:
            raise ValueError('factor_count must be <= 6')
        for name in ('correlation', 'ar_coefficient', 'innovation_std', 'tail_df'):
            value = getattr(self, name)
            if name == 'tail_df' and value is None:
                continue
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not np.isfinite(value)):
                raise ValueError(f'{name} must be finite and real')
        if not 0 <= self.correlation < 1 or abs(self.ar_coefficient) > 0.9:
            raise ValueError('require correlation in [0, 1) and |ar_coefficient| <= 0.9')
        if self.innovation_std <= 0:
            raise ValueError('innovation_std must be positive')
        if self.tail_df is not None and self.tail_df <= 4:
            raise ValueError('tail_df must exceed four')

    @property
    def innovation_covariance(self) -> np.ndarray:
        """Return Q in squared per-observation units."""
        return self.innovation_std ** 2 * (
            (1 - self.correlation) * np.eye(self.num_series) + self.correlation
        )

    @property
    def stationary_covariance(self) -> np.ndarray:
        """Return Q/(1-phi²)."""
        return self.innovation_covariance / (1 - self.ar_coefficient ** 2)

    @property
    def long_run_covariance(self) -> np.ndarray:
        """Return Q/(1-phi)²."""
        return self.innovation_covariance / (1 - self.ar_coefficient) ** 2


@dataclass(frozen=True)
class SignificanceSample:
    """Known-null sample and construction diagnostics.

    Attributes:
        mean_returns: Stationary AR residuals with population mean zero.
        alpha_returns: Residuals plus factors times fixed slopes; intercept zero.
        factors: Independent IID Gaussian factors with volatility 0.015.
        loadings: Fixed factor-by-response slope matrix.
        innovations: Innovations including burn-in, for path verification.
        burn_in: Discarded observations from the initial zero state.
    """

    mean_returns: pd.DataFrame
    alpha_returns: pd.DataFrame
    factors: pd.DataFrame
    loadings: np.ndarray
    innovations: np.ndarray
    burn_in: int


def generate_significance_sample(dgp: SignificanceDgp, seed: int) -> SignificanceSample:
    """Draw from independent local residual/factor RNG streams.

    Args:
        dgp: Model dimensions, dependence, innovation volatility and tail law.
        seed: Nonnegative integer; caller/global RNG state is unchanged.

    Returns:
        Zero-mean/zero-intercept sample. A 1000-observation burn-in bounds the
        initial-state multiplier by 0.9**1000. Exogenous zero-mean factors imply
        intercept asymptotic long-run covariance equals residual long-run covariance.
    """
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    residual_seed, factor_seed = np.random.SeedSequence(int(seed)).spawn(2)
    rng = np.random.default_rng(residual_seed)
    burn_in = 1000
    shape = (dgp.nobs + burn_in, dgp.num_series + 1)
    if dgp.tail_df is None:
        independent = rng.standard_normal(shape)
    else:
        independent = rng.standard_t(dgp.tail_df, size=shape) * np.sqrt(
            (dgp.tail_df - 2) / dgp.tail_df,
        )
    innovations = dgp.innovation_std * (
        np.sqrt(dgp.correlation) * independent[:, :1]
        + np.sqrt(1 - dgp.correlation) * independent[:, 1:]
    )
    residuals = lfilter([1.0], [1.0, -dgp.ar_coefficient], innovations, axis=0)[burn_in:]
    factors = np.random.default_rng(factor_seed).standard_normal(
        (dgp.nobs, dgp.factor_count),
    ) * 0.015
    loadings = 0.2 + 0.1 * ((np.arange(dgp.factor_count)[:, None]
                            + np.arange(dgp.num_series)[None, :]) % 7)
    dates = pd.date_range('1900-01-31', periods=dgp.nobs, freq='ME', name='observation')
    columns = pd.Index([f'H{i:02d}' for i in range(dgp.num_series)], name='hypothesis')
    return SignificanceSample(
        mean_returns=pd.DataFrame(residuals, index=dates, columns=columns),
        alpha_returns=pd.DataFrame(residuals + factors @ loadings, index=dates, columns=columns),
        factors=pd.DataFrame(factors, index=dates,
                             columns=[f'F{i}' for i in range(dgp.factor_count)]),
        loadings=loadings, innovations=innovations, burn_in=burn_in,
    )

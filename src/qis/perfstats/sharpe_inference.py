"""Paired Sharpe-ratio inference using the Ledoit--Wolf (2008) delta method.

Uses unprewhitened Bartlett HAC, not the paper's studentized bootstrap or its
prewhitened quadratic-spectral estimator. DOI: 10.1016/j.jempfin.2008.03.002.
"""

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from qis.utils import regression


@dataclass(frozen=True)
class SharpeDiffTest:
    """Annualised arithmetic Sharpe difference and paired HAC inference.

    Attributes:
        sr1_an: First series' arithmetic Sharpe, using population moments (ddof=0).
        sr2_an: Second series' arithmetic Sharpe, using population moments (ddof=0).
        diff_an: Annualised first-minus-second Sharpe difference.
        se_an: Annualised HAC standard error of that difference.
        t_stat: Difference divided by its standard error; a normal-reference z statistic.
        p_value: Two-sided asymptotic normal p-value for equal Sharpe ratios.
        n_obs: Number of complete paired observations used.
        n_lags: Number of Bartlett autocovariance lags used.
    """

    sr1_an: float
    sr2_an: float
    diff_an: float
    se_an: float
    t_stat: float
    p_value: float
    n_obs: int
    n_lags: int


def sharpe_difference_test(
        returns1: pd.Series,
        returns2: pd.Series,
        n_lags: Optional[int] = 3,
        af: float = 12.0,
) -> SharpeDiffTest:
    """Test equal arithmetic Sharpe ratios on paired periodic returns with HAC.

    Implements the delta-method framework of Ledoit and Wolf (2008), section 3.1,
    with a Bartlett/Newey-West long-run covariance estimate, no prewhitening and
    no finite-sample correction. It is not their studentized bootstrap test.
    Population moments (ddof=0) define both Sharpe estimates and their gradients.
    Positive constant rescaling of either return series leaves inference unchanged.

    Args:
        returns1: First numeric return Series, with unique, sortable, nonmissing index labels.
            Supply simple periodic excess returns to test excess Sharpes; no cash adjustment,
            fee deduction, resampling or return conversion is performed internally.
        returns2: Second return Series on the same sampling grid and return basis.
            Index intersection is sorted and pairs containing NaN are dropped. Lags count
            retained observations, so the caller must resolve material calendar gaps first.
        n_lags: Bartlett truncation, an integer from zero through T-1; defaults to 3,
            matching QIS's regression HAC helpers. Explicit None selects
            floor(4*(T/100)**(2/9)); zero allows nonnormality but ignores serial dependence.
        af: Positive finite periods per year, used only for reporting (default 12 for monthly).
            Both the difference and its standard error scale by sqrt(af); p-values do not.

    Returns:
        Annualised Sharpes, their difference and standard error, normal-reference statistic,
        two-sided p-value, sample size and selected lag count. Identical or positively
        proportional returns have zero difference/standard error and, by convention, z=0, p=1.

    Raises:
        ValueError: If inputs/settings are invalid, fewer than 24 complete pairs remain,
            a return variance is numerically degenerate, or inference is undefined.

    Notes:
        Covariance uses QIS's estimate_hac_mean on the paired influence series. Its
        T/(T-1) correction is removed to retain this test's uncorrected delta-method convention.
        Asymptotic validity requires stationary paired returns, sufficient finite moments
        and weak dependence. A significant result does not correct for strategy selection.
        The field t_stat is retained for familiarity; it does not use Student's t distribution.
    """
    for series in (returns1, returns2):
        if not isinstance(series, pd.Series):
            raise ValueError('returns must be pandas Series')
        if not series.index.is_unique or series.index.hasnans:
            raise ValueError('return indexes must be unique and nonmissing')
        if (not pd.api.types.is_numeric_dtype(series.dtype)
                or pd.api.types.is_complex_dtype(series.dtype)
                or pd.api.types.is_bool_dtype(series.dtype)):
            raise ValueError('returns must be real numeric Series')
        if np.isinf(series.to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError('returns must not contain infinity')
    if (isinstance(af, (bool, np.bool_)) or not isinstance(af, Real)
            or not np.isfinite(af) or af <= 0):
        raise ValueError('af must be positive and finite')
    joint = pd.concat([returns1.rename('r1'), returns2.rename('r2')],
                      axis=1, join='inner', sort=False).sort_index().dropna()
    n_obs = len(joint)
    if n_obs < 24:
        raise ValueError(f'joint sample must have at least 24 observations, got {n_obs}')
    if n_lags is None:
        n_lags = regression.newey_west_lag_rule(n_obs)
    if (isinstance(n_lags, (bool, np.bool_)) or not isinstance(n_lags, Integral)
            or not 0 <= n_lags < n_obs):
        raise ValueError('n_lags must be an integer between zero and T-1')

    # Each column's positive scale cancels from its Sharpe and influence function.
    # Scaling first avoids overflow/underflow in moments for extreme return units.
    values = joint.to_numpy(dtype=float)
    scale = np.max(np.abs(values), axis=0)
    if np.any(scale == 0):
        raise ValueError('degenerate return variance')
    values = values / scale
    mu = np.mean(values, axis=0)
    centered = values - mu
    sigma = np.sqrt(np.mean(centered ** 2, axis=0))
    if np.any(sigma <= 10 * np.finfo(float).eps):
        raise ValueError('degenerate return variance relative to scale')
    sr = mu / sigma

    # Algebraically y_t @ gradient in the four-moment delta method, evaluated
    # using centered, standardized returns to avoid subtracting raw second moments.
    z = centered / sigma
    influence = z - 0.5 * sr * (z ** 2 - 1.0)
    paired = influence[:, 0] - influence[:, 1]
    paired -= paired.mean()  # remove floating-point centering residuals
    diff = float(sr[0] - sr[1])
    tolerance = 64 * np.finfo(float).eps
    if (np.max(np.abs(paired)) <= tolerance * max(1.0, np.max(np.abs(influence)))
            and abs(diff) <= tolerance * max(1.0, np.max(np.abs(sr)))):
        diff, se, statistic, p_value = 0.0, 0.0, 0.0, 1.0
    else:
        # Reuse QIS's Bartlett HAC mean estimator. Remove its one-parameter
        # T/(T-1) variance correction for the uncorrected Sharpe delta method.
        hac_mean = regression.estimate_hac_mean(paired, hac_lags=int(n_lags))
        se = float(hac_mean.hac_se * np.sqrt((n_obs - 1) / n_obs))
        if not np.isfinite(se) or se <= 0:
            raise ValueError('Sharpe difference has no positive finite HAC variance')
        statistic = diff / se
        p_value = float(2 * norm.sf(abs(statistic)))
    sqrt_af = np.sqrt(af)
    return SharpeDiffTest(
        sr1_an=float(sqrt_af * sr[0]), sr2_an=float(sqrt_af * sr[1]),
        diff_an=float(sqrt_af * diff), se_an=float(sqrt_af * se),
        t_stat=float(statistic), p_value=p_value, n_obs=n_obs, n_lags=int(n_lags),
    )

"""
the regime-mixture covariance and the Gaussian-implied regime moments of a benchmark.

When each asset loads on the benchmark with a regime-specific beta and an idiosyncratic residual,
the law of total covariance gives the unconditional covariance from the regime betas and the
benchmark's regime moments alone:

    Sigma = af [ sum_s p_s beta_s beta_s' S_s - (sum_s p_s beta_s m_s)(sum_s p_s beta_s m_s)' ]
            + diag(af idio_var)

with ``m_s`` and ``S_s`` the benchmark's regime mean and second moment. The benchmark enters as
an asset with unit betas and zero residual, and equal regime betas reduce the expression to the
single-factor covariance. ``compute_gaussian_regime_moments`` supplies ``m_s`` and ``S_s`` under a
Gaussian benchmark from one volatility, the zero-estimation variant of these inputs.
"""
# packages
import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Optional, Sequence, Tuple, Union
# qis
from qis.regimes.partition import get_partition_quantiles, get_regime_ids, get_regime_probabilities
from qis.regimes.nulls import _edge_densities


def compute_regime_mixture_covar(betas: pd.DataFrame,
                                 idio_vars: pd.Series,
                                 benchmark_regime_means: pd.Series,
                                 benchmark_regime_second_moments: pd.Series,
                                 af: float,
                                 regime_probs: Optional[pd.Series] = None
                                 ) -> pd.DataFrame:
    """Annualised regime-mixture covariance by the law of total covariance.

    Args:
        betas: regime betas, assets in rows and regimes in columns; include the benchmark with
            unit betas to carry it in the covariance
        idio_vars: per-period residual variance of each asset, zero for the benchmark
        benchmark_regime_means: per-period benchmark mean in each regime
        benchmark_regime_second_moments: per-period benchmark second moment in each regime
        af: annualisation factor of the periodic moments
        regime_probs: probability of each regime; None is ``get_regime_probabilities()``, the
            one-sigma cut, which requires the Bear, Normal and Bull columns

    Returns:
        the annualised covariance, indexed by the assets of ``betas``
    """
    if regime_probs is None:
        regime_probs = get_regime_probabilities()
    regimes = list(betas.columns)
    second = sum(regime_probs[g] * np.outer(betas[g], betas[g]) * benchmark_regime_second_moments[g]
                 for g in regimes)
    mean_vec = sum(regime_probs[g] * betas[g] * benchmark_regime_means[g] for g in regimes)
    factor = (second - np.outer(mean_vec, mean_vec)) * af
    return pd.DataFrame(factor + np.diag(idio_vars.reindex(betas.index) * af),
                        index=betas.index, columns=betas.index)


def compute_gaussian_regime_moments(benchmark_vol: float,
                                    benchmark_mean: float = 0.0,
                                    q: Union[Sequence[float], np.ndarray, None] = None
                                    ) -> Tuple[pd.Series, pd.Series]:
    """Regime means and second moments of a Gaussian benchmark, from its periodic volatility.

    For the bucket between the standard-normal quantiles ``a`` and ``b`` with probability ``p``,
    the conditional mean is ``(phi(a) - phi(b)) / p`` and the conditional second moment
    ``1 + (a phi(a) - b phi(b)) / p`` in units of the volatility. On the one-sigma cut the Bear
    mean is -1.521 sigma, the tail variances 0.200 sigma^2 and the Normal variance 0.288 sigma^2.

    Args:
        benchmark_vol: periodic volatility of the benchmark
        benchmark_mean: periodic mean of the benchmark
        q: partition probabilities; None is the one-sigma cut

    Returns:
        the regime means and the regime second moments, per regime id
    """
    q = get_partition_quantiles(q)
    probs = get_regime_probabilities(q).to_numpy()
    phi = _edge_densities(q=q, nu=None)
    z = np.zeros(q.size)  # z phi(z) is zero at the infinite ends
    for i in range(1, q.size - 1):
        z[i] = norm.ppf(q[i]) if q[i] >= 0.5 else -norm.ppf(1.0 - q[i])
    mean_z = (phi[:-1] - phi[1:]) / probs
    second_z = 1.0 + (z[:-1] * phi[:-1] - z[1:] * phi[1:]) / probs
    ids = get_regime_ids(q)
    means = pd.Series(mean_z * benchmark_vol + benchmark_mean, index=ids)
    variances = pd.Series((second_z - mean_z ** 2) * benchmark_vol ** 2, index=ids)
    return means, variances + means ** 2

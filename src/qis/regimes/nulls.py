"""
the Gaussian and Student-t nulls of the regime Sharpe decomposition.

``compute_regime_sharpe_decomposition`` splits a Sharpe ratio into additive regime contributions
``sr_s = sqrt(af) p_s m_s / sigma``, with the regimes set by the quantiles of a benchmark's
returns. When the asset and the benchmark are jointly Gaussian, the conditional mean of the asset
is linear in the benchmark, and every contribution has a closed form in the asset's Sharpe ratio
``SR`` and its correlation ``rho`` with the benchmark:

    sr_s = p_s SR + rho k_s,
    k_s = sqrt(af) E[Z 1{Z in bucket s}] = sqrt(af) (phi(z_a) - phi(z_b))

for the bucket between the standard-normal quantiles ``z_a`` and ``z_b``. The loadings ``k_s``
sum to zero, so the null contributions sum to ``SR``. On the one-sigma cut the Bear loading is
``-kappa`` and the Bull loading ``+kappa``, with ``kappa = sqrt(af) phi(z_0.84)``: 0.487 for
quarterly and 0.843 for monthly regimes. A joint Student-t pair keeps the linear conditional
mean, and ``phi(z)`` becomes ``(nu + t^2) / (nu - 1) f_nu(t)`` scaled to unit variance.

The convexity premium is the excess of the realised lowest-bucket contribution over its null,
``CP = sr_Bear - (p SR - kappa rho)``. The null holds for partitions of the benchmark's own
returns; for regimes set by volatility or sign the decomposition still adds up, but these
loadings do not apply.

``compute_portfolio_bear_sharpe`` and ``compute_overlay_blend_frontier`` are the closed forms of
the aggregation identity and of a benchmark-overlay blend in these coordinates.
"""
# packages
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as student_t
from typing import Optional, Sequence, Union
# qis
from qis.regimes.partition import (get_partition_quantiles, get_regime_ids,
                                   get_regime_probabilities, _is_symmetric)


def _edge_densities(q: np.ndarray, nu: Optional[float]) -> np.ndarray:
    """Truncated first-moment kernel at each partition edge, zero at the two infinite ends.

    For a standard normal the kernel is phi(z); for a Student-t margin it is
    (nu + t^2) / (nu - 1) f_nu(t), so that E[T 1{a < T < b}] = g(a) - g(b). A symmetric partition
    evaluates its upper half as the mirror image of the lower half, which makes the loadings
    exactly antisymmetric. The Gaussian lower edges go through the upper quantile, -ppf(1 - q), and
    the Student-t edges through the lower one, the quantiles the published kappa values used.
    """
    g = np.zeros(q.size)
    interior = range(1, q.size - 1)
    for i in interior:
        if nu is None:
            z = norm.ppf(q[i]) if q[i] >= 0.5 else -norm.ppf(1.0 - q[i])
            g[i] = norm.pdf(z)
        else:
            t_q = student_t.ppf(q[i], df=nu)
            g[i] = (nu + t_q ** 2) / (nu - 1.0) * student_t.pdf(t_q, df=nu)
    if _is_symmetric(q):
        for i in interior:
            if q[i] > 0.5:
                g[i] = g[q.size - 1 - i]
    return g


def compute_regime_null_loadings(af: float,
                                 q: Union[Sequence[float], np.ndarray, None] = None,
                                 nu: Optional[float] = None,
                                 regime_ids: Optional[Sequence[str]] = None
                                 ) -> pd.Series:
    """Correlation loading of each regime contribution under the Gaussian or Student-t null.

    The null contribution of regime ``s`` is ``p_s SR + rho k_s``, with
    ``k_s = sqrt(af) E[Z 1{Z in bucket s}]`` for a unit-variance margin ``Z``.

    Args:
        af: annualisation factor of the periodic returns, 4 for quarterly regimes
        q: partition probabilities; None is the one-sigma cut
        nu: degrees of freedom of a joint Student-t null, above 2; None is the Gaussian null
        regime_ids: ids of the buckets; None uses the defaults of ``get_regime_ids``

    Returns:
        loading per regime id, summing to zero

    Raises:
        ValueError: if ``af`` is not positive or ``nu`` is not above 2
    """
    if af <= 0.0:
        raise ValueError(f"af must be positive, got {af!r}")
    if nu is not None and nu <= 2.0:
        raise ValueError(f"nu must exceed 2 for a finite-variance t null, got {nu!r}")
    q = get_partition_quantiles(q)
    g = _edge_densities(q=q, nu=nu)
    loadings = np.sqrt(af) * (g[:-1] - g[1:])
    if nu is not None:
        loadings = loadings / np.sqrt(nu / (nu - 2.0))  # unit-variance t margin
    return pd.Series(loadings, index=get_regime_ids(q, regime_ids))


def compute_regime_kappa(af: float,
                         tail_prob: float = 0.16,
                         nu: Optional[float] = None
                         ) -> float:
    """Kappa of the symmetric three-bucket partition, the Sharpe cost of one unit of correlation.

    ``kappa = sqrt(af) phi(z_{1 - tail_prob})`` under the Gaussian null: 0.487 at ``af = 4`` and
    0.843 at ``af = 12`` on the one-sigma cut. A unit-variance Student-t margin puts more mass near
    the centre and in the far tails, so at the one-sigma cut its kappa lies below the Gaussian
    value and rises to it as ``nu`` grows, while at tail probabilities of 10% and less it lies
    above and falls to it.

    Args:
        af: annualisation factor of the periodic returns
        tail_prob: probability of each tail bucket, in (0, 0.5)
        nu: degrees of freedom of a joint Student-t null; None is the Gaussian null

    Returns:
        kappa, the Bull loading and minus the Bear loading

    Raises:
        ValueError: if ``tail_prob`` is outside (0, 0.5)
    """
    if not 0.0 < tail_prob < 0.5:
        raise ValueError(f"tail_prob must be in (0, 0.5), got {tail_prob!r}")
    q = np.array([0.0, tail_prob, 1.0 - tail_prob, 1.0])
    return float(-compute_regime_null_loadings(af=af, q=q, nu=nu).iloc[0])


def calibrate_student_t_nu(excess_kurtosis: float,
                           min_nu: float = 4.5
                           ) -> Optional[float]:
    """Student-t degrees of freedom matching a sample excess kurtosis, ``nu = 4 + 6 / kurt``.

    Args:
        excess_kurtosis: sample excess kurtosis of the benchmark returns
        min_nu: floor of the result, above 4 so that the matched fourth moment exists

    Returns:
        the degrees of freedom, or None when the excess kurtosis is not positive and the Gaussian
        null is the tighter benchmark
    """
    if excess_kurtosis <= 0.0:
        return None
    return float(max(4.0 + 6.0 / excess_kurtosis, min_nu))


def compute_null_regime_contributions(sr: float,
                                      rho: float,
                                      af: float,
                                      q: Union[Sequence[float], np.ndarray, None] = None,
                                      nu: Optional[float] = None,
                                      regime_ids: Optional[Sequence[str]] = None
                                      ) -> pd.Series:
    """Null regime contributions ``p_s SR + rho k_s`` of an asset, which sum to its Sharpe ratio.

    Args:
        sr: annualised Sharpe ratio of the asset
        rho: correlation of the asset with the benchmark
        af: annualisation factor of the periodic returns
        q: partition probabilities; None is the one-sigma cut
        nu: degrees of freedom of a joint Student-t null; None is the Gaussian null
        regime_ids: ids of the buckets; None uses the defaults of ``get_regime_ids``

    Returns:
        null contribution per regime id

    Raises:
        ValueError: if ``rho`` is outside [-1, 1]
    """
    if not -1.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [-1, 1], got {rho!r}")
    loadings = compute_regime_null_loadings(af=af, q=q, nu=nu, regime_ids=regime_ids)
    probs = get_regime_probabilities(q=q, regime_ids=regime_ids)
    return probs * sr + loadings * rho


def compute_convexity_premium(sr_bear: float,
                              sr: float,
                              rho: float,
                              af: float,
                              tail_prob: float = 0.16,
                              nu: Optional[float] = None
                              ) -> float:
    """Convexity premium, the realised Bear contribution less its null ``p SR - kappa rho``.

    Args:
        sr_bear: realised contribution of the lowest benchmark bucket to the Sharpe ratio
        sr: annualised Sharpe ratio of the asset
        rho: correlation of the asset with the benchmark
        af: annualisation factor of the periodic returns
        tail_prob: probability of each tail bucket
        nu: degrees of freedom of a joint Student-t null; None is the Gaussian null

    Returns:
        the premium in annualised Sharpe units
    """
    kappa = compute_regime_kappa(af=af, tail_prob=tail_prob, nu=nu)
    return sr_bear - (tail_prob * sr - kappa * rho)


def compute_portfolio_bear_sharpe(weights: np.ndarray,
                                  vols: np.ndarray,
                                  srs: np.ndarray,
                                  rhos: np.ndarray,
                                  premia: np.ndarray,
                                  portfolio_vol: float,
                                  af: float,
                                  tail_prob: float = 0.16
                                  ) -> float:
    """Bear contribution of a portfolio by the aggregation identity.

    ``sr_bear_p = p SR_p - kappa rho_p + sum_i (w_i sigma_i / sigma_p) CP_i``, where ``SR_p`` and
    ``rho_p`` are the risk-weighted sums of the asset Sharpe ratios and correlations.

    Args:
        weights: asset weights
        vols: annualised asset volatilities
        srs: annualised asset Sharpe ratios
        rhos: asset correlations with the benchmark
        premia: asset convexity premia
        portfolio_vol: annualised portfolio volatility
        af: annualisation factor of the periodic returns
        tail_prob: probability of each tail bucket

    Returns:
        the Bear contribution of the portfolio

    Raises:
        ValueError: if the inputs differ in shape or ``portfolio_vol`` is not positive
    """
    arrays = [np.asarray(x, dtype=float) for x in (weights, vols, srs, rhos, premia)]
    if len({x.shape for x in arrays}) != 1:
        raise ValueError(f"inputs must share one shape, got {[x.shape for x in arrays]!r}")
    if portfolio_vol <= 0.0:
        raise ValueError(f"portfolio_vol must be positive, got {portfolio_vol!r}")
    weights, vols, srs, rhos, premia = arrays
    kappa = compute_regime_kappa(af=af, tail_prob=tail_prob)
    risk_weights = weights * vols / portfolio_vol
    sr_p = float(np.sum(risk_weights * srs))
    rho_p = float(np.sum(risk_weights * rhos))
    premium_p = float(np.sum(risk_weights * premia))
    return tail_prob * sr_p - kappa * rho_p + premium_p


def compute_overlay_blend_frontier(sr_b: float,
                                   vol_b: float,
                                   sr_a: float,
                                   vol_a: float,
                                   rho: float,
                                   af: float,
                                   premium_a: float = 0.0,
                                   overlay_weights: Optional[np.ndarray] = None,
                                   tail_prob: float = 0.16
                                   ) -> pd.DataFrame:
    """Closed-form frontier of the blends ``(1 - x)`` benchmark plus ``x`` overlay.

    Args:
        sr_b: annualised Sharpe ratio of the benchmark
        vol_b: annualised volatility of the benchmark
        sr_a: annualised Sharpe ratio of the overlay
        vol_a: annualised volatility of the overlay
        rho: correlation of the overlay with the benchmark
        af: annualisation factor of the periodic returns
        premium_a: convexity premium of the overlay; zero gives the Gaussian-null frontier
        overlay_weights: overlay weights ``x``; None is 0 to 1 in steps of 0.05
        tail_prob: probability of each tail bucket

    Returns:
        per overlay weight: ``portfolio_vol``, ``sharpe``, ``benchmark_corr`` and ``bear_sharpe``
    """
    if overlay_weights is None:
        overlay_weights = np.linspace(0.0, 1.0, 21)
    x = np.asarray(overlay_weights, dtype=float)
    kappa = compute_regime_kappa(af=af, tail_prob=tail_prob)
    mu = (1.0 - x) * sr_b * vol_b + x * sr_a * vol_a
    vol_p = np.sqrt(np.square((1.0 - x) * vol_b) + np.square(x * vol_a)
                    + 2.0 * x * (1.0 - x) * rho * vol_a * vol_b)
    sr_p = mu / vol_p
    rho_p = ((1.0 - x) * vol_b + x * rho * vol_a) / vol_p
    bear_p = tail_prob * sr_p - kappa * rho_p + x * vol_a / vol_p * premium_a
    return pd.DataFrame(dict(portfolio_vol=vol_p, sharpe=sr_p, benchmark_corr=rho_p,
                             bear_sharpe=bear_p),
                        index=pd.Index(x, name='overlay_weight'))

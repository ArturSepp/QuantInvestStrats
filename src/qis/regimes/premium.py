"""
the regime premium table: decomposition, null and convexity premium per asset.

``compute_regime_premium_table`` takes the frame a regime classifier returns, periodic returns
with a regime column, and reports for every asset its Sharpe ratio, benchmark correlation and
annualised volatility, its additive regime contributions ``sr_s = sqrt(af) p_s m_s / sigma``, the
null of its lowest-bucket (Bear) contribution, the convexity premium, and the benchmark-adjusted
premium ``CP* = CP - rho CP_B``, which nets the benchmark's own departure from the null at the
asset's correlation. ``p_s`` is the realised share of periods in regime ``s``; the null uses the
partition's population probability. The benchmark row carries ``CP* = 0``.

``compute_regime_premium_bootstrap`` resamples each asset's paired returns with the stationary
block bootstrap and reclassifies the regimes inside every resample, so its standard error
includes the sampling error of the regime cutoffs.

Column names follow the regime ids: the one-sigma cut gives ``bear_sharpe``, ``normal_sharpe``,
``bull_sharpe``, ``null_bear_sharpe`` and ``bear_return_pa``, and Q1 to Qn partitions give
``q1_sharpe`` and so on, with the premium columns always on the lowest bucket.
"""
# packages
import numpy as np
import pandas as pd
from typing import Optional, Sequence, Union
# qis
from qis.perfstats.perf_stats import compute_sharpe_arithmetic
from qis.models.bootstrap.bootstrap_numba import BootstrapType, generate_bootstrapped_indices
from qis.regimes.nulls import compute_regime_null_loadings
from qis.regimes.partition import (REGIME_COLUMN, classify_quantile_buckets, get_ordered_regimes,
                                   get_partition_quantiles, get_regime_ids,
                                   get_regime_probabilities)


def compute_regime_premium_table(sampled_returns_with_regime_id: pd.DataFrame,
                                 benchmark: str,
                                 af: float,
                                 q: Union[Sequence[float], np.ndarray, None] = None,
                                 nu: Optional[float] = None,
                                 regime_column: str = REGIME_COLUMN
                                 ) -> pd.DataFrame:
    """Regime decomposition, null and convexity premium of every asset of a sampled frame.

    Moments are equal-weighted over the periods with a regime: ``sigma`` and the regime means
    ``m_s`` over each asset's available returns, ``p_s`` over all classified periods, and the
    Sharpe ratio by ``compute_sharpe_arithmetic`` with ``ddof=1``.

    Args:
        sampled_returns_with_regime_id: periodic returns with a regime column, as returned by
            ``BenchmarkReturnsQuantilesRegime.compute_sampled_returns_with_regime_id`` or
            ``create_sampled_returns_with_regime_id``
        benchmark: name of the benchmark column
        af: annualisation factor of the periodic returns
        q: the partition that produced the regimes; None is the one-sigma cut
        nu: degrees of freedom of an additional Student-t null; None reports the Gaussian only
        regime_column: name of the regime column

    Returns:
        one row per asset and the columns ``sharpe``, ``rho``, ``ann_vol``, one ``<id>_sharpe``
        per regime, ``null_<tail>_sharpe``, ``convexity_premium``, ``cp_star`` and
        ``<tail>_return_pa``, the annualised return contribution of the lowest bucket; with ``nu``
        also ``null_<tail>_sharpe_t`` and ``convexity_premium_t``

    Raises:
        ValueError: if the benchmark is missing or the regime labels are not the ids of ``q``
    """
    if benchmark not in sampled_returns_with_regime_id.columns:
        raise ValueError(f"benchmark {benchmark!r} is not a column of the sampled returns")
    q = get_partition_quantiles(q)
    regime_ids = get_regime_ids(q)
    data = sampled_returns_with_regime_id.dropna(subset=[regime_column])
    labels = get_ordered_regimes(data[regime_column])
    if not set(labels) <= set(regime_ids):
        raise ValueError(f"regime labels {labels} are not the ids {regime_ids} of q={q.tolist()}")
    rets = data.drop(columns=regime_column)
    sigma = rets.std()
    sharpe = compute_sharpe_arithmetic(returns=rets, af=af, ddof=1)
    rho = rets.corrwith(rets[benchmark])
    out = pd.DataFrame({'sharpe': sharpe, 'rho': rho, 'ann_vol': sigma * np.sqrt(af)})
    p_s = data[regime_column].value_counts(normalize=True)
    for regime in regime_ids:
        m_s = rets[data[regime_column] == regime].mean()
        out[f"{regime.lower()}_sharpe"] = np.sqrt(af) * p_s[regime] * m_s / sigma
    tail = regime_ids[0].lower()
    tail_prob = get_regime_probabilities(q).iloc[0]
    loading = compute_regime_null_loadings(af=af, q=q).iloc[0]
    out[f"null_{tail}_sharpe"] = tail_prob * out['sharpe'] + loading * out['rho']
    out['convexity_premium'] = out[f"{tail}_sharpe"] - out[f"null_{tail}_sharpe"]
    out['cp_star'] = out['convexity_premium'] - out['rho'] * out.loc[benchmark, 'convexity_premium']
    out[f"{tail}_return_pa"] = out['ann_vol'] * out[f"{tail}_sharpe"]
    if nu is not None:
        loading_t = compute_regime_null_loadings(af=af, q=q, nu=nu).iloc[0]
        out[f"null_{tail}_sharpe_t"] = tail_prob * out['sharpe'] + loading_t * out['rho']
        out['convexity_premium_t'] = out[f"{tail}_sharpe"] - out[f"null_{tail}_sharpe_t"]
    return out


def compute_regime_premium_bootstrap(returns: pd.DataFrame,
                                     benchmark: str,
                                     af: float,
                                     q: Union[Sequence[float], np.ndarray, None] = None,
                                     block_size: int = 8,
                                     n_boot: int = 2000,
                                     seed: int = 7,
                                     ci: float = 0.95
                                     ) -> pd.DataFrame:
    """Stationary block-bootstrap standard error and interval of each asset's convexity premium.

    Each asset is resampled jointly with the benchmark over their common periods with
    ``BootstrapType.STATIONARY`` (geometric blocks of mean ``block_size``, circular wrap), and
    the regimes are reclassified inside every resample with the ``pd.qcut`` convention, so the
    interval includes the sampling error of the regime cutoffs. Every asset uses the same
    ``seed``.

    Args:
        returns: periodic returns, one column per asset, including the benchmark
        benchmark: name of the benchmark column
        af: annualisation factor of the periodic returns
        q: partition probabilities; None is the one-sigma cut
        block_size: mean block length in periods
        n_boot: number of resamples
        seed: seed of the resampling indices
        ci: coverage of the percentile interval

    Returns:
        one row per asset other than the benchmark and the columns ``premium_se``, the standard
        deviation of the resampled premia with ``ddof=1``, ``premium_ci_low`` and
        ``premium_ci_high``

    Raises:
        ValueError: if an asset has fewer than five blocks of common periods
    """
    q = get_partition_quantiles(q)
    tail_prob = get_regime_probabilities(q).iloc[0]
    loading = compute_regime_null_loadings(af=af, q=q).iloc[0]
    rows = {}
    for asset in [c for c in returns.columns if c != benchmark]:
        joint = returns[[asset, benchmark]].dropna().to_numpy(dtype=float)
        n = len(joint)
        if n < 5 * block_size:
            raise ValueError(f"{asset}: need at least {5 * block_size} common periods, got {n}")
        # generate_bootstrapped_indices returns the samples in the columns
        indices = np.asarray(generate_bootstrapped_indices(num_data_index=n,
                                                           bootstrap_type=BootstrapType.STATIONARY,
                                                           num_samples=n_boot,
                                                           index_length=n,
                                                           block_size=block_size,
                                                           seed=seed))
        premia = np.zeros(n_boot)
        for b in range(n_boot):
            sample = joint[indices[:, b], :]
            r_a, r_b = sample[:, 0], sample[:, 1]
            tail = classify_quantile_buckets(r_b, q=q) == 0
            sigma = np.std(r_a, ddof=1)
            sr = np.sqrt(af) * np.mean(r_a) / sigma
            sr_tail = np.sqrt(af) * np.mean(tail) * np.mean(r_a[tail]) / sigma
            rho = np.corrcoef(r_a, r_b)[0, 1]
            premia[b] = sr_tail - (tail_prob * sr + loading * rho)
        rows[asset] = {'premium_se': float(np.std(premia, ddof=1)),
                       'premium_ci_low': float(np.quantile(premia, (1.0 - ci) / 2.0)),
                       'premium_ci_high': float(np.quantile(premia, (1.0 + ci) / 2.0))}
    return pd.DataFrame.from_dict(rows, orient='index')

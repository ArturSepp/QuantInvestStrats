"""
the benchmark-return partition shared by every function of ``qis.regimes``.

A partition is a vector of probabilities ``q`` running from 0 to 1; bucket ``s`` holds the
periods whose benchmark return lies between the ``q[s]`` and ``q[s + 1]`` sample quantiles. The
default is the one-sigma cut ``[0.0, 0.16, 0.84, 1.0]`` with the ids Bear, Normal and Bull, the
default of ``BenchmarkReturnsQuantilesRegime`` and of ``compute_regime_sharpe_decomposition``.
Other bucket counts use the ordered ids Q1 to Qn, again as the classifier does.

Classification is ``pd.qcut``: buckets are closed on the right and the first includes the sample
minimum, so a return equal to an interior quantile falls in the lower bucket. The numpy
classifier used inside the bootstrap loops, ``classify_quantile_buckets``, follows the same rule,
which matters there because a resample repeats observations and quantiles land on data points.

``create_sampled_returns_with_regime_id`` builds, from a panel of periodic returns, the frame a
``RegimeClassifier`` returns from prices, so that every analytic of ``qis.regimes`` takes one
input format whatever produced the regimes.
"""
# packages
import numpy as np
import pandas as pd
from typing import List, Optional, Sequence, Union
# qis
from qis.perfstats.regime_classifier import RegimeClassifier

ONE_SIGMA_QUANTILES = (0.0, 0.16, 0.84, 1.0)
REGIME_COLUMN = RegimeClassifier.REGIME_COLUMN


def get_partition_quantiles(q: Union[Sequence[float], np.ndarray, None] = None) -> np.ndarray:
    """Validated partition probabilities, the one-sigma cut when none are given.

    Args:
        q: probabilities from 0 to 1, strictly increasing, with at least two buckets

    Returns:
        the probabilities as a float array

    Raises:
        ValueError: if ``q`` does not run from 0 to 1, is not strictly increasing, or has fewer
            than two buckets
    """
    q = np.asarray(ONE_SIGMA_QUANTILES if q is None else q, dtype=float)
    if q.ndim != 1 or q.size < 3:
        raise ValueError(f"q needs at least two buckets, got {q!r}")
    if q[0] != 0.0 or q[-1] != 1.0 or np.any(np.diff(q) <= 0.0):
        raise ValueError(f"q must increase strictly from 0.0 to 1.0, got {q!r}")
    return q


def get_regime_ids(q: Union[Sequence[float], np.ndarray, None] = None,
                   regime_ids: Optional[Sequence[str]] = None
                   ) -> List[str]:
    """Ordered regime ids of a partition, from the lowest benchmark bucket up.

    Args:
        q: partition probabilities; None is the one-sigma cut
        regime_ids: explicit ids, one per bucket; None uses Bear, Normal, Bull for three buckets
            and Q1 to Qn otherwise

    Returns:
        the ids in bucket order

    Raises:
        ValueError: if ``regime_ids`` does not have one id per bucket
    """
    n_buckets = get_partition_quantiles(q).size - 1
    if regime_ids is None:
        if n_buckets == 3:
            return ['Bear', 'Normal', 'Bull']
        return [f"Q{n + 1}" for n in range(n_buckets)]
    regime_ids = [str(x) for x in regime_ids]
    if len(regime_ids) != n_buckets:
        raise ValueError(f"{n_buckets} buckets need {n_buckets} regime ids, got {regime_ids!r}")
    return regime_ids


def _is_symmetric(q: np.ndarray) -> bool:
    """Whether the partition is symmetric about the median, q[i] + q[n - i] = 1."""
    return bool(np.allclose(q + q[::-1], 1.0, rtol=0.0, atol=1e-12))


def get_regime_probabilities(q: Union[Sequence[float], np.ndarray, None] = None,
                             regime_ids: Optional[Sequence[str]] = None
                             ) -> pd.Series:
    """Population probability of each bucket, the width of its probability interval.

    A partition symmetric about the median gives its mirrored buckets exactly equal
    probabilities: ``1 - 0.84`` is not ``0.16`` in floating point, and the difference would
    otherwise leak into every mixture moment.

    Args:
        q: partition probabilities; None is the one-sigma cut
        regime_ids: ids of the buckets; None uses the defaults of ``get_regime_ids``

    Returns:
        probability per regime id, summing to one
    """
    q = get_partition_quantiles(q)
    probs = np.diff(q)
    if _is_symmetric(q):
        n = probs.size
        for i in range(n // 2):
            probs[n - 1 - i] = probs[i]
    return pd.Series(probs, index=get_regime_ids(q, regime_ids))


def classify_quantile_buckets(x: np.ndarray,
                              q: Union[Sequence[float], np.ndarray, None] = None
                              ) -> np.ndarray:
    """Bucket number of each observation, with the ``pd.qcut`` convention, for numpy loops.

    Args:
        x: one-dimensional observations without missing values
        q: partition probabilities; None is the one-sigma cut

    Returns:
        integer bucket of each observation, 0 for the lowest

    Raises:
        ValueError: if the interior quantiles coincide, so that a bucket would be empty
    """
    q = get_partition_quantiles(q)
    edges = np.quantile(np.asarray(x, dtype=float), q)
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError(f"quantile edges are not unique for q={q.tolist()}: {edges.tolist()}")
    # right-closed buckets: a value equal to an interior edge belongs to the lower bucket
    return np.searchsorted(edges[1:-1], x, side='left')


def create_sampled_returns_with_regime_id(returns: pd.DataFrame,
                                          benchmark: str,
                                          q: Union[Sequence[float], np.ndarray, None] = None,
                                          regime_ids: Optional[Sequence[str]] = None
                                          ) -> pd.DataFrame:
    """Periodic returns with the benchmark-quantile regime of each period, from returns.

    The counterpart of ``BenchmarkReturnsQuantilesRegime.compute_sampled_returns_with_regime_id``
    for a caller who holds periodic returns rather than prices: no resampling, and the same
    ``pd.qcut`` classification of the benchmark column. Periods without a benchmark return get no
    regime.

    Args:
        returns: periodic returns, one column per asset, including the benchmark
        benchmark: name of the benchmark column
        q: partition probabilities; None is the one-sigma cut
        regime_ids: ids of the buckets; None uses the defaults of ``get_regime_ids``

    Returns:
        a copy of ``returns`` with the categorical regime column ``RegimeClassifier.REGIME_COLUMN``

    Raises:
        ValueError: if the benchmark is missing, has fewer than three returns, or is too
            degenerate for unique quantile edges
    """
    if benchmark not in returns.columns:
        raise ValueError(f"benchmark {benchmark!r} is not a column of returns: "
                         f"{list(returns.columns)}")
    q = get_partition_quantiles(q)
    labels = get_regime_ids(q, regime_ids)
    x = returns[benchmark]
    x_valid = x.dropna().to_numpy(dtype=float)
    if x_valid.size < 3:
        raise ValueError(f"need at least 3 benchmark returns to classify, got {x_valid.size}")
    if np.unique(np.nanquantile(x_valid, q)).size <= len(labels):
        raise ValueError(f"benchmark {benchmark!r} is degenerate for q={q.tolist()}: "
                         f"the quantile edges are not unique")
    out = returns.copy()
    out[REGIME_COLUMN] = pd.qcut(x=x, q=q, labels=labels)
    return out


def get_ordered_regimes(regime_column: pd.Series) -> List[str]:
    """Regime ids of a sampled frame's regime column, in bucket order.

    Args:
        regime_column: the regime column, categorical as the classifiers produce it or plain

    Returns:
        the categories of a categorical column, otherwise Bear, Normal, Bull when the labels are
        those, otherwise the sorted labels
    """
    if isinstance(regime_column.dtype, pd.CategoricalDtype):
        return [str(x) for x in regime_column.cat.categories]
    labels = set(regime_column.dropna().astype(str))
    default = ['Bear', 'Normal', 'Bull']
    return default if labels <= set(default) else sorted(labels)

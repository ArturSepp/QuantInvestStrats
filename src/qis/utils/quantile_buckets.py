"""
one rule for assigning observations to quantile buckets, used by every quantile classification
in qis.

The regime classifiers, the regime Sharpe decomposition, ``qis.regimes`` and the quantile hue
buckets of the plots all classify through this module, so a period is Bear in one exhibit exactly
when it is Bear in every other. The rule, case by case:

- Partition: an integer ``k`` gives the probabilities ``i / k``; a probability that is not exact
  in binary is rounded up to the next double, as ``pd.qcut`` does. An explicit vector must run
  strictly from 0 to 1.
- Edges: the interior edges are sample quantiles by linear interpolation between order statistics
  at position ``h = (n - 1) p`` (Hyndman-Fan type 7, the numpy and pandas default), computed with
  numpy's floating-point steps. A position within ``POSITION_TOLERANCE`` of a whole number is taken
  as that number, so the order statistic itself is the edge and float noise cannot decide a bucket.
- Assignment: buckets are closed on the right with open outer ends, ``(-inf, e1], (e1, e2], ...,
  (e_k-1, +inf)``; an observation's bucket is the number of interior edges strictly below it.
- Ties: an observation equal to an interior edge falls in the lower bucket. Block-bootstrap
  resamples repeat observations and put them on edges routinely, so this case is not academic.
- Extremes: the sample minimum is in the first bucket and the maximum in the last; with edges from
  another sample, observations beyond them fall in the outer buckets rather than being dropped.
- Missing values: NaN and infinite observations are left out of the edges and get no bucket, code
  -1 or a missing label.
- Occupancy: when the edges are estimated from the data, every bucket must hold an observation,
  else ``EmptyQuantileBucketError`` reports how many are occupied; callers that draw rather than
  estimate may waive the check.

Within these rules the classification equals ``pd.qcut`` on the same data, which the tests of this
module check on continuous, tied and gappy samples. ``BenchmarkReturnsPositiveNegativeRegime`` is a
sign rule rather than a quantile partition and keeps its own definition: a zero return is Positive.
"""
# packages
import numpy as np
import pandas as pd
from typing import List, Optional, Sequence, Union

POSITION_TOLERANCE = 1e-9

QuantileSpec = Union[int, Sequence[float], np.ndarray]


class EmptyQuantileBucketError(ValueError):
    """Raised when estimated quantile edges leave a bucket without observations.

    Attributes:
        num_occupied: number of buckets holding at least one observation
        num_buckets: number of buckets requested
        edges: sorted unique values of the full edges, the sample minimum and maximum included
    """

    def __init__(self, num_occupied: int, num_buckets: int, edges: List[float]):
        """Store the occupancy diagnostics and compose the message."""
        self.num_occupied = num_occupied
        self.num_buckets = num_buckets
        self.edges = edges
        super().__init__(f"only {num_occupied} of {num_buckets} quantile buckets are non-empty "
                         f"(edges={edges})")


def _to_float_array(x: Union[np.ndarray, pd.Series, Sequence[float]]) -> np.ndarray:
    """Float array of the observations, pandas missing values of any dtype as NaN."""
    if isinstance(x, (pd.Series, pd.Index, pd.api.extensions.ExtensionArray)):
        return x.to_numpy(dtype=float, na_value=np.nan)
    return np.asarray(x, dtype=float)


def get_quantile_probabilities(q: QuantileSpec) -> np.ndarray:
    """Probabilities of a partition, from 0 to 1.

    Args:
        q: number of equal buckets, a positive integer, or explicit probabilities increasing
            strictly from 0 to 1

    Returns:
        the probabilities; for an integer the fractions ``i / q``, those not exact in binary
        rounded up to the next double

    Raises:
        ValueError: if ``q`` is not a positive integer or a valid probability vector
    """
    if np.isscalar(q):
        if isinstance(q, (bool, np.bool_)) or not float(q).is_integer() or int(q) < 1:
            raise ValueError(f"q must be a positive integer or probabilities, got {q!r}")
        k = int(q)
        probs = np.linspace(0.0, 1.0, k + 1)
        np.putmask(probs, k * probs != np.arange(k + 1), np.nextafter(probs, 1.0))
        return probs
    probs = np.asarray(q, dtype=float)
    if probs.ndim != 1 or probs.size < 2:
        raise ValueError(f"q needs at least one bucket, got {q!r}")
    if probs[0] != 0.0 or probs[-1] != 1.0 or np.any(np.diff(probs) <= 0.0):
        raise ValueError(f"q must increase strictly from 0.0 to 1.0, got {q!r}")
    return probs


def compute_quantile_edges(x: Union[np.ndarray, pd.Series], q: QuantileSpec) -> np.ndarray:
    """Interior quantile edges of the finite observations.

    Args:
        x: observations; NaN and infinite values are left out
        q: the partition, see ``get_quantile_probabilities``

    Returns:
        the ``k - 1`` interior edges of a ``k``-bucket partition, non-decreasing

    Raises:
        ValueError: if there is no finite observation
    """
    return compute_sample_quantiles(x=x, probs=get_quantile_probabilities(q)[1:-1])


def compute_sample_quantiles(x: Union[np.ndarray, pd.Series],
                             probs: Union[Sequence[float], np.ndarray]
                             ) -> np.ndarray:
    """Sample quantiles of the finite observations by the rule of this module.

    Linear interpolation between order statistics at ``h = (n - 1) p``, with numpy's floating-point
    steps and a position within ``POSITION_TOLERANCE`` of a whole number taken as that number.

    Args:
        x: observations; NaN and infinite values are left out
        probs: probabilities in [0, 1]

    Returns:
        one quantile per probability

    Raises:
        ValueError: if there is no finite observation or a probability is outside [0, 1]
    """
    probs = np.asarray(probs, dtype=float)
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError(f"probabilities must lie in [0, 1], got {probs.tolist()}")
    values = _to_float_array(x)
    values = np.sort(values[np.isfinite(values)])
    n = values.size
    if n == 0:
        raise ValueError("need at least one finite observation to estimate quantile edges")
    position = (n - 1) * probs  # numpy's position for its linear method
    nearest = np.round(position)
    position = np.where(np.abs(position - nearest) <= POSITION_TOLERANCE * np.maximum(1.0, nearest),
                        nearest, position)
    position = np.clip(position, 0.0, n - 1.0)
    below = np.floor(position).astype(int)
    above = np.minimum(below + 1, n - 1)
    weight = position - below
    lower, upper = values[below], values[above]
    step = upper - lower
    # numpy's linear rule, evaluated from the nearer end so that both ends are exact
    return np.where(weight >= 0.5, upper - step * (1.0 - weight), lower + step * weight)


def assign_bucket_codes(x: Union[np.ndarray, pd.Series],
                        edges: Union[Sequence[float], np.ndarray]
                        ) -> np.ndarray:
    """Bucket of each observation for given interior edges, closed on the right with open ends.

    Args:
        x: observations
        edges: interior edges, non-decreasing; estimated from ``x`` or supplied, e.g. from another
            sample or fixed thresholds

    Returns:
        integer bucket per observation, 0 for the lowest, the number of edges strictly below the
        observation; -1 for a NaN or infinite observation
    """
    values = _to_float_array(x)
    codes = np.searchsorted(np.asarray(edges, dtype=float), values, side='left')
    return np.where(np.isfinite(values), codes, -1)


def compute_bucket_codes(x: Union[np.ndarray, pd.Series],
                         q: QuantileSpec,
                         is_require_occupied: bool = True
                         ) -> np.ndarray:
    """Quantile bucket of each observation, with the edges estimated from the observations.

    Args:
        x: observations
        q: the partition, see ``get_quantile_probabilities``
        is_require_occupied: raise when a bucket holds no observation

    Returns:
        integer bucket per observation, -1 for a NaN or infinite observation

    Raises:
        EmptyQuantileBucketError: if ``is_require_occupied`` and a bucket is empty
    """
    edges = compute_quantile_edges(x=x, q=q)
    codes = assign_bucket_codes(x=x, edges=edges)
    if is_require_occupied:
        num_buckets = edges.size + 1
        num_occupied = int(np.unique(codes[codes >= 0]).size)
        if num_occupied < num_buckets:
            values = _to_float_array(x)
            finite = values[np.isfinite(values)]
            full = np.concatenate([[finite.min()], edges, [finite.max()]])
            raise EmptyQuantileBucketError(num_occupied=num_occupied, num_buckets=num_buckets,
                                           edges=np.unique(full).tolist())
    return codes


def classify_quantile_buckets(x: Union[np.ndarray, pd.Series],
                              q: QuantileSpec,
                              labels: Optional[Sequence[str]] = None,
                              is_require_occupied: bool = True
                              ) -> pd.Series:
    """Quantile bucket of each observation as an ordered categorical.

    Args:
        x: observations; a Series keeps its index and name
        q: the partition, see ``get_quantile_probabilities``
        labels: one label per bucket, lowest first; None uses the bucket numbers 0 to k - 1
        is_require_occupied: raise when a bucket holds no observation

    Returns:
        the ordered categorical buckets, every label a category even when unobserved, missing
        for a NaN or infinite observation

    Raises:
        ValueError: if the number of labels is not the number of buckets
        EmptyQuantileBucketError: if ``is_require_occupied`` and a bucket is empty
    """
    num_buckets = get_quantile_probabilities(q).size - 1
    categories = list(range(num_buckets)) if labels is None else list(labels)
    if len(categories) != num_buckets:
        raise ValueError(f"{num_buckets} quantile buckets need {num_buckets} labels, "
                         f"got {len(categories)}")
    codes = compute_bucket_codes(x=x, q=q, is_require_occupied=is_require_occupied)
    buckets = pd.Categorical.from_codes(codes, categories=categories, ordered=True)
    index = x.index if isinstance(x, pd.Series) else None
    name = x.name if isinstance(x, pd.Series) else None
    return pd.Series(buckets, index=index, name=name)

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
``compute_sample_regime_moments`` supplies their empirical counterparts, with the empirical regime
frequencies, and ``compute_regime_mixture_covar_from_sample`` assembles the covariance of a
classified panel in one call: it estimates or takes the regime betas, converts ``idio_vol`` to
per-period variances and inserts the benchmark row.

The expression is equation (10) of Sepp and Kastenholz (2026), derived in their Appendix B. The
residuals are uncorrelated across assets by construction, which overstates diversification when
overlays share a strategy: the paper reads the mixture covariance as a weighting engine for the
allocation and not as a risk forecast.
"""
# packages
import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Optional, Sequence, Tuple, Union
# qis
from qis.regimes.partition import (REGIME_COLUMN, get_ordered_regimes, get_partition_quantiles,
                                  get_regime_ids, get_regime_probabilities)
from qis.regimes.betas import compute_regime_betas
from qis.regimes.nulls import _edge_densities


def compute_regime_mixture_covar(betas: pd.DataFrame,
                                 idio_vars: pd.Series,
                                 benchmark_regime_means: pd.Series,
                                 benchmark_regime_second_moments: pd.Series,
                                 af: float,
                                 regime_probs: Optional[pd.Series] = None
                                 ) -> pd.DataFrame:
    """Annualised regime-mixture covariance by the law of total covariance.

    Equation (10) of Sepp and Kastenholz (2026) from caller-supplied inputs. The default regime
    probabilities are the partition's population probabilities; with sample moments, pass the
    empirical frequencies, as ``compute_regime_mixture_covar_from_sample`` does.

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


def _sample_regime_data(sampled_returns_with_regime_id: pd.DataFrame,
                        benchmark: str, regime_column: str) -> pd.DataFrame:
    """Validate classified benchmark observations without changing the partition."""
    data = sampled_returns_with_regime_id
    if not data.columns.is_unique:
        raise ValueError("sample columns must be unique")
    if benchmark == regime_column or benchmark not in data or regime_column not in data:
        raise ValueError("sample must contain distinct benchmark and regime columns")
    data = data.dropna(subset=[regime_column])
    if data.empty:
        raise ValueError("sample has no classified observations")
    if not np.isfinite(data[benchmark].to_numpy(dtype=float)).all():
        raise ValueError("classified benchmark returns must be finite")
    if not all(isinstance(label, str) for label in data[regime_column]):
        raise ValueError("regime labels must be strings")
    regimes = get_ordered_regimes(data[regime_column])
    if len({label.lower() for label in regimes}) != len(regimes):
        raise ValueError("regime labels must be unique ignoring case")
    if any(not (data[regime_column] == label).any() for label in regimes):
        raise ValueError("sample contains an empty regime")
    return data


def compute_sample_regime_moments(sampled_returns_with_regime_id: pd.DataFrame,
                                  benchmark: str,
                                  regime_column: str = REGIME_COLUMN
                                  ) -> pd.DataFrame:
    """Per-period empirical regime moments of the benchmark on a supplied classification.

    The empirical counterpart of ``compute_gaussian_regime_moments``. Unclassified periods are
    excluded; every classified benchmark return must be finite and every regime occupied. The means
    and second moments are equal-weighted population moments (``ddof=0``), not sample variances.
    The function neither classifies nor resamples, and it uses every period it is given: a rolling
    caller passes only the periods known at the decision date.

    Args:
        sampled_returns_with_regime_id: periodic simple returns with a regime column of string
            labels
        benchmark: name of the benchmark column
        regime_column: name of the regime column

    Returns:
        regimes in rows, in bucket order, and the columns ``probability``, the empirical frequency
        of the classified periods, ``mean`` and ``second_moment``

    Raises:
        ValueError: if the columns, the labels, the classified benchmark returns or the regimes
            are invalid
    """
    data = _sample_regime_data(sampled_returns_with_regime_id, benchmark, regime_column)
    regimes = get_ordered_regimes(data[regime_column])
    labels = data[regime_column]
    grouped = data[benchmark].groupby(labels, observed=True)
    return pd.DataFrame({
        'probability': labels.value_counts(normalize=True),
        'mean': grouped.mean(),
        'second_moment': data[benchmark].pow(2).groupby(labels, observed=True).mean(),
    }).reindex(regimes)


def compute_regime_mixture_covar_from_sample(
        sampled_returns_with_regime_id: pd.DataFrame,
        benchmark: str,
        af: float,
        betas: Optional[pd.DataFrame] = None,
        regime_column: str = REGIME_COLUMN,
) -> pd.DataFrame:
    """Annualised regime-mixture covariance of one complete classified panel.

    Equation (10) of Sepp and Kastenholz (2026) in one call. The betas come from
    ``compute_regime_betas`` unless a frozen sheet is supplied, and ``compute_regime_mixture_covar``
    assembles the covariance with the empirical regime frequencies and the population benchmark
    moments of ``compute_sample_regime_moments``, all over the classified periods. The residual
    volatilities keep the sample convention of ``compute_regime_betas`` (``ddof=1``). Residuals are
    uncorrelated across assets and with the benchmark, and the fitted regime intercepts are
    discarded. A ragged panel is rejected: choose the common sample before classifying it.

    Args:
        sampled_returns_with_regime_id: periodic simple returns with a regime column;
            unclassified periods are excluded and every other asset return must be finite. The
            regime label ``total``, in any case, is reserved by the ``beta_total`` column
        benchmark: name of the benchmark column, inserted first with unit betas and zero residual
        af: annualisation factor of the periodic returns, finite and positive
        betas: optional frozen sheet in the format of ``compute_regime_betas``, indexed by
            exactly the non-benchmark assets, with ``beta_<regime id in lower case>`` columns and
            a finite, non-negative ``idio_vol`` annualised with the same ``af``; other columns
            are ignored. None estimates the betas on this panel, which needs 24 periods and two
            distinct benchmark returns in every regime
        regime_column: name of the regime column

    Returns:
        the annualised covariance, the benchmark first and then the other assets in the panel's
        column order. On a full sample it is descriptive; a rolling caller passes only the
        periods, and frozen betas, known at the decision date, and answers for the estimation
        dates and the annualisation of frozen betas

    Raises:
        ValueError: if the panel, ``af``, the identification of a regression or the frozen betas
            are invalid
    """
    if not np.isfinite(af) or af <= 0.0:
        raise ValueError("af must be finite and positive")
    moments = compute_sample_regime_moments(
        sampled_returns_with_regime_id, benchmark, regime_column)
    data = sampled_returns_with_regime_id.dropna(subset=[regime_column])
    assets = [name for name in data.columns if name not in (benchmark, regime_column)]
    if not np.isfinite(data[assets].to_numpy(dtype=float)).all():
        raise ValueError("classified asset returns must form a complete finite sample")
    regimes = list(moments.index)
    if any(name.lower() == 'total' for name in regimes):
        raise ValueError("regime label 'total' is reserved by the beta_total summary column")
    beta_columns = [f'beta_{name.lower()}' for name in regimes]
    required = beta_columns + ['idio_vol']
    if betas is None:
        if assets:
            for regime in regimes:
                observations = data.loc[data[regime_column] == regime, benchmark]
                if observations.nunique() < 2:
                    raise ValueError(f"{regime}: need two distinct benchmark returns")
            betas = compute_regime_betas(data, benchmark, af, regime_column=regime_column)
        else:
            betas = pd.DataFrame(index=assets, columns=required, dtype=float)
    if (not betas.index.is_unique or not betas.columns.is_unique
            or set(betas.index) != set(assets)):
        raise ValueError("betas must have unique labels and exactly the non-benchmark assets")
    missing = set(required).difference(betas.columns)
    if missing:
        raise ValueError(f"betas are missing required columns: {sorted(missing)}")
    frozen = betas.loc[assets, required].astype(float)
    if not np.isfinite(frozen.to_numpy()).all() or (frozen['idio_vol'] < 0.0).any():
        raise ValueError("betas must be finite and annual idio_vol nonnegative")
    names = [benchmark] + assets
    loadings = frozen[beta_columns].copy()
    loadings.columns = regimes
    loadings.loc[benchmark] = 1.0
    idio_vars = frozen['idio_vol'].pow(2) / af
    idio_vars.loc[benchmark] = 0.0
    return compute_regime_mixture_covar(
        betas=loadings.reindex(names), idio_vars=idio_vars,
        benchmark_regime_means=moments['mean'],
        benchmark_regime_second_moments=moments['second_moment'],
        regime_probs=moments['probability'], af=af,
    )

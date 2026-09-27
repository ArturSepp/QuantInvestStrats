"""
regime-conditional betas of assets on the benchmark, with block-bootstrap standard errors.

Within each regime the asset's returns are regressed on the benchmark's with an intercept, which is
then discarded: forcing the fit through the origin would bias the slope, and expected returns come
from Sharpe ratios rather than regime alphas. The pooled residuals of the piecewise fit give the
idiosyncratic volatility. Under a Gaussian null every regime beta equals the total beta, so their
spread is the non-linearity the regimes reveal.

``compute_regime_betas_bootstrap`` resamples whole rows of the panel with the stationary block
bootstrap, keeping the cross-section, and reclassifies the regimes inside each resample with the
rule of ``qis.utils.quantile_buckets`` before re-estimating.
"""
# packages
import numpy as np
import pandas as pd
from typing import Dict, Sequence, Union
# qis
from qis.models.bootstrap.bootstrap_numba import BootstrapType, generate_bootstrapped_indices
from qis.regimes.partition import (REGIME_COLUMN, create_sampled_returns_with_regime_id,
                                   get_ordered_regimes, get_partition_quantiles, get_regime_ids)


def compute_regime_betas(sampled_returns_with_regime_id: pd.DataFrame,
                         benchmark: str,
                         af: float,
                         min_periods: int = 24,
                         regime_column: str = REGIME_COLUMN
                         ) -> pd.DataFrame:
    """Per-regime OLS betas of every asset on the benchmark, with the intercepts discarded.

    Each asset uses the periods where it and the benchmark both have a return; the regimes are
    those of the sampled frame, set on the whole benchmark sample.

    Args:
        sampled_returns_with_regime_id: periodic returns with a regime column
        benchmark: name of the benchmark column
        af: annualisation factor of the periodic returns, for the idiosyncratic volatility
        min_periods: fewest common periods an asset needs
        regime_column: name of the regime column

    Returns:
        one row per asset other than the benchmark and, per regime id in bucket order,
        ``beta_<id>`` and ``n_<id>``, then ``beta_total`` and ``idio_vol``, the annualised
        standard deviation of the pooled piecewise residuals

    Raises:
        ValueError: if an asset has fewer than ``min_periods`` common periods
    """
    data = sampled_returns_with_regime_id.dropna(subset=[regime_column])
    regimes = get_ordered_regimes(data[regime_column])
    rows = {}
    for asset in [c for c in data.columns if c not in (regime_column, benchmark)]:
        joint = data[[asset, benchmark, regime_column]].dropna(subset=[asset, benchmark])
        if len(joint) < min_periods:
            raise ValueError(f"{asset}: need at least {min_periods} common periods, "
                             f"got {len(joint)}")
        y, x, labels = joint[asset], joint[benchmark], joint[regime_column]
        out: Dict[str, float] = {}
        residuals = []
        for regime in regimes:
            ys, xs = y[labels == regime], x[labels == regime]
            slope, intercept = np.polyfit(xs, ys, 1)
            out[f"beta_{regime.lower()}"] = slope
            out[f"n_{regime.lower()}"] = len(xs)
            residuals.append(ys - (intercept + slope * xs))
        slope_total, _ = np.polyfit(x, y, 1)
        out['beta_total'] = slope_total
        out['idio_vol'] = float(pd.concat(residuals).std() * np.sqrt(af))
        rows[asset] = out
    return pd.DataFrame(rows).T


def compute_regime_betas_bootstrap(returns: pd.DataFrame,
                                   benchmark: str,
                                   af: float,
                                   q: Union[Sequence[float], np.ndarray, None] = None,
                                   block_size: int = 12,
                                   n_boot: int = 2000,
                                   seed: int = 7
                                   ) -> pd.DataFrame:
    """Regime betas of a panel with their stationary block-bootstrap standard errors.

    Whole rows are resampled, so the cross-section of the panel is kept, and the regimes are
    reclassified inside each resample before the betas are re-estimated.

    Args:
        returns: periodic returns without missing values, one column per asset, including the
            benchmark
        benchmark: name of the benchmark column
        af: annualisation factor of the periodic returns
        q: partition probabilities; None is the one-sigma cut
        block_size: mean block length in periods
        n_boot: number of resamples
        seed: seed of the resampling indices

    Returns:
        one row per asset other than the benchmark and, per regime id in bucket order,
        ``beta_<id>``, the estimate on the original panel, and ``beta_<id>_se``, the standard
        deviation of the resampled betas with ``ddof=1``

    Raises:
        ValueError: if the panel has missing values or fewer than five blocks of periods
    """
    if returns.isna().to_numpy().any():
        raise ValueError("returns must have no missing values; align the panel first")
    n = len(returns)
    if n < 5 * block_size:
        raise ValueError(f"need at least {5 * block_size} periods at block_size={block_size}, "
                         f"got {n}")
    q = get_partition_quantiles(q)
    regimes = get_regime_ids(q)

    def estimate(panel: pd.DataFrame) -> pd.DataFrame:
        """regime betas of one panel, with its own classification"""
        sampled = create_sampled_returns_with_regime_id(returns=panel, benchmark=benchmark, q=q)
        return compute_regime_betas(sampled_returns_with_regime_id=sampled, benchmark=benchmark,
                                    af=af, min_periods=0)

    # generate_bootstrapped_indices returns the samples in the columns
    indices = np.asarray(generate_bootstrapped_indices(num_data_index=n,
                                                       bootstrap_type=BootstrapType.STATIONARY,
                                                       num_samples=n_boot,
                                                       index_length=n,
                                                       block_size=block_size,
                                                       seed=seed))
    beta_columns = [f"beta_{regime.lower()}" for regime in regimes]
    draws = {column: [] for column in beta_columns}
    for b in range(n_boot):
        resample = returns.iloc[indices[:, b]].reset_index(drop=True)
        betas = estimate(resample)
        for column in beta_columns:
            draws[column].append(betas[column])
    point = estimate(returns)
    columns = {}
    for column in beta_columns:
        columns[column] = point[column].astype(float)
        columns[f"{column}_se"] = pd.DataFrame(draws[column]).std(ddof=1)
    return pd.DataFrame(columns)

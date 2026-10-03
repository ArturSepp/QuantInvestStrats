"""Historical tail risk for equally weighted observations at their supplied horizon."""

from numbers import Real
from typing import Union

import numpy as np
import pandas as pd


def compute_cvar(returns: Union[pd.Series, pd.DataFrame],
                 confidence_level: float = 0.95) -> Union[float, pd.Series]:
    """Compute historical CVaR (expected shortfall) as a loss-positive quantity.

    Each nonmissing observation has equal probability. Average exactly the worst
    ``1 - confidence_level`` probability mass, with fractional weight on the boundary
    observation. This is the empirical Rockafellar--Uryasev expected shortfall, including
    distributions with ties. It is not the mean of every observation below an interpolated
    sample quantile, which can include the wrong tail mass.

    Args:
        returns: Period returns in decimal units. Series or DataFrame columns are treated
            independently, omitting NaNs. Supply returns at the required risk horizon
            (for example monthly simple returns for one-month CVaR). No return conversion,
            frequency resampling or annualisation is performed.
        confidence_level: Confidence probability strictly between zero and one.

    Returns:
        Scalar for a Series, or a Series indexed by DataFrame columns. A positive value
        denotes a loss; an entirely profitable tail can produce a negative value.
        Empty or entirely missing samples produce NaN.

    Raises:
        TypeError: If returns is not a pandas Series or DataFrame.
        ValueError: If confidence_level is invalid, or a return is infinite or nonnumeric.

    References:
        Rockafellar and Uryasev (2002), Conditional value-at-risk for general loss
        distributions, https://doi.org/10.1016/S0378-4266(02)00271-6.
    """
    if not isinstance(returns, (pd.Series, pd.DataFrame)):
        raise TypeError('returns must be a pandas Series or DataFrame')
    if (not isinstance(confidence_level, Real) or isinstance(confidence_level, bool)
            or not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0):
        raise ValueError('confidence_level must be finite and strictly between zero and one')

    def tail_loss(sample: pd.Series) -> float:
        """Integrate the empirical lower return tail with its exact probability mass."""
        try:
            values = sample.to_numpy(dtype=float, na_value=np.nan)
        except (TypeError, ValueError) as error:
            raise ValueError('returns must be numeric') from error
        if np.isinf(values).any():
            raise ValueError('returns must not contain infinite observations')
        values = np.sort(values[~np.isnan(values)])
        if not len(values):
            return float('nan')
        tail_mass = len(values) * (1.0 - confidence_level)
        weights = np.clip(tail_mass - np.arange(len(values)), 0.0, 1.0)
        return float(-np.dot(values, weights) / tail_mass)

    if isinstance(returns, pd.Series):
        return tail_loss(returns)
    return returns.apply(tail_loss).astype(float)

"""Arithmetic attribution for complete feature-coalition experiments, in return units."""
from collections.abc import Mapping

import numpy as np
import pandas as pd

from qis.portfolio.attribution.model_feature import _shapley_powers, _validate_scenario_keys


def link_return_contributions(contributions: pd.DataFrame, returns: pd.Series) -> pd.Series:
    """Link period P&L on beginning wealth; rows are returns AFTER the baseline date.

    For example, 10% from A then 10% from B contributes 10pp and 11pp, not
    10pp each. Include fees/funding explicitly; a missing leg is not a zero.
    Return decimals, without annualisation or scaling by benchmark wealth.
    """
    if (contributions.empty or not contributions.columns.is_unique
            or not contributions.index.is_unique or not contributions.index.is_monotonic_increasing
            or not contributions.index.equals(returns.index)):
        raise ValueError('expected nonempty aligned returns and unique ordered observations')
    if not (np.isfinite(contributions).all().all() and np.isfinite(returns).all()
            and (returns > -1).all()):
        raise ValueError('returns and contributions must be finite with positive wealth')
    if not np.allclose(contributions.sum(axis=1), returns, atol=1e-12, rtol=1e-12):
        raise ValueError('period contributions do not reconcile to returns')
    wealth_before = (1.0 + returns).cumprod().shift(1, fill_value=1.0)
    return contributions.mul(wealth_before, axis=0).sum()


def shapley_return_contributions(
        scenarios: Mapping[frozenset[str], pd.Series]) -> pd.DataFrame:
    """Allocate full-minus-empty arithmetic contribution vectors over all feature orders.

    Inputs must use the same labelled groups and return convention, e.g. output
    of link_return_contributions on one common interval. This is not a regression
    alpha or the return of a standalone signal portfolio.
    """
    features, coalitions = _validate_scenario_keys(scenarios)
    baseline = scenarios[frozenset()]
    for coalition in coalitions:
        values = scenarios[coalition]
        if (not isinstance(values, pd.Series) or values.empty or not values.index.is_unique
                or not values.index.equals(baseline.index) or not np.isfinite(values).all()):
            raise ValueError('expected finite vectors with identical unique group labels')
    return pd.DataFrame({
        feature: sum(power * scenarios[key] for key, power in
                     _shapley_powers(feature=feature, features=features).items())
        for feature in features
    })


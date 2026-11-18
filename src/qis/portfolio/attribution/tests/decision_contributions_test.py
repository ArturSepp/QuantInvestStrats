"""Independent numerical contracts for arithmetic linked coalition contributions."""
from itertools import combinations, permutations

import numpy as np
import pandas as pd
import pytest

from qis.portfolio.attribution.decision_contributions import (
    link_return_contributions, shapley_return_contributions)


def test_link_on_prior_wealth_not_per_column_compounding():
    """The second asset earns on the first asset's gains, including a signed fee."""
    pnl = pd.DataFrame([[.1, 0., 0.], [0., .1, -.01]], columns=['A', 'B', 'fees'])
    got = link_return_contributions(pnl, pd.Series([.1, .09]))
    np.testing.assert_allclose(got, [.1, .11, -.011])
    assert got.sum() == pytest.approx(1.1 * 1.09 - 1)
    with pytest.raises(ValueError, match='reconcile'):
        link_return_contributions(pnl, pd.Series([.1, .1]))
    with pytest.raises(ValueError):
        link_return_contributions(pnl, pd.Series([.1, np.nan]))
    with pytest.raises(ValueError):
        link_return_contributions(pnl, pd.Series([.1, .1], index=[1,0]))


def test_shapley_against_explicit_activation_orders():
    """Check interactions, dummy feature, completeness, and group-wise reconciliation."""
    names = ('a', 'b', 'dummy')
    scenarios = {}
    for count in range(4):
        for subset in combinations(names, count):
            k = frozenset(subset)
            scenarios[k] = pd.Series([.2 + int('a' in k) + 4*int({'a','b'} <= k),
                                      -.5*int('b' in k)], index=['X','Y'])
    got = shapley_return_contributions(scenarios)
    ref = got * 0.
    for order in permutations(names):
        k = frozenset()
        for name in order:
            next_k = k | {name}
            ref[name] += (scenarios[next_k] - scenarios[k])/6
            k = next_k
    np.testing.assert_allclose(got, ref)
    np.testing.assert_allclose(got.sum(axis=1), scenarios[frozenset(names)]-scenarios[frozenset()])
    scenarios.pop(frozenset({'a'}))
    with pytest.raises(ValueError, match='complete'):
        shapley_return_contributions(scenarios)


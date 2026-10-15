"""Independent numerical and timing checks for the monitoring report."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from qis import (
    GaussianMonteCarlo, GaussianVarSpec as RiskMonitoringSpec,
    compute_gaussian_portfolio_risk as compute_risk_snapshot,
)


@pytest.fixture(scope='module')
def simulation():
    return GaussianMonteCarlo(RiskMonitoringSpec(simulations=200_000))


def test_var_beta_and_te_against_independent_closed_forms(simulation):
    covariance = pd.DataFrame([[0.04, 0.006], [0.006, 0.01]], index=['a', 'b'], columns=['a', 'b'])
    p = pd.Series([0.6, 0.4], index=['a', 'b'])
    b = pd.Series([1.0, 0.0], index=['a', 'b'])
    row = compute_risk_snapshot(covariance, p, b, pd.Timestamp('2020-01-31'), simulation)
    expected_variance = .6**2 * .04 + .4**2 * .01 + 2*.6*.4*.006
    expected_cross = .6*.04 + .4*.006
    assert row['portfolio_var'] == pytest.approx(norm.ppf(.99)*np.sqrt(expected_variance), rel=.012)
    assert row['benchmark_var'] == pytest.approx(norm.ppf(.99)*.2, rel=.012)
    assert row['ex_ante_beta'] == pytest.approx(expected_cross/.04)
    assert row['tracking_error'] == pytest.approx(np.sqrt(.4**2*(.04+.01-2*.006)))
    assert row['var_ratio'] == row['portfolio_var']/row['benchmark_var']


def test_identical_portfolio_and_benchmark_has_exact_unit_ratio(simulation):
    cov = pd.DataFrame([[.04]], index=['a'], columns=['a'])
    weight = pd.Series({'a': 1.0})
    row = compute_risk_snapshot(cov, weight, weight, pd.Timestamp('2020-01-31'), simulation)
    assert row['var_ratio'] == pytest.approx(1.0, abs=1e-14)
    assert row['ex_ante_beta'] == pytest.approx(1.0)
    assert row['tracking_error'] == 0.0


def test_monte_carlo_scaling_and_determinism(simulation):
    same = GaussianMonteCarlo(simulation.spec)
    twice_horizon = GaussianMonteCarlo(replace(simulation.spec, horizon_years=4))
    base = simulation.var(.04, .01, -.005)
    assert same.var(.04, .01, -.005) == base
    assert twice_horizon.var(.04, .01, -.005) == pytest.approx(np.array(base)*2)
    with pytest.raises(ValueError, match='positive semidefinite'):
        simulation.var(.04, .01, .03)


def test_missing_held_risk_is_rejected_and_zero_benchmark_is_unavailable(simulation):
    date = pd.Timestamp('2020-01-31')
    cov = pd.DataFrame([[.04, np.nan], [np.nan, np.nan]], index=['a', 'b'], columns=['a', 'b'])
    with pytest.raises(ValueError):
        compute_risk_snapshot(cov, pd.Series({'b': 1.0}), pd.Series({'a': 1.0}), date, simulation)
    with pytest.raises(ValueError, match='missing covariance'):
        compute_risk_snapshot(cov, pd.Series({'unknown': 1.0}),
                              pd.Series({'a': 1.0}), date, simulation)
    row = compute_risk_snapshot(cov, pd.Series({'a': 1.0}), pd.Series({'a': 0.0}), date, simulation)
    assert np.isnan(row['var_ratio']) and np.isnan(row['ex_ante_beta'])
    assert row['benchmark_var'] == 0

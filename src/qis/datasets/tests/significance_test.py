"""Independent construction checks for controlled calibration fixtures."""

import numpy as np
import pandas as pd
import pytest

from qis.datasets.significance import SignificanceDgp, generate_significance_sample


@pytest.mark.parametrize('phi', [0.0, 0.3, -0.3])
@pytest.mark.parametrize('rho', [0.0, 0.5])
@pytest.mark.parametrize('tail_df', [None, 5.0])
def test_known_dgp_path_covariance_and_zero_intercept(phi, rho, tail_df):
    dgp = SignificanceDgp(60, correlation=rho, ar_coefficient=phi, tail_df=tail_df)
    sample = generate_significance_sample(dgp, seed=20261102)
    state = np.zeros(dgp.num_series)
    reference = []
    for innovation in sample.innovations:
        state = phi * state + innovation
        reference.append(state.copy())
    np.testing.assert_allclose(sample.mean_returns, np.array(reference)[1000:], atol=1e-15)
    np.testing.assert_allclose(sample.alpha_returns - sample.factors.to_numpy() @ sample.loadings,
                               sample.mean_returns, atol=1e-15)
    covariance = dgp.innovation_covariance * sum(phi ** (2 * i) for i in range(1000))
    long_run = covariance * (1 + 2 * sum(phi ** i for i in range(1, 1000)))
    np.testing.assert_allclose(dgp.stationary_covariance, covariance, rtol=1e-14)
    np.testing.assert_allclose(dgp.long_run_covariance, long_run, rtol=1e-14)
    np.testing.assert_allclose(dgp.innovation_covariance[0, 1], rho * 0.02 ** 2)
    assert sample.mean_returns.shape == sample.alpha_returns.shape == (60, 12)
    assert sample.factors.shape == (60, 3) and sample.burn_in == 1000
    assert sample.mean_returns.index.freqstr == 'ME'


def test_dgp_reproducibility_and_global_rng_preservation():
    dgp = SignificanceDgp(24)
    before = np.random.get_state()
    first = generate_significance_sample(dgp, 12)
    again = generate_significance_sample(dgp, 12)
    different = generate_significance_sample(dgp, 13)
    pd.testing.assert_frame_equal(first.mean_returns, again.mean_returns)
    pd.testing.assert_frame_equal(first.factors, again.factors)
    assert not first.mean_returns.equals(different.mean_returns)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])


@pytest.mark.parametrize('settings', [
    {'nobs': 23}, {'nobs': True}, {'num_series': 0}, {'factor_count': 7},
    {'correlation': 1.0}, {'ar_coefficient': 0.95}, {'innovation_std': 0},
    {'tail_df': 4}, {'tail_df': np.inf}, {'correlation': '0.5'},
    {'innovation_std': 1j}, {'tail_df': False},
])
def test_invalid_dgp_parameters(settings):
    with pytest.raises(ValueError):
        SignificanceDgp(**{'nobs': 24, **settings})


@pytest.mark.parametrize('seed', [-1, True, 1.5])
def test_invalid_dgp_seeds(seed):
    with pytest.raises(ValueError, match='seed'):
        generate_significance_sample(SignificanceDgp(24), seed)

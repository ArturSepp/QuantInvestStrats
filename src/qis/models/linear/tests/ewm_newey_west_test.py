"""compute_ewm_newey_west_vol against an independent loop and against compute_ewm_vol.

The variance recursion runs on squared observations and must be seeded on that scale, so with no
lag terms the estimator is exactly the EWM variance. Each lag term is the Bartlett-weighted EWM
of ``x_t x_{t-m}``, doubled for the two sides of the autocovariance, using the same decay as the
variance and scaled by ``lambda^(m/2)``, the geometric mean of the EWM weights of the two dates
it pairs, which makes the estimator positive semidefinite.
"""

# packages
import numpy as np
import pandas as pd
import pytest

# qis
from qis.models.linear.ewm import (InitType, MeanAdjType, compute_ewm_newey_west_vol,
                                   compute_ewm_vol)


def _returns() -> pd.DataFrame:
    """Serially correlated returns with a non-zero mean, two columns."""
    rng = np.random.default_rng(11)
    shocks = rng.standard_normal((120, 2)) * 0.01
    values = shocks + 0.3 * np.vstack([np.zeros((1, 2)), shocks[:-1]]) + 0.002
    return pd.DataFrame(values, index=pd.bdate_range('2021-01-01', periods=120),
                        columns=['a', 'b'])


def _reference(x: np.ndarray, ewm_lambda: float, num_lags: int) -> np.ndarray:
    """Newey-West EWM variance by explicit loops, seeded with the first squared observation.

    The lag-m term carries the factor ``ewm_lambda ** (m / 2)``. Before this release the lag
    terms were unscaled, and the estimator could turn negative (see
    ``ewm_covariance_newey_west_psd_test.py``); the expected values changed with the fix.
    """
    variance = np.empty_like(x)
    variance[0] = x[0] ** 2
    for t in range(1, len(x)):
        variance[t] = ewm_lambda * variance[t - 1] + (1.0 - ewm_lambda) * x[t] ** 2
    adjustment = np.zeros_like(x)
    for m in range(1, num_lags + 1):
        cross = np.zeros_like(x)
        for t in range(1, len(x)):
            product = x[t] * x[t - m] if t >= m else np.zeros_like(x[t])
            cross[t] = ewm_lambda * cross[t - 1] + (1.0 - ewm_lambda) * product
        adjustment += (1.0 - m / (num_lags + 1)) * 2.0 * ewm_lambda ** (m / 2) * cross
    return variance + adjustment


def test_no_lags_equals_ewm_variance() -> None:
    """With num_lags=0 the estimator is the EWM variance of compute_ewm_vol."""
    data = _returns()
    nw_variance, ratio = compute_ewm_newey_west_vol(
        data, num_lags=0, span=10, mean_adj_type=MeanAdjType.NONE, init_type=InitType.X0,
        apply_sqrt=False)
    variance = compute_ewm_vol(data, span=10, mean_adj_type=MeanAdjType.NONE,
                               init_type=InitType.X0, apply_sqrt=False)
    np.testing.assert_allclose(nw_variance.to_numpy(), variance.to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(ratio.to_numpy(), 1.0)


@pytest.mark.parametrize('ewm_lambda,num_lags', [(0.8, 1), (0.9, 3)])
def test_lag_terms_use_the_requested_decay(ewm_lambda: float, num_lags: int) -> None:
    """An explicit ewm_lambda drives both the variance and the lag terms."""
    data = _returns()
    nw_variance, _ = compute_ewm_newey_west_vol(
        data, num_lags=num_lags, ewm_lambda=ewm_lambda, mean_adj_type=MeanAdjType.NONE,
        init_type=InitType.X0, apply_sqrt=False)
    expected = _reference(data.to_numpy(), ewm_lambda=ewm_lambda, num_lags=num_lags)
    np.testing.assert_allclose(nw_variance.to_numpy(), expected, rtol=1e-10, atol=1e-16)


def test_series_input_matches_frame_column() -> None:
    """A Series is estimated exactly as the same column of a DataFrame."""
    data = _returns()
    frame_vol, frame_ratio = compute_ewm_newey_west_vol(data, num_lags=2, span=20)
    series_vol, series_ratio = compute_ewm_newey_west_vol(data['a'], num_lags=2, span=20)
    assert isinstance(series_vol, pd.Series) and series_vol.name == 'a'
    np.testing.assert_allclose(series_vol.to_numpy(), frame_vol['a'].to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(series_ratio.to_numpy(), frame_ratio['a'].to_numpy(), rtol=1e-12)

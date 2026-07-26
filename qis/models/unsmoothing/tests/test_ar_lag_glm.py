"""
Regression tests for the static Getmansky-Lo-Makarov unsmoother, qis.unsmooth_returns_glm.

Two paths are locked here:

  * the estimated path - weights fitted from the sample, which is what the function did before
    ``theta`` existed, so these tests are the guard that adding the argument changed nothing,
  * the fixed-weight path - ``theta`` supplied from outside the series, which is what a panel
    estimate pooled across vintages or a production constant looks like.

Everything runs on a seeded AR(1) draw; no data fixtures, no network.
"""
# packages
import numpy as np
import pandas as pd
import pytest
# qis / project
from qis.models.unsmoothing.ar_lag import unsmooth_returns_glm

THETA = 0.176  # a panel AR(1) estimate; the value is arbitrary here, being supplied not fitted


def _ar1_returns(theta: float = 0.4,
                 num_periods: int = 120,
                 seed: int = 4,
                 ) -> pd.Series:
    """draw a smoothed quarterly series r_t = theta r_{t-1} + e_t."""
    rng = np.random.default_rng(seed)
    values = np.zeros(num_periods)
    for t in range(1, num_periods):
        values[t] = theta * values[t - 1] + rng.normal(0.0, 0.03)
    return pd.Series(values, index=pd.date_range('2000-03-31', periods=num_periods, freq='QE'),
                     name='fund')


def test_fixed_theta_equals_the_closed_form_inversion() -> None:
    """with theta supplied the result is exactly (r_t - theta r_{t-1}) / (1 - theta)."""
    returns = _ar1_returns()
    unsmoothed = unsmooth_returns_glm(returns=returns, theta=THETA)

    values = returns.to_numpy()
    expected = values.copy()
    expected[1:] = (values[1:] - THETA * values[:-1]) / (1.0 - THETA)
    assert np.allclose(unsmoothed.to_numpy(), expected)
    assert unsmoothed.index.equals(returns.index)
    assert unsmoothed.name == returns.name


def test_fixed_theta_preserves_the_leading_observations() -> None:
    """the first q observations have no lags to invert against and are returned unchanged."""
    returns = _ar1_returns()
    unsmoothed = unsmooth_returns_glm(returns=returns, theta=np.array([0.2, 0.1]))
    assert np.allclose(unsmoothed.iloc[:2].to_numpy(), returns.iloc[:2].to_numpy())


def test_fixed_theta_sets_the_diagnostics() -> None:
    """diagnostics report the supplied weights, not a fit."""
    returns = _ar1_returns()
    _, diagnostics = unsmooth_returns_glm(returns=returns, theta=THETA, return_diagnostics=True)
    assert diagnostics.ar_order == 1
    assert np.allclose(diagnostics.theta, np.array([THETA]))
    assert diagnostics.theta_sum == pytest.approx(THETA)
    assert diagnostics.vol_inflation_factor == pytest.approx(1.0 / (1.0 - THETA))
    assert diagnostics.is_severe is False


def test_fixed_theta_skips_the_sample_length_guard() -> None:
    """nothing is estimated, so a series too short to fit AR(3) still inverts."""
    returns = _ar1_returns(num_periods=6)
    with pytest.raises(ValueError, match='insufficient observations'):
        unsmooth_returns_glm(returns=returns, ar_order=3)
    unsmoothed = unsmooth_returns_glm(returns=returns, theta=np.array([0.1, 0.1, 0.05]))
    assert len(unsmoothed) == 6


def test_fixed_theta_applies_to_every_column() -> None:
    """one set of weights, applied column by column, with per-column diagnostics."""
    returns = _ar1_returns()
    panel = pd.concat([returns.rename('a'), (2.0 * returns).rename('b')], axis=1)
    unsmoothed, diagnostics = unsmooth_returns_glm(returns=panel, theta=THETA,
                                                   return_diagnostics=True)
    assert list(unsmoothed.columns) == ['a', 'b']
    assert np.allclose(unsmoothed['b'].to_numpy(), 2.0 * unsmoothed['a'].to_numpy())
    assert set(diagnostics.keys()) == {'a', 'b'}


def test_uninvertible_observation_is_nan_not_the_raw_value() -> None:
    """an observation whose lag is missing cannot be inverted, so it is missing."""
    returns = _ar1_returns()
    returns.iloc[10] = np.nan
    unsmoothed = unsmooth_returns_glm(returns=returns, theta=THETA)
    assert np.isnan(unsmoothed.iloc[10])
    assert np.isnan(unsmoothed.iloc[11]), 'the lag of position 11 is missing'
    assert not np.isnan(unsmoothed.iloc[12])


@pytest.mark.parametrize('bad_theta, message', [
    (1.0, 'singular'),
    (np.array([0.5, 0.5]), 'singular'),
    (np.nan, 'finite'),
    (np.zeros((2, 2)), '1-d array'),
])
def test_invalid_theta_is_rejected(bad_theta: object,
                                   message: str,
                                   ) -> None:
    """a supplied weight that cannot invert is refused with the offending value."""
    with pytest.raises(ValueError, match=message):
        unsmooth_returns_glm(returns=_ar1_returns(), theta=bad_theta)


def test_estimated_path_is_unchanged_by_the_theta_argument() -> None:
    """the default call still fits the weights and lifts the volatility."""
    returns = _ar1_returns(theta=0.5, num_periods=400, seed=11)
    unsmoothed, diagnostics = unsmooth_returns_glm(returns=returns, ar_order=1,
                                                   return_diagnostics=True)
    assert diagnostics.theta_sum == pytest.approx(0.5, abs=0.1)
    assert unsmoothed.std() > returns.std()

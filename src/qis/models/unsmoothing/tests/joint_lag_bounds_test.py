"""Joint unsmoothing bounds must minimize the same weighted regression objective."""
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import lsq_linear

from qis.models.linear.ewm import MeanAdjType, InitType, compute_rolling_mean_adj
from qis.models.unsmoothing import joint_lag


def _correlated_sample():
    """Return own-lag and factor-lag regressors with a binding beta floor."""
    rng = np.random.default_rng(20261024)
    r = np.zeros(100)
    f = np.zeros(100)
    for t in range(1, len(r)):
        r[t] = .8 * r[t - 1] - .45 * f[t - 1] + rng.normal(0, .025)
        f[t] = .9 * r[t] + rng.normal(0, .01)
    index = pd.date_range('2000-03-31', periods=len(r), freq='QE')
    return pd.DataFrame({'asset': r}, index=index), pd.Series(f, index=index)


@pytest.mark.parametrize('mean', [MeanAdjType.NONE, MeanAdjType.EWMA])
@pytest.mark.parametrize('bounds', [(-np.inf, .75, 0., np.inf),
                                    (0., .2, -.1, .1), (-.3, .1, .05, .2)])
def test_joint_bounds_match_weighted_least_squares(mean, bounds):
    """Each released pair agrees with an independent bounded fit, including gaps."""
    returns, factor = _correlated_sample()
    returns.iloc[40] = np.nan
    lo_phi, hi_phi, lo_beta, hi_beta = bounds
    _, phi, beta = joint_lag.adjust_returns_with_joint_unsmoothing(
        returns, factor, span=40, mean_adj_type=mean, warmup_period=4,
        min_ar_coeff=lo_phi, max_ar_coeff=hi_phi,
        min_factor_beta=lo_beta, max_factor_beta=hi_beta,
        apply_ewma_mean_smoother=False, return_diagnostics=True)
    design = pd.concat([returns.asset.shift(), factor.shift(), returns.asset], axis=1)
    design.columns = ['ownlag', 'factorlag', 'y']
    if mean != MeanAdjType.NONE:
        design = compute_rolling_mean_adj(design, mean_adj_type=mean, span=40,
                                          init_type=InitType.ZERO)
    for pos in range(20, len(design)):
        if pd.isna(phi.iloc[pos, 0]):
            continue
        panel = design.iloc[:pos + 1]
        valid = panel.notna().all(axis=1)
        w = (1 - 39 / 41) * (39 / 41) ** (pos - np.flatnonzero(valid.to_numpy()))
        a = panel.loc[valid]
        fit = lsq_linear(a[['ownlag', 'factorlag']].to_numpy() * np.sqrt(w[:, None]),
                         a.y.to_numpy() * np.sqrt(w),
                         bounds=([lo_phi, lo_beta], [hi_phi, hi_beta]), method='bvls', tol=1e-14)
        assert fit.success
        np.testing.assert_allclose([phi.iloc[pos, 0], beta.iloc[pos, 0]], fit.x,
                                   rtol=2e-8, atol=2e-9)


@pytest.mark.parametrize('truth,bounds', [
    ([.2, .1], ([-np.inf, -np.inf], [.75, np.inf])),
    ([.9, -.4], ([-np.inf, 0.], [.75, np.inf])),
    ([1.2, .5], ([0., 0.], [.3, .2])),
    ([-1.2, -.5], ([-.3, -.2], [.3, .2])),
    ([.4, -.2], ([.1, 0.], [.1, np.inf])),
])
def test_bounded_solver_handles_faces_corners_and_fixed_coefficients(truth, bounds):
    """Face enumeration agrees with least squares, including a fixed coefficient."""
    rng = np.random.default_rng(24)
    x = rng.normal(size=(80, 2))
    x[:, 1] += .8 * x[:, 0]
    y = x @ truth + rng.normal(scale=.01, size=80)
    lower, upper = np.array(bounds[0]), np.array(bounds[1])
    actual = joint_lag._fit_bounded_joint_betas(x, y, 40, lower, upper)[-1]
    w = (1 - 39 / 41) * (39 / 41) ** np.arange(79, -1, -1)
    if lower[0] == upper[0]:
        expected = np.array([lower[0], np.clip(
            np.sum(w * x[:, 1] * (y - lower[0] * x[:, 0])) /
            np.sum(w * x[:, 1] ** 2), lower[1], upper[1])])
    else:
        expected = lsq_linear(x * np.sqrt(w[:, None]), y * np.sqrt(w),
                              bounds=(lower, upper), method='bvls', tol=1e-14).x
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-9)


def test_bounded_joint_preserves_rank_guard_and_scale():
    """Singular data stay unavailable and coefficient units do not change the fit."""
    x = np.column_stack([np.arange(1., 31.), np.arange(1., 31.)])
    lower, upper = np.array([-np.inf, 0.]), np.array([.75, np.inf])
    singular = joint_lag._fit_bounded_joint_betas(x, x[:, 0], 40, lower, upper)
    assert np.isnan(singular).all()
    rng = np.random.default_rng(81)
    x = rng.normal(size=(60, 2))
    x[:, 1] += x[:, 0]
    y = x @ [.9, -.4]
    base = joint_lag._fit_bounded_joint_betas(x, y, 40, lower, upper)
    scales = np.array([1e-6, 1e6])
    scaled = joint_lag._fit_bounded_joint_betas(x * scales, y, 40,
                                              lower / scales, upper / scales)
    np.testing.assert_allclose(scaled * scales, base, atol=1e-8, equal_nan=True)


@pytest.mark.parametrize('kwargs', [
    {'min_ar_coeff': .8, 'max_ar_coeff': .75},
    {'min_factor_beta': .3, 'max_factor_beta': .1},
    {'max_ar_coeff': np.nan}, {'min_factor_beta': np.nan},
])
def test_joint_rejects_invalid_bounds(kwargs):
    """Invalid bound combinations fail before estimating any return history."""
    returns, factor = _correlated_sample()
    with pytest.raises(ValueError, match='bound'):
        joint_lag.adjust_returns_with_joint_unsmoothing(returns, factor, **kwargs)


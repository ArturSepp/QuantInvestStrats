"""Positivity rule of the AR(1) residual bootstrap and scale invariance of the AR(1) fit.

Two defects are guarded here, each by a test that fails on it alone:

  - the positivity clamp of ``bootstrap_ar_process`` replaced a non-positive value by the 25%
    quantile of that step's values *across columns*. For one series that quantile is the value
    itself, so the clamp did nothing (a positive series went negative); for a panel it coupled
    independent columns and could itself be negative, and it rewrote mean-zero columns whose
    data are not positive at all. The rule is now per column, and it can be switched off.
  - ``compute_ar_residuals`` treated a regressor as constant when its sample variance was within
    ``numpy.isclose``'s absolute tolerance of 1e-8, so any series with standard deviation below
    1e-4 got a zero slope. The test is now relative to the scale of the data.

Expected paths are recomputed with a plain numpy recursion from the fitted coefficients and the
supplied indices, independently of the compiled kernel.
"""

# packages
import numpy as np
import pandas as pd
import pytest
# qis / project
import qis
from qis.models.bootstrap.bootstrap_numba import (BootstrapOutput,
                                                  bootstrap_ar_process,
                                                  compute_ar_residuals,
                                                  generate_bootstrapped_indices)


def _ar1(theta: float, mean: float, sigma: float, n: int = 400, seed: int = 3) -> pd.Series:
    """A seeded AR(1) around ``mean`` on a month-end index."""
    rng = np.random.default_rng(seed)
    values = np.full(n, mean)
    for t in range(1, n):
        values[t] = mean + theta * (values[t - 1] - mean) + rng.normal(0.0, sigma)
    return pd.Series(values, index=pd.date_range('1990-01-31', periods=n, freq='ME'), name='y')


def _reference_paths(data: pd.DataFrame, indices: np.ndarray, floors: np.ndarray) -> np.ndarray:
    """Run the documented recursion and per-column floor in numpy.

    Returns:
        array of shape (num_samples, index_length, num_columns)
    """
    residuals, intercept, beta = compute_ar_residuals(data)
    start = np.nanmean(data.to_numpy(dtype=float), axis=0)
    paths = np.zeros((indices.shape[1], indices.shape[0], data.shape[1]))
    for m in range(indices.shape[1]):
        level = start.copy()
        for t in range(indices.shape[0]):
            level = intercept + beta * level + residuals[indices[t, m]]
            clamp = np.isfinite(floors) & (level <= 0.0)
            level = np.where(clamp, floors, level)
            paths[m, t] = level
    return paths


# a positive, persistent series close to zero, like a dividend yield: every observation is
# positive (the smallest is 0.002), but about 3% of unconstrained AR(1) path values fall below
# zero, because a path of 1500 steps wanders further than a sample of 400
POSITIVE = _ar1(theta=0.98, mean=0.02, sigma=0.0015, seed=5)
# a mean-zero series: positivity is not a property of its data
MEAN_ZERO = _ar1(theta=0.5, mean=0.0, sigma=0.01, seed=4).rename('z')
INDEX_LENGTH, NUM_SAMPLES = 1500, 6


def _indices(num_rows: int) -> np.ndarray:
    return generate_bootstrapped_indices(num_data_index=num_rows,
                                         bootstrap_type=qis.BootstrapType.STATIONARY,
                                         num_samples=NUM_SAMPLES, index_length=INDEX_LENGTH,
                                         block_size=20, seed=5)


def test_fixture_is_positive_and_its_unconstrained_paths_are_not():
    """The fixture has to exercise the rule: positive data, negative unconstrained paths."""
    assert (POSITIVE > 0.0).all()
    indices = _indices(len(POSITIVE) - 1)
    free = _reference_paths(POSITIVE.to_frame(), indices, floors=np.array([np.nan]))
    assert (free <= 0.0).mean() > 0.01


@pytest.mark.parametrize('bootstrap_output', list(BootstrapOutput))
def test_a_positive_single_series_stays_positive_by_default(bootstrap_output):
    """For one series the old cross-column quantile was the value itself, so it never bit."""
    indices = _indices(len(POSITIVE) - 1)
    sample = bootstrap_ar_process(POSITIVE, bootstrap_output=bootstrap_output,
                                  bootstrapped_indices=indices)
    if bootstrap_output == BootstrapOutput.SERIES_TO_DF:
        actual = sample.to_numpy().T[:, :, None]
    else:
        actual = np.stack([np.asarray(path) for path in sample])
    floor = np.quantile(POSITIVE.to_numpy(), 0.25)
    expected = _reference_paths(POSITIVE.to_frame(), indices, floors=np.array([floor]))
    assert (actual > 0.0).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)
    # the floor is where clamped steps land, and it is the column's own lower quartile
    assert np.isclose(actual, floor).any()


def test_is_positive_false_leaves_the_recursion_unconstrained():
    indices = _indices(len(POSITIVE) - 1)
    sample = bootstrap_ar_process(POSITIVE, bootstrapped_indices=indices, is_positive=False)
    actual = np.stack([np.asarray(path) for path in sample])
    expected = _reference_paths(POSITIVE.to_frame(), indices, floors=np.array([np.nan]))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)
    assert (actual <= 0.0).any()


def test_panel_columns_are_floored_independently():
    """A column's path cannot depend on the level of another column."""
    other = _ar1(theta=0.9, mean=0.05, sigma=0.003, seed=8).rename('other')
    assert (other > 0.0).all()
    first = pd.concat([POSITIVE, other], axis=1)
    second = pd.concat([POSITIVE, 10.0 * other], axis=1)
    indices = _indices(len(first) - 1)
    first_paths = np.stack([np.asarray(p) for p in bootstrap_ar_process(
        first, bootstrapped_indices=indices)])
    second_paths = np.stack([np.asarray(p) for p in bootstrap_ar_process(
        second, bootstrapped_indices=indices)])
    np.testing.assert_allclose(first_paths[:, :, 0], second_paths[:, :, 0], rtol=1e-13)
    assert (first_paths > 0.0).all() and (second_paths > 0.0).all()


def test_a_mean_zero_column_of_a_panel_is_not_constrained():
    """Positivity applies only to columns whose observed values are all positive."""
    panel = pd.concat([POSITIVE, MEAN_ZERO], axis=1)
    indices = _indices(len(panel) - 1)
    actual = np.stack([np.asarray(p) for p in bootstrap_ar_process(
        panel, bootstrapped_indices=indices)])
    floors = np.array([np.quantile(POSITIVE.to_numpy(), 0.25), np.nan])
    expected = _reference_paths(panel, indices, floors=floors)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)
    assert (actual[:, :, 0] > 0.0).all()
    assert (actual[:, :, 1] < 0.0).mean() > 0.3


def test_the_slope_does_not_depend_on_the_units_of_the_series():
    """A persistent series with standard deviation near 1e-4 is not a constant."""
    series = _ar1(theta=0.9, mean=0.0, sigma=4e-5, n=500, seed=3)
    assert np.var(series.to_numpy()[:-1], ddof=1) < 1e-8
    _, intercept_small, beta_small = compute_ar_residuals(series)
    _, intercept_large, beta_large = compute_ar_residuals(1e4 * series)
    values = series.to_numpy()
    target, regressor = values[1:], values[:-1]
    expected = np.cov(target, regressor, ddof=1)[0, 1] / np.var(regressor, ddof=1)
    assert beta_small[0] == pytest.approx(expected, rel=1e-10)
    assert beta_small[0] == pytest.approx(beta_large[0], rel=1e-10)
    assert 1e4 * intercept_small[0] == pytest.approx(intercept_large[0], rel=1e-8)
    assert beta_small[0] == pytest.approx(0.9, abs=0.05)


def test_a_series_constant_up_to_rounding_is_still_constant():
    """Values equal up to a few ulps have no autoregression to estimate."""
    values = np.where(np.arange(60) % 2 == 0, 0.1 + 0.2, 0.3)
    assert np.ptp(values) > 0.0
    series = pd.Series(values, index=pd.date_range('2000-01-31', periods=60, freq='ME'))
    residuals, intercept, beta = compute_ar_residuals(series)
    assert beta[0] == 0.0
    assert intercept[0] == pytest.approx(0.3)
    assert np.allclose(residuals, 0.0)

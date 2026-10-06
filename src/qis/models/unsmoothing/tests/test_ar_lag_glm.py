"""
Regression tests for the static Getmansky-Lo-Makarov unsmoother, qis.unsmooth_returns_glm.

Two paths are locked here:

  * the estimated path - weights fitted from the sample, which is what the function did before
    ``theta`` existed, so these tests are the guard that adding the argument changed nothing,
  * the fixed-weight path - ``theta`` supplied from outside the series, which is what a panel
    estimate pooled across vintages or a production constant looks like.

Tests use seeded draws and literal boundary panels; no data fixtures, no network.
"""
# packages
import numpy as np
import pandas as pd
import pytest
# qis / project
from qis.models.unsmoothing.ar_lag import unsmooth_returns_glm
from qis.models.unsmoothing import ar_lag

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


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'Float64'])
@pytest.mark.parametrize('theta', [[0.2], [0.2, -0.1], [0.2, -0.1, 0.05]])
def test_unsmooth_returns_glm_fixed_panel_matches_literal_recurrence(dtype, theta) -> None:
    """A ragged fund must not change its neighbors' lags, values or diagnostics."""
    panel = pd.DataFrame({
        'complete': [.01, .02, -.03, .04, .05, -.06, .07],
        'leading': [np.nan, np.nan, .03, .04, .05, .06, .07],
        'gap': [.01, .02, np.nan, .04, .05, .06, .07],
        'terminated': [.01, .02, .03, .04, np.nan, np.nan, np.nan],
        'missing': [np.nan] * 7,
        'zero': [0.] * 7,
    }, dtype=dtype, index=pd.date_range('2024-01-01', periods=7, tz='UTC', name='date'))
    panel.columns.name = 'fund'
    panel.attrs['convention'] = 'simple returns'
    before = panel.copy(deep=True)
    weights = np.array(theta)
    values = np.column_stack([panel[col].values.astype(float) for col in panel])
    expected_values = values.copy()
    # Accumulate in lag order: a dot product may round differently from the scalar contract.
    for column in range(values.shape[1]):
        for row in range(len(weights), len(values)):
            correction = 0.0
            for lag, weight in enumerate(weights):
                correction += weight * values[row - lag - 1, column]
            expected_values[row, column] = (
                values[row, column] - correction
            ) / (1.0 - float(weights.sum()))
    expected = pd.DataFrame({col: pd.Series(expected_values[:, i], index=panel.index, name=col)
                             for i, col in enumerate(panel.columns)})
    actual, diagnostics = unsmooth_returns_glm(panel, ar_order=99, theta=weights,
                                               return_diagnostics=True)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    pd.testing.assert_frame_equal(panel, before, check_exact=True)
    pd.testing.assert_frame_equal(
        unsmooth_returns_glm(panel.iloc[:5], theta=weights), actual.iloc[:5], check_exact=True)
    pd.testing.assert_frame_equal(
        unsmooth_returns_glm(panel.iloc[:, ::-1], theta=weights), actual.iloc[:, ::-1],
        check_exact=True)
    assert list(diagnostics) == list(panel)
    assert len({id(diag) for diag in diagnostics.values()}) == len(panel.columns)
    for diag in diagnostics.values():
        assert diag.theta is weights
        assert diag.ar_order == len(weights)
        assert diag.theta_sum == float(weights.sum())
        assert diag.vol_inflation_factor == 1.0 / (1.0 - float(weights.sum()))
        assert diag.is_severe is False
    # Accepted results own their values, but do not promise deep isolation of index objects.
    actual.iloc[0, 0] = 99.
    pd.testing.assert_frame_equal(panel, before, check_exact=True)


def test_unsmooth_returns_glm_fixed_panel_bypasses_scalar_recurrences(monkeypatch) -> None:
    """Guard the optimization without a machine-dependent timing assertion."""
    panel = pd.DataFrame(np.arange(24).reshape(8, 3) / 100)

    def unexpected_scalar(*args, **kwargs):
        pytest.fail('a regular fixed-theta panel must not run per-column scalar recurrences')

    monkeypatch.setattr(ar_lag, '_unsmooth_glm_single', unexpected_scalar)
    actual = unsmooth_returns_glm(panel, theta=[.2, -.1])
    assert actual.shape == panel.shape


@pytest.mark.parametrize('rows, columns', [(0, 2), (1, 2), (2, 2), (3, 0)])
def test_unsmooth_returns_glm_fixed_panel_short_and_empty_shapes(rows, columns) -> None:
    """No complete lag window exists; preserve even the legacy zero-column construction."""
    panel = pd.DataFrame(np.zeros((rows, columns)), index=pd.RangeIndex(5, 5 + rows))
    actual, diagnostics = unsmooth_returns_glm(panel, theta=[.2, -.1, .05],
                                               return_diagnostics=True)
    expected = pd.DataFrame({col: panel[col].astype(float) for col in panel})
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert list(diagnostics) == list(panel)


@pytest.mark.parametrize('theta', [[.2], [-.96], [1.1]])
def test_unsmooth_returns_glm_fixed_panel_multiindex_and_diagnostics(theta) -> None:
    """Reconstruction must keep scalar-column metadata and severe/sign-flipping diagnostics."""
    panel = pd.DataFrame(np.arange(18).reshape(6, 3), columns=pd.MultiIndex.from_tuples(
        [('b', 2), ('a', 1), ('a', 3)], names=['fund', 'class']))
    expected = pd.DataFrame({col: unsmooth_returns_glm(panel[col], theta=theta) for col in panel})
    actual, diagnostics = unsmooth_returns_glm(panel, theta=theta, return_diagnostics=True)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert all(diag.is_severe == (abs(sum(theta)) > .95) for diag in diagnostics.values())
    assert all(diag.vol_inflation_factor == (1 / (1 - sum(theta)) if sum(theta) < 1 else np.inf)
               for diag in diagnostics.values())


@pytest.mark.parametrize('values, theta', [
    ([.1, np.inf, .2, np.nan, .3], [0.]),
    ([1e308, -1e308, 1e308, .1, .2], [2.]),
    ([1e-300, 1e-300, 1e-300, .1, .2], [1e-100]),
])
def test_unsmooth_returns_glm_fixed_panel_preserves_scalar_numerical_errors(values, theta) -> None:
    """Exceptional arithmetic must retain scalar rejection rather than new vector warnings."""
    panel = pd.DataFrame({'a': values, 'b': np.arange(len(values)) / 100})
    before = panel.copy(deep=True)
    with np.errstate(all='raise'):
        with pytest.raises(FloatingPointError) as expected:
            unsmooth_returns_glm(panel['a'], theta=theta)
        with pytest.raises(FloatingPointError) as actual:
            unsmooth_returns_glm(panel, theta=theta)
    assert str(actual.value) == str(expected.value)
    pd.testing.assert_frame_equal(panel, before, check_exact=True)


def test_unsmooth_returns_glm_fixed_panel_duplicate_labels_keep_existing_failure() -> None:
    """An optimization is not authorization to choose new duplicate-label semantics."""
    panel = pd.DataFrame([[.1, .2], [.3, .4]], columns=['a', 'a'])
    with pytest.raises(ValueError, match='truth value'):
        unsmooth_returns_glm(panel, theta=.2)


@pytest.mark.parametrize('panel, theta', [
    (pd.DataFrame({'a': [.1, np.inf, .2, .3]}), [0.]),
    (pd.DataFrame({'a': [1e308, -1e308, .1, .2]}), [2.]),
    (pd.DataFrame({'a': [1e-300, 1e-300, .1, .2]}), [1e-100]),
    (pd.DataFrame({'a': [.1, .2], 'b': [.2, .1]}), [1e308, 1e308]),
])
def test_unsmooth_returns_glm_fixed_panel_preserves_scalar_warnings(panel, theta) -> None:
    """Fallback preserves warning messages/counts, including coefficients with no lag window."""
    with np.errstate(all='warn'):
        # Validate once, like the public panel path, before the per-column oracle.
        with pytest.warns(RuntimeWarning) as expected_warnings:
            weights = ar_lag._validate_fixed_theta(theta)
            expected = pd.DataFrame({
                col: ar_lag._unsmooth_glm_single(panel[col], len(weights), weights)[0]
                for col in panel
            })
        with pytest.warns(RuntimeWarning) as actual_warnings:
            actual = unsmooth_returns_glm(panel, theta=theta)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert [(w.category, str(w.message)) for w in actual_warnings] == [
        (w.category, str(w.message)) for w in expected_warnings]

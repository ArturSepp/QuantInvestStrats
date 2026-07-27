"""
tests for compute_ar_residuals and the AR path of bootstrap_ar_process.

Four tests guard defects that were live before the ``statsmodels.AutoReg``
implementation was replaced, and each was proven to fail on its own defect:

  - ``test_no_key_error_on_a_labelled_params_series``: ``ar_model.params[0]`` was
    a label lookup on a Series indexed ``['const', 'a.L1']``, raising
    ``KeyError: 0`` on pandas 3.0 and taking ``bootstrap_ar_process`` with it
  - ``test_gaps_do_not_produce_nan_residuals``: the model was fitted on
    ``dropna()`` but residualised on the raw array, so three gaps in 120 rows
    left six NaN residuals, one of which enters every draw that samples it
  - ``test_a_gap_does_not_create_a_false_lag_pair``: ``dropna()`` collapsed gaps,
    pairing observations several periods apart. Blanking 30 percent of a 3000
    point AR(1) at theta 0.7 moved the coefficient to 0.607
  - ``test_the_draw_is_over_the_residual_rows_not_the_data_rows``: the draw used
    ``num_data_index=len(data.index)`` against ``len(data.index)-1`` residual
    rows. The consumer is ``@njit`` with bounds checking off, so the largest
    index read adjacent memory instead of raising, and the value entered the
    draw as a residual

The rest pin what must not move: the coefficient agrees with ordinary least
squares on the lag pairs, which is what ``AutoReg(lags=1)`` computed, to 4e-16.
"""

# packages
import numpy as np
import pandas as pd
import pytest
# qis / project
import qis
from qis.models.bootstrap import bootstrap_numba
from qis.models.bootstrap.bootstrap_numba import compute_ar_residuals


def _ar1(theta: float, n: int = 300, sigma: float = 0.02, seed: int = 3) -> pd.Series:
    """a seeded AR(1) on a quarter-end index."""
    rng = np.random.default_rng(seed)
    values = np.zeros(n)
    for t in range(1, n):
        values[t] = theta * values[t - 1] + rng.normal(0.0, sigma)
    # unit='s' because n reaches 3000 quarters, which is 750 years: a nanosecond
    # DatetimeIndex tops out at 2262 and pandas 2.x raises OutOfBoundsDatetime there
    return pd.Series(values, name='a',
                     index=pd.date_range('1950-03-31', periods=n, freq='QE', unit='s'))


def test_no_key_error_on_a_labelled_params_series():
    """The fitted parameters are looked up positionally, not by label.

    ``AutoReg(...).fit().params`` is indexed ``['const', 'a.L1']``, so ``params[0]``
    is a label lookup and raises on pandas 3.0.
    """
    residuals, intercept, beta = compute_ar_residuals(_ar1(0.5))
    assert np.isfinite(residuals).all()
    assert np.isfinite(intercept).all()
    assert np.isfinite(beta).all()


def test_the_estimate_recovers_a_known_coefficient():
    _, _, beta = compute_ar_residuals(_ar1(0.6, n=2000))
    assert beta[0] == pytest.approx(0.6, abs=0.05)


def test_it_matches_ordinary_least_squares_on_the_lag_pairs():
    """The AR(1) conditional MLE is OLS of the series on its own lag."""
    series = _ar1(0.4, n=500)
    _, intercept, beta = compute_ar_residuals(series)
    values = series.to_numpy()
    target, regressor = values[1:], values[:-1]
    expected_beta = np.cov(target, regressor, ddof=1)[0, 1] / np.var(regressor, ddof=1)
    assert beta[0] == pytest.approx(expected_beta, rel=1e-10)
    assert intercept[0] == pytest.approx(np.mean(target) - expected_beta * np.mean(regressor),
                                         rel=1e-10)


def test_residuals_reconstruct_the_series():
    """y_t = intercept + beta y_{t-1} + residual, on every retained pair."""
    series = _ar1(0.5, n=200)
    residuals, intercept, beta = compute_ar_residuals(series)
    values = series.to_numpy()
    reconstructed = intercept[0] + beta[0] * values[:-1] + residuals[:, 0]
    assert np.allclose(reconstructed, values[1:])


def test_residuals_have_no_remaining_autocorrelation():
    residuals, _, _ = compute_ar_residuals(_ar1(0.7, n=2000))
    column = residuals[:, 0]
    autocorrelation = np.corrcoef(column[1:], column[:-1])[0, 1]
    assert abs(autocorrelation) < 0.1


def test_gaps_do_not_produce_nan_residuals():
    """A missing observation must not leave NaN in the residuals.

    The residuals feed the bootstrap directly, so a NaN here becomes a NaN in
    every draw that samples it.
    """
    series = _ar1(0.5, n=120)
    holed = series.copy()
    holed.iloc[[10, 50, 90]] = np.nan
    residuals, _, _ = compute_ar_residuals(holed)
    assert np.isfinite(residuals).all()


def test_a_gap_removes_the_pairs_that_straddle_it():
    """Each NaN invalidates the pair ending at it and the one starting from it."""
    series = _ar1(0.5, n=120)
    holed = series.copy()
    holed.iloc[[10, 50, 90]] = np.nan
    residuals, _, _ = compute_ar_residuals(holed)
    assert len(residuals) == (len(series) - 1) - 6


def test_a_gap_does_not_create_a_false_lag_pair():
    """Dropping a gap must not make the observations either side consecutive.

    Estimating on the pairs that survive a scattered set of gaps should recover
    the same coefficient. Collapsing the gaps instead pairs points two and three
    periods apart, and an AR(1) at those spacings has persistence theta squared
    or lower, so the estimate is dragged towards zero.
    """
    series = _ar1(0.7, n=3000)
    holed = series.copy()
    holed[np.random.default_rng(11).random(len(series)) < 0.3] = np.nan
    _, _, beta_full = compute_ar_residuals(series)
    _, _, beta_holed = compute_ar_residuals(holed)
    # Over 20 gap patterns the surviving pairs stay within 0.020 of the gap-free
    # estimate, while collapsing the gaps never comes closer than 0.068.
    assert beta_holed[0] == pytest.approx(beta_full[0], abs=0.04)


def test_the_draw_is_over_the_residual_rows_not_the_data_rows(monkeypatch):
    """The resampler must not be able to index past the end of the residuals.

    An AR(1) on n observations has n-1 residuals, so drawing with
    ``num_data_index=len(data.index)`` puts the largest index one row past the
    end. The consumer is ``@njit`` with bounds checking off, so that read does
    not raise: it returns adjacent memory, and the value enters the draw as a
    residual. Captured here at the call site, because the symptom downstream is
    a plausible number rather than an error.
    """
    drawn = {}
    original = bootstrap_numba.generate_bootstrapped_indices

    def capture(num_data_index: int, **kwargs) -> np.ndarray:
        drawn['num_data_index'] = num_data_index
        return original(num_data_index=num_data_index, **kwargs)

    monkeypatch.setattr(bootstrap_numba, 'generate_bootstrapped_indices', capture)
    series = _ar1(0.5, n=50)
    qis.bootstrap_ar_process(series, num_samples=4, index_length=49, block_size=10, seed=1)
    residuals, _, _ = compute_ar_residuals(series)
    assert drawn['num_data_index'] == len(residuals)


def test_supplied_indices_are_checked_against_the_residual_rows():
    """Indices drawn elsewhere must fit the residual array of this data.

    ``bootstrap_price_fundamental_data`` draws one index set over ``len(prices.index)-1``
    and passes it to both the price path and the AR path, which is what keeps the two
    resampled together. A gap in the fundamental data shortens the residuals below that
    length, and ``get_bootstrap_ar_data_list`` is ``@njit`` with bounds checking off, so
    the overshoot reads adjacent memory instead of raising. The call site has to refuse
    it. Shipped in 5.2.1; 5.2.0 went out without it.
    """
    series = _ar1(0.5, n=120)
    holed = series.copy()
    holed.iloc[[10, 50, 90]] = np.nan
    # in bounds for a gap-free series of this length, past the end once gaps remove pairs
    indices = np.full((5, 2), len(series) - 2, dtype=int)
    with pytest.raises(ValueError, match='draw them over the residual rows'):
        qis.bootstrap_ar_process(holed, bootstrapped_indices=indices)


def test_bootstrap_ar_process_runs_and_returns_finite_paths():
    """The end-to-end path completes and carries no NaN or garbage."""
    series = _ar1(0.5, n=200)
    sample = qis.bootstrap_ar_process(series, num_samples=5, index_length=199,
                                      block_size=20, seed=1)
    paths = [np.asarray(path) for path in sample]
    assert len(paths) == 5
    assert all(np.isfinite(path).all() for path in paths)


def test_a_multi_column_panel_keeps_its_rows_aligned():
    """Rows are resampled jointly, so a row must be complete across columns."""
    frame = pd.concat([_ar1(0.5, seed=1).rename('a'), _ar1(0.2, seed=2).rename('b')], axis=1)
    residuals, intercept, beta = compute_ar_residuals(frame)
    assert residuals.shape == (len(frame) - 1, 2)
    assert len(intercept) == len(beta) == 2
    assert beta[0] > beta[1]


def test_a_constant_series_is_not_an_autoregression():
    """No variation in the regressor means no coefficient to estimate."""
    flat = pd.Series(np.full(50, 0.01),
                     index=pd.date_range('2000-03-31', periods=50, freq='QE'))
    residuals, intercept, beta = compute_ar_residuals(flat)
    assert beta[0] == 0.0
    assert intercept[0] == pytest.approx(0.01)
    assert np.allclose(residuals, 0.0)


def test_too_few_usable_pairs_is_refused():
    short = pd.Series([0.01, 0.02, 0.03],
                      index=pd.date_range('2000-03-31', periods=3, freq='QE'))
    with pytest.raises(ValueError, match='at least 3 complete lag pairs'):
        compute_ar_residuals(short.iloc[:2])

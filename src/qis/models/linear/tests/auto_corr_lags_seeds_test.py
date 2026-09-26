"""Lag-zero, seed, warm-up and degenerate-input contracts of the autocorrelation estimators."""

# packages
import numpy as np
import pandas as pd
import pytest

# qis
from qis.models.linear.auto_corr import (
    compute_autocorrelation_at_int_periods,
    compute_ewm_matrix_autocorr,
    compute_ewm_vector_autocorr,
    compute_ewm_vector_autocorr_df,
    compute_path_autocorr,
    compute_path_autocorr_given_lags,
    compute_path_lagged_corr,
    compute_path_lagged_corr_given_lags,
    estimate_acf_from_paths,
)
from qis.models.linear.ewm import InitType


def _pair(num_rows: int = 150, seed: int = 21):
    rng = np.random.default_rng(seed)
    a1 = rng.standard_normal(num_rows)
    a2 = 0.7 * a1 + rng.standard_normal(num_rows)
    return a1, a2


def test_lagged_corr_lag_zero_is_contemporaneous_correlation() -> None:
    """With a1 != a2, entry 0 is corr(a1, a2), not one."""
    a1, a2 = _pair()
    corr = compute_path_lagged_corr(a1=a1, a2=a2, num_lags=4)
    assert corr[0] == pytest.approx(np.corrcoef(a1, a2)[0, 1], abs=1e-12)
    for k in range(1, 4):
        assert corr[k] == pytest.approx(np.corrcoef(a1[k:], a2[:-k])[0, 1], abs=1e-12)


def test_autocorr_lag_zero_stays_one() -> None:
    """The autocorrelation kernels keep lag 0 at exactly one."""
    a1, a2 = _pair()
    np.testing.assert_array_equal(compute_path_autocorr(a=a1, num_lags=3)[0], 1.0)
    panel = np.column_stack([a1, a2])
    np.testing.assert_array_equal(compute_path_autocorr(a=panel, num_lags=3)[0], 1.0)


def test_given_lags_accepts_lag_zero() -> None:
    """Lag 0 returns the contemporaneous correlation, and one for an autocorrelation."""
    a1, a2 = _pair()
    lags = np.array([0, 1, 5])
    corr = compute_path_lagged_corr_given_lags(a1=a1, a2=a2, lags=lags)
    expected = [np.corrcoef(a1, a2)[0, 1]] + [np.corrcoef(a1[k:], a2[:-k])[0, 1] for k in (1, 5)]
    np.testing.assert_allclose(corr, expected, atol=1e-12)
    auto = compute_path_autocorr_given_lags(a=np.column_stack([a1, a2]), lags=lags)
    np.testing.assert_array_equal(auto[:, 0], 1.0)
    assert auto[0, 1] == pytest.approx(np.corrcoef(a1[1:], a1[:-1])[0, 1], abs=1e-12)


def test_acf_from_paths_names_dispersion_std() -> None:
    """The dispersion across paths is named 'std' and uses ddof=0."""
    paths = pd.DataFrame(np.random.default_rng(1).standard_normal((200, 3)))
    acfs, mean, std = estimate_acf_from_paths(paths=paths, nlags=4, is_pacf=False)
    assert mean.name == 'mean' and std.name == 'std'
    np.testing.assert_allclose(std.to_numpy(), acfs.std(axis=1, ddof=0).to_numpy(), atol=1e-14)


def test_matrix_autocorr_single_column_has_nan_off_diagonal() -> None:
    """With one column there is no off-diagonal entry, so its mean is NaN, not a crash."""
    a = np.random.default_rng(2).standard_normal((50, 1))
    diagonal, off_diagonal = compute_ewm_matrix_autocorr(a=a, aggregation_type='mean')
    assert np.all(np.isnan(off_diagonal))
    assert np.all(np.isfinite(diagonal[1:]))


@pytest.mark.parametrize('lag', [1, 3])
def test_rows_before_lag_are_nan(lag: int) -> None:
    """Both EWM kernels report NaN, not zero, before the first lagged pair exists."""
    a = np.random.default_rng(4).standard_normal((40, 3))
    vector = compute_ewm_vector_autocorr(a=a, lag=lag)
    diagonal, off_diagonal = compute_ewm_matrix_autocorr(a=a, lag=lag)
    assert np.all(np.isnan(vector[:lag])) and np.all(np.isfinite(vector[lag:]))
    assert np.all(np.isnan(diagonal[:lag])) and np.all(np.isnan(off_diagonal[:lag]))
    assert np.all(np.isfinite(diagonal[lag:]))


def test_vector_autocorr_is_point_in_time_by_default() -> None:
    """A run on a prefix reproduces the prefix of the full run; the seed is not full-sample."""
    x = pd.Series(np.random.default_rng(6).standard_normal(400)).cumsum().diff().fillna(0.0)
    full = compute_ewm_vector_autocorr_df(data=x, span=30)
    prefix = compute_ewm_vector_autocorr_df(data=x.iloc[:120], span=30)
    pd.testing.assert_series_equal(prefix, full.iloc[:120], atol=1e-12, rtol=0.0)


def test_vector_autocorr_equals_matrix_diagonal() -> None:
    """With zero seeds the vector estimator is the diagonal of the matrix estimator."""
    a = np.random.default_rng(8).standard_normal((80, 1))
    vector = compute_ewm_vector_autocorr(a=a, ewm_lambda=0.9, lag=1)
    diagonal, _ = compute_ewm_matrix_autocorr(a=a, ewm_lambda=0.9, lag=1)
    np.testing.assert_allclose(vector[:, 0], diagonal, atol=1e-12, equal_nan=True)


def test_vector_autocorr_full_sample_seed_on_request() -> None:
    """var_init_type=InitType.VAR restores the full-sample variance seed."""
    a = np.random.default_rng(9).standard_normal(60)
    lam = 0.9
    out = compute_ewm_vector_autocorr(a=a, ewm_lambda=lam, lag=1, var_init_type=InitType.VAR)
    cross, second = 0.0, np.var(a)
    expected = np.full(60, np.nan)
    for t in range(1, 60):
        cross = (1 - lam) * a[t - 1] * a[t] + lam * cross
        second = (1 - lam) * a[t] ** 2 + lam * second
        expected[t] = cross / second
    np.testing.assert_allclose(out[:, 0], expected, atol=1e-12, equal_nan=True)
    with pytest.raises(ValueError, match='var_init_type'):
        compute_ewm_vector_autocorr(a=a, var_init_type=InitType.MEAN)


def test_block_autocorr_rejects_smoothing_span_clearly() -> None:
    """The unimplemented ewma_smoothin_span option is rejected with a message naming it."""
    data = pd.DataFrame({'x': np.random.default_rng(1).standard_normal(100)})
    with pytest.raises(NotImplementedError, match='ewma_smoothin_span'):
        compute_autocorrelation_at_int_periods(data=data, span=5, ewma_smoothin_span=3)


def test_block_autocorr_demean_has_no_effect() -> None:
    """demean subtracts a constant from each column, which a Pearson correlation ignores."""
    data = pd.DataFrame({'x': 0.3 + np.random.default_rng(12).standard_normal(100)})
    with_demean = compute_autocorrelation_at_int_periods(data=data, span=5, demean=True)
    without = compute_autocorrelation_at_int_periods(data=data, span=5, demean=False)
    pd.testing.assert_series_equal(with_demean, without, atol=1e-12, rtol=0.0)

"""Positive semidefiniteness, point-in-time seeds and decay of the EWM covariance estimators.

The Newey-West estimators weight the lag-k cross moment by lambda^(k/2), the geometric mean of
the EWM weights of the two dates it pairs. The estimator is then a quadratic form in the
EWM-weighted observations with the Bartlett (Fejer) kernel, so it cannot be negative.
"""
# packages
import numpy as np
import pandas as pd
import pytest

# qis
from qis.models.linear.ewm import (NanBackfill, compute_ewm_covar, compute_ewm_covar_newey_west,
                                   compute_ewm_covar_tensor,
                                   compute_ewm_covar_tensor_vol_norm_returns,
                                   compute_ewm_newey_west_vol)


def _nw_matrix_reference(x: np.ndarray, ewm_lambda: float, num_lags: int) -> np.ndarray:
    """Newey-West EWM covariance at the last row by explicit loops, zero seeds."""
    n = x.shape[1]
    covar = np.zeros((n, n))
    for row in x:
        covar = ewm_lambda * covar + (1.0 - ewm_lambda) * np.outer(row, row)
    for lag in range(1, num_lags + 1):
        cross = np.zeros((n, n))
        for t in range(len(x)):
            product = np.outer(x[t], x[t - lag]) if t >= lag else np.zeros((n, n))
            cross = ewm_lambda * cross + (1.0 - ewm_lambda) * product
        weight = (1.0 - lag / (num_lags + 1)) * ewm_lambda ** (lag / 2)
        covar = covar + weight * (cross + cross.T)
    return covar


def test_newey_west_covariance_lags_use_the_given_decay() -> None:
    """With only ewm_lambda given, the lag terms use it rather than the 0.94 default."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal((150, 2)) * 0.01
    actual = compute_ewm_covar_newey_west(x, num_lags=2, ewm_lambda=0.5)
    np.testing.assert_allclose(actual, _nw_matrix_reference(x, 0.5, 2), rtol=1e-12)


def test_newey_west_covariance_passes_nan_backfill_to_the_lags() -> None:
    """DEFLATED_FFILL on a gappy panel equals the estimator on the zero-filled panel."""
    rng = np.random.default_rng(4)
    x = rng.standard_normal((120, 2)) * 0.01
    x[[10, 11, 50], 0] = np.nan
    gappy = compute_ewm_covar_newey_west(x, num_lags=2, ewm_lambda=0.8,
                                         nan_backfill=NanBackfill.DEFLATED_FFILL)
    filled = compute_ewm_covar_newey_west(np.nan_to_num(x), num_lags=2, ewm_lambda=0.8)
    np.testing.assert_allclose(gappy, filled, rtol=1e-12)


def test_newey_west_covariance_is_positive_semidefinite() -> None:
    """Alternating decaying returns, the counterexample for the unweighted lags, stay PSD."""
    t = np.arange(800)
    x = np.column_stack([(-1.0) ** t * 0.94 ** (t / 2), (-1.0) ** t * 0.94 ** (t / 2) * 0.5])
    covar = compute_ewm_covar_newey_west(x, num_lags=1, ewm_lambda=0.94)
    assert np.linalg.eigvalsh(covar).min() >= -1e-12 * np.abs(covar).max()


def test_newey_west_vol_counterexample_is_non_negative_with_closed_form() -> None:
    """For x_t = (-1)^t lambda^(t/2) and one lag the estimator is exactly lambda^t."""
    rows = np.arange(1000)
    alternating = pd.Series((-1.0) ** rows * 0.94 ** (rows / 2))
    variance, ratio = compute_ewm_newey_west_vol(alternating, num_lags=1, ewm_lambda=0.94,
                                                 apply_sqrt=False)
    np.testing.assert_allclose(variance, 0.94 ** rows, rtol=1e-8)
    vol, _ = compute_ewm_newey_west_vol(alternating, num_lags=1, ewm_lambda=0.94)
    assert vol.notna().all()


@pytest.mark.parametrize('nan_backfill', list(NanBackfill))
def test_newey_west_vol_is_non_negative_with_gaps(nan_backfill: NanBackfill) -> None:
    """Random gaps under every policy leave the corrected variance non-negative."""
    rng = np.random.default_rng(8)
    shocks = rng.standard_normal((600, 4)) * 0.01
    x = shocks - 0.8 * np.vstack([np.zeros((1, 4)), shocks[:-1]])  # strongly negative AC
    x[rng.random(x.shape) < 0.15] = np.nan
    variance, ratio = compute_ewm_newey_west_vol(x, num_lags=3, span=20, apply_sqrt=False,
                                                 nan_backfill=nan_backfill)
    finite = variance[np.isfinite(variance)]
    assert finite.size > 0 and finite.min() >= 0.0
    assert np.nanmin(ratio) >= 0.0


def test_newey_west_vol_deflated_ffill_is_a_zero_observation() -> None:
    """DEFLATED_FFILL equals the estimator on the series with the gaps set to zero."""
    rng = np.random.default_rng(9)
    x = rng.standard_normal(200) * 0.01
    x[[20, 21, 90]] = np.nan
    gappy, _ = compute_ewm_newey_west_vol(x, num_lags=2, ewm_lambda=0.9, apply_sqrt=False,
                                          nan_backfill=NanBackfill.DEFLATED_FFILL)
    filled, _ = compute_ewm_newey_west_vol(np.nan_to_num(x), num_lags=2, ewm_lambda=0.9,
                                           apply_sqrt=False)
    np.testing.assert_allclose(gappy, filled, rtol=1e-12)


def test_newey_west_vol_ffill_stops_time_at_a_gap() -> None:
    """FFILL equals the estimator on the observed rows only, held through the gaps."""
    rng = np.random.default_rng(10)
    x = rng.standard_normal(200) * 0.01
    gaps = np.array([20, 21, 90])
    x[gaps] = np.nan
    gappy, _ = compute_ewm_newey_west_vol(x, num_lags=2, ewm_lambda=0.9, apply_sqrt=False,
                                          nan_backfill=NanBackfill.FFILL)
    compressed, _ = compute_ewm_newey_west_vol(x[np.isfinite(x)], num_lags=2, ewm_lambda=0.9,
                                               apply_sqrt=False)
    np.testing.assert_allclose(gappy[np.isfinite(x)], compressed, rtol=1e-12)
    np.testing.assert_allclose(gappy[gaps], gappy[gaps - 1 - np.array([0, 1, 0])])


def test_newey_west_vol_zero_fill_restarts_after_a_gap() -> None:
    """ZERO_FILL erases the history: after a gap the estimator restarts from a zero state."""
    rng = np.random.default_rng(11)
    x = rng.standard_normal(120) * 0.01
    x[60] = np.nan
    gappy, _ = compute_ewm_newey_west_vol(x, num_lags=2, ewm_lambda=0.9, apply_sqrt=False,
                                          nan_backfill=NanBackfill.ZERO_FILL)
    restart, _ = compute_ewm_newey_west_vol(x[61:], num_lags=2, ewm_lambda=0.9,
                                            init_value=0.0, apply_sqrt=False)
    assert gappy[60] == 0.0
    np.testing.assert_allclose(gappy[61:], restart, rtol=1e-12)
    nan_filled, _ = compute_ewm_newey_west_vol(x, num_lags=2, ewm_lambda=0.9, apply_sqrt=False,
                                               nan_backfill=NanBackfill.NAN_FILL)
    assert np.isnan(nan_filled[60])
    np.testing.assert_allclose(nan_filled[61:], restart, rtol=1e-12)


def _asynchronous_panel() -> np.ndarray:
    rng = np.random.default_rng(12)
    base = rng.standard_normal((400, 1))
    a = 0.9 * base + 0.3 * rng.standard_normal((400, 3))
    a[rng.random(a.shape) < 0.2] = np.nan  # holidays that differ across assets
    return a * 0.01


def test_default_covariance_tensor_is_positive_semidefinite_with_asynchronous_gaps() -> None:
    """The default missing-data policy keeps every matrix of the path PSD."""
    tensor = compute_ewm_covar_tensor(_asynchronous_panel(), span=20)
    for matrix in tensor[1:]:
        eigenvalues = np.linalg.eigvalsh(matrix)
        assert eigenvalues.min() >= -1e-12 * eigenvalues.max()
    corr = compute_ewm_covar_tensor(_asynchronous_panel(), span=20, is_corr=True)
    assert np.nanmax(np.abs(corr)) <= 1.0 + 1e-12


def test_default_covariance_at_last_date_is_positive_semidefinite() -> None:
    """The FFILL counterexample: the default policy decays the held entries and stays PSD."""
    seed = np.array([[1.0, 0.99], [0.99, 1.0]])
    covar = compute_ewm_covar(np.array([[np.nan, 0.0]]), ewm_lambda=0.94, covar0=seed)
    np.testing.assert_allclose(covar, 0.94 * seed)
    held = compute_ewm_covar(np.array([[np.nan, 0.0]]), ewm_lambda=0.94, covar0=seed,
                             nan_backfill=NanBackfill.FFILL)
    assert np.linalg.eigvalsh(held).min() < 0.0  # explicit FFILL keeps the documented risk


def test_vol_normalised_tensor_is_point_in_time() -> None:
    """Every output on a prefix of the sample equals the same rows of the full-sample output."""
    rng = np.random.default_rng(13)
    a = rng.standard_normal((300, 3)) * 0.01
    full = compute_ewm_covar_tensor_vol_norm_returns(a, span=36)
    prefix = compute_ewm_covar_tensor_vol_norm_returns(a[:100], span=36)
    for whole, part in zip(full, prefix):
        np.testing.assert_allclose(whole[:100], part, rtol=1e-12)
    np.testing.assert_allclose(full[2][0], np.abs(a[0]))  # X0 seed: first vol is |r_0|


def test_vol_normalised_tensor_is_corr_switches_the_second_output() -> None:
    """is_corr changes the normalised tensor to a correlation; the covariance is unchanged."""
    rng = np.random.default_rng(14)
    a = rng.standard_normal((200, 3)) * 0.01
    cov, norm, vols = compute_ewm_covar_tensor_vol_norm_returns(a, span=36)
    cov_c, corr, vols_c = compute_ewm_covar_tensor_vol_norm_returns(a, span=36, is_corr=True)
    np.testing.assert_allclose(cov_c, cov, rtol=1e-12)
    np.testing.assert_allclose(np.diagonal(corr, axis1=1, axis2=2), 1.0)
    np.testing.assert_allclose(np.diagonal(cov, axis1=1, axis2=2), vols ** 2, rtol=1e-12)

"""Regression tests for warning-free variance and correlation normalization."""
import warnings

import numpy as np
import pytest

from qis.models.linear.ewm import (
    compute_ewm_covar,
    compute_ewm_covar_newey_west,
    compute_ewm_covar_tensor,
    compute_ewm_covar_tensor_vol_norm_returns,
)
from qis.models.linear.ewm_winsor_outliers import ewm_winsdor_markovian_score
from qis.models.linear.pca import compute_eigen_portfolio_weights
from qis.utils.np_ops import covar_to_corr


def test_ewm_correlation_uses_canonical_zero_variance_convention() -> None:
    """A wholly constant panel has undefined correlations rather than a zero matrix."""
    observations = np.zeros((8, 3))
    covariance = compute_ewm_covar(observations, span=3, is_corr=False)

    actual = compute_ewm_covar(observations, span=3, is_corr=True)

    np.testing.assert_allclose(actual, covar_to_corr(covariance), equal_nan=True)


def test_ewm_correlation_preserves_positive_variance_result() -> None:
    """Canonical normalization leaves the established finite EWM result unchanged."""
    observations = np.array([
        [0.01, -0.02],
        [0.03, 0.01],
        [-0.02, 0.04],
        [0.01, 0.02],
    ])
    covariance = compute_ewm_covar(observations, span=3, is_corr=False)

    actual = compute_ewm_covar(observations, span=3, is_corr=True)

    np.testing.assert_allclose(actual, covar_to_corr(covariance), atol=1.0e-14)


def test_all_ewm_correlation_paths_share_the_invalid_variance_convention() -> None:
    """Terminal, Newey-West, tensor, and vol-normalized paths agree on undefined rows."""
    observations = np.zeros((8, 2))

    newey_west = compute_ewm_covar_newey_west(
        observations,
        num_lags=2,
        span=3,
        is_corr=True,
    )
    tensor = compute_ewm_covar_tensor(observations, span=3, is_corr=True)
    _, normalized_tensor, _ = compute_ewm_covar_tensor_vol_norm_returns(
        observations,
        span=3,
        is_corr=True,
    )

    assert np.isnan(newey_west).all()
    assert np.isnan(tensor).all()
    assert np.isnan(normalized_tensor).all()


def test_markovian_score_masks_zero_variance_without_warning() -> None:
    """The outlier score remains missing until a strictly positive variance exists."""
    observations = np.array([[0.0, 0.0], [1.0, 0.0]])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        *_, score = ewm_winsdor_markovian_score(
            observations,
            init_value=np.zeros(2),
            init_var=np.zeros(2),
            span=3,
        )

    assert np.isnan(score[1]).all()


@pytest.mark.parametrize(
    "covariance",
    [
        np.diag([1.0, 0.0]),
        np.diag([1.0, np.nan]),
        np.array([[1.0, 1.0], [1.0, 1.0]]),
    ],
)
def test_eigen_portfolios_reject_undefined_unit_variance_weights(
        covariance: np.ndarray,
        ) -> None:
    """Zero/non-finite asset variance or a zero eigenvalue makes unit scaling undefined."""
    with pytest.raises(ValueError, match="unit-variance eigen-portfolios"):
        compute_eigen_portfolio_weights(covariance)


def test_eigen_portfolios_remain_orthogonal_and_unit_variance() -> None:
    """The guarded implementation preserves the PCA portfolio normalization contract."""
    covariance = np.array([[0.04, 0.012], [0.012, 0.09]])

    weights = compute_eigen_portfolio_weights(covariance)

    portfolio_covariance = weights @ covariance @ weights.T
    np.testing.assert_allclose(portfolio_covariance, np.identity(2), atol=1.0e-12)

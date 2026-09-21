"""Regression tests for covariance-to-correlation normalization."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from qis.utils.np_ops import covar_to_corr


def test_covar_to_corr_preserves_array_type_and_finite_correlations() -> None:
    """Normalize an ordinary covariance array without changing its container type."""
    covariance = np.array([[0.04, 0.012], [0.012, 0.09]])

    actual = covar_to_corr(covariance)

    assert isinstance(actual, np.ndarray)
    np.testing.assert_allclose(actual, np.array([[1.0, 0.2], [0.2, 1.0]]))


def test_covar_to_corr_marks_undefined_rows_without_warnings() -> None:
    """Return missing correlations for exact-zero and missing variances without warnings."""
    labels = ["RISKY", "ZERO", "MISSING"]
    covariance = pd.DataFrame(
        np.diag([0.04, 0.0, np.nan]),
        index=pd.Index(labels, name="asset"),
        columns=pd.Index(labels, name="asset"),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        actual = covar_to_corr(covariance)

    expected = pd.DataFrame(
        np.full((3, 3), np.nan),
        index=covariance.index,
        columns=covariance.columns,
    )
    expected.loc["RISKY", "RISKY"] = 1.0
    pd.testing.assert_frame_equal(actual, expected)


def test_covar_to_corr_treats_roundoff_negative_variance_as_zero() -> None:
    """Treat a negative diagonal within the matrix-scaled tolerance as zero variance."""
    covariance = np.diag([0.04, -1.0e-20])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        actual = covar_to_corr(covariance)

    np.testing.assert_allclose(
        actual,
        np.array([[1.0, np.nan], [np.nan, np.nan]]),
        equal_nan=True,
    )


def test_covar_to_corr_rejects_materially_negative_variance() -> None:
    """Reject a negative diagonal that cannot be explained by floating-point round-off."""
    covariance = np.diag([0.04, -1.0e-4])

    with pytest.raises(ValueError, match="materially negative"):
        covar_to_corr(covariance)


def test_covar_to_corr_rejects_non_square_input() -> None:
    """Reject inputs that cannot represent a covariance matrix."""
    with pytest.raises(ValueError, match="square"):
        covar_to_corr(np.ones((2, 3)))

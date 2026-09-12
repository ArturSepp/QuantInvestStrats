"""Verify that numeric pandas storage does not change scalar OLS estimates.

The compact representation matrix covers the one-regressor containers accepted by
``estimate_ols_alpha_beta``. Nullable dtypes matter because statsmodels cannot fit the object
design matrix that pandas can otherwise construct when an intercept is added.
"""

import warnings
from typing import Union

import numpy as np
import pandas as pd
import pytest

from qis.utils.regression import estimate_ols_alpha_beta


OlsInput = Union[np.ndarray, pd.Series, pd.DataFrame]


def _regressor(storage: str) -> OlsInput:
    """Build equivalent explanatory data in each supported one-regressor container."""
    values = [0.0, 1.0, 2.0, 3.0]
    if storage == "numpy-vector":
        return np.asarray(values, dtype=np.float64)
    if storage == "numpy-column":
        return np.asarray(values, dtype=np.float64)[:, None]
    if storage == "series-float64":
        return pd.Series(values, dtype=np.float64, name="x")
    if storage == "series-Float64":
        return pd.Series(values, dtype=pd.Float64Dtype(), name="x")
    if storage == "series-Int64":
        return pd.Series([0, 1, 2, 3], dtype=pd.Int64Dtype(), name="x")
    if storage == "frame-float64":
        return pd.DataFrame({"x": pd.Series(values, dtype=np.float64)})
    if storage == "frame-Float64":
        return pd.DataFrame({"x": pd.Series(values, dtype=pd.Float64Dtype())})
    raise ValueError(f"unknown storage {storage!r}")


@pytest.mark.parametrize(
    "storage",
    [
        "numpy-vector",
        "numpy-column",
        "series-float64",
        "series-Float64",
        "series-Int64",
        "frame-float64",
        "frame-Float64",
    ],
)
def test_estimate_ols_alpha_beta_is_invariant_to_numeric_storage(storage: str) -> None:
    """Match the independently known line for every supported numeric representation."""
    x = _regressor(storage)
    y = pd.Series([1.0, 3.0, 5.0, 7.0], name="y")
    x_before = x.copy()
    y_before = y.copy(deep=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        alpha, beta, r_squared, alpha_pvalue = estimate_ols_alpha_beta(x=x, y=y)

    np.testing.assert_allclose(
        [alpha, beta, r_squared],
        [1.0, 2.0, 1.0],
        rtol=0.0,
        atol=1.0e-12,
    )
    assert 0.0 <= alpha_pvalue <= 1.0
    if isinstance(x, pd.Series):
        pd.testing.assert_series_equal(x, x_before)
    elif isinstance(x, pd.DataFrame):
        pd.testing.assert_frame_equal(x, x_before)
    else:
        np.testing.assert_array_equal(x, x_before)
    pd.testing.assert_series_equal(y, y_before)


def test_estimate_ols_alpha_beta_filters_paired_nullable_missing_rows() -> None:
    """Drop paired missing rows before fitting the independently known finite line."""
    x = pd.Series([0.0, 1.0, pd.NA, 3.0, 4.0], dtype=pd.Float64Dtype(), name="x")
    y = pd.Series([1.0, 3.0, pd.NA, 7.0, 9.0], dtype=pd.Float64Dtype(), name="y")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        alpha, beta, r_squared, _ = estimate_ols_alpha_beta(x=x, y=y)

    np.testing.assert_allclose(
        [alpha, beta, r_squared],
        [1.0, 2.0, 1.0],
        rtol=0.0,
        atol=1.0e-12,
    )


def test_estimate_ols_alpha_beta_preserves_no_intercept_semantics() -> None:
    """Keep the established zero-alpha convention when the intercept is disabled."""
    x = pd.Series([1.0, 2.0, 3.0, 4.0], dtype=pd.Float64Dtype(), name="x")
    y = pd.Series([2.0, 4.0, 6.0, 8.0], dtype=pd.Float64Dtype(), name="y")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = estimate_ols_alpha_beta(x=x, y=y, fit_intercept=False)

    np.testing.assert_allclose(actual, [0.0, 2.0, 1.0, 0.0], rtol=0.0, atol=1.0e-12)


def test_estimate_ols_alpha_beta_keeps_nonnumeric_fallback() -> None:
    """Do not turn the numeric-storage normalization into string coercion."""
    x = pd.Series(["0", "1", "2", "3"], name="x")
    y = pd.Series([1.0, 3.0, 5.0, 7.0], name="y")
    x_before = x.copy(deep=True)
    y_before = y.copy(deep=True)

    with pytest.warns(UserWarning, match="problem with x="):
        actual = estimate_ols_alpha_beta(x=x, y=y)

    assert actual == (0.0, 0.0, 0.0, 0.0)
    pd.testing.assert_series_equal(x, x_before)
    pd.testing.assert_series_equal(y, y_before)

"""Undefined OLS alpha/beta statistics are NaN, never an exception or a silent zero.

``estimate_ols_alpha_beta`` documents a warn-and-return fallback for a fit it cannot perform. A
regressor that does not vary (a constant non-zero benchmark return, a single observation, an
all-zero benchmark) leaves the intercept and the slope unidentified. statsmodels then either
drops the intercept (``add_constant`` skips a constant column) or returns a minimum-norm
solution with a zero slope. Neither is an estimate, so the function returns NaN with a warning.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

import qis
from qis.utils.regression import (
    estimate_ols_alpha_beta,
    fit_multivariate_ols,
    fit_ols,
    reg_model_params_to_str,
)


@pytest.mark.parametrize(
    ("x", "y"),
    [
        (np.full(5, 0.01), np.array([0.02, -0.01, 0.03, 0.0, 0.01])),
        (np.array([0.5]), np.array([0.3])),
        (np.zeros(5), np.array([0.02, -0.01, 0.03, 0.0, 0.01])),
    ],
    ids=["constant-nonzero-regressor", "single-observation", "zero-regressor"],
)
def test_unidentified_regression_warns_and_returns_nan(x: np.ndarray, y: np.ndarray) -> None:
    """Return four NaN values with a warning instead of raising or reporting a zero slope."""
    with pytest.warns(UserWarning, match="not identified"):
        actual = estimate_ols_alpha_beta(x=x, y=y)

    assert len(actual) == 4
    assert np.isnan(actual).all()


def test_failed_fit_returns_nan_not_a_significant_zero_pvalue() -> None:
    """A failed fit must not report an alpha p-value of zero, which reads as significant."""
    x = pd.Series(["0", "1", "2", "3"], name="x")
    y = pd.Series([1.0, 3.0, 5.0, 7.0], name="y")

    with pytest.warns(UserWarning, match="problem with x="):
        actual = estimate_ols_alpha_beta(x=x, y=y)

    assert np.isnan(actual).all()


def test_no_intercept_fit_reports_zero_alpha_and_undefined_pvalue() -> None:
    """Alpha is zero by construction without an intercept, but its p-value is undefined."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([2.0, 4.1, 5.9, 8.0])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        alpha, beta, r_squared, alpha_pvalue = estimate_ols_alpha_beta(
            x=x, y=y, fit_intercept=False)

    assert alpha == 0.0
    np.testing.assert_allclose(beta, x @ y / (x @ x), rtol=0.0, atol=1.0e-12)
    assert 0.0 < r_squared <= 1.0
    assert np.isnan(alpha_pvalue)


def test_benchmark_table_survives_constant_benchmark_returns() -> None:
    """The benchmark table reports NaN regression statistics for a riskless benchmark."""
    index = pd.date_range("2020-01-31", periods=6, freq="ME")
    prices = pd.DataFrame(
        {
            "Benchmark": 2.0 ** np.arange(6),  # every simple return is exactly 1.0
            "Asset": 100.0 * np.exp(np.cumsum([0.0, 0.02, -0.01, 0.03, 0.0, 0.01])),
        },
        index=index,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.warns(UserWarning, match="not identified"):
            table = qis.compute_ra_perf_table_with_benchmark(
                prices=prices,
                benchmark="Benchmark",
                perf_params=qis.PerfParams(freq="ME", freq_reg="ME"),
            )

    assert np.isnan(table.loc["Asset", "Alpha"])
    assert np.isnan(table.loc["Asset", "Beta"])


def test_fit_multivariate_ols_returns_prediction_first() -> None:
    """The documented return order is prediction, parameters, label."""
    index = pd.date_range("2024-01-31", periods=5, freq="ME")
    x = pd.DataFrame({"f": [0.0, 1.0, 2.0, 3.0, 4.0]}, index=index)
    y = pd.Series([1.0, 3.0, 5.0, 7.0, 9.0], index=index, name="y")

    prediction, params, label = fit_multivariate_ols(x=x, y=y, verbose=False)

    pd.testing.assert_index_equal(prediction.index, index)
    assert list(params.index) == ["intercept", "f"]
    np.testing.assert_allclose(params.to_numpy(), [1.0, 2.0], atol=1.0e-12)
    assert label.startswith("y=")


def test_legend_annualises_alpha_linearly_like_the_tables() -> None:
    """``alpha_an_factor`` prints AN times the periodic alpha, as ``PerfStat.ALPHA_AN`` does."""
    x = np.array([-0.04, -0.01, 0.0, 0.02, 0.05])
    y = 0.013 + 0.9 * x  # monthly alpha 1.3%: 12 x 1.3% = 15.6%, expm1(0.156) = 16.9%
    model = fit_ols(x=x, y=y)
    np.testing.assert_allclose(model.params, [0.013, 0.9], atol=1.0e-12)

    label = reg_model_params_to_str(reg_model=model, order=1, alpha_an_factor=12)

    assert label == "y=+0.90X+16%, R²=100%"


def test_legend_default_prints_periodic_alpha_with_alpha_format() -> None:
    """Without ``alpha_an_factor`` the per-period intercept is printed with ``alpha_format``."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.013, 1.013, 2.013, 3.013])
    model = sm.OLS(y, sm.add_constant(x)).fit()

    assert reg_model_params_to_str(reg_model=model, order=1) == "y=+1.00X+0.01, R²=100%"
    assert reg_model_params_to_str(
        reg_model=model, order=1, alpha_format="{0:+0.1%}") == "y=+1.00X+1.3%, R²=100%"

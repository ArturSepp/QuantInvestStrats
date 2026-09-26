"""compute_masked_covar_corr computes the covariance of each pair on that pair's overlap."""

# packages
import numpy as np
import pandas as pd
import pytest

# qis
from qis.models.linear.corr_cov_matrix import compute_masked_covar_corr

NAN = np.nan


def test_covariance_centres_each_pair_on_its_overlap() -> None:
    """X=(0,2,4,6,.), Y=(.,1,3,2,4): overlap means 4 and 2 give covariance 1, as in pandas."""
    pair = pd.DataFrame({'X': [0, 2, 4, 6, NAN], 'Y': [NAN, 1, 3, 2, 4]}, dtype=float)
    covar = compute_masked_covar_corr(data=pair, is_covar=True)
    pd.testing.assert_frame_equal(covar, pair.cov(), rtol=1e-12, atol=1e-14)
    assert covar.loc['X', 'Y'] == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(np.diag(covar), [20.0 / 3.0, 5.0 / 3.0], atol=1e-12)


def test_covariance_bias_uses_overlap_count() -> None:
    """bias=True divides the overlap sum of products by the overlap count."""
    pair = pd.DataFrame({'X': [0, 2, 4, 6, NAN], 'Y': [NAN, 1, 3, 2, 4]}, dtype=float)
    covar = compute_masked_covar_corr(data=pair, is_covar=True, bias=True)
    overlap = pair.dropna()
    expected_xy = np.cov(overlap['X'], overlap['Y'], bias=True)[0, 1]
    assert covar.loc['X', 'Y'] == pytest.approx(expected_xy, abs=1e-12)
    assert covar.loc['X', 'X'] == pytest.approx(np.var([0, 2, 4, 6]), abs=1e-12)
    assert covar.loc['Y', 'Y'] == pytest.approx(np.var([1, 3, 2, 4]), abs=1e-12)


def test_pair_without_overlap_is_nan() -> None:
    """A pair with no common date has an undefined covariance, as its correlation already had."""
    data = pd.DataFrame({'X': [0, 2, 4, NAN, NAN], 'Y': [NAN, NAN, NAN, 1, 3]}, dtype=float)
    covar = compute_masked_covar_corr(data=data, is_covar=True)
    corr = compute_masked_covar_corr(data=data, is_covar=False)
    assert np.isnan(covar.loc['X', 'Y']) and np.isnan(covar.loc['Y', 'X'])
    assert np.isnan(corr.loc['X', 'Y'])
    assert covar.loc['X', 'X'] == pytest.approx(4.0, abs=1e-12)


def test_ndarray_input_matches_dataframe() -> None:
    """The ndarray path returns the same numbers as an ndarray."""
    rng = np.random.default_rng(11)
    values = rng.standard_normal((40, 3))
    values[:10, 0] = NAN
    values[25:, 2] = NAN
    covar_np = compute_masked_covar_corr(data=values, is_covar=True)
    assert isinstance(covar_np, np.ndarray)
    np.testing.assert_allclose(covar_np, pd.DataFrame(values).cov().to_numpy(), atol=1e-14)


def test_complete_panel_unchanged() -> None:
    """Without NaN the covariance is the common-sample np.cov."""
    values = np.random.default_rng(3).standard_normal((30, 4))
    for bias in (False, True):
        np.testing.assert_allclose(compute_masked_covar_corr(data=values, bias=bias),
                                   np.cov(values, rowvar=False, bias=bias), atol=1e-15)

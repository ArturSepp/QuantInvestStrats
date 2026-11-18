"""Finite-history trend slopes against independent weighted regression and limiting kernels."""
import numpy as np
import pandas as pd
import pytest

import qis


def direct_fit(y, span=12, min_periods=2):
    """Reference: whiten an intercept and signed row age, then solve by least squares."""
    ans = np.full(len(y), np.nan)
    for t in range(len(y)):
        ok = np.isfinite(y[:t + 1])
        if not ok[-1] or ok.sum() < min_periods:
            continue
        age = t - np.flatnonzero(ok)
        w = ((span - 1) / (span + 1)) ** (age / 2)
        design = np.column_stack((np.ones_like(age), -age))
        ans[t] = np.linalg.lstsq(w[:, None] * design, w * y[:t + 1][ok], rcond=None)[0][1]
    return ans


def test_sparse_prefixes_and_pandas_labels():
    """Gaps age old data, warm-up counts finite levels and future rows change nothing."""
    t = np.arange(83.)
    y = pd.DataFrame({'A': np.sin(t / 7), 'B': np.cos(t), 'empty': np.nan},
                     index=pd.date_range('2010-01-31', periods=len(t), freq='ME'))
    y.iloc[[0, 2, 18, 19, 20, 57], 0] = np.nan
    y.iloc[4:20, 1] = np.inf
    result = qis.compute_ewm_trend_slope(y, span=12, min_periods=13)
    ref = y.apply(lambda col: direct_fit(col.to_numpy(), min_periods=13))
    pd.testing.assert_frame_equal(result, ref, rtol=1e-12, atol=1e-13)
    pd.testing.assert_frame_equal(qis.compute_ewm_trend_slope(y.iloc[:35], min_periods=13),
                                  result.iloc[:35])
    pd.testing.assert_series_equal(qis.compute_ewm_trend_slope(y.A, min_periods=13), result.A)


def test_units_offsets_gaps_and_unchanged_shape():
    """Large offsets and long gaps must not erase a linear slope or fabricate zero scores."""
    t = np.arange(120.)
    x = np.column_stack((1e12 + 0.125 * t, 5 - t, np.ones(len(t))))
    x[7:53] = np.nan
    got = qis.compute_ewm_trend_slope(x)
    assert np.isnan(got[0]).all() and np.isnan(got[7:53]).all()
    rows = np.flatnonzero(np.isfinite(x[:, 0]))[1:]
    np.testing.assert_allclose(got[rows], np.tile([0.125, -1, 0], (len(rows), 1)), atol=1e-13)
    np.testing.assert_equal(qis.compute_ewm_trend_slope(x[:, 0]), got[:, 0])
    assert qis.compute_ewm_trend_slope(np.empty((0, 3))).shape == (0, 3)


def test_return_kernel_is_a_limit_not_a_startup_rule():
    """Double smoothing increments is correct at equilibrium but not at the second level."""
    t = np.arange(300.)
    r = pd.Series(np.cos(t / 5) / 10 + 0.02)
    levels = r.cumsum()
    direct = qis.compute_ewm_trend_slope(levels)
    double = levels.diff().fillna(0).ewm(span=12, adjust=False).mean().ewm(
        span=12, adjust=False).mean()
    assert direct.iloc[1] == pytest.approx(r.iloc[1])
    assert abs(double.iloc[1] - direct.iloc[1]) > 0.01
    np.testing.assert_allclose(direct.iloc[-60:], double.iloc[-60:], atol=1e-13)


@pytest.mark.parametrize('kw', [{'span': 1}, {'span': np.inf}, {'span': True},
                                {'min_periods': 1}, {'min_periods': 2.5}])
def test_invalid_parameters(kw):
    """A slope needs two distinct times, positive prior mass, and an integer finite count."""
    with pytest.raises(ValueError):
        qis.compute_ewm_trend_slope(np.ones(10), **kw)


def test_invalid_inputs():
    """Do not flatten a tensor or guess the chronology of a labelled series."""
    with pytest.raises(ValueError):
        qis.compute_ewm_trend_slope(np.ones((2, 2, 2)))
    with pytest.raises(ValueError):
        qis.compute_ewm_trend_slope(pd.Series([1., 2.], index=[2, 1]))

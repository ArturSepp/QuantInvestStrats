"""Factor units must survive mixed transforms and legacy log defaults."""
import numpy as np
import pandas as pd
import pytest

from qis.perfstats.config import ReturnTypes
from qis.perfstats import returns as rt
from qis.utils.regression import estimate_ewma_alpha_beta_hac


def test_mixed_factor_units():
    prices = pd.DataFrame({'VIX': [20., 25., 22.5], 'Equity': [100., 110., 99.]},
                          index=pd.date_range('2020-01-31', periods=3, freq='ME'))
    actual = rt.to_factor_returns(prices, {'VIX': ReturnTypes.DIFFERENCE,
                                          'Equity': ReturnTypes.LOG})
    np.testing.assert_allclose(actual.VIX.iloc[1:], [5., -2.5])
    np.testing.assert_allclose(actual.Equity.iloc[1:], np.log([1.1, .9]))
    pd.testing.assert_frame_equal(rt.to_factor_returns(prices),
                                  rt.to_returns(prices, is_log_returns=True))
    with pytest.raises(ValueError, match='exactly'):
        rt.to_factor_returns(prices, {'VIX': ReturnTypes.DIFFERENCE})
    with pytest.raises(ValueError, match='override'):
        rt.to_factor_returns(prices, {'VIX': ReturnTypes.DIFFERENCE,
                                     'Equity': ReturnTypes.LOG}, is_log_returns=True)


def test_difference_preserves_signed_and_missing_endpoints():
    prices = pd.DataFrame({'rate': [-1., 1., np.nan, 2.]},
                          index=pd.date_range('2020-01-31', periods=4, freq='ME'))
    actual = rt.to_factor_returns(prices, {'rate': ReturnTypes.DIFFERENCE}, ffill_nans=False)
    assert actual.rate.iloc[1] == 2.
    assert actual.rate.iloc[[0, 2, 3]].isna().all()


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("dtype", ["UInt8", "UInt64"])
def test_to_factor_returns_preserves_point_change_regression_sign(dtype):
    """Later estimator conversion cannot repair an unsigned factor decrease."""
    levels = [20, 18, 22, 21, 25, 19, 23, 17, 24, 20, 26, 21, 27]
    dates = pd.date_range("2025-01-31", periods=len(levels), freq="ME")
    prices = pd.DataFrame({"VIX": pd.array(levels, dtype=dtype)}, index=dates)
    # An exact linear response has slope -0.002 under any positive WLS weighting.
    changes = [int(right) - int(left) for left, right in zip(levels[:-1], levels[1:])]
    response = pd.DataFrame({"fund": [-0.002 * change for change in changes]}, index=dates[1:])
    expected = pd.Series(changes, dtype="Int64", index=dates[1:], name="VIX")
    actual = rt.to_factor_returns(prices, {"VIX": ReturnTypes.DIFFERENCE}).VIX.iloc[1:]
    pd.testing.assert_series_equal(actual, expected, check_exact=True)
    fit = estimate_ewma_alpha_beta_hac(actual, response, span=36)
    np.testing.assert_allclose(fit.beta, [-0.002], rtol=0.0, atol=1e-15)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("return_type", [ReturnTypes.DIFFERENCE, ReturnTypes.LOG])
def test_to_factor_returns_preserves_sparse_integer_storage(return_type):
    """The factor wrapper must retain sparse storage and its declared return units."""
    dates = pd.date_range("2025-01-31", periods=3, freq="ME")
    fill = 1 if return_type == ReturnTypes.LOG else 0
    prices = pd.DataFrame({"VIX": pd.arrays.SparseArray([20, 18, 22],
                          dtype=pd.SparseDtype("uint8", fill))}, index=dates)
    values = ([np.nan, -2.0, 4.0] if return_type == ReturnTypes.DIFFERENCE
              else [np.nan, np.log(0.9), np.log(11 / 9)])
    expected = pd.Series(values, dtype=pd.SparseDtype("float64", 0), index=dates, name="VIX")
    actual = rt.to_factor_returns(prices, {"VIX": return_type}, ffill_nans=False)
    pd.testing.assert_series_equal(actual.VIX, expected, rtol=0.0, atol=1e-15)

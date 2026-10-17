"""Factor units must survive mixed transforms and legacy log defaults."""
import numpy as np
import pandas as pd
import pytest

from qis.perfstats.config import ReturnTypes
from qis.perfstats import returns as rt


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

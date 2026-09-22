"""Exact-zero policy for FX-adjusted panel returns."""

import numpy as np
import pandas as pd
import pytest

from qis.market_data import FxRatesData


@pytest.mark.parametrize('per_asset_frequency', [False, True])
def test_zero_return_policy_preserves_opt_in_zeros(per_asset_frequency: bool) -> None:
    """The default skips exact zeros while opt-out retains them without moving other returns."""
    dates = pd.date_range('2024-01-31', periods=4, freq='ME')
    data = FxRatesData(
        fx_spots=pd.DataFrame({'USD': 1.0}, index=dates),
        domestic_rates=pd.DataFrame({'USD': 0.0}, index=dates),
    )
    prices = pd.DataFrame({'FLAT': [100.0, 101.0, 101.0, 102.0],
                           'MOVING': [100.0, 102.0, 103.0, 104.0]}, index=dates)
    assets = prices.columns
    kwargs = dict(prices=prices, hedge_ratios=pd.Series(0.0, index=assets),
                  local_ccys=pd.Series('USD', index=assets), reference_ccy='USD',
                  freq=pd.Series('ME', index=assets) if per_asset_frequency else 'ME',
                  is_log_returns=False)

    default = data.compute_fx_adjusted_returns(**kwargs)['ME']
    explicit_default = data.compute_fx_adjusted_returns(
        **kwargs, zero_return_to_nan=True)['ME']
    retained = data.compute_fx_adjusted_returns(
        **kwargs, zero_return_to_nan=False)['ME']

    pd.testing.assert_frame_equal(default, explicit_default)
    assert np.isnan(default.loc[dates[2], 'FLAT'])
    assert retained.loc[dates[2], 'FLAT'] == 0.0
    pd.testing.assert_series_equal(default['MOVING'], retained['MOVING'])
    pd.testing.assert_frame_equal(default.replace({np.nan: 0.0}).iloc[1:],
                                  retained.fillna(0.0).iloc[1:])

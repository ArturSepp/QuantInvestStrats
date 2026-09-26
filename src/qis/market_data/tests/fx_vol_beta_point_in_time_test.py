"""``compute_fx_vol_beta`` is point in time.

Its EWMA beta used ``compute_ewm_cross_xy`` with the full-sample mean-square seed of the FX
variance, so the early hedge betas depended on later FX returns.
"""

# packages
import numpy as np
import pandas as pd

# qis
import qis


def test_fx_beta_does_not_depend_on_later_data() -> None:
    rng = np.random.default_rng(11)
    index = pd.date_range('2012-01-31', periods=96, freq='ME')
    fx_returns = rng.normal(0.0, 0.025, size=96)
    local_returns = -0.3 * fx_returns + rng.normal(0.004, 0.03, size=96)
    fx = pd.Series(1.1 * np.exp(np.cumsum(fx_returns)), index=index, name='EURUSD')
    asset = pd.Series(100.0 * np.exp(np.cumsum(local_returns)), index=index, name='asset')
    _, beta_full = qis.compute_fx_vol_beta(asset_price_local_ccy=asset,
                                           local_to_reference_fx_rate=fx)
    _, beta_prefix = qis.compute_fx_vol_beta(asset_price_local_ccy=asset.iloc[:30],
                                             local_to_reference_fx_rate=fx.iloc[:30])
    pd.testing.assert_series_equal(beta_full.iloc[:len(beta_prefix)], beta_prefix)

"""estimate_dimson_beta reports the standard error and t-statistic of the Dimson beta."""

# packages
import numpy as np
import pandas as pd
import pytest

# qis
from qis.models.unsmoothing.dimson_beta import estimate_dimson_beta

LEGACY_COLUMNS = ['beta_0', 'beta_dimson', 'smoothing_ratio', 't_beta_0', 'sum_lag_beta',
                  't_sum_lag', 'ar1', 'r2', 'n_obs']


def _stale_asset(seed: int = 17, num_obs: int = 180):
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2005-01-31', periods=num_obs, freq='ME')
    market = pd.Series(0.04 * rng.standard_normal(num_obs), index=dates, name='market')
    noise = pd.Series(0.02 * rng.standard_normal(num_obs), index=dates)
    asset = (0.5 * market + 0.3 * market.shift(1) + noise).rename('asset')
    return asset, market


def test_dimson_beta_standard_error_is_classical_quadratic_form() -> None:
    """se_beta_dimson is sqrt(iota' Cov(b) iota) with iota selecting every market slope."""
    asset, market = _stale_asset()
    fit = estimate_dimson_beta(asset_returns=asset, market_returns=market, num_lags=2)
    assert fit.columns.tolist()[:len(LEGACY_COLUMNS)] == LEGACY_COLUMNS
    lags = pd.concat({f'l{k}': market.shift(k) for k in range(3)}, axis=1)
    data = pd.concat([asset, lags], axis=1).dropna()
    x = np.column_stack([np.ones(len(data)), data.iloc[:, 1:].to_numpy()])
    y = data['asset'].to_numpy()
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    cov = resid @ resid / (len(y) - x.shape[1]) * np.linalg.inv(x.T @ x)
    iota = np.array([0.0, 1.0, 1.0, 1.0])
    se = np.sqrt(iota @ cov @ iota)
    row = fit.loc['asset']
    assert row['se_beta_dimson'] == pytest.approx(se, rel=1e-10)
    assert row['t_beta_dimson'] == pytest.approx(coef[1:].sum() / se, rel=1e-10)


def test_without_lags_dimson_statistics_equal_contemporaneous_ones() -> None:
    """num_lags=0: beta_dimson is beta_0, sum_lag_beta is 0 and t_sum_lag is NaN."""
    asset, market = _stale_asset()
    row = estimate_dimson_beta(asset_returns=asset, market_returns=market, num_lags=0).loc['asset']
    assert row['beta_dimson'] == pytest.approx(row['beta_0'], abs=1e-14)
    assert row['t_beta_dimson'] == pytest.approx(row['t_beta_0'], rel=1e-12)
    assert row['sum_lag_beta'] == 0.0 and np.isnan(row['t_sum_lag'])


def test_short_sample_row_has_nan_inference() -> None:
    """An asset below min_obs gets NaN for the new fields as for the others."""
    asset, market = _stale_asset(num_obs=20)
    row = estimate_dimson_beta(asset_returns=asset, market_returns=market).loc['asset']
    assert np.isnan(row['se_beta_dimson']) and np.isnan(row['t_beta_dimson'])

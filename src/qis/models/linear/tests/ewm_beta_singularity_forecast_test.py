"""EWM beta estimators: scale-free singularity handling, point-in-time seeds and the forecast."""
# packages
import inspect

import numpy as np
import pandas as pd
import pytest

# qis
from qis.models.linear.ewm import (CrossXyType, InitType, NanBackfill,
                                   compute_ewm_alpha_r2_given_prediction,
                                   compute_ewm_beta_alpha_forecast, compute_ewm_cross_xy,
                                   compute_ewm_xy_beta_tensor, compute_one_factor_ewm_betas)


def _factor_asset(scale: float, n: int = 400) -> tuple:
    rng = np.random.default_rng(21)
    factor = rng.standard_normal(n) * scale
    asset = 2.0 * factor + rng.standard_normal(n) * scale * 0.02
    return factor, asset


def test_beta_tensor_is_scale_free() -> None:
    """A factor with 0.5bp volatility gets beta 2, as the same data scaled by 1000 does."""
    factor, asset = _factor_asset(scale=5e-5)
    betas = compute_ewm_xy_beta_tensor(factor, asset, span=36)
    scaled = compute_ewm_xy_beta_tensor(factor * 1000.0, asset * 1000.0, span=36)
    np.testing.assert_allclose(betas[21:], scaled[21:], rtol=1e-9)
    np.testing.assert_allclose(betas[-1, 0, 0], 2.0, atol=0.01)


def test_zero_variance_factor_gives_nan_only_for_that_factor() -> None:
    """A factor with no variation has a NaN beta; the others come from the reduced system."""
    factor, asset = _factor_asset(scale=0.01)
    factors = np.column_stack([factor, np.zeros_like(factor)])
    betas = compute_ewm_xy_beta_tensor(factors, asset, span=36, warmup_period=5)
    single = compute_ewm_xy_beta_tensor(factor, asset, span=36, warmup_period=5)
    assert np.isnan(betas[6:, 1, 0]).all()
    np.testing.assert_allclose(betas[6:, 0, 0], single[6:, 0, 0], rtol=1e-12)


def test_collinear_factors_give_nan_not_a_cross_moment() -> None:
    """Two identical factors make the system singular: the betas are NaN."""
    factor, asset = _factor_asset(scale=0.01)
    factors = np.column_stack([factor, factor])
    betas = compute_ewm_xy_beta_tensor(factors, asset, span=36, warmup_period=5)
    assert np.isnan(betas[6:]).all()


@pytest.mark.parametrize('cross_xy_type', [CrossXyType.BETA, CrossXyType.CORR])
def test_cross_xy_ratios_are_scale_free(cross_xy_type: CrossXyType) -> None:
    """compute_ewm_cross_xy does not mask a small-scale factor as a zero denominator."""
    factor, asset = _factor_asset(scale=5e-5, n=200)
    small = compute_ewm_cross_xy(factor, asset, span=36, cross_xy_type=cross_xy_type)
    large = compute_ewm_cross_xy(factor * 1000, asset * 1000, span=36,
                                 cross_xy_type=cross_xy_type)
    np.testing.assert_allclose(small, large, rtol=1e-9)


def _monthly() -> tuple:
    rng = np.random.default_rng(22)
    index = pd.date_range('2010-01-31', periods=120, freq='ME')
    x = pd.Series(rng.standard_normal(120) * 0.04, index=index, name='bench')
    y = pd.DataFrame({'a': 1.3 * x + rng.standard_normal(120) * 0.02 + 0.001,
                      'b': -0.4 * x + rng.standard_normal(120) * 0.01}, index=index)
    return x, y


def test_beta_alpha_forecast_default_is_point_in_time() -> None:
    """Truncating the sample does not change any earlier output under the default seed."""
    x, y = _monthly()
    full = compute_ewm_beta_alpha_forecast(x, y, span=36)
    prefix = compute_ewm_beta_alpha_forecast(x.iloc[:60], y.iloc[:60], span=36)
    for whole, part in zip(full, prefix):
        np.testing.assert_allclose(whole.iloc[:60].to_numpy(), part.to_numpy(), rtol=1e-12)
    assert inspect.signature(compute_ewm_beta_alpha_forecast).parameters[
        'init_type'].default == InitType.X0


def test_beta_alpha_forecast_prediction_uses_information_to_t_minus_one() -> None:
    """The prediction at t is beta_{t-1} x_t + alpha_{t-1} and does not see y_t."""
    x, y = _monthly()
    betas, alphas, prediction, *_ = compute_ewm_beta_alpha_forecast(x, y, span=36)
    expected = betas.shift(1).multiply(x, axis=0) + alphas.shift(1)
    np.testing.assert_allclose(prediction.to_numpy(), expected.to_numpy(), rtol=1e-12)
    assert prediction.iloc[0].isna().all()
    shocked = y.copy()
    shocked.iloc[70] += 1.0
    shocked_prediction = compute_ewm_beta_alpha_forecast(x, shocked, span=36)[2]
    np.testing.assert_allclose(shocked_prediction.iloc[:71], prediction.iloc[:71], rtol=1e-12)


def test_beta_alpha_forecast_moments_honour_nan_backfill() -> None:
    """DEFLATED_FFILL treats a missing factor return as zero in both beta moments."""
    x, y = _monthly()
    gappy = x.copy()
    gappy.iloc[[30, 31, 80]] = np.nan
    betas = compute_ewm_beta_alpha_forecast(gappy, y, span=36,
                                            nan_backfill=NanBackfill.DEFLATED_FFILL)[0]
    filled = compute_ewm_beta_alpha_forecast(gappy.fillna(0.0), y, span=36)[0]
    np.testing.assert_allclose(betas.to_numpy(), filled.to_numpy(), rtol=1e-12)


def test_beta_alpha_forecast_factor_variance_is_labelled_by_asset() -> None:
    """The factor-variance frame carries the asset column labels it is aligned with."""
    x, y = _monthly()
    x_var = compute_ewm_beta_alpha_forecast(x, y, span=36)[3]
    assert list(x_var.columns) == ['a', 'b']


def test_public_ewm_docstrings_document_their_arguments() -> None:
    """The ledger's undocumented public names now carry Google-style sections."""
    from qis.models.linear.ewm import InitType as init_type_enum
    from qis.models.linear.ewm import compute_ewm_sharpe
    assert 'Attributes:' in (init_type_enum.__doc__ or '')
    for fn in (compute_ewm_alpha_r2_given_prediction, compute_one_factor_ewm_betas,
               compute_ewm_sharpe):
        assert 'Args:' in (fn.__doc__ or '') and 'Returns:' in (fn.__doc__ or '')
    assert 'warmup_period' in compute_one_factor_ewm_betas.__doc__

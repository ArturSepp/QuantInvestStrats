"""Container and argument contracts of the EWM wrappers: every documented input type works."""
# packages
import numpy as np
import pandas as pd
import pytest

# qis
from qis.models.linear.ewm import (CrossXyType, MeanAdjType, compute_ewm_covar,
                                   compute_ewm_cross_xy, compute_ewm_long_short_filter,
                                   compute_ewm_vol, compute_one_factor_ewm_betas,
                                   compute_roll_mean, compute_rolling_mean_adj)


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    values = rng.standard_normal((300, 2)) * 0.01
    values[[3, 40], 0] = np.nan
    return pd.DataFrame(values, index=pd.bdate_range('2020-01-01', periods=300),
                        columns=['x', 'y'])


def test_insample_mean_accepts_pandas_and_ignores_missing_values() -> None:
    """INSAMPLE is the full-sample nanmean, broadcast to every row, in the input's container."""
    frame = _frame()
    expected = frame.mean()  # pandas skips NaN
    mean = compute_roll_mean(frame, mean_adj_type=MeanAdjType.INSAMPLE)
    assert isinstance(mean, pd.DataFrame) and mean.index.equals(frame.index)
    np.testing.assert_allclose(mean.to_numpy(), np.tile(expected.to_numpy(), (300, 1)))
    series_mean = compute_roll_mean(frame['x'], mean_adj_type=MeanAdjType.INSAMPLE)
    assert isinstance(series_mean, pd.Series) and series_mean.name == 'x'
    np.testing.assert_allclose(series_mean.to_numpy(), expected['x'])
    array_mean = compute_roll_mean(frame.to_numpy(), mean_adj_type=MeanAdjType.INSAMPLE)
    np.testing.assert_allclose(array_mean, np.tile(expected.to_numpy(), (300, 1)))
    adjusted = compute_rolling_mean_adj(frame, mean_adj_type=MeanAdjType.INSAMPLE)
    np.testing.assert_allclose(adjusted.to_numpy(), (frame - expected).to_numpy())


def test_vol_floor_accepts_series_and_one_dimensional_arrays() -> None:
    """The rolling-quantile floor gives the same path for a Series, a 1-d array and a frame."""
    frame = _frame()
    kwargs = dict(span=31, vol_floor_quantile=0.16, vol_floor_quantile_roll_period=100)
    from_frame = compute_ewm_vol(frame, **kwargs)['y'].to_numpy()
    from_series = compute_ewm_vol(frame['y'], **kwargs)
    from_array = compute_ewm_vol(frame['y'].to_numpy(), **kwargs)
    assert isinstance(from_series, pd.Series) and from_array.shape == (300,)
    np.testing.assert_allclose(from_series.to_numpy(), from_frame)
    np.testing.assert_allclose(from_array, from_frame)
    unfloored = compute_ewm_vol(frame['y'], span=31)
    assert (from_series >= unfloored - 1e-15).all() and (from_series > unfloored).any()


@pytest.mark.parametrize('cross_xy_type', list(CrossXyType))
def test_cross_xy_accepts_every_documented_container(cross_xy_type: CrossXyType) -> None:
    """Series x DataFrame, Series x Series and 1-d arrays equal the one-column frame result."""
    frame = _frame()
    kwargs = dict(span=20, cross_xy_type=cross_xy_type)
    reference = compute_ewm_cross_xy(frame[['x']], frame[['y']], **kwargs)['y'].to_numpy()

    series_frame = compute_ewm_cross_xy(frame['x'], frame[['y']], **kwargs)
    assert isinstance(series_frame, pd.DataFrame) and list(series_frame.columns) == ['y']
    np.testing.assert_allclose(series_frame['y'].to_numpy(), reference)

    series_series = compute_ewm_cross_xy(frame['x'], frame['y'], **kwargs)
    assert isinstance(series_series, pd.Series) and series_series.name == 'y'
    np.testing.assert_allclose(series_series.to_numpy(), reference)

    arrays = compute_ewm_cross_xy(frame['x'].to_numpy(), frame['y'].to_numpy(), **kwargs)
    assert arrays.shape == (300,)
    np.testing.assert_allclose(arrays, reference)


def test_cross_xy_series_factor_is_paired_with_every_asset_column() -> None:
    """A Series factor against a DataFrame is the same as tiling the factor across columns."""
    frame = _frame()
    assets = frame[['y']].assign(z=frame['y'] * 2.0 + 0.001)
    factor = frame['x']
    actual = compute_ewm_cross_xy(factor, assets, span=20, cross_xy_type=CrossXyType.BETA)
    tiled = pd.DataFrame({c: factor for c in assets.columns})
    expected = compute_ewm_cross_xy(tiled, assets, span=20, cross_xy_type=CrossXyType.BETA)
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy())


def test_cross_xy_rejects_frame_factor_with_series_asset() -> None:
    """The undocumented DataFrame x Series pairing fails with a clear TypeError."""
    frame = _frame()
    with pytest.raises(TypeError, match='x_data'):
        compute_ewm_cross_xy(frame[['x']], frame['y'])


def test_one_dimensional_covariance_honours_is_corr() -> None:
    """A single cross-section is normalised to a correlation matrix when asked."""
    corr = compute_ewm_covar(np.array([0.01, -0.02]), ewm_lambda=0.94, is_corr=True)
    np.testing.assert_allclose(corr, [[1.0, -1.0], [-1.0, 1.0]])


def test_long_short_filter_accepts_one_dimensional_arrays() -> None:
    """A 1-d ndarray is filtered exactly like the corresponding Series."""
    frame = _frame()
    series = compute_ewm_long_short_filter(frame['y'], long_span=63, short_span=5)
    array = compute_ewm_long_short_filter(frame['y'].to_numpy(), long_span=63, short_span=5)
    np.testing.assert_allclose(array, series.to_numpy())


def test_one_factor_betas_error_message_names_the_indices() -> None:
    """The index-mismatch message is formatted, not the literal '{x.index}'."""
    x = pd.Series([1.0, 2.0], index=[0, 1])
    y = pd.DataFrame({'a': [1.0, 2.0]}, index=[5, 6])
    with pytest.raises(ValueError) as error:
        compute_one_factor_ewm_betas(x, y)
    assert '{x.index}' not in str(error.value) and '5' in str(error.value)


def test_one_factor_betas_exposes_warmup_period() -> None:
    """warmup_period is passed to the beta tensor; the default keeps rows t <= 20 missing."""
    frame = _frame()
    default = compute_one_factor_ewm_betas(frame['x'], frame[['y']], span=20)
    short = compute_one_factor_ewm_betas(frame['x'], frame[['y']], span=20, warmup_period=5)
    assert default.iloc[:21].isna().all().all() and default.iloc[21:].notna().all().all()
    assert short.iloc[:6].isna().all().all() and short.iloc[6:].notna().all().all()
    np.testing.assert_allclose(short.iloc[21:], default.iloc[21:])

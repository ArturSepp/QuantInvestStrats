"""Point-in-time default seed and optional warm-up mask of ``compute_ewm_cross_xy``.

The variance recursions of a beta or correlation were seeded by default with the full-sample mean
square (``InitType.MEAN``), so early ratios depended on later data. The default is now the first
finite square (``InitType.X0``). ``warmup_period`` masks each column until it has more joint
observations than the warm-up count.
"""

# packages
import numpy as np
import pandas as pd
import pytest

# qis
import qis
from qis.models.linear.ewm import InitType

_SPAN = 12
_LAMBDA = 1.0 - 2.0 / (_SPAN + 1.0)


def _panel() -> tuple:
    rng = np.random.default_rng(7)
    index = pd.date_range('2015-01-31', periods=120, freq='ME')
    x = pd.DataFrame(rng.normal(0.0, 0.02, size=(120, 2)), index=index, columns=['f1', 'f2'])
    y = pd.DataFrame(0.8 * x.to_numpy() + rng.normal(0.0, 0.01, size=(120, 2)), index=index,
                     columns=['a1', 'a2'])
    return x, y


@pytest.mark.parametrize('cross_xy_type', [qis.CrossXyType.BETA, qis.CrossXyType.CORR])
def test_default_ratio_is_point_in_time(cross_xy_type) -> None:
    """The first 40 rows do not change when the later 80 are removed."""
    x, y = _panel()
    full = qis.compute_ewm_cross_xy(x_data=x, y_data=y, span=_SPAN, cross_xy_type=cross_xy_type)
    prefix = qis.compute_ewm_cross_xy(x_data=x.iloc[:40], y_data=y.iloc[:40], span=_SPAN,
                                      cross_xy_type=cross_xy_type)
    pd.testing.assert_frame_equal(full.iloc[:40], prefix)


def test_default_variance_seed_is_the_first_square() -> None:
    x, y = _panel()
    default = qis.compute_ewm_cross_xy(x_data=x, y_data=y, span=_SPAN,
                                       cross_xy_type=qis.CrossXyType.BETA)
    explicit = qis.compute_ewm_cross_xy(x_data=x, y_data=y, span=_SPAN,
                                        cross_xy_type=qis.CrossXyType.BETA,
                                        var_init_type=InitType.X0)
    pd.testing.assert_frame_equal(default, explicit)
    # zero-seeded cross moment over the X0-seeded factor variance on the first row
    x0, y0 = x.iloc[0].to_numpy(), y.iloc[0].to_numpy()
    np.testing.assert_allclose(default.iloc[0].to_numpy(), (1.0 - _LAMBDA) * y0 / x0, rtol=1e-12)


def test_warmup_period_masks_each_column_from_its_own_first_observation() -> None:
    x, y = _panel()
    y.iloc[:10, 1] = np.nan  # the second asset starts on row 10
    unmasked = qis.compute_ewm_cross_xy(x_data=x, y_data=y, span=_SPAN,
                                        cross_xy_type=qis.CrossXyType.BETA)
    masked = qis.compute_ewm_cross_xy(x_data=x, y_data=y, span=_SPAN,
                                      cross_xy_type=qis.CrossXyType.BETA, warmup_period=5)
    assert masked['a1'].iloc[:5].isna().all()
    pd.testing.assert_series_equal(masked['a1'].iloc[5:], unmasked['a1'].iloc[5:])
    assert masked['a2'].iloc[:15].isna().all()
    pd.testing.assert_series_equal(masked['a2'].iloc[15:], unmasked['a2'].iloc[15:])


def test_warmup_period_for_series_and_arrays() -> None:
    x, y = _panel()
    series = qis.compute_ewm_cross_xy(x_data=x['f1'], y_data=y['a1'], span=_SPAN,
                                      cross_xy_type=qis.CrossXyType.CORR, warmup_period=3)
    assert series.iloc[:3].isna().all() and series.iloc[3:].notna().all()
    array = qis.compute_ewm_cross_xy(x_data=x.to_numpy(), y_data=y.to_numpy(), span=_SPAN,
                                     warmup_period=3)
    assert np.isnan(array[:3]).all() and np.isfinite(array[3:]).all()

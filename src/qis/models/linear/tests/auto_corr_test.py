"""compute_autocorr_df honours its lag count."""

# packages
import numpy as np
import pandas as pd
import pytest

# qis
from qis.models.linear.auto_corr import compute_autocorr_df


@pytest.mark.parametrize('num_lags', [5, 20, 30])
def test_autocorr_df_returns_requested_lags(num_lags: int) -> None:
    """Rows are lags 0..num_lags-1 and match the lagged Pearson correlation of each column."""
    rng = np.random.default_rng(7)
    data = pd.DataFrame(rng.standard_normal((200, 2)), columns=['x', 'y'])
    acf = compute_autocorr_df(df=data, num_lags=num_lags)
    assert list(acf.index) == list(range(num_lags))
    assert list(acf.columns) == ['x', 'y']
    for lag in range(1, num_lags):
        for column in data.columns:
            values = data[column].to_numpy()
            expected = np.corrcoef(values[lag:], values[:-lag])[0, 1]
            assert acf.loc[lag, column] == pytest.approx(expected, abs=1e-12)
    np.testing.assert_array_equal(acf.loc[0].to_numpy(), 1.0)


def test_autocorr_df_series_keeps_name() -> None:
    """A Series input returns a Series of the requested length with its name."""
    series = pd.Series(np.sin(np.arange(100) / 3.0), name='wave')
    acf = compute_autocorr_df(df=series, num_lags=7)
    assert isinstance(acf, pd.Series) and acf.name == 'wave' and len(acf) == 7

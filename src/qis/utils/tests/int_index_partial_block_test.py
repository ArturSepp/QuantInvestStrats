"""Integer-block aggregation drops an incomplete first block.

``df_resample_at_int_index`` counts blocks of ``sample_size`` rows back from the last row, so
the first block holds only ``T mod sample_size`` rows. A sum over that short block is not an
observation of the ``sample_size``-row aggregate and is dropped; with ``func=None`` the block's
last row is a genuine level observation on the block grid and is kept.
``qis.compute_autocorrelation_at_int_periods`` inherits the rule for summed returns.
"""

import numpy as np
import pandas as pd

import qis
from qis.utils.df_freq import df_resample_at_int_index


_INDEX = pd.bdate_range('2024-01-01', periods=10)


def test_summed_blocks_exclude_the_incomplete_first_block() -> None:
    """Ten rows in blocks of three give three complete sums."""
    df = pd.DataFrame({'x': np.arange(10.0)}, index=_INDEX)

    sums = df_resample_at_int_index(df=df, func=np.nansum, sample_size=3)

    np.testing.assert_allclose(sums['x'].to_numpy(), [1.0 + 2 + 3, 4.0 + 5 + 6, 7.0 + 8 + 9])
    assert list(sums.index) == [_INDEX[3], _INDEX[6], _INDEX[9]]


def test_last_value_blocks_keep_the_first_level() -> None:
    """Levels on the block grid keep the last row of the short first block."""
    df = pd.DataFrame({'x': np.arange(10.0)}, index=_INDEX)

    levels = df_resample_at_int_index(df=df, func=None, sample_size=3)

    np.testing.assert_allclose(levels['x'].to_numpy(), [0.0, 3.0, 6.0, 9.0])


def test_block_autocorrelation_uses_complete_blocks_only() -> None:
    """The block autocorrelation equals the lag-one Pearson correlation of complete sums."""
    rng = np.random.default_rng(11)
    dates = pd.bdate_range('2020-01-01', periods=1003)
    returns = pd.DataFrame({'a': rng.standard_normal(1003)}, index=dates)

    actual = qis.compute_autocorrelation_at_int_periods(data=returns, span=10)

    complete = returns['a'].to_numpy()[3:].reshape(-1, 10).sum(axis=1)
    expected = np.corrcoef(complete[1:], complete[:-1])[0, 1]
    assert abs(float(actual.iloc[0]) - expected) < 1e-12


def test_block_sums_are_np_nansum_of_each_block() -> None:
    """Each block sum is np.nansum of its rows, bit for bit, on every pandas version."""
    rng = np.random.default_rng(3)
    df = pd.DataFrame(0.0004 + 0.01 * rng.standard_normal((2600, 4)),
                      index=pd.bdate_range('2010-01-01', periods=2600), columns=list('abcd'))
    df.iloc[rng.random(df.shape) < 0.05] = np.nan
    df.iloc[:40, 2] = np.nan  # a late start leaves all-NaN blocks, whose sum is zero

    sums = df_resample_at_int_index(df=df, func=np.nansum, sample_size=21)

    rows = df.to_numpy()[len(df) % 21:].reshape(-1, 21, 4)
    expected = np.array([[np.nansum(block[:, j]) for j in range(4)] for block in rows])
    np.testing.assert_array_equal(sums.to_numpy(), expected)
    assert (sums['c'].iloc[:1] == 0.0).all()

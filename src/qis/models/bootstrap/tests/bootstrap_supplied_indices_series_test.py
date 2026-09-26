"""Series inputs, supplied index arrays and random state of the bootstrap entry points.

Each test guards one defect and fails on it alone:

  - ``bootstrap_data`` on a Series with the default ``DF_TO_LIST_ARRAYS`` output raised a numba
    ``TypingError``, because the kernel indexes a two-dimensional array;
  - ``bootstrap_price_data`` with ``SERIES_TO_DF`` repeated the anchor ``num_samples`` times, so
    supplied indices with another number of columns raised a broadcast ``ValueError``;
  - supplied indices outside the rows of the data were read by an ``@njit`` kernel without
    bounds checking and returned adjacent memory as data.

The seed contract is pinned too: ``seed`` drives numba's generator, and numpy's global
generator is neither read nor changed.
"""

# packages
import numpy as np
import pandas as pd
import pytest
# qis / project
from qis.models.bootstrap.bootstrap_numba import (BootstrapOutput,
                                                  BootstrapType,
                                                  bootstrap_data,
                                                  bootstrap_price_data,
                                                  generate_bootstrapped_indices)

DATES = pd.date_range('2024-01-01', periods=8, freq='D')
PRICES = pd.Series([100.0, 101.0, 99.0, 102.0, 104.0, 103.0, 105.0, 107.0], index=DATES,
                   name='asset')
INDICES = np.array([[0, 6], [3, 2], [5, 5], [1, 0], [6, 4]], dtype=np.int64)


def test_bootstrap_data_accepts_a_series_for_list_output():
    """A Series resamples as a one-column panel."""
    returns = PRICES.pct_change().dropna()
    sample = bootstrap_data(data=returns, bootstrapped_indices=INDICES)
    assert len(sample) == INDICES.shape[1]
    for m, path in enumerate(sample):
        assert path.shape == (INDICES.shape[0], 1)
        np.testing.assert_array_equal(path[:, 0], returns.to_numpy()[INDICES[:, m]])


def test_series_price_paths_follow_the_supplied_index_columns():
    """The anchor is repeated once per supplied path, whatever ``num_samples`` says."""
    paths = bootstrap_price_data(prices=PRICES, bootstrap_output=BootstrapOutput.SERIES_TO_DF,
                                 bootstrapped_indices=INDICES)
    assert isinstance(paths, pd.DataFrame)
    assert paths.columns.tolist() == ['path_1', 'path_2']
    levels = PRICES.to_numpy()
    returns = levels[1:] / levels[:-1] - 1.0
    for m in range(INDICES.shape[1]):
        # row 0 is the anchor; row k compounds the returns drawn at index rows 1..k
        growth = np.cumprod(1.0 + returns[INDICES[1:, m]])
        expected = levels[-1] * np.concatenate([[1.0], growth])
        np.testing.assert_allclose(paths.iloc[:, m].to_numpy(), expected, rtol=1e-13)


@pytest.mark.parametrize('bootstrap_output', list(BootstrapOutput))
@pytest.mark.parametrize('bad', [7, 12, -1])
def test_supplied_indices_outside_the_data_are_refused(bootstrap_output, bad):
    returns = PRICES.pct_change().dropna()
    indices = INDICES.copy()
    indices[2, 1] = bad
    with pytest.raises(ValueError, match='bootstrapped_indices'):
        bootstrap_data(data=returns, bootstrap_output=bootstrap_output,
                       bootstrapped_indices=indices)


def test_seed_does_not_touch_numpy_global_state():
    """numba keeps its own generator: numpy's seed neither changes nor is changed by a draw."""
    kwargs = dict(num_data_index=50, bootstrap_type=BootstrapType.STATIONARY, num_samples=3,
                  index_length=40, block_size=5, seed=9)
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    first = generate_bootstrapped_indices(**kwargs)
    np.testing.assert_array_equal(np.random.get_state()[1], before)
    np.random.seed(456)
    second = generate_bootstrapped_indices(**kwargs)
    np.testing.assert_array_equal(first, second)

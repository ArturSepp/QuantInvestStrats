"""
Regression tests for the resampling primitive in qis.models.bootstrap.bootstrap_numba.

These lock the properties the samplers are supposed to have, because none of them is visible
from a shape check:

  * every scheme returns ``(index_length, num_samples)`` indices inside ``[0, num_data_index)``,
  * STATIONARY blocks wrap around the end of the sample rather than being cut short, which is
    what makes the resample stationary (Politis-Romano 1994),
  * ``min_block_size`` floors the drawn block length,
  * FIXED_BLOCK draws blocks of exactly ``block_size``,
  * the same index array applied to two aligned panels resamples them jointly - the paired
    resampling that ``generate_bootstrapped_indices`` is public for.

No data fixtures, no network: the samplers take a length, not a panel.
"""
# packages
import numpy as np
import pandas as pd
import pytest
# qis / project
from qis.models.bootstrap.bootstrap_numba import (BootstrapType,
                                                  BootstrapOutput,
                                                  bootstrap_data,
                                                  generate_bootstrapped_indices)

NUM_DATA_INDEX = 40
INDEX_LENGTH = 200
NUM_SAMPLES = 4


@pytest.mark.parametrize('bootstrap_type', list(BootstrapType))
def test_shape_and_range(bootstrap_type: BootstrapType) -> None:
    """every scheme fills the requested array with valid positions into the source data."""
    indices = generate_bootstrapped_indices(num_data_index=NUM_DATA_INDEX,
                                            bootstrap_type=bootstrap_type,
                                            num_samples=NUM_SAMPLES,
                                            index_length=INDEX_LENGTH,
                                            block_size=8,
                                            seed=3)
    assert indices.shape == (INDEX_LENGTH, NUM_SAMPLES)
    assert indices.min() >= 0
    assert indices.max() < NUM_DATA_INDEX


@pytest.mark.parametrize('bootstrap_type', list(BootstrapType))
def test_seed_is_reproducible(bootstrap_type: BootstrapType) -> None:
    """the same seed gives the same draw; a different seed does not."""
    kwargs = dict(num_data_index=NUM_DATA_INDEX, bootstrap_type=bootstrap_type,
                  num_samples=NUM_SAMPLES, index_length=INDEX_LENGTH, block_size=8)
    same = generate_bootstrapped_indices(seed=5, **kwargs)
    again = generate_bootstrapped_indices(seed=5, **kwargs)
    other = generate_bootstrapped_indices(seed=6, **kwargs)
    assert np.array_equal(same, again)
    assert not np.array_equal(same, other)


def test_unknown_type_raises() -> None:
    """an unhandled scheme reports the value it was given."""
    with pytest.raises(ValueError, match='not implemented'):
        generate_bootstrapped_indices(num_data_index=NUM_DATA_INDEX,
                                      bootstrap_type='stationary',  # a string, not the Enum
                                      num_samples=NUM_SAMPLES,
                                      index_length=INDEX_LENGTH)


def test_stationary_blocks_wrap() -> None:
    """
    a block running off the end of the sample continues at the start.

    Under wrapping, an observation at the last position is followed by position 0 whenever the
    block has not ended - so the (last -> first) transition rate is of the order of
    ``1 - 1/block_size``. Under the truncating implementation the block always ended there and
    the next position was a fresh uniform draw, giving a rate of ``1/num_data_index`` = 0.025.
    """
    block_size = 10
    indices = generate_bootstrapped_indices(num_data_index=NUM_DATA_INDEX,
                                            bootstrap_type=BootstrapType.STATIONARY,
                                            num_samples=200,
                                            index_length=INDEX_LENGTH,
                                            block_size=block_size,
                                            seed=11)
    at_last = indices[:-1, :] == NUM_DATA_INDEX - 1
    wraps_to_first = np.logical_and(at_last, indices[1:, :] == 0)
    wrap_rate = float(np.sum(wraps_to_first) / np.sum(at_last))
    assert wrap_rate > 0.5, f"blocks are not wrapping, rate {wrap_rate:.3f}"


def test_stationary_positions_are_uniform() -> None:
    """
    every observation is drawn about equally often.

    This is the consequence of wrapping that matters for inference: with truncation the last
    observations could only ever appear in short blocks, so the tail of the sample was
    under-represented and the realised block length was not geometric there.
    """
    indices = generate_bootstrapped_indices(num_data_index=NUM_DATA_INDEX,
                                            bootstrap_type=BootstrapType.STATIONARY,
                                            num_samples=500,
                                            index_length=INDEX_LENGTH,
                                            block_size=10,
                                            seed=13)
    counts = np.bincount(indices.flatten(), minlength=NUM_DATA_INDEX)
    frequencies = counts / float(indices.size)
    max_deviation = float(np.max(np.abs(frequencies - 1.0 / NUM_DATA_INDEX)) * NUM_DATA_INDEX)
    assert max_deviation < 0.1, f"positions not uniform, max relative deviation {max_deviation:.3f}"


def test_min_block_size_floors_the_block() -> None:
    """
    with the floor set to the output length the whole draw is one wrapped block.

    Choosing ``min_block_size == index_length`` makes the property checkable exactly: every
    column must be ``(start + k) mod num_data_index``.
    """
    indices = generate_bootstrapped_indices(num_data_index=NUM_DATA_INDEX,
                                            bootstrap_type=BootstrapType.STATIONARY,
                                            num_samples=NUM_SAMPLES,
                                            index_length=INDEX_LENGTH,
                                            block_size=5,
                                            min_block_size=INDEX_LENGTH,
                                            seed=17)
    for column in range(NUM_SAMPLES):
        start = indices[0, column]
        expected = (start + np.arange(INDEX_LENGTH)) % NUM_DATA_INDEX
        assert np.array_equal(indices[:, column], expected)


def test_fixed_block_draws_exact_blocks() -> None:
    """every block is exactly block_size consecutive positions, modulo the sample length."""
    block_size = 8
    assert INDEX_LENGTH % block_size == 0, 'the test needs whole blocks'
    indices = generate_bootstrapped_indices(num_data_index=NUM_DATA_INDEX,
                                            bootstrap_type=BootstrapType.FIXED_BLOCK,
                                            num_samples=NUM_SAMPLES,
                                            index_length=INDEX_LENGTH,
                                            block_size=block_size,
                                            seed=19)
    for column in range(NUM_SAMPLES):
        for start_row in range(0, INDEX_LENGTH, block_size):
            block = indices[start_row:start_row + block_size, column]
            expected = (block[0] + np.arange(block_size)) % NUM_DATA_INDEX
            assert np.array_equal(block, expected)


def test_iid_is_not_serially_dependent() -> None:
    """IID draws are unconditional, so consecutive positions increment by one only by chance."""
    indices = generate_bootstrapped_indices(num_data_index=NUM_DATA_INDEX,
                                            bootstrap_type=BootstrapType.IID,
                                            num_samples=50,
                                            index_length=INDEX_LENGTH,
                                            seed=23)
    increments = (indices[1:, :] - indices[:-1, :]) % NUM_DATA_INDEX
    consecutive_rate = float(np.mean(increments == 1))
    assert consecutive_rate < 0.1, f"IID draw looks blocked, rate {consecutive_rate:.3f}"


def test_paired_resampling_keeps_panels_aligned() -> None:
    """
    one index array applied to two aligned panels resamples them jointly.

    This is the contract that makes ``generate_bootstrapped_indices`` public: a factor panel and
    a residual panel must be resampled with the same draw or their joint structure is lost.
    """
    dates = pd.date_range('2020-01-31', periods=NUM_DATA_INDEX, freq='ME')
    first = pd.DataFrame(np.random.default_rng(1).normal(size=(NUM_DATA_INDEX, 3)),
                         index=dates, columns=['a', 'b', 'c'])
    second = 2.0 * first  # any deterministic map: alignment is what is being tested

    indices = generate_bootstrapped_indices(num_data_index=NUM_DATA_INDEX,
                                            bootstrap_type=BootstrapType.STATIONARY,
                                            num_samples=NUM_SAMPLES,
                                            index_length=INDEX_LENGTH,
                                            block_size=6,
                                            seed=29)
    first_samples = bootstrap_data(data=first,
                                   bootstrap_output=BootstrapOutput.DF_TO_LIST_ARRAYS,
                                   bootstrapped_indices=indices)
    second_samples = bootstrap_data(data=second,
                                    bootstrap_output=BootstrapOutput.DF_TO_LIST_ARRAYS,
                                    bootstrapped_indices=indices)
    assert len(first_samples) == NUM_SAMPLES
    for first_sample, second_sample in zip(first_samples, second_samples):
        assert np.allclose(second_sample, 2.0 * first_sample)

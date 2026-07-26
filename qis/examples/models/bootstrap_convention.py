"""
what an unstated convention costs: block resampling that truncates against one that wraps.

Two implementations of "the stationary bootstrap" can differ in one detail that no output
reports. A block drawn near the end of the sample either wraps around to the start, as in
Politis and Romano (1994), or is cut short there. Both run, both return an index array of the
right shape, and neither prints the choice it made.

This script measures what the choice costs. It reports how often each observation is drawn under
each convention, then applies both to a series whose drift rises through the sample, which is
where an uneven draw turns into a biased statistic.

Run it with ``python -m qis.examples.models.bootstrap_convention``. No network, no data file,
about three seconds including the numba compile.
"""
# packages
import numpy as np
from enum import Enum
from numba import njit
# qis / project
import qis as qis
from qis.models.bootstrap.bootstrap_numba import set_seed

NUM_DATA_INDEX = 250       # a 250-period sample, e.g. twenty years of monthly returns
BLOCK_SIZE = 20            # mean block length
NUM_SAMPLES = 400          # independent draws
INDEX_LENGTH = 250         # each draw is the length of the source sample
SEED = 7
PERIODS_PER_YEAR = 260


# qis.generate_bootstrapped_indices deliberately NOT used for this leg: the point of the script
# is to exhibit the convention qis stopped using at 5.1.0, so the superseded sampler has to be
# written out. This is the one case where reimplementing a stack primitive is the subject rather
# than a defect.
@njit
def draw_truncating_indices(num_data_index: int,
                            num_samples: int,
                            index_length: int,
                            block_size: int,
                            seed: int
                            ) -> np.ndarray:
    """
    the superseded sampler: a block that reaches the end of the sample is cut short there.

    Args:
        num_data_index: number of observations in the source sample
        num_samples: number of independent draws
        index_length: length of each draw
        block_size: mean block length, geometric
        seed: seed for the numba random state

    Returns:
        integer indices of shape ``(index_length, num_samples)``
    """
    np.random.seed(seed)
    set_seed(seed)
    indices = np.zeros((index_length, num_samples), dtype=np.int64)
    for column in np.arange(num_samples):
        previous_row, next_row = 0, 0
        while next_row < index_length - 1:
            start = np.random.randint(0, num_data_index)
            drawn_block = np.random.geometric(1.0 / block_size)
            end_in_data = np.minimum(start + drawn_block, num_data_index)  # the truncation
            next_row = np.minimum(previous_row + (end_in_data - start), index_length)
            filled = next_row - previous_row
            indices[previous_row:next_row, column] = np.arange(start, start + filled)
            previous_row = next_row
    return indices


def relative_draw_frequency(indices: np.ndarray,
                            num_data_index: int
                            ) -> np.ndarray:
    """
    how often each source observation was drawn, relative to a uniform draw.

    Args:
        indices: index array of shape ``(index_length, num_samples)``
        num_data_index: number of observations in the source sample

    Returns:
        one relative frequency per source observation; 1.0 means drawn exactly as often as
        uniform sampling would draw it
    """
    counts = np.bincount(indices.flatten(), minlength=num_data_index)
    return counts / float(indices.size) * num_data_index


def make_trending_returns(num_periods: int = NUM_DATA_INDEX,
                          seed: int = 3
                          ) -> np.ndarray:
    """
    returns whose drift rises through the sample, so an uneven draw becomes a biased mean.

    A flat-drift series would hide the effect: every observation has the same expectation, so it
    does not matter which ones are over-drawn. Drift that changes through the sample is the
    common case in practice, and it is what makes the convention observable in a reported number.

    Args:
        num_periods: length of the series
        seed: seed for the generator

    Returns:
        the return series
    """
    generator = np.random.default_rng(seed)
    drift = 0.0004 + np.linspace(0.0, 0.0016, num_periods)
    return drift + generator.normal(0.0, 0.01, num_periods)


class LocalTest(Enum):
    DRAW_FREQUENCY = 1      # how evenly each convention samples the source
    STATISTIC_BIAS = 2      # what that does to a reported mean return


def run_local_test(local_test: LocalTest) -> None:

    truncating = draw_truncating_indices(num_data_index=NUM_DATA_INDEX,
                                         num_samples=NUM_SAMPLES,
                                         index_length=INDEX_LENGTH,
                                         block_size=BLOCK_SIZE,
                                         seed=SEED)
    circular = qis.generate_bootstrapped_indices(num_data_index=NUM_DATA_INDEX,
                                                 bootstrap_type=qis.BootstrapType.STATIONARY,
                                                 num_samples=NUM_SAMPLES,
                                                 index_length=INDEX_LENGTH,
                                                 block_size=BLOCK_SIZE,
                                                 seed=SEED)
    decile = NUM_DATA_INDEX // 10

    if local_test == LocalTest.DRAW_FREQUENCY:
        print(f"relative draw frequency, {NUM_DATA_INDEX} observations, mean block {BLOCK_SIZE}, "
              f"{NUM_SAMPLES} draws (1.00 = uniform)\n")
        print(f"{'convention':14s}{'first obs':>12s}{'first decile':>14s}{'last decile':>13s}")
        for name, indices in (('truncating', truncating), ('circular', circular)):
            frequency = relative_draw_frequency(indices=indices, num_data_index=NUM_DATA_INDEX)
            print(f"{name:14s}{frequency[0]:12.3f}{frequency[:decile].mean():14.3f}"
                  f"{frequency[-decile:].mean():13.3f}")
        print("\nTruncation is not neutral: a block can only ever run forwards, so the earliest "
              "observations\nare reachable as a block start and almost never as a continuation.")

    elif local_test == LocalTest.STATISTIC_BIAS:
        returns = make_trending_returns()
        source_mean = returns.mean()
        print(f"source series: {NUM_DATA_INDEX} periods, mean {source_mean * 1e4:.2f} bp, "
              f"drift rising through the sample\n")
        print(f"{'convention':14s}{'resampled mean':>17s}{'bias':>10s}{'bias p.a.':>12s}")
        for name, indices in (('truncating', truncating), ('circular', circular)):
            resampled_mean = returns[indices].mean(axis=0).mean()
            bias = resampled_mean - source_mean
            print(f"{name:14s}{resampled_mean * 1e4:14.2f} bp{bias * 1e4:+9.2f}"
                  f"{bias * PERIODS_PER_YEAR * 100:+11.2f}%")
        print("\nThe truncating convention over-weights the late, higher-drift part of the "
              "sample.\nNeither implementation reports which convention it used.")

    else:
        raise ValueError(f"unknown case: {local_test!r}")


if __name__ == '__main__':

    for case in LocalTest:
        print(f"\n===== {case.name} =====")
        run_local_test(local_test=case)

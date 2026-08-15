import time
import numpy as np
from enum import Enum
from qis.utils.np_ops import (RollFillType, np_shift, running_mean, np_nonan_weighted_avg,
                              compute_expanding_power,
                              find_nearest)


class LocalTests(Enum):
    SHIFT_TEST = 1
    CUM_POWER = 2
    ROLLING = 3
    WA = 4
    ARRAY_RANK = 5
    NEAREST = 6


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    if local_test == LocalTests.SHIFT_TEST:
        test_array = np.array([str(n) for n in range(10)])
        print('test_array')
        print(test_array)

        print('shift=2')
        for roll_fill_type in RollFillType:
            print(roll_fill_type)
            print(np_shift(a=test_array, shift=2, roll_fill_type=roll_fill_type))

        print('shift=-2')
        for roll_fill_type in RollFillType:
            print(roll_fill_type)
            print(np_shift(a=test_array, shift=-2, roll_fill_type=roll_fill_type))

    elif local_test == LocalTests.CUM_POWER:
        tic = time.perf_counter()
        for _ in np.arange(20):
            b = compute_expanding_power(n=10000000, power_lambda=0.97, reverse_columns=True)
        toc = time.perf_counter()
        print(f"{toc - tic} secs to run")
        print(b)

    elif local_test == LocalTests.ROLLING:
        x = np.array([1.0, 2.0, np.nan, np.nan, 3.0, 4.0, 5.0])
        xx = running_mean(x=x, n=2)
        print(xx)

    elif local_test == LocalTests.WA:
        x = np.array([1.0, 2.0, np.nan, np.nan, 3.0, 4.0, 5.0])
        weights = np.arange(len(x))
        print(x)
        print(weights)
        xx = np_nonan_weighted_avg(a=x, weights=weights)
        print(xx)

    elif local_test == LocalTests.ARRAY_RANK:
        a = np.array([1.0, 2.0, 10.0, 20.0, 3.0, 4.0, 5.0])
        array_rank = np.argsort(a).argsort()  # ranks by smallest value
        print(np.argsort(a))
        array_idx_rank = {array_rank[n]: n for n in np.arange(a.shape[0])}  # assign rank to idx
        array_idx_rank = dict(sorted(array_idx_rank.items()))  # sort by rank
        print(array_idx_rank)

    elif local_test == LocalTests.NEAREST:
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        print(a)
        print(f"x={2.1}, nearest={find_nearest(a=a, value=2.1)}")
        print(f"x={2.1}, nearest={find_nearest(a=a, value=2.1, is_equal_or_largest=True)}")
        print(f"x={2.0}, nearest={find_nearest(a=a, value=2.0, is_equal_or_largest=True)}")


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.SHIFT_TEST)

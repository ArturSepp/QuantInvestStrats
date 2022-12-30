"""
times
"""
import time
import functools
import numpy as np


def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        if run_time < 60.0:
            print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        else:
            minuts = np.floor(run_time/60.0)
            secs = run_time - 60.0*minuts
            print(f"Finished {func.__name__!r} in {minuts:.0f}m {secs:.0f}secs")
        return value
    return wrapper_timer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from qis.utils.dates import TimePeriod
from qis.utils.sampling import split_to_train_live_samples, split_to_samples


class LocalTests(Enum):
    SAMPLE_DATES = 1
    SPLIT_TO_SAMPLES = 2


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    if local_test == LocalTests.SAMPLE_DATES:
        time_period = TimePeriod(start='31Dec2018', end='31Dec2020')

        ts_index = time_period.to_pd_datetime_index(freq='ME')
        train_live_samples = split_to_train_live_samples(ts_index=ts_index, model_update_freq='ME', roll_period=12)
        train_live_samples.print()

    elif local_test == LocalTests.SPLIT_TO_SAMPLES:
        time_period = TimePeriod(start='31Dec2010', end='31Dec2020')

        ts_index = time_period.to_pd_datetime_index(freq='B')
        data = pd.DataFrame(data=np.random.normal(0, 1.0, (len(ts_index), 1)), index=ts_index, columns=['id1'])

        data_samples = split_to_samples(data=data, sample_freq='YE')
        print(data_samples)

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.SPLIT_TO_SAMPLES)

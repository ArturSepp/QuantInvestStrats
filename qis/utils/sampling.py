"""
utiliti
"""
# packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from dataclasses import dataclass
from typing import NamedTuple, Dict, Union

# qis
from qis.utils.dates import TimePeriod


class TrainLivePeriod(NamedTuple):
    train: TimePeriod
    live: TimePeriod


@dataclass
class TrainLiveSamples:
    train_live_dates: Dict[TimePeriod, TrainLivePeriod] = None

    def __post_init__(self):
        self.train_live_dates = {}

    def add(self, date: TimePeriod, train_live_period: TrainLivePeriod):
        self.train_live_dates[date] = train_live_period

    def print(self):
        for key, samples in self.train_live_dates.items():
            print(f"{key}: train={samples.train.to_str()}, live={samples.live.to_str()}")


def split_to_train_live_samples(ts_index: Union[pd.DatetimeIndex, pd.Index],
                                model_update_freq: 'str' = 'ME',
                                roll_period: int = 12
                                ) -> TrainLiveSamples:

    update_dates = pd.date_range(start=ts_index[0],
                                 end=ts_index[-1],
                                 freq=model_update_freq,
                                 closed='right')

    train_live_samples = TrainLiveSamples()
    for idx, date in enumerate(update_dates):
        # x_data is shifted backward
        # last point in y_data is applied for forecast
        # data used for estimation is [last_update, current update date]
        model_update_date = update_dates[idx - 1]

        if idx > roll_period + 1:
            date_t_0 = update_dates[idx-roll_period-1]
            train = TimePeriod(start=date_t_0, end=model_update_date).shift_end_date_by_days(backward=True)
            live = TimePeriod(start=model_update_date, end=date).shift_start_date_by_days(backward=False)

            train_live_samples.add(model_update_date, TrainLivePeriod(train, live))

    return train_live_samples


def split_to_samples(data: Union[pd.DataFrame, pd.Series],
                     sample_freq: 'str' = 'YE',
                     start_to_one: bool = False
                     ) -> Dict[pd.Timestamp, pd.DataFrame]:
    data1 = data.resample(sample_freq).last()
    ts_index = data1.index
    update_dates = pd.date_range(start=ts_index[0],
                                 end=ts_index[-1],
                                 freq=sample_freq)
    data_samples = {}
    for idx, date in enumerate(update_dates):
        if idx > 1 and date < ts_index[-1]:
            period_data = data.loc[update_dates[idx-1]: date]
            if start_to_one:
                period_data = period_data.divide(period_data.iloc[0])
            data_samples[date] = period_data

    return data_samples


def get_data_samples_df(data: Union[pd.DataFrame, pd.Series],
                        sample_freq: 'str' = 'YE',
                        start_to_one: bool = False
                        ) -> pd.DataFrame:
    data_samples = {}
    data_samples_dict = split_to_samples(data, sample_freq=sample_freq, start_to_one=start_to_one)
    for key, kdata in data_samples_dict.items():
        data_samples[key] = kdata.reset_index(drop=True)
    data_samples_df = pd.DataFrame.from_dict(data_samples)
    data_samples_df = data_samples_df.ffill()
    return data_samples_df


class UnitTests(Enum):
    SAMPLE_DATES = 1
    SPLIT_TO_SAMPLES = 2


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.SAMPLE_DATES:
        time_period = TimePeriod(start='31Dec2018', end='31Dec2020')

        ts_index = time_period.to_pd_datetime_index(freq='ME')
        train_live_samples = split_to_train_live_samples(ts_index=ts_index, model_update_freq='ME', roll_period=12)
        train_live_samples.print()

    elif unit_test == UnitTests.SPLIT_TO_SAMPLES:
        time_period = TimePeriod(start='31Dec2010', end='31Dec2020')

        ts_index = time_period.to_pd_datetime_index(freq='B')
        data = pd.DataFrame(data=np.random.normal(0, 1.0, (len(ts_index), 1)), index=ts_index, columns=['id1'])

        data_samples = split_to_samples(data=data, sample_freq='YE')
        print(data_samples)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.SPLIT_TO_SAMPLES

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

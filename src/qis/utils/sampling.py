"""Calendar-period slicing for dated pandas objects.

``split_to_samples`` is the supported implementation. ``TrainLivePeriod`` and
``TrainLiveSamples`` remain temporarily as deprecated public compatibility containers.
"""
# packages
import warnings
from dataclasses import dataclass
from typing import NamedTuple, Dict, Union

import pandas as pd
# qis
from qis.utils.dates import TimePeriod


class TrainLivePeriod(NamedTuple):
    """Deprecated pair of training and live periods.

    Use an application-specific structure containing two ``TimePeriod`` values instead.

    Attributes:
        train: Estimation period.
        live: Subsequent out-of-sample period.

    Warns:
        DeprecationWarning: On construction; the tuple remains available through qis 5.x.
    """
    train: TimePeriod
    live: TimePeriod


_train_live_period_new = TrainLivePeriod.__new__


def _deprecated_train_live_period_new(
        cls: type[TrainLivePeriod],
        train: TimePeriod,
        live: TimePeriod,
) -> TrainLivePeriod:
    warnings.warn(
        "TrainLivePeriod is deprecated and will be removed in qis 6.0; use an "
        "application-specific pair of TimePeriod values",
        DeprecationWarning,
        stacklevel=2,
    )
    return _train_live_period_new(cls, train, live)


# NamedTuple rejects an in-class __new__ override. Wrapping its generated constructor here keeps
# the exact tuple class, signature, fields, repr, and pickle identity during the deprecation cycle.
TrainLivePeriod.__new__ = staticmethod(_deprecated_train_live_period_new)


@dataclass
class TrainLiveSamples:
    """Deprecated mapping of update dates to training/live periods.

    Use a standard mapping from update dates to application-specific train/live period pairs.

    Attributes:
        train_live_dates: Mapping populated through ``add``.

    Warns:
        DeprecationWarning: On construction; the container remains available through qis 5.x.
    """
    train_live_dates: Dict[TimePeriod, TrainLivePeriod] = None

    def __post_init__(self) -> None:
        warnings.warn(
            "TrainLiveSamples is deprecated and will be removed in qis 6.0; use a standard "
            "mapping of update dates to train/live TimePeriod pairs",
            DeprecationWarning,
            stacklevel=2,
        )
        self.train_live_dates = {}

    def add(self, date: TimePeriod, train_live_period: TrainLivePeriod) -> None:
        self.train_live_dates[date] = train_live_period

    def print(self) -> None:
        for key, samples in self.train_live_dates.items():
            print(f"{key}: train={samples.train.to_str()}, live={samples.live.to_str()}")


def split_to_samples(data: Union[pd.DataFrame, pd.Series],
                     sample_freq: str = 'YE',
                     start_to_one: bool = False
                     ) -> Dict[pd.Timestamp, Union[pd.DataFrame, pd.Series]]:
    """Slice a dated pandas object into established calendar-boundary samples.

    The output retains the historical QIS contract: generated calendar boundaries start from
    the third boundary and exclude the final, potentially incomplete period. Pandas label slicing
    includes an observation on the preceding boundary, so adjacent samples can share that anchor.

    Args:
        data: Series or DataFrame with a monotonic ``DatetimeIndex``.
        sample_freq: Pandas calendar frequency used to generate sample boundaries.
        start_to_one: Whether to divide each sample by its first observation.

    Returns:
        Insertion-ordered mapping from each included period end to the corresponding slice. The
        value type matches ``data``.
    """
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

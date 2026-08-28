"""Calendar-period slicing for dated pandas objects."""
# packages
from typing import Dict, Union

import pandas as pd


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

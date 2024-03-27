"""
utilities to apply frequencies
"""
import numpy as np
import pandas as pd
from enum import Enum
from typing import Optional, Union, Callable, Literal

import qis.utils.dates as da
from qis.utils.df_str import df_index_to_str

# Arguments for fillna()
FillnaOptions = Literal["bfill", "ffill", "pad"]


def df_asfreq(df: Union[pd.DataFrame, pd.Series],
              freq: Optional[str] = 'QE',
              method: FillnaOptions = 'ffill',
              fill_na_method: Optional[FillnaOptions] = 'ffill',
              inclusive: Optional[str] = None,
              include_start_date: bool = False,
              include_end_date: bool = False,
              tz: str = None
              ) -> Union[pd.DataFrame, pd.Series]:
    """
    wrapper to asfreq with closed period
    closed{None, ‘left’, ‘right’}, optional : Make the interval closed with respect to the given frequency
    to the ‘left’, ‘right’, or both sides (None, the default).
    note that using include_start_date and include_end_date will disable given index freq
    """
    if freq is None:
        return df
    elif len(df.index) >=3 and freq == pd.infer_freq(df.index):
        return df

    freq_index = da.generate_dates_schedule(time_period=da.get_time_period(df=df, tz=tz), freq=freq,
                                            include_start_date=include_start_date,
                                            include_end_date=include_end_date)
    if freq_index.empty:
        print(f"in df_asfreq: cannot resample with freq={freq} with start={df.index[0]}, end={df.index[-1]} - using start/end")
        freq_index = freq_index.insert(0, df.index[0])
        freq_index = freq_index.insert(1, df.index[-1])
    if include_start_date:
        if freq_index[0] != df.index[0]:
            freq_index = freq_index.insert(0, df.index[0])
    if include_end_date:  # close on last date
        if freq_index[-1] != df.index[-1]:
            freq_index = freq_index.append(df.index[-1:])
    freq_data = df.reindex(index=freq_index, method=method)

    if fill_na_method is not None:
        if fill_na_method == 'ffill':
            freq_data = freq_data.ffill()
        elif fill_na_method == 'bfill':
            freq_data = freq_data.bfill()
        else:
            raise NotImplementedError(f"fill_na_method={fill_na_method}")
    return freq_data


def agg_remained_data_on_right(df: Union[pd.DataFrame, pd.Series],
                               data: Union[pd.DataFrame, pd.Series],
                               agg_func: Optional[Callable[[pd.DataFrame], pd.Series]]  # for none use last
                               ) -> Union[pd.DataFrame, pd.Series]:
    if df.index[-1] < data.index[-1]:  # some data left on right
        remained_data_on_right = data.loc[df.index[-1]:]
        if df.index[-1] in remained_data_on_right.index:  # data_f.index[-1] can be included in previous sample
            remained_data_on_right = remained_data_on_right.drop(df.index[-1])
        if agg_func is not None:
            agg_remained_data_on_right = remained_data_on_right.apply(agg_func)
        else:
            agg_remained_data_on_right = remained_data_on_right.iloc[-1]
        df = df.append(agg_remained_data_on_right)
    else:
        df = df
    return df


def df_resample_at_other_index(df: Union[pd.DataFrame, pd.Series],
                               other_index: Union[pd.DatetimeIndex, pd.Index],
                               agg_func: Callable[[pd.DataFrame], pd.Series] = np.nanmean,
                               fill_na_method: FillnaOptions = 'ffill',
                               include_end_date: bool = False
                               ) -> Union[pd.DataFrame, pd.Series]:
    """
    given the time index of another time series, aggregate data at frequency of the index
    """
    if not isinstance(other_index, pd.DatetimeIndex):
        raise TypeError (f"other_index type = {type(other_index)} must be pd.DatetimeIndex")

    freq = pd.infer_freq(other_index)
    if freq is None:
        raise ValueError(f"could not infer frequency for index = {other_index}")

    data_f = df.resample(freq).apply(agg_func)

    if include_end_date:
        data_f = agg_remained_data_on_right(df=data_f, data=df, agg_func=agg_func)

    if fill_na_method is not None:
        if fill_na_method == 'ffill':
            data_f = data_f.reindex(index=other_index).ffill()
        elif fill_na_method == 'bfill':
            data_f = data_f.reindex(index=other_index).bfill()
        else:
            raise NotImplementedError(f"fill_na_method={fill_na_method}")

    return data_f


def df_resample_at_freq(df: Union[pd.DataFrame, pd.Series],
                        freq: str = 'QE',
                        fill_na_method: FillnaOptions = 'ffill',
                        agg_func: Optional[Callable[[pd.DataFrame], pd.Series]] = np.nanmean,  # if none use last
                        include_end_date: bool = False
                        ) -> Union[pd.DataFrame, pd.Series]:
    """
    wrapper to asfreq with closed period
    problem with resample it can put last date above last date in data
    """
    # other index date is strictly below the last date in data
    insample_index = pd.date_range(start=df.index[0], end=df.index[-1], freq=freq)
    in_sample_data = df.loc[:insample_index[-1]]
    if agg_func is not None:
        data_f = in_sample_data.resample(freq).apply(agg_func)
    else:
        data_f = in_sample_data.resample(freq).last()

    if include_end_date:
        data_f = agg_remained_data_on_right(df=data_f, data=df, agg_func=agg_func)

    if fill_na_method is not None:  # final check
        if fill_na_method == 'ffill':
            data_f = data_f.ffill()
        elif fill_na_method == 'bfill':
            data_f = data_f.bfill()
        else:
            raise NotImplementedError(f"fill_na_method={fill_na_method}")

    return data_f


def df_resample_at_int_index(df: pd.DataFrame,
                             func: Callable = np.nansum,
                             sample_size: int = 5
                             ) -> pd.DataFrame:
    """
    resample dataframe at evenly spaced discrete index
    """
    assert isinstance(sample_size, int)
    original_index = df.index
    df = df.reset_index(drop=True)  # need integer range index
    sampler = (df.index.to_series() / sample_size).astype(int)
    df = df.groupby(sampler).agg(func, axis=0)
    if isinstance(df, pd.Series):  # if input is one columns
        df = df.to_frame()

    # define sampled index
    sampled_index = original_index[sample_size-1::sample_size]
    if len(sampled_index) < len(df.index):
        sampled_index = sampled_index.append(pd.Index([original_index[-1]]))
    df = df.set_index(sampled_index)
    return df


class UnitTests(Enum):
    AS_FREQ = 1
    RESAMPLE = 2
    INT_INDEX = 3


def run_unit_test(unit_test: UnitTests):

    time_period = da.TimePeriod('1Jan2020', '1Dec2020')
    daily_index = time_period.to_pd_datetime_index(freq='B')
    df = pd.DataFrame(data=np.tile(np.array([1.0, 2.0]), (len(daily_index), 1)), index=daily_index, columns=['1', '2'])
    print(df)

    if unit_test == UnitTests.AS_FREQ:
        freq_data = df_asfreq(df=df, freq='YE')
        print(freq_data)
        freq_data = df_asfreq(df=df, freq='YE', include_end_date=True)
        print(freq_data)
        print(type(freq_data.index))

        freq_data_s = df_index_to_str(freq_data)
        print(freq_data_s)
        print(type(freq_data_s.index))

    elif unit_test == UnitTests.RESAMPLE:

        time_period1 = da.TimePeriod('1Jan2020', '1Jan2021')
        other_index = time_period1.to_pd_datetime_index(freq='QE')
        print(other_index)
        freq_data1 = df_resample_at_other_index(df=df, other_index=other_index, agg_func=np.nansum)
        print(freq_data1)
        print(freq_data1.index)

        freq_data2 = df_resample_at_freq(df=df, freq='QE', agg_func=np.nansum, include_end_date=True)
        print(freq_data2)
        print(freq_data2.index)

    elif unit_test == UnitTests.INT_INDEX:
        df = df_resample_at_int_index(df=df, sample_size=21)
        print(df)


if __name__ == '__main__':

    unit_test = UnitTests.INT_INDEX

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

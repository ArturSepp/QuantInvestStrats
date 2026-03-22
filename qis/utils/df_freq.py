"""
utilities to apply frequencies
"""
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Union, Callable, Literal

import qis.utils.dates as da

# Arguments for fillna()
FillnaOptions = Literal["bfill", "ffill", "pad"]


def _apply_fill(df: Union[pd.DataFrame, pd.Series],
                fill_na_method: Optional[FillnaOptions]
                ) -> Union[pd.DataFrame, pd.Series]:
    """apply forward or backward fill to a dataframe or series"""
    if fill_na_method is None:
        return df
    if fill_na_method in ('ffill', 'pad'):
        return df.ffill()
    elif fill_na_method == 'bfill':
        return df.bfill()
    else:
        raise ValueError(f"unsupported fill_na_method={fill_na_method}")


def df_asfreq(df: Union[pd.DataFrame, pd.Series],
              freq: Optional[str] = 'QE',
              method: FillnaOptions = 'ffill',
              fill_na_method: Optional[FillnaOptions] = 'ffill',
              inclusive: Optional[str] = None,
              include_start_date: bool = False,
              include_end_date: bool = False,
              tz: Optional[str] = None
              ) -> Union[pd.DataFrame, pd.Series]:
    """
    Wrapper to asfreq with closed period.

    Reindexes df onto a date schedule generated at the given freq,
    with optional inclusion of the original start/end dates.

    Args:
        df: input time series
        freq: pandas frequency string; None returns df unchanged
        method: fill method passed to pd.DataFrame.reindex()
        fill_na_method: residual fill applied after reindex (handles leading/trailing NaNs)
        inclusive: reserved, currently unused
        include_start_date: if True, ensures df's first date is in the output index
        include_end_date: if True, ensures df's last date is in the output index
        tz: timezone string passed to date schedule generation

    Note:
        Using include_start_date / include_end_date may produce an irregular index.
    """
    if freq is None or df.empty:
        return df

    # pd.infer_freq requires >= 3 points and can raise on irregular indices
    if len(df.index) >= 3:
        try:
            inferred = pd.infer_freq(df.index)
        except (TypeError, ValueError):
            inferred = None
        if inferred is not None and inferred == freq:
            return df

    freq_index = da.generate_dates_schedule(
        time_period=da.get_time_period(df=df, tz=tz),
        freq=freq,
        include_start_date=include_start_date,
        include_end_date=include_end_date
    )

    if freq_index.empty:
        warnings.warn(
            f"df_asfreq: cannot resample with freq={freq} over "
            f"[{df.index[0]}, {df.index[-1]}]; falling back to endpoints"
        )
        freq_index = pd.DatetimeIndex([df.index[0], df.index[-1]])

    # ensure boundary dates are present when requested
    if include_start_date and freq_index[0] != df.index[0]:
        freq_index = freq_index.insert(0, df.index[0])
    if include_end_date and freq_index[-1] != df.index[-1]:
        freq_index = freq_index.insert(len(freq_index), df.index[-1])

    freq_data = df.reindex(index=freq_index, method=method)
    freq_data = _apply_fill(freq_data, fill_na_method)
    return freq_data


def agg_remained_data_on_right(df: Union[pd.DataFrame, pd.Series],
                               data: Union[pd.DataFrame, pd.Series],
                               agg_func: Optional[Callable[[pd.DataFrame], pd.Series]]  # for None use last
                               ) -> Union[pd.DataFrame, pd.Series]:
    """
    If data extends beyond df's last date, aggregate the remaining tail
    and append it to df.
    """
    if df.index[-1] >= data.index[-1]:
        return df

    remained_data_on_right = data.loc[df.index[-1]:]
    # df.index[-1] may already be included in the previous resample bucket
    if df.index[-1] in remained_data_on_right.index:
        remained_data_on_right = remained_data_on_right.drop(df.index[-1])

    if remained_data_on_right.empty:
        return df

    if agg_func is not None:
        agg_row = remained_data_on_right.apply(agg_func)
    else:
        agg_row = remained_data_on_right.iloc[-1]

    df = pd.concat([df, agg_row.to_frame().T if isinstance(agg_row, pd.Series) and isinstance(df, pd.DataFrame) else pd.DataFrame([agg_row], index=[remained_data_on_right.index[-1]])])
    return df


def df_resample_at_other_index(df: Union[pd.DataFrame, pd.Series],
                               other_index: Union[pd.DatetimeIndex, pd.Index],
                               agg_func: Callable[[pd.DataFrame], pd.Series] = np.nanmean,
                               fill_na_method: FillnaOptions = 'ffill',
                               include_end_date: bool = False
                               ) -> Union[pd.DataFrame, pd.Series]:
    """
    Given the time index of another time series, aggregate data at frequency of the index.
    """
    if not isinstance(other_index, pd.DatetimeIndex):
        raise TypeError(f"other_index type = {type(other_index)} must be pd.DatetimeIndex")

    freq = pd.infer_freq(other_index)
    if freq is None:
        raise ValueError(f"could not infer frequency for index = {other_index}")

    data_f = df.resample(freq).apply(agg_func)

    if include_end_date:
        data_f = agg_remained_data_on_right(df=data_f, data=df, agg_func=agg_func)

    data_f = data_f.reindex(index=other_index)
    data_f = _apply_fill(data_f, fill_na_method)
    return data_f


def df_resample_at_freq(df: Union[pd.DataFrame, pd.Series],
                        freq: str = 'QE',
                        fill_na_method: FillnaOptions = 'ffill',
                        agg_func: Optional[Callable[[pd.DataFrame], pd.Series]] = np.nanmean,  # if None use last
                        include_end_date: bool = False
                        ) -> Union[pd.DataFrame, pd.Series]:
    """
    Wrapper to resample with closed period.

    Problem with resample: it can generate dates beyond the last observation.
    This clips to in-sample and optionally appends the tail.
    """
    if df.empty:
        return df

    insample_index = pd.date_range(start=df.index[0], end=df.index[-1], freq=freq)
    if insample_index.empty:
        warnings.warn(
            f"df_resample_at_freq: no periods for freq={freq} in "
            f"[{df.index[0]}, {df.index[-1]}]"
        )
        return df

    in_sample_data = df.loc[:insample_index[-1]]
    if agg_func is not None:
        data_f = in_sample_data.resample(freq).apply(agg_func)
    else:
        data_f = in_sample_data.resample(freq).last()

    if include_end_date:
        data_f = agg_remained_data_on_right(df=data_f, data=df, agg_func=agg_func)

    data_f = _apply_fill(data_f, fill_na_method)
    return data_f


def df_resample_at_int_index(df: pd.DataFrame,
                             func: Optional[Callable] = np.nansum,
                             sample_size: int = 5
                             ) -> pd.DataFrame:
    """
    Resample dataframe at evenly spaced discrete index with intervals of sample_size.

    The grouping is reversed so the last group always has a full cycle.
    func is the accumulating function; None takes the last row per group.
    """
    if not isinstance(sample_size, int) or sample_size <= 0:
        raise ValueError(f"sample_size must be a positive integer, got {sample_size}")
    if sample_size == 1:
        return df

    original_index = df.index
    df = df.reset_index(drop=True)
    int_index = df.index

    # reverse grouping so the last bucket is always complete
    sampler = (int_index.to_series() / sample_size).astype(int)
    sampler = pd.Series(sampler.values[-1] - sampler.values[::-1], index=sampler.index)

    if func is not None:
        df = df.groupby(sampler, sort=False).agg(func)
    else:
        df = df.groupby(sampler, sort=False).last()

    if isinstance(df, pd.Series):
        df = df.to_frame()

    # recover the original datetime index: take the last timestamp per group
    sampled_index = pd.Series(original_index, index=int_index).groupby(sampler, sort=False).last()
    df.index = sampled_index.values
    return df

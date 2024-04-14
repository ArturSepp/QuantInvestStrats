"""
Implement core for contacenation of time series
"""
# packages
import warnings
import numpy as np
import pandas as pd
from enum import Enum
from typing import Optional, Union, List, Tuple

# qis
import qis.utils.df_ops as dfo
import qis.utils.np_ops as npo
import qis.utils.struct_ops as sop
import qis.perfstats.returns as ret


class MergingMethods(Enum):
    LEVELS = 1
    RETURNS = 2
    VOL_SCALED_RETURNS = 3


def bfill_timeseries(df_newer: Union[pd.DataFrame, pd.Series],  # more recent data
                     df_older: Union[pd.DataFrame, pd.Series],  # older price is preserved to the end
                     freq: str = 'B',
                     fill_method: Optional[str] = None,  # for return use to_zero, else ffill
                     is_prices: bool = False
                     ) -> Union[pd.DataFrame, pd.Series]:
    """
    append column-wise older df data to df new data
    nb output columns are always data_newer columns
    """
    is_series_out = False
    if isinstance(df_newer, pd.Series) and isinstance(df_older, pd.Series):
        # will be error if not same type
        df_newer = df_newer.to_frame()
        df_older = df_older.to_frame(name=df_newer.columns[0])
        is_series_out = True
    elif isinstance(df_newer, pd.DataFrame) and isinstance(df_older, pd.DataFrame):
        pass
    else:
        raise NotImplementedError(f"type1={type(df_newer)}, type2={type(df_older)}") 
    
    if is_prices:
        terminal_value = dfo.get_last_nonnan_values(df_newer)
        if np.any(np.isnan(terminal_value)):
            terminal_value_old = dfo.get_last_nonnan_values(df_older[df_newer.columns])
            terminal_value = np.where(np.isnan(terminal_value), terminal_value_old, terminal_value)

        df_newer = ret.to_returns(df_newer)
        df_older = ret.to_returns(df_older, is_first_zero=True)  # the time series will start from first day of df_older
    else:
        terminal_value = None

    bfill_datas = []
    for column in df_newer:
        newer = df_newer[column]
        if column in df_older.columns:
            older = df_older[column]
            if np.all(newer.isna()): # all new data is none, use old
                bfill_data = older
            else:
                older_start = dfo.get_first_last_nonnan_index(older)
                newer_start = dfo.get_first_last_nonnan_index(newer)
                # print(f"{column}\n{older_start}\n{newer_start}")
                if older_start < newer_start:  # bffill
                    bffill_part = older[:newer_start].iloc[:-1]  # first filerr to newer start and out of last overlap
                    bfill_data = pd.concat([bffill_part, newer[newer_start:]], axis=0)
                else:
                    bfill_data = newer
        else:
            bfill_data = newer

        # just in case
        if bfill_data.index.is_unique is False:  # check if index is unique
            bfill_data = bfill_data.iloc[bfill_data.index.duplicated(keep='last') == False]

        if fill_method is not None:
            start = dfo.get_first_last_nonnan_index(bfill_data)
            if fill_method == 'to_zero':
                bfill_data[start:] = bfill_data[start:].fillna(value=0.0)
            else:
                bfill_data[start:] = bfill_data[start:].ffill()

        bfill_datas.append(bfill_data)
    bfill_datas = pd.concat(bfill_datas, axis=1).sort_index()

    if is_prices:
        bfill_datas = ret.returns_to_nav(returns=bfill_datas,
                                         init_period=None,
                                         terminal_value=terminal_value)

    if pd.infer_freq(bfill_datas.index) != freq:
        bfill_datas = bfill_datas.asfreq(freq, method='ffill').ffill()

    if is_series_out:
        bfill_datas = bfill_datas.iloc[:, 0]

    return bfill_datas


def append_time_series(df_newer: Union[pd.DataFrame, pd.Series],  # more recent data
                       df_older: Union[pd.DataFrame, pd.Series],  # older price is preserved to the end
                       numerical_check_columns: List[str] = None
                       ) -> Tuple[Union[pd.DataFrame, pd.Series], Optional[pd.Series]]:
    """
    append time series by colomns
    force alignment of columns
    """
    is_series = False
    if isinstance(df_newer, pd.Series) and isinstance(df_older, pd.Series):
        is_series = True
        df_newer = df_newer.to_frame()
        df_older = df_older.to_frame()
    elif isinstance(df_newer, pd.DataFrame) and isinstance(df_older, pd.DataFrame):
        pass
    else:
        raise ValueError(f"{type(df_older)} not aligned with {type(df_newer)}")

    sop.assert_list_subset(large_list=df_newer.columns.to_list(),
                           list_sample=df_older.columns.to_list())

    if df_older.index[0] >= df_newer.index[0]:  # old index is older than new, no need to do anything
        new_df = df_newer
        diff = None

    elif df_older.index[-1] >= df_newer.index[0]:  # append
        t0 = df_newer.index[0]
        t1 = df_older.index[-1]
        overlap_old = df_older.loc[t0:t1, :]
        overlap_new = df_newer.loc[t0:t1, :]

        if numerical_check_columns is not None:
            diff = np.abs(overlap_old[numerical_check_columns] - overlap_new[numerical_check_columns]).mean(0)
            # if np.any(np.greater(diff, 1e-0)):
            #    print(f"differences detected {diff}")
        else:
            diff = None
        new_df = pd.concat([df_older.loc[:t0, :], df_newer], axis=0)
    else:
        new_df = pd.concat([df_older, df_newer], axis=0)
        diff = None

    # just in case
    if new_df.index.is_unique is False:  # check if index is unique
        new_df = new_df.iloc[new_df.index.duplicated(keep='last')==False]

    if is_series:
        new_df = new_df.iloc[:, 0]

    return new_df, diff


def replace_nan_by_median(df: pd.DataFrame,
                          is_replace_zeros: bool = True
                          ) -> pd.DataFrame:
    """
    fill non nan using median of other columns
    """
    if is_replace_zeros:
        df = df.replace(0.0, np.nan)
    np_data = df.to_numpy()
    data_med = npo.np_array_to_df_columns(a=np.nanmedian(np_data, axis=1), n_col=len(df.columns))
    np_data_fill = np.where(np.isnan(np_data), data_med, np_data)
    data_clean = pd.DataFrame(np_data_fill, index=df.index, columns=df.columns)
    return data_clean


def df_fill_first_nan_by_cross_median(df: pd.DataFrame,
                                      is_replace_zeros: bool = True
                                      ) -> pd.DataFrame:
    """
    before first non nan use median of other columns
    after that use ffill
    """
    df = df.copy()
    if is_replace_zeros:
        df = df.replace(0.0, np.nan)

    # for each column find first nonan
    first_nonnan_index = dfo.get_first_before_nonnan_index(df)
    merged_data = pd.DataFrame(index=df.index, columns=df.columns)
    for idx, column in enumerate(df.columns):
        its_first_nonnan_index = first_nonnan_index[idx]
        with warnings.catch_warnings():  # silence All-NaN slice encountered
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median_backfill = np.nanmedian(df.loc[:its_first_nonnan_index, :].to_numpy(), axis=1)
        merged_data.loc[:its_first_nonnan_index, column] = median_backfill
        merged_data.loc[its_first_nonnan_index:, column] = df.loc[its_first_nonnan_index:, column].to_numpy()

    # fillnans with ffill in data after
    with pd.option_context("future.no_silent_downcasting", True):
        merged_data = merged_data.ffill().infer_objects(copy=False)

    return merged_data


def df_price_fill_first_nan_by_cross_median(prices: pd.DataFrame) -> pd.DataFrame:
    """
    before first non nan use median of other columns
    after that use ffill
    """
    prices = prices.ffill()
    returns = ret.to_returns(prices=prices, is_first_zero=False)
    returns_fill = df_fill_first_nan_by_cross_median(df=returns, is_replace_zeros=False)
    returns_fill = returns_fill.fillna(0.0)
    bfilled_data = ret.returns_to_nav(returns=returns_fill, terminal_value=prices.iloc[-1, :])
    return bfilled_data

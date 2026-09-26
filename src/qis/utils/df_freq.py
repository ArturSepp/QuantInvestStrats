"""
resampling a panel onto another frequency or onto another series' index, with the closed-period
convention made explicit. ``df_asfreq`` samples the last observation at or before each scheduled
date, while ``df_resample_at_freq`` and ``df_resample_at_other_index`` aggregate within the period
by ``agg_func`` and, under ``include_end_date``, carry a final partial period through
``agg_remained_data_on_right`` rather than discarding it. ``df_resample_at_int_index`` groups rows
into blocks of a fixed count, counted back from the last row, and drops an incomplete first block
when it aggregates, since a partial sum is not an observation of the block aggregate.

A panel is a mapping from date to value, so the row order it arrives in carries no information:
``df_asfreq`` sorts an index that is not chronological rather than resampling along it.
"""
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Union, Callable, Literal

import qis.utils.dates as da

# Arguments for fillna()
FillnaOptions = Literal["bfill", "ffill", "pad"]


def validate_calendar_index(data: Union[pd.DataFrame, pd.Series],
                            argument_name: str
                            ) -> None:
    """Validate the date axis required by calendar operations.

    Empty objects remain valid schema declarations because they contribute no observations to a
    schedule. The validator deliberately leaves sorting and duplicate-date policy to the calling
    operation.

    Args:
        data: Series or DataFrame about to undergo calendar scheduling or timestamp arithmetic.
        argument_name: Public parameter name used in deterministic error messages.

    Raises:
        TypeError: If a nonempty object does not use a ``DatetimeIndex``.
        ValueError: If a nonempty object's index contains ``NaT``.
    """
    if data.empty:
        return
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError(f"{argument_name} must use a DatetimeIndex for calendar operations")
    if data.index.hasnans:
        raise ValueError(f"{argument_name} index must not contain NaT")


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

    Reindexes df onto completed calendar boundaries generated at the given frequency. The
    original start or terminal partial-period observation is included only when explicitly
    requested.

    Args:
        df: Input time series. Calendar resampling requires a nonempty object to use a
            ``DatetimeIndex`` without ``NaT``; arbitrary indexes remain valid when ``freq=None``.
        freq: pandas frequency string; None returns df unchanged
        method: fill method passed to pd.DataFrame.reindex()
        fill_na_method: Fill applied before and after reindexing when the input frequency differs
            from ``freq``. This lets an exact-boundary missing value use the latest earlier value.
            Already-periodic input is returned without changing its missing-value mask: on its
            own grid a missing value is a missing observation, and ``qis.to_returns`` and
            ``qis.prices_at_freq`` inherit this rule.
        inclusive: reserved, currently unused
        include_start_date: If True, include ``df``'s first observation date.
        include_end_date: If True, include ``df``'s last observation date, representing a terminal
            partial period when it is not a regular boundary.
        tz: timezone string passed to date schedule generation

    Raises:
        TypeError: If calendar resampling is requested for a nonempty object without a
            ``DatetimeIndex``.
        ValueError: If calendar resampling is requested for a nonempty object whose index contains
            ``NaT``.

    Note:
        With both inclusion flags false, a history shorter than one complete period returns an
        empty object with the input schema. Using either flag may produce an irregular index.
        A df whose index is not in chronological order is sorted before resampling.
    """
    if freq is None or df.empty:
        return df

    # Validate the structural requirement only when a calendar schedule will be constructed.
    validate_calendar_index(df, argument_name="df")

    # Everything below assumes the panel is in chronological order: the pre-reindex ffill
    # carries values forward in ROW order, and reindex(method='ffill') raises
    # "index must be monotonic increasing or decreasing" out of pandas on an unsorted index.
    #
    # Such a panel reaches here more easily than it used to. pandas 3.0 changed
    # pd.concat(axis=1, sort=False) - how a benchmark series and a strategy nav on different
    # calendars get joined - to leave the union of the two DatetimeIndexes in appearance order:
    # the dates carried only by the second frame land after the last date of the first. pandas
    # 2.2 returned a sorted union from the same call, and concat with no explicit sort= still
    # sorts today under a deprecation that ends that.
    #
    # Sorting is a repair rather than a convention: a price panel is a mapping from date to
    # value, and the order its rows arrive in carries no information.
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    # pd.infer_freq requires >= 3 points and can raise on irregular indices
    if len(df.index) >= 3:
        try:
            inferred = pd.infer_freq(df.index)
        except (TypeError, ValueError):
            inferred = None
        if inferred is not None and inferred == freq:
            return df

    # Fill on the source grid before boundary reindexing. In particular, pandas
    # reindex(method=...) does not replace a NaN already stored exactly at a requested boundary.
    # The same-frequency shortcut above deliberately preserves an existing missing-value mask.
    df = _apply_fill(df, fill_na_method)

    freq_index = da.generate_dates_schedule(
        time_period=da.get_time_period(df=df, tz=tz),
        freq=freq,
        include_start_date=include_start_date,
        include_end_date=include_end_date
    )

    if freq_index.empty:
        return df.iloc[0:0].copy()

    # Pre-fill NaN values in df BEFORE the reindex.
    #
    # Rationale: pd.DataFrame.reindex(index=freq_index, method='ffill')
    # looks back through the INPUT INDEX LABELS — not the values — and
    # copies whatever value sits at the nearest preceding label. So when a
    # target date (e.g. a holiday Friday at 'W-FRI' resample) is itself
    # present in df.index with an explicit NaN value (as in yfinance data
    # that returns NaN on US holidays), reindex returns NaN — it does not
    # "skip past the NaN" to find the previous valid observation.
    #
    # Applying _apply_fill on df first carries the last known value
    # forward through the explicit-NaN rows, so the reindex picks up the
    # correct close-to-close anchor at each target date. This matches the
    # convention of `df.resample(freq).last()` on a ffilled series.
    #
    # The post-reindex _apply_fill below stays in place: it handles
    # leading/trailing NaNs introduced by reindex when target dates fall
    # outside the observed range.
    freq_index = freq_index.rename(df.index.name)
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
                             sample_size: int = 5,
                             drop_incomplete_first: Optional[bool] = None
                             ) -> pd.DataFrame:
    """Resample a panel into consecutive blocks of ``sample_size`` rows.

    Blocks are counted back from the last row, so the last block is always complete and the
    first holds the remaining ``T mod sample_size`` rows when ``T`` is not a multiple of
    ``sample_size``.

    Args:
        df: Panel in chronological row order
        func: Aggregation applied to each block, such as ``np.nansum`` for returns; None takes
            the last row of each block, which samples levels on the block grid
        sample_size: Positive number of rows per block
        drop_incomplete_first: Whether to drop an incomplete first block. None drops it when
            ``func`` aggregates, because an aggregate of fewer rows, such as a partial sum of
            returns, is not an observation of the ``sample_size``-row aggregate, and keeps it
            when ``func`` is None, because its last row is a level on the block grid

    Returns:
        One row per block, labelled with the last timestamp of the block

    Raises:
        ValueError: If ``sample_size`` is not a positive integer.
    """
    if not isinstance(sample_size, int) or sample_size <= 0:
        raise ValueError(f"sample_size must be a positive integer, got {sample_size}")
    if sample_size == 1:
        return df
    if drop_incomplete_first is None:
        drop_incomplete_first = func is not None
    if drop_incomplete_first:
        df = df.iloc[len(df.index) % sample_size:]

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

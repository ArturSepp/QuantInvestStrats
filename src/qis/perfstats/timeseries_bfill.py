"""
splicing time series together: a Brownian-bridge interpolation and two backfill joins.

``interpolate_infrequent_returns`` fills the gaps of an infrequently reported series onto the
grid of a frequent pivot series, drawing increments from a Brownian bridge whose innovations
come from the pivot path, so the result hits every reported value exactly while carrying the
timing of a real market. The interpolated path is a plausible history, not the true one.

``bfill_timeseries`` extends a newer panel backwards with an older one column by column. Output
columns are always the newer panel's, and with ``is_prices=True`` the join is made in return
space and rebuilt into a nav anchored on the newer panel's last non-nan level, so the recent
level is preserved rather than the old one. ``append_time_series`` is the plain concatenation
over a shared overlap and requires the older columns to be a subset of the newer ones.
"""
# packages
import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple, cast
# qis
import qis.utils.df_ops as dfo
import qis.utils.np_ops as npo
import qis.utils.struct_ops as sop
import qis.perfstats.returns as ret
import qis.models.linear.ewm as ewm
from qis.utils.df_ops import df_ffill_negatives


def interpolate_infrequent_returns(infrequent_returns: Union[pd.Series, pd.DataFrame],
                                   pivot_returns: pd.Series,
                                   span: int = 12,
                                   annualization_factor: float = 260,
                                   is_to_log_returns: bool = False,
                                   vol_adjustment: float = 1.15  # adjust vol of the bridge
                                   ) -> Union[pd.Series, pd.DataFrame]:
    """
    backfill an infrequently reported series onto a frequent grid with a Brownian bridge.

    A quarterly private-market series cannot be risk-analysed against daily public markets, and
    forward-filling it makes it look riskless between reports. This interpolates instead: the
    increments between two reported values are drawn from a Brownian bridge whose innovations come
    from the path of ``pivot_returns``, so the interpolated series carries the timing of a real
    market rather than a smooth line, while still hitting each reported value exactly.

    The interpolated path is a plausible history, not the true one. Statistics computed on it
    inherit the bridge's assumptions, so it belongs in a risk model and not in a performance report.

    Args:
        infrequent_returns: the reported series, with gaps between reports. A DataFrame is handled
            column by column
        pivot_returns: a frequent series whose path supplies the innovations; the grid it is
            observed on is the grid the result is returned on
        span: EWM span used to estimate the volatility of the infrequent series
        annualization_factor: periods per year of ``pivot_returns``
        is_to_log_returns: treat the inputs as log returns
        vol_adjustment: multiplier on the bridge volatility. Above one compensates for the bridge
            understating the volatility of the unobserved path

    Returns:
        the interpolated returns on the index of ``pivot_returns``, in the shape of the input
    """
    # call recursion here
    if isinstance(infrequent_returns, pd.DataFrame):
        infrequent_return_backfills = {}
        for column in infrequent_returns.columns:
            ds = infrequent_returns[column].dropna()
            infrequent_return_backfills[column] = interpolate_infrequent_returns(infrequent_returns=ds,
                                                                                 pivot_returns=pivot_returns,
                                                                                 span=span,
                                                                                 annualization_factor=annualization_factor,
                                                                                 is_to_log_returns=is_to_log_returns,
                                                                                 vol_adjustment=vol_adjustment)
        infrequent_return_backfills = pd.DataFrame.from_dict(infrequent_return_backfills, orient='columns')
        return infrequent_return_backfills

    # ensure no nans in infrequent_returns
    if np.any(np.isnan(infrequent_returns)):
        raise ValueError(f"infrequent_returns contains nans")

    # transform to cumulative
    if is_to_log_returns:
        infrequent_returns = np.log(1.0+infrequent_returns)
    infrequent_cumulative = infrequent_returns.cumsum()

    # starting time
    date0 = infrequent_returns.index[0]
    # pivot brownian starting from date0
    pivot_brownian = (pivot_returns - ewm.compute_ewm(data=pivot_returns, span=span)) / ewm.compute_ewm_vol(data=pivot_returns, span=span)
    pivot_brownian = pivot_brownian.loc[date0:, ]
    pivot_brownian = (pivot_brownian - np.nanmean(pivot_brownian)) / np.nanstd(pivot_brownian)  # path to (0, 1) brownian

    # add running times
    seconds_per_year = annualization_factor * 24 * 60 * 60  # days, hours, minute, seconds
    t = pd.Series((infrequent_returns.index - date0).total_seconds() / seconds_per_year, index=infrequent_returns.index)
    t1 = t.shift(-1)
    dt = t1 - t

    # the index of df = index of pivot_brownian
    df = pd.concat([pivot_brownian,
                    infrequent_cumulative.rename('x_i'), infrequent_cumulative.shift(-1).rename('x_i+1'),
                    t.rename('t_i'), t1.rename('t_i+1'), dt.rename('dt_i')], axis=1, sort=True)
    df['t'] = (df.index - date0).total_seconds() / seconds_per_year
    df = df.ffill()  # ffill data to cover nans for infrequent series

    # compute bridge mean and stdev
    bridge_mean = ((df['t_i+1']-df['t']) * df['x_i'] + (df['t']-df['t_i']) * df['x_i+1'] ) / df['dt_i']
    # extrapolate last values when df['x_i'] = df['x_i+1']
    bridge_mean = bridge_mean.where(np.equal(df['x_i'], df['x_i+1']) == False, other=np.nan)
    bridge_mean[infrequent_cumulative.index[-1]] = infrequent_cumulative.iloc[-1]  # enter last observed value
    bridge_mean = bridge_mean.ffill()  # extrapolate last value

    bridge_stdev = np.nanstd(infrequent_returns)*np.sqrt(((df['t_i+1']-df['t'])*(df['t']-df['t_i'])) / df['dt_i'])
    # simulate backfill
    infrequent_cumulative_backfill = bridge_mean + vol_adjustment*bridge_stdev * df[pivot_brownian.name]
    # compute returns
    infrequent_return_backfill = infrequent_cumulative_backfill.diff(1)
    if is_to_log_returns:
        infrequent_return_backfill = np.expm1(infrequent_return_backfill)

    return infrequent_return_backfill


def bfill_timeseries(df_newer: Union[pd.DataFrame, pd.Series],  # more recent data
                     df_older: Union[pd.DataFrame, pd.Series],  # older price is preserved to the end
                     freq: str = 'B',
                     fill_method: Optional[str] = None,  # None, 'to_zero', or 'ffill' for returns
                     is_prices: bool = False
                     ) -> Union[pd.DataFrame, pd.Series]:
    """Extend newer time series backward with older provider histories.

    For price DataFrames, a newer all-missing column without an older counterpart remains
    entirely missing.

    Args:
        df_newer: Newer Series or DataFrame whose labels and columns define the output. Its rows
            are interpreted in increasing date order without modifying the caller's object.
        df_older: Older object of the same pandas type, used before each newer history begins. Its
            rows are also interpreted in increasing date order without modifying the caller.
        freq: Frequency of the returned date grid.
        fill_method: Return-gap policy. ``None`` preserves missing returns, ``'to_zero'`` fills
            missing returns with zero after each column begins, and ``'ffill'`` carries its last
            observed return forward. For price inputs, the policy is applied in return space.
        is_prices: Whether the supplied data are price levels. Expanded price grids carry the
            last observed level forward.

    Returns:
        Chronologically ordered backfilled data on the requested grid, matching the newer input's
        pandas type, labels, and column order.

    Raises:
        ValueError: If ``fill_method`` is not ``None``, ``'to_zero'``, or ``'ffill'``.
        NotImplementedError: If the newer and older inputs are not both Series or both
            DataFrames.
    """
    # Reject unsupported policies before the fallback branch can silently treat them as ffill.
    if fill_method is not None and (
            not isinstance(fill_method, str) or fill_method not in ('to_zero', 'ffill')):
        raise ValueError(
            f"fill_method must be None, 'to_zero', or 'ffill', got {fill_method!r}"
        )

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

    # Provider row order carries no information; boundaries and fills must run chronologically.
    if not df_newer.index.is_monotonic_increasing:
        df_newer = df_newer.sort_index()
    if not df_older.index.is_monotonic_increasing:
        df_older = df_older.sort_index()
    
    price_fallback_columns = []
    if is_prices:

        # make sure no negative prices
        df_newer = df_ffill_negatives(df_newer)
        df_older = df_ffill_negatives(df_older)
        newer_prices = cast(pd.DataFrame, df_newer)
        older_prices = cast(pd.DataFrame, df_older)
        price_fallback_columns = [
            column for column in newer_prices.columns
            if column in older_prices.columns
            and newer_prices[column].isna().all()
            and older_prices[column].notna().any()
        ]

        terminal_value = dfo.get_last_nonnan_values(df_newer)
        if np.any(np.isnan(terminal_value)):
            # Preserve the newer schema when an older terminal history is unavailable.
            aligned_older = older_prices.reindex(columns=newer_prices.columns)
            terminal_value_old = dfo.get_last_nonnan_values(aligned_older)
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
                older_start = dfo.get_nonnan_index(older)
                newer_start = dfo.get_nonnan_index(newer)
                # print(f"{column}\n{older_start}\n{newer_start}")
                if older_start < newer_start:  # bffill
                    bffill_part = older.loc[older.index < newer_start]  # retain earlier dates
                    bfill_data = pd.concat([bffill_part, newer[newer_start:]], axis=0)
                else:
                    bfill_data = newer
        else:
            bfill_data = newer

        # just in case
        if bfill_data.index.is_unique is False:  # check if index is unique
            bfill_data = bfill_data.iloc[bfill_data.index.duplicated(keep='last') == False]

        # Price gaps must be resolved in return space before levels are reconstructed.
        if fill_method is not None and is_prices:
            start = dfo.get_nonnan_index(bfill_data)
            if fill_method == 'to_zero':
                bfill_data[start:] = bfill_data[start:].fillna(value=0.0)
            else:
                bfill_data[start:] = bfill_data[start:].ffill()

        bfill_datas.append(bfill_data)
    bfill_datas = pd.concat(bfill_datas, axis=1, sort=True).sort_index()

    if is_prices:
        bfill_datas = ret.returns_to_nav(returns=bfill_datas,
                                         init_period=None,
                                         terminal_value=terminal_value)
        # Carry available older-only histories independently of grid-resampling side effects.
        for column in price_fallback_columns:
            bfill_datas[column] = bfill_datas[column].ffill()

    # Short splices have no inferable frequency but can still be expanded safely.
    inferred_freq = pd.infer_freq(bfill_datas.index) if len(bfill_datas.index) >= 3 else None
    if inferred_freq != freq:
        if is_prices:
            # Carry the latest source price even when its date is outside the target grid.
            bfill_datas = bfill_datas.asfreq(freq, method='ffill').ffill()
        else:
            bfill_datas = bfill_datas.asfreq(freq)

    if fill_method is not None and is_prices is False:
        # Apply return policies after expansion so inserted dates follow the selected convention.
        for column in bfill_datas:
            # An all-missing column has no observation from which its fill policy can begin.
            if not bfill_datas[column].notna().any():
                continue
            start = dfo.get_nonnan_index(bfill_datas[column])
            if fill_method == 'to_zero':
                bfill_datas.loc[start:, column] = bfill_datas.loc[start:, column].fillna(0.0)
            else:
                bfill_datas.loc[start:, column] = bfill_datas.loc[start:, column].ffill()

    if is_series_out:
        bfill_datas = bfill_datas.iloc[:, 0]

    return bfill_datas


def append_time_series(df_newer: Union[pd.DataFrame, pd.Series],  # more recent data
                       df_older: Union[pd.DataFrame, pd.Series],  # older price is preserved to the end
                       numerical_check_columns: List[str] = None
                       ) -> Tuple[Union[pd.DataFrame, pd.Series], Optional[pd.Series]]:
    """Append older history before newer data with newer values winning overlaps.

    Args:
        df_newer: Newer Series or DataFrame. Rows are interpreted in increasing index order
            without modifying the caller's object.
        df_older: Older object of the same pandas type. Its DataFrame columns must be a subset of
            the newer columns, and its rows are also interpreted in increasing index order.
        numerical_check_columns: Columns for which to return mean absolute differences over the
            aligned overlap. ``None`` skips the diagnostic.

    Returns:
        A chronologically ordered appended object matching the input pandas type, plus the
        optional overlap-difference Series. Newer values take precedence on shared dates;
        duplicate dates within a provider retain the last supplied observation.

    Raises:
        ValueError: If the inputs have different pandas types or their converted columns cannot
            be aligned.
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

    # Row storage order carries no information; boundaries and overlap checks are chronological.
    if not df_newer.index.is_monotonic_increasing:
        df_newer = df_newer.sort_index(kind='stable')
    if not df_older.index.is_monotonic_increasing:
        df_older = df_older.sort_index(kind='stable')

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
            diff = np.abs(overlap_old[numerical_check_columns] - overlap_new[numerical_check_columns]).mean(axis=0)
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

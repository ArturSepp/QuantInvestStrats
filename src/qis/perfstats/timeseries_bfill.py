"""
splicing time series together: a Brownian-bridge interpolation and two backfill joins.

``interpolate_infrequent_returns`` places an infrequently reported series on the grid of a
frequent pivot series with a Brownian bridge on log levels whose innovations come from the pivot
path, so the result reproduces every reported return exactly, point in time, while carrying the
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
import qis.utils.df_freq as dfr
import qis.utils.np_ops as npo
import qis.perfstats.returns as ret
from qis.utils.df_ops import df_ffill_negatives


def interpolate_infrequent_returns(infrequent_returns: Union[pd.Series, pd.DataFrame],
                                   pivot_returns: pd.Series,
                                   span: int = 12,
                                   annualization_factor: float = 260,
                                   is_to_log_returns: bool = False,
                                   vol_adjustment: float = 1.0
                                   ) -> Union[pd.Series, pd.DataFrame]:
    """
    backfill an infrequently reported series onto a frequent grid with a Brownian bridge.

    A quarterly private-market series cannot be risk-analysed against daily public markets, and
    forward-filling it makes it look riskless between reports. This interpolates instead: between
    two reports the log NAV follows a Brownian bridge pinned to both reported values, and the
    bridge increments take their timing from ``pivot_returns``, so the interpolated series moves
    with a real market rather than along a smooth line.

    Report ``b`` is placed on the last pivot date on or before its report date. Over the ``n``
    pivot dates of a report interval, with reported log return ``l``, the interpolated log
    returns are ``x_j = l / n + vol_adjustment * sqrt(v) * e_j``. Here ``e_j`` are the pivot
    returns of the interval, demeaned and scaled so that ``sum(e_j**2) = n - 1``, the expected
    sum of squares of the demeaned increments of a unit Brownian motion, and ``v`` is the
    per-pivot-period variance of the reported returns: an EWM, with span ``span`` reports, of
    ``l**2 / n`` over the intervals up to and including the current one. Because ``e`` sums to
    zero, the interval's log returns sum exactly to ``l``. Their sum of squares is
    ``l**2 / n + vol_adjustment**2 * v * (n - 1)``, so with ``vol_adjustment=1`` the increments
    carry, per pivot period, the variance of the reported returns under the square-root-of-time
    rule, and under a Brownian motion observed at the report dates they are, on average,
    serially uncorrelated.

    The interval ending at a report uses only the pivot returns inside it and the reports up to
    and including it, so the interpolated history up to a report date does not change when later
    data arrive. It is a plausible history, not the true one, and inherits the bridge's
    assumptions: it belongs in a risk model and not in a performance report.

    Args:
        infrequent_returns: the reported returns, one per report date, in the convention set by
            ``is_to_log_returns``. A Series must not contain missing values; a DataFrame is
            handled column by column after dropping each column's missing values
        pivot_returns: a frequent series whose path supplies the timing of the increments; the
            result is returned on its index. Missing pivot returns contribute no deviation
        span: EWM span, in reports, of the per-pivot-period variance of the reported returns
        annualization_factor: periods per year of ``pivot_returns``. Both the bridge variance and
            the reported-return variance are measured on the pivot clock, so the interpolated
            path does not depend on this value; the bridge volatility per year is
            ``vol_adjustment * sqrt(annualization_factor * v)``
        is_to_log_returns: False (default) for simple returns, which are converted with
            ``log1p`` for the bridge and returned as simple returns that compound exactly to each
            reported return; True for log returns, returned as log returns that sum exactly to
            each reported return
        vol_adjustment: multiplier on the bridge volatility. The default 1 keeps the variance of
            the reported returns; a value above one adds variance, for example to offset the
            smoothing of appraisal-based reports

    Returns:
        the interpolated returns on the index of ``pivot_returns``, in the shape of the input.
        Dates up to the first placed report and after the last one are missing. A report
        interval that contains no pivot date is merged with the next one, whose increments then
        reproduce the two reports together

    Raises:
        TypeError: If a nonempty input does not use a ``DatetimeIndex``.
        ValueError: If a nonempty input index contains ``NaT``, or if ``infrequent_returns``
            contains missing values.
    """
    # Both paths perform timestamp slicing and subtraction, so validate their grids up front.
    dfr.validate_calendar_index(infrequent_returns, argument_name="infrequent_returns")
    dfr.validate_calendar_index(pivot_returns, argument_name="pivot_returns")

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

    if not pivot_returns.index.is_monotonic_increasing:
        pivot_returns = pivot_returns.sort_index()
    if not infrequent_returns.index.is_monotonic_increasing:
        infrequent_returns = infrequent_returns.sort_index()

    # The bridge runs on log levels, so compounding in the simple-return mode is exact.
    reported = infrequent_returns.to_numpy(dtype=float)
    log_reported = reported if is_to_log_returns else np.log1p(reported)
    log_levels = np.cumsum(log_reported)

    # Place each report on the last pivot date on or before it. A report before the first pivot
    # date only sets the starting level; later reports sharing a pivot date with an earlier one
    # are merged into the next interval, so no return lands on a date before it occurred.
    positions = pivot_returns.index.searchsorted(infrequent_returns.index, side='right') - 1
    anchor_positions: List[int] = []
    anchor_levels: List[float] = []
    for position, level in zip(positions, log_levels):
        if position < 0:
            if len(anchor_positions) > 0:
                anchor_positions[-1], anchor_levels[-1] = -1, level
            else:
                anchor_positions.append(-1)
                anchor_levels.append(level)
        elif len(anchor_positions) == 0 or position > anchor_positions[-1]:
            anchor_positions.append(int(position))
            anchor_levels.append(level)

    pivot_values = pivot_returns.to_numpy(dtype=float)
    increments = np.full(len(pivot_values), np.nan)
    ewm_lambda = 1.0 - 2.0 / (span + 1.0)
    per_period_variance = np.nan
    for start, end, level_start, level_end in zip(anchor_positions[:-1], anchor_positions[1:],
                                                  anchor_levels[:-1], anchor_levels[1:]):
        num_periods = end - start
        interval_return = level_end - level_start
        # Point-in-time EWM of the per-period variance implied by the reported returns.
        variance = interval_return ** 2 / num_periods
        if np.isnan(per_period_variance):
            per_period_variance = variance
        else:
            per_period_variance = ewm_lambda * per_period_variance + (1.0 - ewm_lambda) * variance

        # Demeaned pivot returns scaled to the expected sum of squares of a unit Brownian
        # bridge's increments; they sum to zero, so the reported return is matched exactly.
        interval_pivot = pivot_values[start + 1:end + 1]
        is_observed = np.isfinite(interval_pivot)
        deviations = np.zeros(num_periods)
        if np.any(is_observed):
            deviations[is_observed] = (interval_pivot[is_observed]
                                       - np.mean(interval_pivot[is_observed]))
        sum_squares = float(np.sum(np.square(deviations)))
        if num_periods > 1 and sum_squares > 0.0:
            innovations = deviations * np.sqrt((num_periods - 1) / sum_squares)
        else:
            innovations = np.zeros(num_periods)

        increments[start + 1:end + 1] = (interval_return / num_periods
                                         + vol_adjustment * np.sqrt(per_period_variance)
                                         * innovations)

    if not is_to_log_returns:
        increments = np.expm1(increments)
    return pd.Series(increments, index=pivot_returns.index, name=infrequent_returns.name)


def _add_price_return_anchors(price_returns: pd.DataFrame,
                              prices: pd.DataFrame,
                              prior_returns: Optional[pd.DataFrame] = None
                              ) -> None:
    """Retain first observed provider prices as zero-return NAV anchors.

    Args:
        price_returns: Fresh return frame to update in place.
        prices: Provider prices whose first finite level may require an anchor.
        prior_returns: Earlier-provider returns that can already anchor a newer history.
    """
    for column, price_series in prices.items():
        if price_series.isna().all():
            continue
        first_price_date = cast(pd.Timestamp, dfo.get_nonnan_index(price_series))
        has_prior_anchor = False
        if prior_returns is not None and column in prior_returns.columns:
            prior_return_series = prior_returns[column]
            prior_observations = prior_return_series.loc[:first_price_date]
            has_prior_anchor = bool(prior_observations.notna().any())
        if not has_prior_anchor:
            # A zero return preserves the supplied level without inventing price movement.
            price_returns.loc[first_price_date, column] = 0.0


def bfill_timeseries(df_newer: Union[pd.DataFrame, pd.Series],  # more recent data
                     df_older: Union[pd.DataFrame, pd.Series],  # older price is preserved to the end
                     freq: str = 'B',
                     fill_method: Optional[str] = None,  # None, 'to_zero', or 'ffill' for returns
                     is_prices: bool = False
                     ) -> Union[pd.DataFrame, pd.Series]:
    """Extend newer time series backward with older provider histories.

    For price inputs, a provider's first observed level supplies a zero-return NAV anchor when no
    usable earlier return exists. A newer all-missing DataFrame column without an older
    counterpart remains entirely missing.

    Args:
        df_newer: Newer Series or DataFrame whose labels and columns define the output. Its rows
            are interpreted in increasing date order without modifying the caller's object. A
            nonempty provider requires a ``DatetimeIndex`` without ``NaT``. A zero-row object
            retains its declared schema; a zero-column DataFrame is invalid.
        df_older: Older object of the same pandas type, used before each newer history begins. Its
            rows are also interpreted in increasing date order without modifying the caller. A
            nonempty provider requires a ``DatetimeIndex`` without ``NaT``. A zero-row object or
            zero-column DataFrame is treated as an unavailable provider.
        freq: Frequency of the returned date grid.
        fill_method: Return-gap policy. ``None`` preserves missing returns, ``'to_zero'`` fills
            missing returns with zero after each column begins, and ``'ffill'`` carries its last
            observed return forward. For price inputs, the policy is applied in return space.
        is_prices: Whether the supplied data are price levels. First observed provider prices are
            retained when no earlier return can anchor them, and expanded grids carry the last
            observed level forward.

    Returns:
        Chronologically ordered backfilled data on the requested grid with matching frequency
        metadata, preserving the newer input's pandas type, labels, and column order. If one
        provider has no observations, the available provider supplies the result under the newer
        schema and existing frequency and fill policies; if both have no observations, an owned
        empty object with that schema is returned.

    Raises:
        TypeError: If a nonempty provider does not use a ``DatetimeIndex``.
        ValueError: If a nonempty provider index contains ``NaT``, the newer DataFrame has no
            columns, or ``fill_method`` is not ``None``, ``'to_zero'``, or ``'ffill'``.
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

    # Only providers contributing observations need a valid calendar axis; empty schemas remain.
    dfr.validate_calendar_index(df_newer, argument_name="df_newer")
    dfr.validate_calendar_index(df_older, argument_name="df_older")

    # Provider row order carries no information; boundaries and fills must run chronologically.
    if not df_newer.index.is_monotonic_increasing:
        df_newer = df_newer.sort_index()
    if not df_older.index.is_monotonic_increasing:
        df_older = df_older.sort_index()

    if df_newer.shape[1] == 0:
        raise ValueError("df_newer must contain at least one column")

    newer_has_no_rows = len(df_newer.index) == 0
    older_has_no_data = len(df_older.index) == 0 or df_older.shape[1] == 0
    if newer_has_no_rows and older_has_no_data:
        # With no observations, the newer declaration is the complete output contract.
        empty_result = df_newer.asfreq(freq)
        if is_series_out:
            empty_series = cast(pd.Series, empty_result.iloc[:, 0])
            return empty_series
        return empty_result
    if older_has_no_data:
        # Reuse the established single-provider path without letting empty dates affect the grid.
        df_older = df_newer
    elif newer_has_no_rows:
        # The newer schema still controls labels, order, and dtypes when only older data exist.
        df_newer = df_newer.reindex(index=df_older.index)
        df_newer.update(df_older)
        df_older = df_newer

    # Price reconstruction must retain dates even when their calculated return is missing.
    provider_index = df_newer.index.union(df_older.index).sort_values()
    
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

        df_newer = cast(pd.DataFrame, ret.to_returns(newer_prices))
        df_older = cast(pd.DataFrame, ret.to_returns(older_prices))
        # Anchor older prices first so only genuinely unanchored newer histories are initialized.
        _add_price_return_anchors(price_returns=df_older, prices=older_prices)
        _add_price_return_anchors(price_returns=df_newer,
                                  prices=newer_prices,
                                  prior_returns=df_older)
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
        # Preserve price dates whose missing returns still delimit a valid provider history.
        bfill_datas = bfill_datas.reindex(provider_index)
        bfill_datas = ret.returns_to_nav(returns=bfill_datas,
                                         init_period=None,
                                         terminal_value=terminal_value)
        # Restore trailing provider dates removed by between-NaN NAV cleanup before fallback fill.
        bfill_datas = bfill_datas.reindex(provider_index)
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
    elif cast(pd.DatetimeIndex, bfill_datas.index).freq is None:
        # A regular result should advertise the requested cadence regardless of input metadata.
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
        df_newer: Newer Series or DataFrame. Its Series name or DataFrame columns, axis names,
            column order, and declared dtypes define the result. Rows are interpreted in
            increasing index order without modifying the caller's object. A zero-row object may
            still declare the result, but a DataFrame must contain at least one column.
        df_older: Older object of the same pandas type. A Series is interpreted under the newer
            Series name. Nonempty DataFrame columns must be unique and a subset of the newer
            columns; missing newer columns remain missing in the historical prefix. A zero-row or
            zero-column older DataFrame is treated as unavailable. Rows are interpreted in
            increasing index order without modifying the caller's object.
        numerical_check_columns: Columns for which to return mean absolute differences over the
            aligned overlap after duplicate dates use the same stable keep-last selection as the
            returned splice. Labels must belong to the newer schema and their requested order is
            preserved. ``None`` skips the diagnostic.

    Returns:
        A chronologically ordered appended object matching the input pandas type, plus the
        optional overlap-difference Series. Newer values take precedence on shared dates;
        duplicate dates within a provider retain the last supplied observation. An unavailable
        provider returns an independently owned result under the newer declaration and no overlap
        diagnostic.

    Raises:
        ValueError: If the inputs have different pandas types, the newer DataFrame has no columns,
            either DataFrame has duplicate columns, older columns fall outside the newer schema,
            or a diagnostic label falls outside the newer schema.
    """
    is_series = False
    series_name = None
    if isinstance(df_newer, pd.Series) and isinstance(df_older, pd.Series):
        is_series = True
        series_name = df_newer.name
        df_newer = df_newer.to_frame()
        # Series values are structurally compatible regardless of their provider-specific names.
        df_older = df_older.to_frame()
        df_older.columns = df_newer.columns.copy()
    elif isinstance(df_newer, pd.DataFrame) and isinstance(df_older, pd.DataFrame):
        pass
    else:
        raise ValueError(f"{type(df_older)} not aligned with {type(df_newer)}")

    # Validate provider declarations before endpoint access or pandas alignment can fail indirectly.
    newer_index_name = df_newer.index.name
    newer_columns_name = df_newer.columns.name
    if len(df_newer.columns) == 0:
        raise ValueError("df_newer must contain at least one column")
    if not df_newer.columns.is_unique:
        raise ValueError("df_newer columns must be unique")
    if not df_older.columns.is_unique:
        raise ValueError("df_older columns must be unique")

    # Row storage order carries no information; boundaries and overlap checks are chronological.
    if not df_newer.index.is_monotonic_increasing:
        df_newer = df_newer.sort_index(kind='stable')
    if not df_older.index.is_monotonic_increasing:
        df_older = df_older.sort_index(kind='stable')

    # Resolve each provider's date mapping before boundary checks and overlap diagnostics.
    if not df_newer.index.is_unique:
        df_newer = df_newer.iloc[~df_newer.index.duplicated(keep='last')]
    if not df_older.index.is_unique:
        df_older = df_older.iloc[~df_older.index.duplicated(keep='last')]

    older_has_no_data = len(df_older.index) == 0 or len(df_older.columns) == 0
    if not older_has_no_data:
        missing_older_columns = [column for column in df_older.columns
                                 if column not in df_newer.columns]
        if missing_older_columns:
            raise ValueError(f"df_older columns not found in df_newer: {missing_older_columns}")
    if numerical_check_columns is not None:
        missing_numerical_columns = [column for column in numerical_check_columns
                                     if column not in df_newer.columns]
        if missing_numerical_columns:
            raise ValueError("numerical_check_columns not found in df_newer: "
                             f"{missing_numerical_columns}")

    # Start from an owned newer result, then align any usable older provider before splicing.
    new_df = df_newer.copy()
    diff = None
    if not older_has_no_data:
        df_older = df_older.reindex(columns=df_newer.columns)

    if older_has_no_data:
        pass
    elif len(df_newer.index) == 0:
        # Preserve newer-declared dtypes while filling them from the available older history.
        new_df = df_newer.reindex(index=df_older.index)
        new_df.update(df_older)
        diff = None
    # No append is needed when the older provider cannot extend the newer history backwards.
    elif df_older.index[0] >= df_newer.index[0]:
        new_df = df_newer.copy()
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

    # Restore the newer provider's public metadata after concatenation and duplicate removal.
    new_df.index.name = newer_index_name
    new_df.columns.name = newer_columns_name
    if is_series:
        new_df = new_df.iloc[:, 0].copy()
        new_df.name = series_name

    return new_df, diff

"""
weights from a score panel: row normalisations, equal weighting and long-short indicators.

``df_to_weight_allocation_sum1`` is the general normaliser - each row divided by its nansum, so
signs survive and a row holding shorts still sums to one. ``df_to_long_only_allocation_sum1``
zeroes negatives first and returns an all-zero row when nothing is positive.
``df_to_equal_weight_allocation`` weights over the assets live on each date: a nan excludes an
asset from that date's denominator rather than counting it as a zero weight.

``df_to_top_bottom_n_indicators`` returns +1 for the ``num_top_assets`` largest values in a row,
-1 for the smallest and 0 between - indicators, not weights summing to one, so they need
normalising before use as an allocation.

``generate_static_weights_schedule`` is the one function here that takes prices rather than
scores: it turns a fixed allocation into the rebalancing weight frame the backtester consumes,
allocating over the instruments live on each rebalancing date. ``align_weights_to_columns`` is
the shared normaliser for the weight argument that both it and ``qis.backtest_model_portfolio``
accept, so the two cannot disagree on what a Dict, a List or an np.ndarray means.

Turning a weight frame into a portfolio is ``qis.backtest_model_portfolio``, not this module.
"""
import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, List

# qis
import qis.utils.df_groups as dfg
from qis.utils.dates import generate_rebalancing_indicators
from qis.utils.struct_ops import assert_list_subset


def align_weights_to_columns(weights: Union[Dict[str, float], List[float], np.ndarray,
                                            pd.Series],
                             columns: pd.Index
                             ) -> pd.Series:
    """
    align a weight specification to a set of instrument columns.

    A Dict or pd.Series is aligned by name, so the order of the specification does not matter and
    a column the specification does not name carries nan, which every consumer treats as no
    position. A List or np.ndarray is positional and must match the column count, since there is
    no name to align on. This is the normaliser :func:`generate_static_weights_schedule` and
    ``qis.backtest_model_portfolio`` share, so that one weight argument means one thing.

    Args:
        weights: target weights as a Dict or pd.Series keyed by ticker, or as a List or
            np.ndarray in column order
        columns: instrument columns to align to, normally ``prices.columns``

    Returns:
        weights indexed by ``columns``, nan where the specification does not name a column

    Raises:
        ValueError: if a named specification carries a name outside ``columns``, or a positional
            specification does not match the number of columns or is not one dimensional
        NotImplementedError: if ``weights`` is of an unsupported type
    """
    if isinstance(weights, pd.Series):  # map to dict
        weights = weights.to_dict()

    if isinstance(weights, Dict):
        assert_list_subset(large_list=columns.to_list(),
                           list_sample=list(weights.keys()),
                           message="weights columns must be aligned with price columns")
        return pd.Series(columns.map(weights), index=columns, dtype=float)

    if isinstance(weights, List):  # map to np
        weights = np.array(weights)

    if isinstance(weights, np.ndarray):
        if weights.shape[0] != len(columns):
            raise ValueError("number of weights must be aligned with number of price columns")
        if len(weights.shape) > 1:
            raise ValueError("only single aray is allowed")
        return pd.Series(weights, index=columns, dtype=float)

    raise NotImplementedError(f"unsupported weights type = {type(weights)}")


def generate_static_weights_schedule(prices: pd.DataFrame,
                                     weights: Union[Dict[str, float], List[float], np.ndarray,
                                                    pd.Series],
                                     rebalancing_freq: str = 'QE',
                                     is_rescale_to_live_universe: bool = True,
                                     is_preserve_total_exposure: bool = True,
                                     include_start_date: bool = True,
                                     num_warmup_periods: Optional[int] = None
                                     ) -> pd.DataFrame:
    """
    turn a fixed allocation into a rebalancing weight frame over the live universe.

    A static allocation stated once cannot be applied to a panel whose instruments start and stop
    at different dates: an instrument with no price on a rebalancing date is not traded, and
    ``qis.backtest_model_portfolio`` leaves its weight in the cash balance, which is the correct
    contract and rarely the intended allocation. This reallocates that weight over the
    instruments that are priced on each rebalancing date, and returns the frame to pass back in
    as ``weights``.

    The universe is read at the rebalancing date and nowhere else, so the construction is point
    in time: a price observed at *t* is information available at *t*. Instruments are
    reallocated at rebalancings only - an instrument going live between two rebalancings is
    admitted at the next one, not when it starts.

    Rescaling preserves the total exposure of the specification rather than forcing the row to
    one, so a book that is 90% invested by design stays 90% invested when an instrument is
    missing instead of being silently levered to 100%. The two are the same whenever the
    specification sums to one.

    Args:
        prices: instrument prices, columns are tickers. A nan marks an instrument as not
            allocable on that date
        weights: target weights, aligned to ``prices.columns`` by :func:`align_weights_to_columns`
        rebalancing_freq: calendar anchor for the rebalancing dates, passed to
            :func:`generate_rebalancing_indicators`. The returned dates are the observation dates
            of ``prices``, so the frame can never rebalance more often than the price panel
        is_rescale_to_live_universe: reallocate the weight of a missing instrument over the live
            ones. False returns the specification with missing instruments set to zero, which is
            the cash-residual behaviour with an explicit 0.0 rather than a nan in the reported
            weights
        is_preserve_total_exposure: scale the live weights to the total exposure of
            ``weights``. False forces every row to sum to one
        include_start_date: mark the first price date as a rebalancing date, so the allocation
            is invested from inception. Note this defaults to True where the corresponding
            ``is_rebalanced_at_first_date`` of ``qis.backtest_model_portfolio`` defaults to False
        num_warmup_periods: leading observations excluded from the schedule

    Returns:
        weights indexed by the rebalancing dates that exist in ``prices.index``, columns as
        ``prices.columns``. A date with no live instrument carries all zeros

    Raises:
        ValueError: if ``prices`` is not a pd.DataFrame, if the specification has no net
            exposure to preserve, or if the live weights cancel on a rebalancing date so that
            the rescaling is undefined
    """
    if not isinstance(prices, pd.DataFrame):
        raise ValueError(f"prices type={type(prices)} must be pd.Dataframe")

    static_weights = align_weights_to_columns(weights=weights, columns=prices.columns).fillna(0.0)
    rebalancing_dates = generate_rebalancing_indicators(df=prices,
                                                        freq=rebalancing_freq,
                                                        include_start_date=include_start_date,
                                                        num_warmup_periods=num_warmup_periods,
                                                        return_true_only=True).index
    is_live = prices.loc[rebalancing_dates, :].notna()
    live_weights = is_live.multiply(static_weights, axis=1)  # zero where an instrument is unpriced

    if is_rescale_to_live_universe:
        target_exposure = float(static_weights.sum()) if is_preserve_total_exposure else 1.0
        if np.isclose(target_exposure, 0.0):
            raise ValueError(f"weights sum to {target_exposure} so there is no total exposure to "
                             f"preserve: rescaling a book with no net exposure is undefined, pass "
                             f"is_rescale_to_live_universe=False")
        live_sums = live_weights.sum(axis=1).to_numpy()
        is_cancelling = np.logical_and(is_live.to_numpy().any(axis=1),
                                       np.isclose(live_sums, 0.0))
        if np.any(is_cancelling):
            date = rebalancing_dates[is_cancelling][0]
            raise ValueError(f"live weights sum to {live_sums[is_cancelling][0]} at "
                             f"{date:%d%b%Y} while instruments are priced: rescaling to the "
                             f"target exposure {target_exposure} is undefined because the live "
                             f"weights cancel")
        scale = np.divide(target_exposure, live_sums,
                          out=np.zeros_like(live_sums),
                          where=np.isclose(live_sums, 0.0) == False)
        live_weights = live_weights.multiply(scale, axis=0)

    return live_weights


def df_to_equal_weight_allocation(df: Union[pd.Series, pd.DataFrame],
                                  freq: str = None,
                                  index: pd.DatetimeIndex = None
                                  ) -> Union[pd.Series, pd.DataFrame]:
    """
    equal weights across the assets observed on each date.

    Equal-weight over the live universe rather than over the column count: an asset that has not
    started or has been delisted carries a NaN and is excluded from that date's denominator, so the
    weights still sum to one as the universe changes.

    Args:
        df: panel whose non-NaN entries define which assets are live on each date
        freq: resample the weights to this frequency, forward-filling. None keeps the input dates
        index: reindex the weights onto this index, forward-filling. Ignored when ``freq`` is given

    Returns:
        weights in the same shape, each row summing to one over the live assets
    """
    equal_weight_allocation = df_to_weight_allocation_sum1(df=df_nans_to_one_zero(df=df))
    if freq is not None:
        equal_weight_allocation = equal_weight_allocation.asfreq(freq, method='ffill')
    elif index is not None:
        equal_weight_allocation = equal_weight_allocation.reindex(index, method='ffill')
    return equal_weight_allocation


def df_to_weight_allocation_sum1(df: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """
    normalise each row to sum to one.

    Signed values are kept, so a row with short positions produces weights summing to one that
    include negatives - this is the general normaliser. Use
    :func:`df_to_long_only_allocation_sum1` when negatives should be dropped instead.

    Args:
        df: scores or notionals, one column per asset. NaN is treated as zero in the row sum and
            the resulting weight is zero

    Returns:
        weights in the same shape, each row summing to one
    """
    if isinstance(df, pd.Series):
        weights = df.divide(np.nansum(df.to_numpy(dtype=float), axis=0)).fillna(0.0)
    else:
        row_sums = np.nansum(df.to_numpy(dtype=float), axis=1, keepdims=True)
        weights = df.divide(row_sums).fillna(0.0)
    return weights


def df_to_long_only_allocation_sum1(df: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """
    normalise each row to sum to one after setting negative values to zero.

    The long-only counterpart of :func:`df_to_weight_allocation_sum1`: negatives are dropped before
    normalising rather than after, so the surviving positives carry the full weight. A row with no
    positive value returns all zeros rather than dividing by zero.

    Args:
        df: scores, one column per asset. Non-positive entries and NaN become zero weight

    Returns:
        non-negative weights in the same shape, each row summing to one or to zero
    """
    weighted_score = np.where(df.to_numpy() > 0.0, df, 0.0)
    if isinstance(df, pd.Series):
        weights = weighted_score / np.nansum(weighted_score)
        weights = pd.Series(weights, index=df.index).fillna(0.0)
    else:
        row_sums = np.nansum(weighted_score, axis=1, keepdims=True)
        # NumPy 2.x: use out= to avoid division-by-zero artifacts in all-nonpositive rows.
        weights = np.divide(
            weighted_score, row_sums,
            out=np.zeros_like(weighted_score, dtype=float),
            where=row_sums != 0.0,
        )
        weights = pd.DataFrame(weights, index=df.index, columns=df.columns).fillna(0.0)
    return weights


def df_nans_to_one_zero(df: pd.DataFrame) -> pd.DataFrame:
    """
    return 1 if is set is not None and 0 otherwise
    """
    data = np.where(np.isfinite(df.to_numpy(dtype=np.float64)), 1.0, 0.0)
    return pd.DataFrame(data=data, index=df.index, columns=df.columns)


def mult_df_columns_with_vector(df: pd.DataFrame,
                                vector: pd.Series,
                                is_norm: bool = False,
                                nan_fill_zero: bool = False
                                ) -> pd.DataFrame:
    """
    multiply data set with vector column-wise and normalize accounting for nans in data
    data can be indicators data with False/True
    """
    conv = df.multiply(vector)

    if is_norm:  # nan sum of row of all nans is zero
        nump_data = conv.to_numpy(dtype=np.float64)
        column_sum = np.nansum(nump_data, axis=1, keepdims=True)  # column vector: column_sum=0.0 if all rows are nans
        nan_ind = np.all(np.isnan(nump_data), axis=1, keepdims=True)  # column vector to trace rows with all nans
        div_cond = np.logical_and(np.isclose(column_sum, 0.0) == False, nan_ind == False)
        # NumPy 2.x: divide on ndarray with explicit out buffer, then rebuild the DataFrame.
        divided = np.divide(
            nump_data, column_sum,
            out=np.full_like(nump_data, np.nan),
            where=div_cond,
        )
        conv = pd.DataFrame(divided, index=conv.index, columns=conv.columns)
        if np.any(nan_ind):
            conv.loc[nan_ind.flatten(), :] = np.nan  # rows with nans = nans

    if nan_fill_zero:
        conv = conv.fillna(0)

    return conv


def mult_df_columns_with_vector_group(df: pd.DataFrame,
                                      vector: pd.Series,
                                      group_data: pd.Series,
                                      is_norm: bool = False,
                                      nan_fill_zero: bool = False,
                                      return_df: bool = False
                                      ) -> Union[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    convolve/multiply data set with vector column-wise and normalize with grouping accounting for nans in data
    """
    group_dict = dfg.split_df_by_groups(df=df, group_data=group_data)
    group_conv = {}
    for group, g_data in group_dict.items():
        group_conv[group] = mult_df_columns_with_vector(df=g_data,
                                                        vector=vector.loc[g_data.columns],
                                                        is_norm=is_norm,
                                                        nan_fill_zero=nan_fill_zero)
    if return_df:
        group_conv = pd.concat([v for k, v in group_conv.items()], axis=1, sort=True)[df.columns]

    return group_conv


def df_to_top_bottom_n_indicators(df: Union[pd.Series, pd.DataFrame],
                                  num_top_assets: int = 15,
                                  is_top_and_bottom: bool = True
                                  ) -> Union[pd.Series, pd.DataFrame]:
    """
    assign unit weight to ranked rows at most num_top_assets
    nan values are ignored
    """
    if len(df.columns) < 2*num_top_assets:
        raise ValueError(f"{len(df.columns)} must exceed {2*num_top_assets}")

    def series_to_top_n_indicators(data: pd.Series) -> pd.Series:
        ranked_row_with_nans = data.sort_values(ascending=False)
        ranked_row_without_nans = ranked_row_with_nans.dropna()
        if is_top_and_bottom:
            ranked_row_without_nans[:num_top_assets] = 1.0  # top
            ranked_row_without_nans[-num_top_assets:] = -1.0  # bottom
            ranked_row_without_nans[num_top_assets:-num_top_assets] = 0.0  # mid  will overwight boundaries
        else:
            ranked_row_without_nans[:num_top_assets] = 1.0  # top
            ranked_row_without_nans[num_top_assets:] = 0.0
        ranked_row = ranked_row_without_nans.reindex(index=ranked_row_with_nans.index)
        ranked_row = ranked_row.sort_index()
        return ranked_row

    if isinstance(df, pd.Series):
        ranked_data = series_to_top_n_indicators(data=df)[df.index]
    else:
        columns = df.columns.copy()
        ranked_rows = {}
        for idx, row in df.iterrows():
            ranked_rows[idx] = series_to_top_n_indicators(data=row)
        ranked_data = pd.DataFrame.from_dict(ranked_rows, orient='index')[columns]
    return ranked_data

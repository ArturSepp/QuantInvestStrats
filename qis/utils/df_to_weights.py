"""
weights from a score panel: row normalisations, equal weighting and long-short indicators.

``df_to_weight_allocation_sum1`` is the general normaliser - each row divided by its nansum, so
signs survive and a row holding shorts still sums to one. ``df_to_long_only_allocation_sum1``
zeroes negatives first and returns an all-zero row when nothing is positive.
``df_to_equal_weight_allocation`` weights over the assets live on each date: a nan excludes an
asset from that date's denominator rather than counting it as a zero weight.

``df_to_top_bottom_n_indicators`` returns +1 for the ``num_top_assets`` largest values in a row,
-1 for the smallest and 0 between - indicators, not weights summing to one, so they need
normalising before use as an allocation. ``compute_long_short_ind_by_row`` also lives here.

Turning a weight frame into a portfolio is ``qis.backtest_model_portfolio``, not this module.
"""
import numpy as np
import pandas as pd
from typing import Union, Optional, Dict
from enum import Enum

# qis
import qis.utils.df_groups as dfg


class WeightMethod(str, Enum):
    EQUAL_WEIGHT = 'EqualWight'
    PROPORTIONAL = 'Proportional'
    SQRT_PROPORTIONAL = 'Sqrt '


def compute_long_only_portfolio_weights(data: pd.DataFrame,
                                        weight_method: WeightMethod = WeightMethod.EQUAL_WEIGHT
                                        ) -> pd.DataFrame:
    """
    compute [0, 1] weights data
    """
    if weight_method == WeightMethod.EQUAL_WEIGHT:
        weights = df_to_equal_weight_allocation(df=data)
    elif weight_method == WeightMethod.PROPORTIONAL:
        weights = df_to_weight_allocation_sum1(df=data)
    elif weight_method == WeightMethod.SQRT_PROPORTIONAL:
        weights = df_to_weight_allocation_sum1(df=np.sqrt(data))
    else:
        raise TypeError(f"not implemented method {weight_method}")
    return weights


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
        group_conv = pd.concat([v for k, v in group_conv.items()], axis=1)[df.columns]

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


def fill_long_short_signal(rank_data: np.ndarray,
                           leg_size: int
                           ) -> np.ndarray:
    """
    place +1 for top, -1 for bottom
    """
    signal0 = np.zeros_like(rank_data)
    signal1 = np.ones_like(rank_data)
    bottom_quantile = leg_size
    upper_quantile = rank_data.shape[0] - leg_size - 1
    signal = np.where(rank_data > upper_quantile, signal1, signal0)  # +1 top ranks
    signal = np.where(rank_data < bottom_quantile, -signal1, signal)  # -1 smallest quantile
    return signal


def compute_long_short_ind_by_row(data: pd.DataFrame,
                                  cut_off_quantile: float = 0.1,
                                  min_leg_size: int = 2,
                                  leg_size: Optional[int] = None
                                  ) -> pd.DataFrame:
    """
    get cross sectional indicators
    """
    if len(data.columns) == 1:
        raise ValueError(f"one column is not supported for ranking")
    elif len(data.columns) == 2:
        leg_size = 0
    else:
        pass

    # compute quantiles
    if leg_size is None:
        leg_size = np.maximum(np.floor(cut_off_quantile * len(data.columns)),
                              min_leg_size)

    rank_data = data.rank(axis=1, method='first', na_option='keep', ascending=False)

    signal = fill_long_short_signal(rank_data=rank_data.to_numpy(), leg_size=leg_size)
    cross_sectional_indicators = pd.DataFrame(data=signal, columns=data.columns, index=data.index)

    return cross_sectional_indicators


def compute_long_short_ind(data: np.ndarray,
                           cut_off_quantile: float = 0.1,
                           min_leg_size: int = 2,
                           leg_size: Optional[int] = None
                           ) -> np.ndarray:
    """
    row wise operator to assing -/+1 for ranked data in df
    """
    if not data.ndim == 1:
        raise ValueError(f"ndim must be 1")

    n = data.shape[0]
    # compute quantiles
    if leg_size is None:
        leg_size = np.maximum(np.floor(cut_off_quantile * n), min_leg_size)

    rank_data = pd.DataFrame(data).rank(axis=1, method='first', na_option='keep', ascending=False)

    signal = fill_long_short_signal(rank_data=rank_data.to_numpy(), leg_size=leg_size)

    return signal

"""
different methods to generate signal and portfolio weights from df
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


def get_weights(data: pd.DataFrame, weight_method: WeightMethod = WeightMethod.EQUAL_WEIGHT) -> pd.DataFrame:
    if weight_method == WeightMethod.EQUAL_WEIGHT:
        weights = df_to_equal_weight_allocation(df=data)
    elif weight_method == WeightMethod.PROPORTIONAL:
        weights = df_to_weight_allocation_sum1(df=data)
    elif weight_method == WeightMethod.SQRT_PROPORTIONAL:
        weights = df_to_weight_allocation_sum1(df=np.sqrt(data))
    else:
        raise TypeError(f"not implemented method {weight_method}")
    return weights


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
        conv = np.divide(conv, column_sum, where=div_cond)  # divide by column sum where possible
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


def df_to_equal_weight_allocation(df: pd.DataFrame) -> pd.DataFrame:
    indicator = np.isfinite(df).astype(np.float64)  # 0.0 or 1.0
    equal_weight_allocation = mult_df_columns_with_vector(df=indicator,
                                                          vector=pd.Series(1.0, index=df.columns),
                                                          is_norm=True,
                                                          nan_fill_zero=True)
    return equal_weight_allocation


def df_to_max_score(df: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """
    normalized rows by cross-sectional max
    """
    if isinstance(df, pd.Series):
        score = df.divide(np.nanmax(df, axis=0))
    else:
        score = df.divide(np.nanmax(df, axis=1, keepdims=True))
    return score


def df_to_top_n_indicators(df: Union[pd.Series, pd.DataFrame],
                           num_top_assets: int = 15
                           ) -> Union[pd.Series, pd.DataFrame]:
    """
    assign unit weight to ranked rows
    """
    def series_to_top_n_indicators(data: pd.Series) -> pd.Series:
        ranked_row = data.sort_values(ascending=False)
        ranked_row[:num_top_assets], ranked_row[num_top_assets:] = 1.0, 0.0
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


def df_nans_to_one_zero(df: pd.DataFrame) -> pd.DataFrame:
    """
    return 1 if is set is not None and 0 otherwise
    """
    data = np.where(np.isfinite(df.to_numpy(dtype=np.float64)), 1.0, 0.0)
    return pd.DataFrame(data=data, index=df.index, columns=df.columns)


def df_to_weight_allocation_sum1(df: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """
    normalized rows by cross-sectional sum
    """
    if isinstance(df, pd.Series):
        weights = df.divide(np.nansum(df, axis=0)).fillna(0.0)
    else:
        weights = df.divide(np.nansum(df, axis=1, keepdims=True)).fillna(0.0)
    return weights


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
    get cross sectional
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
    row wise operator
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


class UnitTests(Enum):
    CONV = 1
    EW_ALLOC = 2


def run_unit_test(unit_test: UnitTests):

    import qis.utils.df_ops as dfo

    if unit_test == UnitTests.CONV:
        constituent_dict = {'f1': np.array((0, np.nan, 1, 1)),
                            'f2': np.array((0, np.nan, np.nan, 1)),
                            'f3': np.array((0, np.nan, 1, 1)),
                            'f4': np.array((0, np.nan, 1, 1)),
                            'f5': np.array((0, np.nan, np.nan, 1)),
                            'f6': np.array((0, np.nan, 1, 1))}

        constituent_prices = pd.DataFrame.from_dict(constituent_dict, orient='columns')
        constituent_data = dfo.df_indicator_like(constituent_prices)
        print(constituent_data)

        desc_dict = {'f1': ('eq', 1.0),
                     'f2': ('fi', 0.5),
                     'f3': ('fi', 1.5),
                     'f4': ('fx', 1.0),
                     'f5': ('fx', 2.0),
                     'f6': ('fx', 3.0)}

        desc_data = pd.DataFrame.from_dict(desc_dict, orient='index', columns=['ac', 'acw'])
        print(desc_data)

        conv = mult_df_columns_with_vector(df=constituent_data, vector=desc_data['acw'])
        print(f"conv:\n{conv}")

        conv_norm = mult_df_columns_with_vector(df=constituent_data, vector=desc_data['acw'], is_norm=True)
        print(f"conv_norm:\n{conv_norm}")

        ac_conv = mult_df_columns_with_vector_group(df=constituent_data, vector=desc_data['acw'],
                                                    group_data=desc_data['ac'])
        for ac, data in ac_conv.items():
            print(f"{ac}:\n{data}")

        ac_conv = mult_df_columns_with_vector_group(df=constituent_data, vector=desc_data['acw'],
                                                    group_data=desc_data['ac'],
                                                    is_norm=True)
        for ac, data in ac_conv.items():
            print(f"norm {ac}:\n{data}")

        ac_conv_pd = mult_df_columns_with_vector_group(df=constituent_data, vector=desc_data['acw'],
                                                       group_data=desc_data['ac'],
                                                       is_norm=True,
                                                       return_df=True)
        print(f"ac_conv_pd:\n{ac_conv_pd}")

    elif unit_test == UnitTests.EW_ALLOC:
        constituent_dict = {'f1': np.array((0, np.nan, 2, 3)),
                            'f2': np.array((0, np.nan, np.nan, 4)),
                            'f3': np.array((0, np.nan, 1, 1)),
                            'f4': np.array((0, np.nan, 1, 1)),
                            'f5': np.array((0, np.nan, np.nan, 1)),
                            'f6': np.array((0, np.nan, 1, 1))}

        constituent_prices = pd.DataFrame.from_dict(constituent_dict, orient='columns')
        print(constituent_prices)
        weights = df_to_equal_weight_allocation(constituent_prices)
        print(weights)


if __name__ == '__main__':

    unit_test = UnitTests.CONV

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

"""
common pandas operations
"""
# built in
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, List, Optional, Dict, Tuple, Type, Any, Literal
from enum import Enum

# qis
import qis.utils.np_ops as npo
import qis.utils.struct_ops as sop


def df_zero_like(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(data=np.zeros(df.shape), index=df.index, columns=df.columns)


def df_ones_like(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(data=np.ones(df.shape), index=df.index, columns=df.columns)


def df_indicator_like(df: pd.DataFrame, type: Type = bool) -> pd.DataFrame:
    """
    return df_indicator = True is set is not None and False otherwise
    """
    data = np.where(np.isnan(df.to_numpy(dtype=np.float64)), False, True).astype(type)
    return pd.DataFrame(data=data, index=df.index, columns=df.columns)


def dfs_indicators(dfs: List[pd.DataFrame], type: Type = bool) -> pd.DataFrame:
    """
    return df_indicator = True if list dfs is not none
    nb: need to check that dfs are aligned
    """
    def to_indicator(df: pd.DataFrame) -> np.ndarray:
        return np.where(np.isnan(df.to_numpy(dtype=np.float64)), False, True)
    indicators = to_indicator(df=dfs[0])
    for df in dfs[1:]:
        indicators = np.logical_and(indicators, to_indicator(df=df))
    return pd.DataFrame(data=indicators, index=dfs[0].index, columns=dfs[0].columns).astype(type)


def df_filter_like(df: pd.DataFrame,
                   df_filter: pd.DataFrame
                   ) -> pd.DataFrame:
    """
    take a subset of set with data conditioned on non nans in
    """
    df_filter = df_filter[df.columns].reindex(index=df.index)
    df_out = df.where(cond=df_indicator_like(df=df_filter), other=np.nan)
    return df_out


def df_joint_indicator(indicator1: pd.DataFrame,
                       indicator2: Union[np.ndarray, pd.DataFrame],
                       to_float: bool = False
                       ) -> pd.DataFrame:
    """
    return df_indicator = True if indicator1 is True and indicator2 is True
    """
    if isinstance(indicator2, pd.DataFrame):
        np_indicator2 = indicator2.to_numpy(dtype=bool)
    elif isinstance(indicator2, np.ndarray):
        np_indicator2 = indicator2.astype(bool)
    else:
        raise ValueError(f"unsupported type {type(indicator2)}")

    data = np.logical_and(indicator1.to_numpy(dtype=bool), np_indicator2)

    if to_float:
        data = data.astype(np.float64)

    return pd.DataFrame(data=data, index=indicator1.index, columns=indicator1.columns)


def df_indicator_like_other(indicator: pd.DataFrame,
                            other: pd.DataFrame
                            ) -> pd.DataFrame:
    """
    return set = other if indicator is True and np.nan otherwise
    """
    data = np.where(indicator.to_numpy(dtype=bool), other.to_numpy(), np.nan)
    return pd.DataFrame(data=data, index=other.index, columns=other.columns)


def series_to_dict(ds: pd.Series, inverse: bool = False) -> Dict:
    """
    convert pd series to dict {index: value} and reverse
    important only uniue values
    """
    sop.assert_list_unique(lsdata=ds.index.to_list())
    data_dict = ds.to_dict()
    if inverse:
        sop.assert_list_unique(lsdata=ds.to_list())  # values need to be unique
        data_dict = {v: k for k, v in data_dict.items()}
    return data_dict


def norm_df_by_ax_mean(data: Union[np.ndarray, pd.DataFrame],
                       proportiontocut: Optional[float] = None,  # for treaming mean,
                       is_zero_mean: bool = True,
                       axis: int = 1
                       ) -> pd.DataFrame:
    """
    to unit norm data
    axis: 0: time-series row-wise mean; 1: cross-sectional columns-wise mean
    """

    if isinstance(data, pd.DataFrame):
        a = data.to_numpy()
    else:
        a = data

    if is_zero_mean:
        if proportiontocut is not None:
            mean = npo.np_array_to_df_columns(a=stats.trim_mean(a=a,
                                                                proportiontocut=proportiontocut, axis=axis),
                                              n_col=len(data.columns))
        else:
            mean = np.nanmean(a=a, axis=axis, keepdims=True)
        a = a - mean

    std = np.nanstd(a=a, axis=axis, ddof=1, keepdims=True)
    norm_data = np.divide(a, std, where=np.greater(std, 0.0))

    if isinstance(data, pd.DataFrame):
        norm_data = pd.DataFrame(data=norm_data, columns=data.columns, index=data.index)

    return norm_data


def get_first_before_non_nan_index(df: Union[pd.Series, pd.DataFrame],
                                   return_index_for_all_nans: int = -1  # last = -1, first = 0
                                   ) -> Union[pd.Timestamp, List[pd.Timestamp]]:
    """
    return first index if no nans in pandas
    """
    if isinstance(df, pd.DataFrame):
        first_before_non_nan_index = []
        for column in df:
            column_data = df[column]
            non_nan_data = column_data.loc[np.isnan(column_data.to_numpy()) == False]
            if not non_nan_data.empty:  # returns first index of nan
                fist_non_nan_index = non_nan_data.index[0]
            else:  # all values are nans so return the last index of original data
                fist_non_nan_index = df.index[return_index_for_all_nans]

            first_before_non_nan_index.append(fist_non_nan_index)

    elif isinstance(df, pd.Series):
        non_nan_data = df.loc[np.isnan(df.to_numpy()) == False]
        if not non_nan_data.empty:
            first_before_non_nan_index = non_nan_data.index[0]
        else:
            first_before_non_nan_index = df.index[return_index_for_all_nans]
    else:
        raise ValueError(f"unsoported data type = {type(df)}")

    return first_before_non_nan_index


def drop_first_nan_data(df: Union[pd.Series, pd.DataFrame],
                        is_oldest: bool = True  # the olderst, the eldest (all non nnans)
                        ) -> Union[pd.Series, pd.DataFrame]:
    """
    drop data before first nonnan either at max or at min for pandas
    the rest of data can still contain occasional nans
    """
    first_non_nan_index = get_first_before_non_nan_index(df=df)

    if isinstance(df, pd.DataFrame):
        if is_oldest:
            joint_start = min(first_non_nan_index)
        else:
            joint_start = max(first_non_nan_index)
        new_data = df[joint_start:].copy()
    else:
        new_data = df.loc[first_non_nan_index:].copy()

    return new_data


def get_first_last_non_nan_date(df: Union[pd.Series, pd.DataFrame], is_first: bool = True) -> Union[pd.Timestamp, np.ndarray]:
    """
    return pd.Timestamp or np.nan
    """
    def get_series_non_nan(ds: pd.Series) -> pd.Timestamp:
        null_ind = ds.isnull()
        good_ind = null_ind == False
        if np.all(good_ind):
            if is_first:  # no nans nans return first:
                fist_non_nan_index = ds.index[0]
            else:  # last
                fist_non_nan_index = ds.index[-1]
        elif np.all(null_ind):
            if is_first:  # all nans return last:
                fist_non_nan_index = ds.index[-1]
            else:
                fist_non_nan_index = ds.index[0]
        else:
            if is_first:
                fist_non_nan_index = ds[good_ind].index[0]
            else:
                fist_non_nan_index = ds[good_ind].index[-1]
        return fist_non_nan_index

    if isinstance(df, pd.Series):
        fist_non_nan_index = get_series_non_nan(df)
    elif isinstance(df, pd.DataFrame):
        values = []
        for column in df:
            values.append(get_series_non_nan(df[column]))
        fist_non_nan_index = np.array(values)
    else:
        raise ValueError(f"unsupported data type = {type(df)}")

    return fist_non_nan_index


def missing_zero_ratios_after_first_non_nan(df: Union[pd.Series, pd.DataFrame],
                                            zero_cutoff: float = 1e-12
                                            ) -> Tuple[np.ndarray, np.ndarray]:
    """
   compute ration of missing data
    """
    if isinstance(df, pd.Series):
        df = df.to_frame()

    missing = []
    zeros = []
    for column in df:
        column_data = df[column]
        nonnan_cond = column_data.isnull() == False
        fist_non_nan_index = column_data.loc[nonnan_cond].index[0]
        after_data = column_data.loc[fist_non_nan_index:]
        missing.append(after_data.isnull().sum() / len(after_data))
        zeros.append(after_data.abs().lt(zero_cutoff).sum() / len(after_data))

    missing_ratio = np.array(missing)
    zeros_ratio = np.array(zeros)

    return missing_ratio, zeros_ratio


def get_first_non_nan_values(df: Union[pd.Series, pd.DataFrame]) -> Union[np.ndarray, float]:

    if df.empty:
        raise ValueError(f"data is empty:\n {df}")

    def get_non_nan_values_series(sdata: pd.Series) -> float:
        x0 = sdata.iloc[0]
        if not np.isnan(x0):
            values = x0
        else:
            nonnan_column_data = sdata[~sdata.isnull()]
            if nonnan_column_data.empty:
                raise ValueError(f"all nans in {df}")
            values = nonnan_column_data[0]
        return values

    if isinstance(df, pd.DataFrame):
        x0 = df.iloc[0, :].to_numpy()
        if np.all(np.isnan(x0) == False):  # first entries are non nan -> most expected
            values = x0
        else:
            values = []
            for column in df:
                values.append(get_non_nan_values_series(sdata=df[column]))
            values = np.array(values)

    elif isinstance(df, pd.Series):
            values = get_non_nan_values_series(sdata=df)

    else:
        raise ValueError(f"unsupported data type = {type(df)}")

    return values


def get_last_non_nan_values(df: Union[pd.Series, pd.DataFrame]) -> Union[np.ndarray, float]:
    if df.empty:
        raise ValueError(f"data is empty:\n {df}")

    def get_non_nan_values_series(ds: pd.Series) -> float:
        x1 = ds.iloc[-1]
        if not pd.isna(x1):
            values = x1
        else:
            nonnan_column_data = ds[~ds.isnull()]
            if nonnan_column_data.empty:
                # print(f"in get_last_non_nan_values: all nans in {ds}")
                values = np.nan
            else:
                values = nonnan_column_data[-1]
        return values

    if isinstance(df, pd.DataFrame):
        x1 = df.iloc[-1, :].to_numpy()
        if np.all(pd.isna(x1) == False):  # first entries are non nan -> most expected
            values = x1
        else:
            values = []
            for column in df:
                values.append(get_non_nan_values_series(ds=df[column]))
            values = np.array(values)

    elif isinstance(df, pd.Series):
        values = get_non_nan_values_series(ds=df)

    else:
        raise ValueError(f"unsupported data type = {type(df)}")

    return values


def get_last_non_nan(df: Union[pd.Series, pd.DataFrame]) -> pd.Series:
    values = get_last_non_nan_values(df=df)
    if isinstance(df, pd.DataFrame):
        ds = pd.Series(values, index=df.columns)
    else:
        ds = pd.Series(values, index=df.name)
    return ds


def get_df_subset_index(df: pd.DataFrame,
                        sub_index_data: Union[pd.Series, pd.DataFrame]
                        ) -> pd.DataFrame:
    """
    get subset of pandas based on smaller index in sub_index_data
    """
    return df.iloc[np.in1d(df.index.to_list(), sub_index_data.index.to_list())]


def multiply_df_by_dt(df: Union[pd.DataFrame, pd.Series],
                      dates: Union[pd.DatetimeIndex, pd.Index] = None,
                      lag: Optional[int] = None,
                      is_actual_calendar_dt: bool = True,
                      af: float = 365.0
                      ) -> Union[pd.DataFrame, pd.Series]:
    """
    to compute rate adjustment with data - rate:
    get data at dates index and adjust by dt if needed
    adjust data by time spread:
    data = dt*data
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"data index must be DateTimeIndex: {df.index}")

    if lag is not None:
        if len(df.index) > lag:
            df = df.shift(lag)

    if dates is not None:
        df = df.reindex(index=dates, method='ffill')

    # apply dt multiplication
    if len(df.index) > 1:
        if is_actual_calendar_dt:
            delta = np.append(0.0, (df.index[1:] - df.index[:-1]).days / af)
        else:
            delta = 1.0 / af
        df = df.multiply(delta, axis=0)
    else:
        warnings.warn(f"in adjust_data_with_dt: lengh of data index is one - cannot adjust by dt")
        return df

    return df


def np_txy_tensor_to_pd_dict(np_tensor_txy: np.ndarray,
                             dateindex: Union[pd.Index, pd.DatetimeIndex],
                             factor_names: Union[pd.Index, List[str]],
                             asset_names: Union[pd.Index, List[str]],
                             is_factor_view: bool = True
                             ) -> Dict[str, pd.DataFrame]:
    """
    np_tensor_txy is an output of factor model: betas = [time, factor, asset]
    given timeindex such that #timeindex = time
    convert to:
    factor view = dict{factor: set[timeindex, asset]}  for each factor exp by assets
    asset view = dict{asset: set[timeindex, factor]}  for each asset exp by factor
    """
    if np_tensor_txy.ndim != 3:
        raise TypeError(f"np_tensor must ne 3-d tensor")
    if len(dateindex) != np_tensor_txy.shape[0]:
        raise TypeError(f"time dimension must be equal")
    if len(factor_names) != np_tensor_txy.shape[1]:
        raise TypeError(f"factor dimension must be equal")
    if len(asset_names) != np_tensor_txy.shape[2]:
        raise TypeError(f"asset dimension must be equal")

    if is_factor_view:  # {factor_id: pd.DataFrame(factor loadings)}
        factor_loadings = {}
        for factor_idx, factor in enumerate(factor_names):
            factor_loadings[factor] = pd.DataFrame(data=np_tensor_txy[:, factor_idx, :],
                                                   index=dateindex,
                                                   columns=asset_names)

    else:  # {asset_id: pd.DataFrame(factor loadings)}
        factor_loadings = {}
        for asset_idx, asset in enumerate(asset_names):
            factor_loadings[asset] = pd.DataFrame(data=np_tensor_txy[:, :, asset_idx],
                                                  index=dateindex,
                                                  columns=factor_names)
    return factor_loadings


def factor_dict_to_asset_dict(factor_loadings: Dict[str, pd.DataFrame],
                              asset_names: Optional[Union[pd.Index, List[str]]] = None  # column names in set
                              ) -> Dict[str, pd.DataFrame]:
    """
    revert the dictionary Dict[factor/model: set[assets] to Dict[assets: set factor)
    """
    if asset_names is None:
        asset_names = factor_loadings[list(factor_loadings.keys())[0]].columns

    asset_factor_loadings = {}
    for asset in asset_names:
        asset_factor_loading = []
        for factor, factor_data in factor_loadings.items():
            asset_factor_loading.append(factor_data[asset].rename(factor))
        asset_factor_loadings[asset] = pd.concat(asset_factor_loading, axis=1)

    return asset_factor_loadings


def df_time_dict_to_pd(factor_loadings_dict: Dict[pd.Timestamp, pd.DataFrame],
                       is_factor_view: bool = True,
                       is_zeros_to_nan: bool = False
                       ) -> Dict[str, pd.DataFrame]:
    """
    {time: set[factor, asset}
    convert to:
    factor view = dict{factor: set[timeindex, asset]}
    asset view = dict{asset: set[timeindex, factor]}
    """
    data0 = next(iter(factor_loadings_dict.values()))
    factor_names = data0.index
    asset_names = data0.columns
    dateindex = pd.DatetimeIndex(factor_loadings_dict.keys())

    np_tensor_txy = np.full((len(dateindex), len(factor_names), len(asset_names)), np.nan)
    for idx, data in enumerate(factor_loadings_dict.values()):
        np_tensor_txy[idx, :, :] = data.to_numpy()

    if is_zeros_to_nan:
        np_tensor_txy = np.where(np.isclose(np_tensor_txy, 0.0), np.nan, np_tensor_txy)

    factor_loadings = np_txy_tensor_to_pd_dict(np_tensor_txy=np_tensor_txy,
                                               dateindex=dateindex,
                                               factor_names=factor_names,
                                               asset_names=asset_names,
                                               is_factor_view=is_factor_view)
    return factor_loadings


def dfs_to_upper_lower_diag(df_upper: pd.DataFrame,
                            df_lower: pd.DataFrame,
                            diagonal: pd.Series
                            ) -> pd.DataFrame:
    """
    create up/low/diag copz
    nb must be square matrizes
    """
    n_rows = len(df_upper.index)
    out = df_upper.copy()
    for i in range(n_rows):
        out.iloc[i, i] = diagonal.iloc[i]
        for j in range(n_rows):
            if i < j:
                out.iloc[i, j] = df_upper.iloc[i, j]
            elif i > j:
                out.iloc[i, j] = df_lower.iloc[i, j]
            else:
                pass
    return out


def df_concat_with_tz(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    can only do if common tz is found
    """
    # find common tz
    tz = None
    for df in dfs:
        tz1 = df.index.tz
        if tz is not None:
            if tz1 != tz:
                raise ValueError(f"{tz1} {tz}")
            else:
                tz = tz1
        else:
            tz = tz1

    # localize
    for df in dfs:
        if df.index.tz is None:
            df.index = df.index.tz_localize(tz)

    return pd.concat(dfs, axis=1)


def compute_last_score(df: Union[pd.DataFrame, pd.Series], is_percent: bool = True) -> pd.Series:
    """
    columnwise score for last value in data
    use column wise loop with percentileofscore (as it supports only float for score)
    """
    if isinstance(df, pd.Series):
        df = df.to_frame()

    percentiles = []
    for column in df.columns:
        x = df[column].dropna()
        percentiles.append(stats.percentileofscore(a=x, score=x.iloc[-1], kind='rank'))
    percentiles = pd.Series(percentiles, index=df.columns)
    if is_percent:
        percentiles = percentiles.multiply(0.01)

    return percentiles


def align_dfs_dict_with_df(dfd: Dict[Any, pd.DataFrame],
                           df: pd.DataFrame,
                           method: str = 'ffill'
                           ) -> Dict[Any, pd.DataFrame]:
    """
    align dict of dataframes with with given df
    """
    alignment_columns = df.columns
    alignment_index = df.index
    aligned_dfd = {}
    for key, df in dfd.items():
        if df is not None:  # align with other pandas in dict by filtered value
            aligned_dfd[key] = df[alignment_columns].reindex(index=alignment_index).fillna(method=method)
        else:
            aligned_dfd[key] = None
    return aligned_dfd


def align_column_to_index(df_column: pd.DataFrame,
                          df_index: pd.DataFrame,
                          join: str = 'inner',
                          is_strict_join: bool = True
                          ) -> pd.DataFrame:
    """
    align dataframe columns to data_index
    join='inner' with strict match
    join='outer' without strict match so missing data has na
    return is df = data_index with columns matched to data_column
    important is that columns in data_column (narrower) correspond to data_index index
    """
    # 1 - make pd. Series indeexd by columns of data_column
    data_columns0 = pd.Series(data=df_column.columns, index=df_column.columns)

    # 2-align data columns to the index
    data_index1, data_columns1 = df_index.align(other=data_columns0, join=join, axis=0)

    if join == 'inner' and is_strict_join:
        if len(df_column.columns) != len(data_index1.index):
            print(df_column.columns)
            print(data_index1.index)
            raise ValueError(f"columns in data_column and index are not matched")

    return data_index1


def align_index_to_index(df1: Union[pd.DataFrame, pd.Series],
                         df2: Union[pd.DataFrame, pd.Series],
                         join: Literal['inner', 'outer', 'left', 'right'] = 'inner'
                         ) -> Tuple[Union[pd.DataFrame, pd.Series], Union[pd.DataFrame, pd.Series]]:
    """
    align dataframes
    join='inner' with strict match
    join='outer' without strict match so missing data has na
    """
    data_index1_, data_index2_ = df1.align(other=df2, join=join, axis=0)

    if join == 'inner':

        if isinstance(df1, pd.Series) and isinstance(df2, pd.Series):
            if len(df1.index) != len(data_index1_.index) \
                    or len(df2.index) != len(data_index2_.index):
                raise ValueError(f"Series1.index{df1.index} and Series2.index{df2.index} are not matched")

        elif isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame):
            if len(df1.index) != len(data_index1_.index) \
                    or len(df1.columns) != len(data_index1_.columns) \
                    or len(df2.index) != len(data_index2_.index) \
                    or len(df2.columns) != len(data_index2_.columns):
                raise ValueError(f"data1{df1} and data2 index{df2}"
                                 f"or columns are not matched:")

        elif isinstance(df1, pd.DataFrame) and isinstance(df2, pd.Series):
            if len(df1.index) != len(data_index1_.index) \
                    or len(df1.columns) != len(data_index1_.columns) \
                    or len(df2.index) != len(data_index2_.index):
                raise ValueError(f"data1{df1} and data2{df2} index are not matched")

        elif isinstance(df1, pd.Series) and isinstance(df2, pd.DataFrame):
            if len(df2.index) != len(data_index2_.index) \
                    or len(df2.columns) != len(data_index2_.columns) \
                    or len(df1.index) != len(data_index1_.index):
                raise ValueError(f"data1{df1} and data2{df2} index are not matched")

        else:
            raise TypeError('unimplemented type')

    return data_index1_, data_index2_


def align_df1_to_df2(df1: pd.DataFrame,
                     df2: pd.DataFrame,
                     join: Literal['inner', 'outer', 'left', 'right'] = 'inner',
                     axis: Optional[int] = None
                     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    align dataframes
    join='inner' with strict match
    join='outer' without strict match so missing data has na
    """
    if isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame):
        pass
    else:
        raise TypeError(f"df1 and df2 must be dataframes")

    df1_, df2_ = df1.align(other=df2, join=join, axis=axis)

    if join == 'inner':
        if len(df1.index) != len(df1_.index) or len(df2.index) != len(df2_.index):
            raise ValueError(f"df1 and df2 index are not matched: {df1.index} vs {df2.index}")

        elif len(df1.columns) != len(df1_.columns) or len(df2.columns) != len(df2_.columns):
            raise ValueError(f"df1 and df2 columns are not matched: {df1.columns} vs {df2.columns}")

    return df1_, df2_


def df12_merge_with_tz(df1: pd.DataFrame,
                       df2: pd.DataFrame,
                       tz: Optional[str] = None,
                       freq: str = 'B'
                       ) -> pd.DataFrame:
    """
    tz sensetive alignment
    """
    # localize index and convert to common time zeone
    df1 = df1.copy()
    df1.index = df1.index.tz_localize(tz=tz)
    df2 = df2.copy()
    df2.index = df2.index.tz_localize(tz=tz)

    # concat
    dfs = pd.concat([df1, df2], axis=1)
    dfs = dfs.fillna(method='ffill').asfreq(freq=freq, method='ffill')
    dfs.index = dfs.index# .tz_localize(tz=tz)
    return dfs


def merge_on_column(df1: pd.DataFrame,
                    df2: pd.DataFrame,
                    is_drop_y: bool = True
                    ) -> pd.DataFrame:
    """
    merge on homogeneous column with preservation of indices and drop dublicated, _y columns
    """
    joint_data = pd.merge(left=df1, right=df2,
                          left_index=True, right_index=True,
                          suffixes=('', '_y'), how='inner')
    if is_drop_y:
        unique_columns = sop.merge_lists_unique(list1=df1.columns.to_list(), list2=df2.columns.to_list())
        joint_data = joint_data[unique_columns]
    return joint_data


def concat_at_start_date(dfs: List[Union[pd.Series, pd.DataFrame]],
                         start_date: pd.Timestamp
                         ) -> pd.DataFrame:
    data = pd.concat(dfs, axis=1).loc[start_date:]
    return data


class UnitTests(Enum):
    ALIGN = 1
    SCORES = 2


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.ALIGN:

        desc_dict1 = {'f0': (0.5, 0.5), 'f1': (0.5, 0.5), 'f2': (0.5, 0.5), 'f3': (0.5, 0.5), 'f4': (0.5, 0.5)}
        df1 = pd.DataFrame.from_dict(desc_dict1, orient='index', columns=['as1', 'as2'])

        desc_dict2 = {'f1': (1.0, 1.0), 'f2': (1.0, 1.0), 'f5': (1.0, 1.0), 'f6': (1.0, 1.0)}
        df2 = pd.DataFrame.from_dict(desc_dict2, orient='index', columns=['as1', 'as3'])

        print(f"df1=\n{df1}")
        print(f"df2=\n{df2}")

        df1_, df2_ = align_df1_to_df2(df1=df1, df2=df2, join='outer', axis=0)
        print(f"df1_=\n{df1_}")
        print(f"df2_=\n{df2_}")

        df1_, df2_ = align_df1_to_df2(df1=df1, df2=df2, join='outer', axis=None)
        print(f"df1_=\n{df1_}")
        print(f"df2_=\n{df2_}")

    elif unit_test == UnitTests.SCORES:
        np.random.seed(1)
        nrows, ncol = 20, 5
        df = pd.DataFrame(data=np.random.normal(0.0, 1.0, size=(nrows, ncol)),
                          columns=[f"id{n + 1}" for n in range(ncol)])
        print(df)
        percentiles = compute_last_score(df=df)
        print(percentiles)


if __name__ == '__main__':

    unit_test = UnitTests.ALIGN

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

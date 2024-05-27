"""
implement group by operations on df
"""
# packages
import numpy as np
import pandas as pd
from typing import Union, List, Dict, Callable, Optional
from enum import Enum

# qis
import qis.utils.struct_ops as sop
import qis.utils.df_agg as dfa


def get_group_dict(group_data: pd.Series,
                   index_data: Union[pd.Index, List[str], np.ndarray] = None,
                   group_order: List[str] = None,
                   total_column: Union[str, None] = None
                   ) -> Dict[str, List[str]]:
    """
    take group_data and return dictionary of index tickers mapped to group_data
    with index_data can do double mappings group_data to key, then key:index_data
    """
    if not isinstance(group_data, pd.Series):
        raise ValueError(f"group_data type={type(group_data)} must be series")

    if np.any(group_data.isnull()):
        raise ValueError(f"group data is nan\ngroup_data=\n{group_data}:"
                         f"\n{group_data.loc[np.isnan(group_data.isnull())]}")

    if index_data is not None:
        if isinstance(index_data, list) or isinstance(index_data, np.ndarray):  # convert to Index
            index_data = pd.Index(index_data)
        elif not isinstance(index_data, pd.Index):
            raise TypeError(f"unsupported type {type(index_data)}")
        if np.any(index_data.isnull()):
            raise ValueError(f"index_data is nan:\n{index_data[index_data.isnull()]}")

    group_data_np = group_data.to_numpy()
    data_groups = list(np.unique(group_data_np))
    if group_order is not None:
        groups = sop.merge_lists_unique(list1=group_order, list2=data_groups)
    else:
        groups = data_groups

    # fill group dict
    group_dict = {}
    if total_column is not None:
        group_tickers = group_data.index
        if index_data is not None:  # match group_tickers in index_data: necessary if index_data != group_data.index
            group_tickers = index_data[np.in1d(ar1=index_data, ar2=group_tickers, assume_unique=True)]
        group_dict[total_column] = group_tickers

    # match group to instruments in group_data
    for group in groups:
        if group in data_groups:
            group_instruments = group_data.index[np.in1d(ar1=group_data_np, ar2=[group], assume_unique=True)]
            if index_data is not None:  # match group_tickers in index_data: necessary if index_data != group_data.index
                group_instruments = index_data[np.in1d(ar1=index_data, ar2=group_instruments, assume_unique=True)]
            group_dict[group] = group_instruments.to_list()

    return group_dict


def split_df_by_groups(df: pd.DataFrame,
                       group_data: pd.Series,
                       total_column: Union[str, None] = None,
                       group_order: List[str] = None
                       ) -> Dict[str, pd.DataFrame]:
    """
    take pandas data which are index=time series, data = assets
    slit data according to grouping by rows in group_data[asset_group_column] in group_data
    the match of columns in data to rows in group_data is done using group_data.index or group_data[ticker_column_to_match]
    """
    # group by descriptive data
    group_dict = get_group_dict(group_data=group_data,
                                index_data=df.columns.to_numpy(),
                                group_order=group_order,
                                total_column=total_column)

    grouped_data = {}
    for key, columns in group_dict.items():
        if len(columns) > 0:  # get subset of pandas by columns, copy() to break view
            grouped_data[key] = df[columns].copy()

    return grouped_data


def agg_df_by_groups_ax1(df: pd.DataFrame,
                         group_data: pd.Series,
                         agg_func: Callable[[pd.DataFrame], pd.Series] = np.nansum,
                         total_column: Union[str, None] = None,
                         group_order: List[str] = None
                         ) -> pd.DataFrame:
    """
    take df[index=time series, data = asset data]
    slit data according to grouping by rows in group_data[asset_group_column] in group_data
    agg data by groups
    """
    # group by descriptive data
    group_dict = get_group_dict(group_data=group_data,
                                index_data=df.columns.to_numpy(),
                                group_order=group_order,
                                total_column=total_column)

    grouped_data = {}
    for key, columns in group_dict.items():
        if len(columns) > 0:  # get subset of pandas by columns, copy() to break view
            grouped_data[key] = df[columns].apply(agg_func, axis=1).rename(key)
    grouped_data = pd.DataFrame.from_dict(grouped_data, orient='columns')
    return grouped_data


def agg_df_by_groups(df: pd.DataFrame,
                     group_data: pd.Series,
                     group: str = None,
                     agg_func: Callable[[pd.DataFrame], pd.Series] = dfa.nansum,
                     total_column: Union[str, None] = None,
                     is_total_first: bool = True,
                     group_order: List[str] = None,
                     axis: Optional[int] = 1  # axis for integration
                     ) -> pd.DataFrame:
    """
    take pandas data which are index=time series, data = instruments
    slit data according to grouping by rows in group_data[group_column] in group_data
    the match of columns in data to rows in group_data is done using group_data.index or group_data[ticker_column_to_match]
    agg_func: pd.DataFrame -> Union[pd.Series, pd.DataFrame] is function for aggregation
    """
    # group by descriptive data
    group_dict = get_group_dict(group_data=group_data,
                                index_data=df.columns.to_numpy(),
                                group_order=group_order)

    # apply agg_func to grouped pandas
    agg_grouped_datas = []
    if group is None:  # output pandas with columns by dictionary keys
        for key, group_columns in group_dict.items():
            if len(group_columns) > 0:
                agg_grouped_datas.append(agg_func(df[group_columns], axis=axis).rename(key))
        agg_grouped_data = pd.concat(agg_grouped_datas, axis=1)

    else:  # just take the group
        if group in group_dict.keys():
            agg_grouped_data = agg_func(df[group_dict[group]]).to_frame(name=group)
        else:
            raise ValueError(f"{group} in keys {group_dict.keys()}")

    # insert total
    if total_column is not None:
        if not isinstance(total_column, str):
            raise TypeError(f"in agg_data_by_groups: {total_column} must be string")
        total = agg_func(df)
        if is_total_first:
            agg_grouped_data.insert(loc=0, column=total_column, value=total.values)
        else:
            agg_grouped_data.insert(loc=len(agg_grouped_data.columns), column=total_column, value=total.values)
    return agg_grouped_data


def agg_df_by_group_with_avg(df: pd.DataFrame,
                             group_data: pd.Series,
                             group_order: List[str] = None,
                             agg_func: Callable = dfa.nanmean,
                             agg_func_id: str = 'mean',
                             total_column: str = 'Universe mean'
                             ) -> Dict[str, pd.DataFrame]:
    """
    create ac grouped data dict with universes median and group medians
    """
    grouped_data = split_df_by_groups(df=df,
                                      group_data=group_data,
                                      group_order=group_order,
                                      total_column=None)

    group_avg = agg_df_by_groups(df=df,
                                 group_data=group_data,
                                 agg_func=agg_func,
                                 total_column=total_column)

    # add group median to grouped_data
    grouped_data_avg = dict()
    grouped_data_avg[total_column] = group_avg
    for ac_id, ac_data in grouped_data.items():
        data_avg = group_avg[ac_id].rename(f"{ac_id} {agg_func_id}")
        ac_data = pd.concat([data_avg, ac_data], axis=1)
        grouped_data_avg[ac_id] = ac_data

    return grouped_data_avg


def fill_df_with_group_avg(df: pd.DataFrame,
                           group_data: pd.Series,
                           agg_func: Callable[[pd.DataFrame], pd.Series] = dfa.nanmean,
                           group_order: List[str] = None
                           ) -> pd.DataFrame:
    """
    compute group avg and return data for each column with corresponding group avg
    """
    group_avg = agg_df_by_groups(df=df,
                                 group_data=group_data,
                                 group_order=group_order,
                                 agg_func=agg_func,
                                 total_column=None)
    avg_data = {}
    for column in df.columns:
        avg_data[column] = group_avg[group_data[column]]
    avg_data = pd.DataFrame.from_dict(avg_data, orient='columns')
    return avg_data


def sort_df_by_index_group(df: pd.DataFrame,
                           group_column: str,
                           sort_column: str,
                           ascending: bool = False,
                           group_order: List[str] = None
                           ) -> pd.DataFrame:

    group_dict = get_group_dict(group_data=df[group_column],
                                group_order=group_order)
    sorted_datas = []
    for group, tickers in group_dict.items():
        sorted_datas.append(df.loc[tickers, :].sort_values(by=[sort_column], ascending=ascending))
    sorted_data = pd.concat(sorted_datas, axis=0)

    return sorted_data


class UnitTests(Enum):
    GROUP = 1


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.GROUP:

        group_data = pd.Series(dict(SPY='Equities', QQQ='Equities', EEM='Equities', TLT='Bonds',
                                    IEF='Bonds', SHY='Bonds', LQD='Credit', HYG='HighYield', GLD='Gold'))

        group_dict = get_group_dict(group_data=group_data)
        print(f"group_dict=\n{group_dict}")

        group_dict_ordered = get_group_dict(group_data=group_data, group_order=list(group_data.unique()))
        print(f"group_dict_ordered=\n{group_dict_ordered}")

        group_dict_subset = get_group_dict(group_data=group_data,
                                           index_data=group_data.index[:5].to_list(),
                                           group_order=list(group_data.unique()))
        print(f"group_dict_subset=\n{group_dict_subset}")


if __name__ == '__main__':

    unit_test = UnitTests.GROUP

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

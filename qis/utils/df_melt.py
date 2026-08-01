"""
reshaping a wide panel into the long form seaborn's scatter and box functions expect: one row per
observation, carrying a ``hue`` column. ``melt_scatter_data_with_xvar`` repeats a named column of
the frame as x, ``melt_scatter_data_with_xdata`` takes x from outside it, and
``melt_signed_paired_df`` returns a dict keyed by ``SignCondition``: TREND and REVERSION for the
same-sign and opposite-sign pairs, NONE for their union rather than for a third bucket.
"""
# packages
import pandas as pd
from enum import Enum
from typing import Dict, Optional, List
# qis
import qis.utils.np_ops as npo


def melt_scatter_data_with_xvar(df: pd.DataFrame,
                                xvar_str: str,
                                y_column: str = 'Strategy returns',
                                hue_name: str = 'hue'
                                ) -> pd.DataFrame:
    """
    column with xvar_str will be repeated to melted y_column
    index is ignored
    """
    # check xvar_str exists
    if xvar_str not in df.columns:
        raise ValueError(f"xvar_str='{xvar_str}' not found in df.columns={df.columns.tolist()}")

    # check y_column does not conflict with existing columns
    ex_xvar_columns = [c for c in df.columns if c != xvar_str]
    if y_column in ex_xvar_columns:
        raise ValueError(f"y_column='{y_column}' conflicts with existing df columns={ex_xvar_columns}. "
                         f"Choose a different y_column name.")

    # check hue_name does not conflict
    if hue_name in df.columns:
        raise ValueError(f"hue_name='{hue_name}' conflicts with existing df columns={df.columns.tolist()}. "
                         f"Choose a different hue_name.")

    df = df.dropna()
    ex_benchmark_data = df.drop(xvar_str, axis=1)
    scatter_data = pd.melt(df,
                           value_vars=ex_benchmark_data.columns,
                           id_vars=[xvar_str],
                           var_name=hue_name,
                           value_name=y_column)
    # move hue to last position
    columns_ex_hue = scatter_data.columns.to_list()
    columns_ex_hue.remove(hue_name)
    scatter_data = scatter_data[columns_ex_hue + [hue_name]]
    return scatter_data


def melt_scatter_data_with_xdata(df: pd.DataFrame,
                                 xdata: pd.Series,
                                 y_column: str = 'Vars',
                                 hue_name: str = 'hue'
                                 ) -> pd.DataFrame:
    """
    df will be melted using xdata
    must be same indices
    """

    joint_data = pd.concat([xdata, df], axis=1, sort=True)
    scatter_data = pd.melt(joint_data,
                           value_vars=df.columns,
                           id_vars=[xdata.name],
                           var_name=hue_name,
                           value_name=y_column)

    # move hue to last position
    columns_ex_hue = scatter_data.columns.to_list()
    columns_ex_hue.remove(hue_name)
    scatter_data = scatter_data[columns_ex_hue+[hue_name]]

    return scatter_data


def melt_df_by_columns(df: pd.DataFrame,
                       x_index_var_name: Optional[str] = 'date',
                       y_var_name: str = 'returns',
                       hue_var_name: str = 'instrument',
                       hue_order: List[str] = None,
                       ) -> pd.DataFrame:
    """
    index is added to hue variables
    df melted to hue = [index_values, column_values, data_values]
    -> df.columns = [x_index_var_name, hue_var_name, y_var_name]
    """
    df = df.dropna(axis=1, how='all')
    df.index.name = x_index_var_name # set to match id_vars
    box_data = pd.melt(df.reset_index(),
                       id_vars=x_index_var_name,
                       value_vars=df.columns.to_list(),
                       value_name=y_var_name,
                       var_name=hue_var_name)

    if hue_order is not None:  # sort by hue
        sort_column = 'sort_column'
        name_sort = {key: idx for idx, key in enumerate(hue_order)}
        box_data[sort_column] = box_data[x_index_var_name].map(name_sort)
        box_data = box_data.sort_values(by=sort_column).drop(sort_column, axis=1)

    box_data = box_data.loc[pd.isna(box_data[y_var_name]) == False, :]  # important to exclude nans

    return box_data


def melt_paired_df(indicator: pd.DataFrame,
                   observations: pd.DataFrame,
                   signal_name: str = 'signal',
                   ra_return_name: str = 'ra_return',
                   hue_name: str = 'hue'
                   ) -> pd.DataFrame:
    # melt to pandas
    col_union = observations.columns.intersection(indicator.columns)
    # indicator = indicator[col_union]
    observations = observations[indicator.columns]
    temp_hue = f"{hue_name}_y"
    x_pd = pd.melt(indicator, value_vars=indicator.columns.to_list(), var_name=hue_name, value_name=signal_name)
    y_pd = pd.melt(observations, value_vars=observations.columns.to_list(), var_name=temp_hue, value_name=ra_return_name)
    # concat
    scatter_data = pd.concat([x_pd, y_pd], axis=1, sort=False).dropna()
    scatter_data = scatter_data.drop(columns=[temp_hue])
    scatter_data = scatter_data.sort_values(by=signal_name)
    return scatter_data


class SignCondition(Enum):
    NONE = 'None'
    TREND = 'Trend'
    REVERSION = 'Reversion'


def melt_signed_paired_df(observations: pd.DataFrame,
                          indicator: pd.DataFrame,
                          signal_name: str = 'signal',
                          ra_return_name: str = 'ra_return',
                          hue_name: str = 'hue'
                          ) -> Dict[SignCondition, pd.DataFrame]:
    # melt to pandas
    scatter_data = melt_paired_df(indicator=indicator,
                                  observations=observations,
                                  signal_name=signal_name,
                                  ra_return_name=ra_return_name,
                                  hue_name=hue_name)

    joint_cond, trend_cond, rev_cond = npo.compute_paired_signs(x=scatter_data[signal_name].to_numpy(),
                                                                y=scatter_data[ra_return_name].to_numpy())

    data_out = {SignCondition.NONE: scatter_data.loc[joint_cond, :],
                SignCondition.TREND: scatter_data.loc[trend_cond, :],
                SignCondition.REVERSION: scatter_data.loc[rev_cond, :]}

    return data_out

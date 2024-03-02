"""
analytic for melting of pandas
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
    ex_benchmark_data = df.drop(xvar_str, axis=1)

    scatter_data = pd.melt(df,
                           value_vars=ex_benchmark_data.columns,
                           id_vars=[xvar_str],
                           var_name=hue_name,
                           value_name=y_column)

    # move hue to last position
    columns_ex_hue = scatter_data.columns.to_list()
    columns_ex_hue.remove(hue_name)
    scatter_data = scatter_data[columns_ex_hue+[hue_name]]

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

    joint_data = pd.concat([xdata, df], axis=1)
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
    scatter_data = pd.concat([x_pd, y_pd], axis=1).dropna()
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


class UnitTests(Enum):
    PD_MELT = 1
    SCATTER_DATA = 2
    MELT_DF_BY_COLUMNS = 3


def run_unit_test(unit_test: UnitTests):
    from qis.test_data import load_etf_data
    prices = load_etf_data().dropna().asfreq('QE', method='ffill')
    returns = prices.pct_change()

    if unit_test == UnitTests.PD_MELT:
        df = pd.DataFrame(data=[[0, 1, 2.5], [2, 3, 5.0]],
                          index=['cat', 'dog'],
                          columns=['weight', 'height', 'age'])
        print(df)
        melted = pd.melt(df, value_vars=df.columns, var_name='myVarname', value_name='myValname')
        print(f"melted=\n{melted}")

        scatter_data = melt_scatter_data_with_xvar(df=df, xvar_str='age', y_column='weight_height')
        print(f"scatter_data=\n{scatter_data}")

        box_data = melt_df_by_columns(df=df, x_index_var_name='animal', hue_var_name='hue_features', y_var_name='observations')
        print(f"box_data=\n{box_data}")

    elif unit_test == UnitTests.SCATTER_DATA:
        scatter_data = melt_scatter_data_with_xvar(df=returns, xvar_str='SPY')
        print(scatter_data)

    elif unit_test == UnitTests.MELT_DF_BY_COLUMNS:
        box_data = melt_df_by_columns(df=returns)
        print(box_data)


if __name__ == '__main__':

    unit_test = UnitTests.PD_MELT

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

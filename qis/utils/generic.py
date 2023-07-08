
# packages
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple, Optional, Callable, Dict, List

# qis
import qis.file_utils as fu
import qis.utils.df_groups as dfg
import qis.utils.df_agg as dfa
import qis.utils.struct_ops as sop
import qis.utils.df_str as dff

DATE_FORMAT = '%d%b%y' # short y for meta


class ValueType(Enum):
    FLOAT0 = 0
    FLOAT = 1
    FLOAT2 = 2
    FLOAT4 = 14
    COMMA_FLOAT0 = 3
    PRICE = 4
    PERCT = 5
    PERCT0 = 6
    DATE = 7
    INT = 8
    STR = 9
    WEIGHT = 10
    SHARPE = 11
    MARG_SHARPE = 12


class ColVar(NamedTuple):
    """
    variable characteristic for columns variable for table
    """
    name: str
    short: str = None
    short_n: str = None
    value_type: ValueType = ValueType.FLOAT
    agg_func: Optional[Callable] = None

    def to_str(self, short: bool = False, short_n: bool = False, **kwargs) -> str:
        if short and self.short is not None:
            name = self.short
        elif short_n and self.short_n is not None:
            name = self.short_n
        else:
            name = self.name
        return name

    def to_format(self,
                  digits_to_show: int = 2,
                  sharpe_digits: int = 2,
                  price_format: Optional[str] = None,
                  # date_format: str = DATE_FORMAT,
                  **kwargs
                  ) -> str:

        var_format = '{:.2f}'

        if self.value_type == ValueType.FLOAT:
            if digits_to_show == 1:
                var_format = '{:.1f}'
            elif digits_to_show == 4:
                var_format = '{:.4f}'
            else:
                var_format = '{:.2f}'
        elif self.value_type == ValueType.FLOAT0:
            var_format = '{:.0f}'
        elif self.value_type == ValueType.FLOAT2:
            var_format = '{:.2f}'
        elif self.value_type == ValueType.FLOAT4:
            var_format = '{:.4f}'
        elif self.value_type == ValueType.COMMA_FLOAT0:
            var_format = '{:,.0f}'
        elif self.value_type == ValueType.PRICE:
            if price_format is not None:
                var_format = price_format
            else:
                var_format = '{:.2f}'

        elif self.value_type == ValueType.PERCT:
            if digits_to_show == 0:
                var_format = '{:.0%}'
            elif digits_to_show == 1:
                var_format = '{:.1%}'
            else:
                var_format = '{:.2%}'

        elif self.value_type == ValueType.PERCT0:
            var_format = '{:,.0%}'

        elif self.value_type == ValueType.DATE:
             var_format = DATE_FORMAT  # independent

        elif self.value_type == ValueType.INT:
             var_format = '{:.0f}'

        elif self.value_type == ValueType.STR:
             var_format = '{}'

        elif self.value_type == ValueType.WEIGHT:
             var_format = '{:.2%}'

        elif self.value_type == ValueType.SHARPE:
            if sharpe_digits == 2:
                var_format = '{:.2f}'
            else:
                var_format = '{:.1f}'

        elif self.value_type == ValueType.MARG_SHARPE:
            var_format = '{:.3f}'

        else:
            TypeError(f"unsupported type = {self.value_type}")

        return var_format


@dataclass
class ColumnData:
    """
    keep data for producing meta and aggregation
    """
    column: ColVar
    data: pd.Series
    is_total_mean: bool = True


def column_datas_to_df(column_datas: Dict[str, ColumnData],
                       weight: Optional[pd.Series] = None,
                       agg_column: Optional[str] = None,
                       group_order: Optional[List[str]] = None,
                       sort_column: str = None,
                       ascending: bool = False,
                       is_to_str: bool = True,
                       total_name: str = 'Total',
                       new_index_col: str = None,
                       agg_columns_filling: Optional[Dict[str, str]] = {'Name': 'Avg'}
                       ) -> pd.DataFrame:
    """
    aggregate dictionary of column vars to pandas with aggregation
    """
    datas = []
    for _, column_data in column_datas.items():
        datas.append(column_data.data.rename(column_data.column.to_str()))
    table_data = pd.concat(datas, axis=1)

    if weight is None:
        weight = pd.Series(1.0, index=table_data.index)

    # now sort
    if sort_column is not None and agg_column is not None:
        table_data = dfg.sort_df_by_index_group(df=table_data,
                                                group_column=agg_column,
                                                sort_column=sort_column,
                                                group_order=group_order,
                                                ascending=ascending)

    if new_index_col is not None:
        table_data = table_data.set_index(new_index_col, drop=True)
        column_datas.pop(new_index_col)

    # create dict for ac aggregation
    if agg_column is not None:
        agg_dict = {}
        table_data1 = table_data.copy()
        for _, column_data in column_datas.items():
            if column_data.column.agg_func is not None:
                if column_data.column.agg_func == dfa.sum_weighted:
                    table_data1[column_data.column.to_str()] = table_data1[column_data.column.to_str()].multiply(weight)
                    agg_dict[column_data.column.to_str()] = np.nansum
                else:
                    agg_dict[column_data.column.to_str()] = column_data.column.agg_func

        if len(agg_dict.keys()) > 0:
            try:
                agg_table = table_data1.groupby(agg_column).agg(agg_dict)  # use agg as apply func for columns
            except AttributeError:
                raise AttributeError(f"dublicated columnes\n{agg_column}")

            # arrange by ac order add name
            if group_order is not None and agg_table is not None:
                groups = sop.merge_lists_unique(list1=group_order, list2=list(np.unique(agg_table.index)))
                groups1 = [x for x in groups if x in agg_table.index]
                agg_table = agg_table.loc[groups1, :]

            # add total as means
            port_table = agg_table.mean(axis=0).to_frame(total_name).T
            # replace with sum
            for _, column_data in column_datas.items():
                if column_data.is_total_mean is False:
                    port_table[column_data.column.to_str()] = np.nansum(agg_table[column_data.column.to_str()])

            agg_table_data = pd.concat([table_data, agg_table, port_table], axis=0)

            # for asset class data in column name add avg for clarity
            if agg_columns_filling is not None:
                for col, value in agg_columns_filling.items():
                    if col in agg_table_data.columns:
                        agg_table_data.loc[agg_table.index, col] = value
                        agg_table_data.loc[port_table.index, col] = value

        else:
            print("no vars for aggregation")
            agg_table_data = table_data
    else:
        agg_table_data = table_data

    if is_to_str:  # convert to str:
        for _, column_data in column_datas.items():
            agg_table_data[column_data.column.to_str()] = dff.series_to_str(
                ds=agg_table_data[column_data.column.to_str()],
                var_format=column_data.column.to_format(digits_to_show=2))
    return agg_table_data


@dataclass
class DfOutDict:
    df_out_dict: Dict[str, pd.DataFrame] = None
    last_data: Optional[pd.DataFrame] = None

    def __post_init__(self):
        self.df_out_dict = {}

    def __getitem__(self, key: str) -> pd.DataFrame:
        # for df_out_dict[key]
        return self.df_out_dict[key]

    def append(self, df: pd.DataFrame, name: str) -> None:
        if name not in self.df_out_dict.keys():
            self.df_out_dict[name] = df
        else:
            raise ValueError(f"{name} exist in {self.df_out_dict.keys()}")

    def set_last_df(self, data: pd.DataFrame) -> None:
        # way to first make a placement and then append it
        self.last_data = data

    def append_last_df(self, name: str) -> None:
        # way to first make a placement and then append it
        if self.last_data is None:
            raise ValueError(f"self.last_data is None")
        self.append(df=self.last_data, name=name)
        self.last_data = None

    def print(self):
        for key, df in self.df_out_dict.items():
            print(f"{key}")
            print(df)

    def save(self, file_name: str) -> None:
        if len(self.df_out_dict.keys()) > 0:
            file_path = fu.save_df_to_excel(data=self.df_out_dict, file_name=file_name)
            print(f"saved output data to excel:\n {file_path}")


class EnumMap(Enum):
    """
    abstract enum with a map function
    """
    @classmethod
    def map_to_value(cls, name):
        """
        given name return value
        """
        for k, v in cls.__members__.items():
            if k == name:
                return v
        raise ValueError(f"nit in enum {name}")


class DotDict(dict):
    """
    dot.notation access to dictionary attributes
    """
    def __getattr__(self, attr):
        return self.get(attr)
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)
        self.__dict__ = self

    # def __setattr__(self, name, value):
    #    self._data[name] = value


class UnitTests(Enum):
    DOT_DICT = 1


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.DOT_DICT:
        this = DotDict({'me': 3, 'you': 10})
        print(this)
        print(this.me)
        print(this.you)

        for k, v in this.items():
            print(f"{k}: {v}")

        this['me1'] = 6
        this.me2 = 12
        for k, v in this.items():
            print(f"{k}: {v}")


if __name__ == '__main__':

    unit_test = UnitTests.DOT_DICT

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

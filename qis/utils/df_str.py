# packages
import numpy as np
import pandas as pd
from functools import partial
from enum import Enum
from pandas.core.dtypes.common import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_string_dtype
from tabulate import tabulate
from typing import List, Optional, Union, Dict

from qis.utils.dates import DATE_FORMAT

EMPTY_NUM = ' '


# Formatting string
def get_fmt_str(x, fill):
    return '{message: >{fill}}'.format(message=x, fill=fill)


def float_to_str(x: float,
                 var_format: str = '{:.2f}',
                 is_exclude_nans: bool = True
                 ) -> str:
    """
    float to str dendent on nan and var_format
    use as pd.Series().apply(lambda x: float_to_str(x, var_format))
    """
    if is_exclude_nans and isinstance(x, float) and np.isnan(x):
        return EMPTY_NUM
    elif isinstance(x, str):
        return x
    else:
        return var_format.format(x)


def str_to_float(x: str) -> float:
    """
    str to float
    use as pd.Series().apply(lambda x: str_to_float(x))
    """
    if isinstance(x, float) and np.isnan(x):  # just in case
        x = np.nan
    elif isinstance(x, str) and x == EMPTY_NUM:
        x = np.nan
    else:
        try:
            x = float(x.replace(',', '').replace('%', ''))
        except:  # the string contains words
            x = np.nan
    return x


def date_to_str(x: pd.Timestamp, var_format: str = DATE_FORMAT) -> str:
    """
    float to str dendent on nan and var_format
    use as pd.Series().apply(lambda x: float_to_str(x, var_format))
    """
    if x is None or x is pd.NaT or pd.notnull(x)==False:
        x_str = EMPTY_NUM
    else:  # var_format.format(x) does not work for date str
        #if isinstance(x, pd.Timestamp):
        #    x_str = x.strftime(var_format)
        #else:
        x_str = pd.Timestamp(x).strftime(var_format)
    return x_str


def series_to_str(ds: pd.Series,
                  var_format: str = '{:.2f}',
                  is_exclude_nans: bool = True
                  ) -> pd.Series:
    """
    pd.Series to string using float_to_str
    """
    if not isinstance(ds, pd.Series):
        print(ds)
        raise TypeError(f"data type= {type(ds)} must be pd.Series")

    if is_string_dtype(ds):
        pd_series = ds.copy()
    elif is_datetime(ds):
        # pycharm geeting wrong typecheck https://youtrack.jetbrains.com/issue/PY-43841
        pd_series = ds.apply(lambda x: date_to_str(x, var_format))
    else:
        pd_series = ds.apply(lambda x: float_to_str(x, var_format, is_exclude_nans))

    if is_exclude_nans:
        pd_series[pd_series.isna()] = EMPTY_NUM

    return pd_series


def series_to_date_str(ds: pd.Series,
                       var_format: str = DATE_FORMAT,
                       is_exclude_nans: bool = True
                       ) -> pd.Series:
    """
    pd.Series to string using float_to_str
    """
    if not isinstance(ds, pd.Series):
        print(ds)
        raise TypeError(f"data type= {type(ds)} must be pd.Series")

    pd_series = ds.apply(lambda x: date_to_str(x, var_format))
    if is_exclude_nans:
        pd_series[pd_series.isna()] = EMPTY_NUM
    return pd_series


def series_to_numeric(ds: pd.Series) -> np.ndarray:
    if ds.dtypes == np.float64:
        a = ds.to_numpy()
    elif ds.dtypes == str or isinstance(ds.dtypes, object):
        a = ds.apply(lambda x: str_to_float(x)).to_numpy()
    else:
        raise ValueError(f"unsupported value type: {ds.dtypes}")
    return a


def df_to_numeric(df: pd.DataFrame) -> np.ndarray:
    a = df.apply(lambda x: series_to_numeric(x)).to_numpy()
    return a


def df_to_str(df: pd.DataFrame,
              var_format: str = '{:.2f}',
              var_formats: Union[List[Optional[str]], Dict[str, str]] = None,  # specific for each column
              is_exclude_nans: bool = True
              ) -> pd.DataFrame:
    """
    pd.DataFrame to string using float_to_str
    """
    if var_formats is not None:
        if isinstance(var_formats, list):
            if not len(var_formats) == len(df.columns):
                raise ValueError(f"match len of var_formats {var_formats} with {df.columns}")
        elif isinstance(var_formats, dict):
            var_formats_ = []
            for column in df.columns:
                if column in var_formats.keys():
                    var_formats_.append(var_formats[column])
                else:
                    var_formats_.append(None)
            var_formats = var_formats_
        else:
            raise ValueError(f"{var_formats} is not supported")
    else:
        var_formats = [var_format]*len(df.columns)
    df = df.copy()
    for column, var_format in zip(df.columns, var_formats):
        if var_format is not None:
            df[column] = series_to_str(ds=df[column], var_format=var_format, is_exclude_nans=is_exclude_nans)
    return df


def timeseries_df_to_str(df: pd.DataFrame,
                         freq: Optional[str] = 'QE',
                         date_format: str = '%b-%y',
                         var_format: str = '{:.0%}',
                         var_formats: List[str] = None,  # specific for each column
                         transpose: bool = True
                         ) -> pd.DataFrame:
    """
    time series df to table str
    """
    if freq is not None:
        df = df.resample(freq).last()
    df.index = df.index.strftime(date_format)
    if transpose:
        df = df.T
    df = df_to_str(df, var_format=var_format, var_formats=var_formats)
    return df


def df_with_ci_to_str(df: pd.DataFrame,
                      df_ci: pd.DataFrame,
                      var_format: str = '{:.2f}',
                      is_exclude_nans: bool = True,
                      sep: str = u"\u00B1"
                      ) -> pd.DataFrame:
    """
    pd.DataFrame to string with ci intervals
    """
    df_out = pd.DataFrame(index=df.index, columns=df.columns)
    for column in df.columns:
        val = series_to_str(ds=df[column], var_format=var_format, is_exclude_nans=is_exclude_nans)
        ci = series_to_str(ds=df_ci[column], var_format=var_format, is_exclude_nans=is_exclude_nans)
        df_out[column] = join_str_series(ds1=val, ds2=ci, sep=sep)
    return df_out


def join_str_series(ds1: pd.Series, ds2: pd.Series, sep: str = u"\u00B1") -> np.ndarray:
    """
    joint str series with default separator = +-
    """
    out = np.empty(len(ds1.index), dtype=object)
    for idx, (n1, n2) in enumerate(zip(ds1.to_numpy(), ds2.to_numpy())):
        out[idx] = f"{n1}{sep}{n2}"
    return out


def df_all_to_str(df: pd.DataFrame, index_name: str = '') -> str:
    # Max character length per column
    df.index.name = index_name
    df = df.reset_index()
    s = df.astype(str).agg(lambda x: x.str.len()).max()

    pad = 10  # How many spaces between
    fmts = {}
    for idx, c_len in s.items():
        # Deal with MultIndex tuples or simple string labels.
        if isinstance(idx, tuple):
            lab_len = max([len(str(x)) for x in idx])
        else:
            lab_len = len(str(idx))

        fill = max(lab_len, c_len) + pad - 1
        fmts[idx] = partial(get_fmt_str, fill=fill)
    df_str = df.apply(fmts)
    stats_str = tabulate(df_str, showindex=False, floatfmt='.2f', headers=df.columns)
    return stats_str


def series_values_to_str(ds: pd.Series, include_index: bool = True) -> str:
    data_dict = ds.to_dict()
    data_str = ""
    for k, v in data_dict.items():
        if include_index:
            data_str += f"{k}: {v}, "
        else:
            data_str += f"{v}, "
    return data_str


def df_index_to_str(df: pd.DataFrame,
                    freq: str = 'QE',
                    data_str: str = 'Q%q-%y'
                    ) -> pd.DataFrame:
    df.index = pd.PeriodIndex(pd.to_datetime(df.index).date, freq=freq).strftime(data_str)
    return df


def idx_to_alphabet(idx: int = 1, capitalise: bool = True) -> str:
    """
    map index to alphabet character
    """
    if capitalise:
        return chr(ord('@') + idx)
    else:
        return chr(ord('`') + idx)


class UnitTests(Enum):
    DF_TO_STR = 1


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.DF_TO_STR:
        df = pd.DataFrame({
            "c1": ("a", "bb", "ccc", "dddd", "eeeeee"),
            "c2": (11, 22, 33, 44, 55),
            "a3235235235": [1, 2, 3, 4, 5]
        })
        print(df)

        fmts = df_all_to_str(df)
        print(fmts)

        stats_str = tabulate(df, showindex=True, floatfmt='.2f', headers=df.columns)
        print(stats_str)


if __name__ == '__main__':

    unit_test = UnitTests.DF_TO_STR

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

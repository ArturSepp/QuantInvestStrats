"""
analytics for dataframe aggregation
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, Callable, List, Optional, Tuple, Literal, Dict
from enum import Enum

# qis
import qis.utils.np_ops as npo


def nanmean(df: pd.DataFrame, axis: Literal[0, 1] = 1) -> pd.Series:
    data_np = npo.to_finite_np(data=df, fill_value=np.nan)
    if axis == 0:
        nanmean_data = pd.Series(data=np.nanmean(data_np, axis=0), index=df.columns, name='nanmean')
    else:
        nanmean_data = pd.Series(data=np.nanmean(data_np, axis=1), index=df.index, name='nanmean')
    return nanmean_data


def nanmedian(df: pd.DataFrame, axis: Literal[0, 1] = 1) -> pd.Series:
    data_np = npo.to_finite_np(data=df, fill_value=np.nan)
    if axis == 0:
        nanmean_data = pd.Series(data=np.nanmedian(data_np, axis=0), index=df.columns, name='nanmedian')
    else:
        nanmean_data = pd.Series(data=np.nanmedian(data_np, axis=1), index=df.index, name='nanmedian')
    return nanmean_data


def nansum(df: pd.DataFrame, axis: Literal[0, 1] = 1) -> pd.Series:
    data_np = npo.to_finite_np(data=df, fill_value=np.nan)
    if axis == 0:
        nansum_data = pd.Series(data=np.nansum(data_np, axis=0), index=df.columns, name='nansum')
    else:
        nansum_data = pd.Series(data=np.nansum(data_np, axis=1), index=df.index, name='nansum')
    return nansum_data


def nansum_positive(df: pd.DataFrame, axis: Literal[0, 1] = 1) -> pd.Series:
    signed_np_data = get_signed_np_data(df=df, is_positive=True)
    if axis == 0:
        nansum_data = pd.Series(data=np.nansum(signed_np_data, axis=0), index=df.columns, name='nansum_positive')
    else:
        nansum_data = pd.Series(data=np.nansum(signed_np_data, axis=1), index=df.index, name='nansum_positive')
    return nansum_data


def nanmean_positive(df: pd.DataFrame, axis: Literal[0, 1] = 1) -> pd.Series:
    signed_np_data = get_signed_np_data(df=df, is_positive=True)
    if axis == 0:
        nanmean_data = pd.Series(data=np.nanmean(signed_np_data, axis=0), index=df.columns, name='nanmean_positive')
    else:
        nanmean_data = pd.Series(data=np.nanmean(signed_np_data, axis=1), index=df.index, name='nanmean_positive')
    return nanmean_data


def nansum_clip(df: pd.DataFrame,
                a_min: Optional[float] = None,
                a_max: Optional[float] = None,
                is_min_max_clip_fill: bool = True  # will be filled at min or max
                ) -> pd.Series:
    data_np = npo.to_finite_np(data=df, fill_value=np.nan, a_min=a_min, a_max=a_max, is_min_max_clip_fill=is_min_max_clip_fill)
    nanmean_data = pd.Series(data=np.nansum(data_np, axis=1), index=df.index, name='nansum_clip')
    return nanmean_data


def nanmean_clip(df: pd.DataFrame,
                 a_min: Optional[float] = None,
                 a_max: Optional[float] = None,
                 is_min_max_clip_fill: bool = True  # will be filled at min or max
                 ) -> pd.Series:
    data_np = npo.to_finite_np(data=df, fill_value=np.nan, a_min=a_min, a_max=a_max, is_min_max_clip_fill=is_min_max_clip_fill)
    nanmean_data = pd.Series(data=np.nanmean(data_np, axis=1), index=df.index, name='nanmean_clip')
    return nanmean_data


def nansum_negative(df: pd.DataFrame, axis: Literal[0, 1] = 1) -> pd.Series:
    signed_np_data = get_signed_np_data(df=df, is_positive=False)
    nansum_data = pd.Series(data=np.nansum(signed_np_data, axis=axis), index=df.index, name='nansum_negative')
    return nansum_data


def abssum(df: pd.DataFrame) -> pd.Series:
    data_np = npo.to_finite_np(data=df, fill_value=np.nan)
    nansum_data = pd.Series(data=np.nansum(np.abs(data_np), axis=1), index=df.index, name='abssum')
    return nansum_data


def abssum_positive(df: pd.DataFrame) -> pd.Series:
    signed_np_data = get_signed_np_data(df=df, is_positive=True)
    nansum_data = pd.Series(data=np.nansum(np.abs(signed_np_data), axis=1), index=df.index, name='abssum_positive')
    return nansum_data


def abssum_negative(df: pd.DataFrame) -> pd.Series:
    signed_np_data = get_signed_np_data(df=df, is_positive=False)
    nansum_data = pd.Series(data=np.nansum(np.abs(signed_np_data), axis=1), index=df.index, name='abssum_negative')
    return nansum_data


def sum_weighted(df: pd.Series, weights: pd.Series) -> float:
    return np.nansum(np.multiply(df, weights))


def get_signed_np_data(df: pd.DataFrame,
                       is_positive: bool = True
                       ) -> np.ndarray:
    a = npo.to_finite_np(data=df, fill_value=np.nan)
    if is_positive:
        signed_np_data = np.where(np.greater(a, 0.0), a, np.nan)
    else:
        signed_np_data = np.where(np.less(a, 0.0), a, np.nan)
    return signed_np_data


def agg_median_mad(df: pd.DataFrame,
                   median_col: str = 'Median',
                   mad_col: str = 'Mad std',
                   ratio_col: str = 'Mad std %',
                   is_zeros_to_nan: bool = True,
                   scale: float = 0.67449
                   ) -> Tuple[pd.Series, pd.Series, pd.Series]:

    np_data = df.to_numpy()
    if is_zeros_to_nan:
        np_data = np.where(np.isclose(np_data, 0.0), np.nan, np_data)
    median = np.nanmedian(np_data, axis=1)
    mad = stats.median_abs_deviation(np_data, axis=1, nan_policy='omit', scale=scale)
    ratio = np.divide(mad, median, where=np.isfinite(median))
    median = pd.Series(median, index=df.index, name=median_col)
    mad = pd.Series(mad, index=df.index, name=mad_col)
    ratio = pd.Series(ratio, index=df.index, name=ratio_col)
    return median, mad, ratio


def agg_data_by_axis(df: pd.DataFrame,
                     total_column: Union[str, None] = None,
                     is_total_column_first: bool = False,  # last of not
                     agg_func: Callable[[pd.DataFrame, int], pd.Series] = np.nansum,
                     agg_total_func: Callable[[pd.Series], pd.Series] = np.nansum,
                     axis: int = 0
                     ) -> pd.Series:
    """
    take pandas data which are index=time series, column = assets
    agg_func: pd.DataFrame -> Union[pd.Series, pd.DataFrame] is function
    aggregation by axis=0 -> pd.Series[columns]
    aggregation by axis=1 -> pd.Series[index]
    """
    agg_data = pd.Series(data=agg_func(df, axis=axis), index=df.columns)
    # insert total
    if total_column is not None:
        if not isinstance(total_column, str):
            raise TypeError(f"in agg_data_by_groups: total_column must be string")
        # agg sum by columns
        agg_total = pd.Series(data=agg_total_func(agg_data), index=[total_column])

        if is_total_column_first == 0:
            agg_data = pd.concat([agg_total, agg_data])
        else:
            agg_data = pd.concat([agg_data, agg_total])

    return agg_data


def agg_dfs(dfs: List[pd.DataFrame],
            agg_func: Callable = np.nanmean
            ) -> pd.DataFrame:
    """
    compute average of same shaped pandas
    index must be the same
    not very efficient
    """
    # create pandas indexed by index*column with columns = len(datas)
    pd_data = pd.concat([df.stack() for df in dfs], axis=1)
    # apply mean to aggregate columns
    pd_avg = pd_data.apply(lambda x: agg_func(x.to_numpy()), axis=1)

    # transfrom to dataframe of original index and columns
    avg_data = pd_avg.unstack()

    return avg_data


def last_row(df: pd.DataFrame, axis: Literal[0, 1] = 0) -> np.ndarray:
    if axis == 0:
        ds = df.iloc[-1, :]
    else:
        ds = df.iloc[:, -1]
    return ds.to_numpy()


def compute_df_desc_data(df: pd.DataFrame,
                         funcs: Dict[str, Callable] = {'avg': np.nanmean, 'min': np.nanmin, 'max': np.nanmax, 'last': last_row},
                         axis: Literal[0, 1] = 0
                         ) -> pd.DataFrame:
    desc_data = {}
    for key, func in funcs.items():
        if axis == 0:
            desc_data[key] = pd.Series(func(df, axis=0), index=df.columns)
        else:
            desc_data[key] = pd.Series(func(df, axis=1), index=df.index)
    if axis == 0:
        desc_data = pd.DataFrame.from_dict(desc_data, orient='index')
    else:
        desc_data = pd.DataFrame.from_dict(desc_data, orient='columns')
    return desc_data


class UnitTests(Enum):
    STACK = 1
    NAN_MEAN = 2
    TEST3 = 3
    DESC_DF = 4


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.STACK:
        df = pd.DataFrame(data=[[0, 1], [2, 3]],
                          index=['cat', 'dog'],
                          columns=['weight', 'height'])
        print(df)
        stacked = df.stack()
        print(stacked)
        print(type(stacked.index))

        # melt to
        melted = pd.melt(df, value_vars=df.columns, var_name='myVarname', value_name='myValname')
        print(melted)

    elif unit_test == UnitTests.NAN_MEAN:

        # 4 * 3 matrix
        a = np.array([[np.nan, np.nan, np.nan],
                      [np.nan, np.nan, 1.0],
                      [np.nan, 2.0, 2.0],
                      [2.0, 3.0, 4.0]])

        #print(f"a={a[0]}\nmean(axis=0)={nan_func_to_data(a=a[0], func=np.nanmean, axis=0)};")

        print(f"a={a}\nmean(axis=0)={npo.nan_func_to_data(a=a, func=npo.np_nanmean, axis=0)};")

        print(f"a={a}\nmean(axis=1)={npo.nan_func_to_data(a=a, func=npo.np_nanmean, axis=1)};")

        pd_a = pd.DataFrame(a)
        print(f"pd_a={pd_a};")

        lambda_mean = pd_a.apply(lambda x: x.mean(), axis=1)
        # pd_a_pd = pd_a.apply(lambda x: np.nanmean(x), axis=1)
        print(f"lambda_mean\n{lambda_mean};")

        lambda_std = pd_a.apply(lambda x: x.std(), axis=1)
        # pd_a_pd = pd_a.apply(lambda x: np.nanstd(x), axis=1)
        print(f"lambda_std\n{lambda_std};")

    elif unit_test == UnitTests.TEST3:
        df = pd.DataFrame({'Date': ['2015-05-08', '2015-05-07', '2015-05-06', '2015-05-05', '2015-05-08', '2015-05-07',
                                    '2015-05-06', '2015-05-05'],
                           'Sym': ['aapl', 'aapl', 'aapl', 'aapl', 'aaww', 'aaww', 'aaww', 'aaww'],
                           'Data2': [11, 8, 10, 15, 110, 60, 100, 40],
                           'Data3': [5, 8, 6, 1, 50, 100, 60, 120]})
        print(df)
        df['Data4'] = df['Data3'].groupby(df['Date']).transform('sum')
        print(df)

    elif unit_test == UnitTests.DESC_DF:
        df = pd.DataFrame({'Date': ['2015-05-08', '2015-05-07', '2015-05-06', '2015-05-05', '2015-05-08', '2015-05-07',
                                    '2015-05-06', '2015-05-05'],
                           'Data2': [11, 8, 10, 15, 110, 60, 100, 40],
                           'Data3': [5, 8, 6, 1, 50, 100, 60, 120]}).set_index('Date')
        print(df)
        df_desc_data = compute_df_desc_data(df=df, axis=0)
        print(df_desc_data)


if __name__ == '__main__':

    unit_test = UnitTests.DESC_DF

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

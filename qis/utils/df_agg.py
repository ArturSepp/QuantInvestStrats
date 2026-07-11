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
import qis.utils.df_ops as dfo


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
    # NumPy 2.x: supply `out=` so non-finite-median rows get nan, not uninitialized memory.
    ratio = np.divide(
        mad, median,
        out=np.full_like(mad, np.nan, dtype=float),
        where=np.isfinite(median),
    )
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


def last_row(df: pd.DataFrame, axis: Literal[0, 1] = 0, is_nonan: bool = True) -> np.ndarray:
    if axis == 0:
        if is_nonan:
            ds = dfo.get_last_nonnan_values(df=df)
        else:
            ds = df.iloc[-1, :].to_numpy()
    else:
        ds = df.iloc[:, -1].to_numpy()
    return ds


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


def nanmean_weighted(df: pd.DataFrame,
                     weights: pd.Series,
                     axis: Literal[0, 1] = 1,
                     ) -> pd.Series:
    """
    weighted average across `axis`, with weights RESCALED at each line to sum
    to one over the non-nan entries only (nan/inf data are excluded and the
    remaining weights renormalised). A line with no finite data — or whose
    finite entries all carry zero weight — returns nan.
    axis=1 -> weights indexed by df.columns, result indexed by df.index
    axis=0 -> weights indexed by df.index,   result indexed by df.columns
    """
    data_np = npo.to_finite_np(data=df, fill_value=np.nan)
    keys = df.columns if axis == 1 else df.index
    out_index = df.index if axis == 1 else df.columns
    w = weights.reindex(keys).to_numpy(dtype=np.float64)
    w = np.where(np.isfinite(w), w, 0.0)
    w = w[np.newaxis, :] if axis == 1 else w[:, np.newaxis]
    mask = np.isfinite(data_np)
    wmat = np.where(mask, w, 0.0)
    wsum = np.nansum(wmat, axis=axis, keepdims=True)
    num = np.nansum(np.where(mask, data_np, 0.0) * wmat, axis=axis, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        avg = np.where(wsum > 0.0, num / wsum, np.nan)
    return pd.Series(np.squeeze(avg, axis=axis), index=out_index, name='nanmean_weighted')

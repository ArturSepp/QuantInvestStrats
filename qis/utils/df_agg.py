"""
analytics for dataframe aggregation

all df_* aggregators consume a pd.DataFrame and return a pd.Series.
non-finite entries (nan and +-inf) are excluded from the aggregation: the data are passed
through npo.to_finite_np() first, so +-inf is mapped to nan and then skipped.

axis convention, which is deliberately the opposite of the pandas default:
  axis=1 (default) -> aggregate across columns, result indexed by df.index   (cross-sectional)
  axis=0           -> aggregate across rows,    result indexed by df.columns (time series)
cross-sectional aggregation dominates in qis, so it is the default.
"""
# packages
import functools
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, Callable, List, Optional, Tuple, Literal, Dict
# qis
import qis.utils.np_ops as npo
import qis.utils.df_ops as dfo


def _validate_axis(axis: Literal[0, 1]) -> None:
    """raise on an axis outside {0, 1}: numpy would silently accept -1 and None"""
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis!r}")


def _to_agg_series(agg_data: np.ndarray,
                   df: pd.DataFrame,
                   axis: Literal[0, 1],
                   name: str
                   ) -> pd.Series:
    """attach the index that matches the aggregated axis: axis=0 collapses rows -> df.columns"""
    index = df.columns if axis == 0 else df.index
    return pd.Series(data=agg_data, index=index, name=name)


def _nanmean(a: np.ndarray, axis: Literal[0, 1]) -> np.ndarray:
    """np.nanmean over an all-nan slice returns nan, which is the intended result here:
    suppress the RuntimeWarning rather than leaking it to the caller"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _nanmedian(a: np.ndarray, axis: Literal[0, 1]) -> np.ndarray:
    """np.nanmedian over an all-nan slice returns nan, which is the intended result here"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        return np.nanmedian(a, axis=axis)


def _get_signed_np_data(df: pd.DataFrame,
                        is_positive: bool = True
                        ) -> np.ndarray:
    """mask out entries of the opposite sign, so they are skipped by the nan-aware aggregators"""
    a = npo.to_finite_np(data=df, fill_value=np.nan)
    if is_positive:
        signed_np_data = np.where(np.greater(a, 0.0), a, np.nan)
    else:
        signed_np_data = np.where(np.less(a, 0.0), a, np.nan)
    return signed_np_data


def df_nanmean(df: pd.DataFrame, axis: Literal[0, 1] = 1) -> pd.Series:
    """mean of finite entries along axis"""
    _validate_axis(axis)
    data_np = npo.to_finite_np(data=df, fill_value=np.nan)
    return _to_agg_series(agg_data=_nanmean(data_np, axis=axis), df=df, axis=axis, name='nanmean')


def df_nanmedian(df: pd.DataFrame, axis: Literal[0, 1] = 1) -> pd.Series:
    """median of finite entries along axis"""
    _validate_axis(axis)
    data_np = npo.to_finite_np(data=df, fill_value=np.nan)
    return _to_agg_series(agg_data=_nanmedian(data_np, axis=axis), df=df, axis=axis, name='nanmedian')


def df_nansum(df: pd.DataFrame, axis: Literal[0, 1] = 1) -> pd.Series:
    """sum of finite entries along axis"""
    _validate_axis(axis)
    data_np = npo.to_finite_np(data=df, fill_value=np.nan)
    return _to_agg_series(agg_data=np.nansum(data_np, axis=axis), df=df, axis=axis, name='nansum')


def df_nansum_positive(df: pd.DataFrame, axis: Literal[0, 1] = 1) -> pd.Series:
    """sum of strictly positive finite entries along axis"""
    _validate_axis(axis)
    signed_np_data = _get_signed_np_data(df=df, is_positive=True)
    return _to_agg_series(agg_data=np.nansum(signed_np_data, axis=axis), df=df, axis=axis, name='nansum_positive')


def df_nansum_negative(df: pd.DataFrame, axis: Literal[0, 1] = 1) -> pd.Series:
    """sum of strictly negative finite entries along axis"""
    _validate_axis(axis)
    signed_np_data = _get_signed_np_data(df=df, is_positive=False)
    return _to_agg_series(agg_data=np.nansum(signed_np_data, axis=axis), df=df, axis=axis, name='nansum_negative')


def df_nanmean_positive(df: pd.DataFrame, axis: Literal[0, 1] = 1) -> pd.Series:
    """mean of strictly positive finite entries along axis"""
    _validate_axis(axis)
    signed_np_data = _get_signed_np_data(df=df, is_positive=True)
    return _to_agg_series(agg_data=_nanmean(signed_np_data, axis=axis), df=df, axis=axis, name='nanmean_positive')


def df_nanmean_negative(df: pd.DataFrame, axis: Literal[0, 1] = 1) -> pd.Series:
    """mean of strictly negative finite entries along axis"""
    _validate_axis(axis)
    signed_np_data = _get_signed_np_data(df=df, is_positive=False)
    return _to_agg_series(agg_data=_nanmean(signed_np_data, axis=axis), df=df, axis=axis, name='nanmean_negative')


def df_nansum_clip(df: pd.DataFrame,
                   a_min: Optional[float] = None,
                   a_max: Optional[float] = None,
                   is_min_max_clip_fill: bool = True,  # entries outside [a_min, a_max] are filled at the bound
                   axis: Literal[0, 1] = 1
                   ) -> pd.Series:
    """sum of finite entries along axis, after clipping the data to [a_min, a_max]"""
    _validate_axis(axis)
    data_np = npo.to_finite_np(data=df,
                               fill_value=np.nan,
                               a_min=a_min,
                               a_max=a_max,
                               is_min_max_clip_fill=is_min_max_clip_fill)
    return _to_agg_series(agg_data=np.nansum(data_np, axis=axis), df=df, axis=axis, name='nansum_clip')


def df_nanmean_clip(df: pd.DataFrame,
                    a_min: Optional[float] = None,
                    a_max: Optional[float] = None,
                    is_min_max_clip_fill: bool = True,  # entries outside [a_min, a_max] are filled at the bound
                    axis: Literal[0, 1] = 1
                    ) -> pd.Series:
    """mean of finite entries along axis, after clipping the data to [a_min, a_max]"""
    _validate_axis(axis)
    data_np = npo.to_finite_np(data=df,
                               fill_value=np.nan,
                               a_min=a_min,
                               a_max=a_max,
                               is_min_max_clip_fill=is_min_max_clip_fill)
    return _to_agg_series(agg_data=_nanmean(data_np, axis=axis), df=df, axis=axis, name='nanmean_clip')


def df_abssum(df: pd.DataFrame, axis: Literal[0, 1] = 1) -> pd.Series:
    """sum of absolute values of finite entries along axis: sum |x_i|"""
    _validate_axis(axis)
    data_np = npo.to_finite_np(data=df, fill_value=np.nan)
    return _to_agg_series(agg_data=np.nansum(np.abs(data_np), axis=axis), df=df, axis=axis, name='abssum')


def df_abssum_positive(df: pd.DataFrame, axis: Literal[0, 1] = 1) -> pd.Series:
    """sum of absolute values of strictly positive finite entries along axis"""
    _validate_axis(axis)
    signed_np_data = _get_signed_np_data(df=df, is_positive=True)
    return _to_agg_series(agg_data=np.nansum(np.abs(signed_np_data), axis=axis), df=df, axis=axis,
                          name='abssum_positive')


def df_abssum_negative(df: pd.DataFrame, axis: Literal[0, 1] = 1) -> pd.Series:
    """sum of absolute values of strictly negative finite entries along axis"""
    _validate_axis(axis)
    signed_np_data = _get_signed_np_data(df=df, is_positive=False)
    return _to_agg_series(agg_data=np.nansum(np.abs(signed_np_data), axis=axis), df=df, axis=axis,
                          name='abssum_negative')


def df_nanmean_weighted(df: pd.DataFrame,
                        weights: pd.Series,
                        axis: Literal[0, 1] = 1
                        ) -> pd.Series:
    """
    weighted average across axis, with weights rescaled at each line to sum to one over the
    finite entries only: mu_t = sum_i w_i x_ti / sum_i w_i 1{x_ti finite}.
    a line with no finite data, or whose finite entries all carry zero weight, returns nan.
    axis=1 -> weights indexed by df.columns, result indexed by df.index
    axis=0 -> weights indexed by df.index,   result indexed by df.columns
    """
    _validate_axis(axis)
    data_np = npo.to_finite_np(data=df, fill_value=np.nan)
    keys = df.columns if axis == 1 else df.index
    w = weights.reindex(keys).to_numpy(dtype=np.float64)
    w = np.where(np.isfinite(w), w, 0.0)
    w = w[np.newaxis, :] if axis == 1 else w[:, np.newaxis]
    mask = np.isfinite(data_np)
    wmat = np.where(mask, w, 0.0)
    wsum = np.nansum(wmat, axis=axis, keepdims=True)
    num = np.nansum(np.where(mask, data_np, 0.0) * wmat, axis=axis, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        avg = np.where(wsum > 0.0, num / wsum, np.nan)
    return _to_agg_series(agg_data=np.squeeze(avg, axis=axis), df=df, axis=axis, name='nanmean_weighted')


def series_nansum_weighted(data: pd.Series, weights: pd.Series) -> float:
    """weighted sum of finite entries: sum_i w_i x_i, with non-finite terms skipped"""
    if not isinstance(data, pd.Series):
        raise ValueError(f"data must be pd.Series, got {type(data)!r}")
    if not isinstance(weights, pd.Series):
        raise ValueError(f"weights must be pd.Series, got {type(weights)!r}")
    return float(np.nansum(np.multiply(data, weights)))


def df_last_row(df: pd.DataFrame,
                axis: Literal[0, 1] = 0,
                is_nonan: bool = True
                ) -> np.ndarray:
    """last row (axis=0) or last column (axis=1); is_nonan takes the last finite value per key"""
    _validate_axis(axis)
    if axis == 0:
        if is_nonan:
            ds = dfo.get_last_nonnan_values(df=df)
        else:
            ds = df.iloc[-1, :].to_numpy()
    else:
        ds = df.iloc[:, -1].to_numpy()
    return ds


def agg_median_mad(df: pd.DataFrame,
                   median_col: str = 'Median',
                   mad_col: str = 'Mad std',
                   ratio_col: str = 'Mad std %',
                   is_zeros_to_nan: bool = True,
                   scale: float = 0.67449,
                   axis: Literal[0, 1] = 1
                   ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """cross-sectional median, median absolute deviation, and their ratio mad / median"""
    _validate_axis(axis)
    np_data = df.to_numpy()
    if is_zeros_to_nan:
        np_data = np.where(np.isclose(np_data, 0.0), np.nan, np_data)
    median = _nanmedian(np_data, axis=axis)
    mad = stats.median_abs_deviation(np_data, axis=axis, nan_policy='omit', scale=scale)
    # NumPy 2.x: supply out= so non-finite-median entries get nan, not uninitialized memory
    ratio = np.divide(mad, median,
                      out=np.full_like(mad, np.nan, dtype=float),
                      where=np.isfinite(median))
    median = _to_agg_series(agg_data=median, df=df, axis=axis, name=median_col)
    mad = _to_agg_series(agg_data=mad, df=df, axis=axis, name=mad_col)
    ratio = _to_agg_series(agg_data=ratio, df=df, axis=axis, name=ratio_col)
    return median, mad, ratio


def agg_data_by_axis(df: pd.DataFrame,
                     total_column: Optional[str] = None,
                     is_total_column_first: bool = False,
                     agg_func: Callable[..., np.ndarray] = np.nansum,
                     agg_total_func: Callable[..., float] = np.nansum,
                     axis: Literal[0, 1] = 0
                     ) -> pd.Series:
    """
    aggregate df with agg_func along axis, optionally appending a total entry
    axis=0 -> pd.Series indexed by df.columns
    axis=1 -> pd.Series indexed by df.index
    """
    _validate_axis(axis)
    agg_data = _to_agg_series(agg_data=agg_func(df, axis=axis), df=df, axis=axis, name='agg')

    if total_column is not None:
        if not isinstance(total_column, str):
            raise ValueError(f"total_column must be str, got {type(total_column)!r}")
        agg_total = pd.Series(data=[agg_total_func(agg_data)], index=[total_column])
        if is_total_column_first:
            agg_data = pd.concat([agg_total, agg_data])
        else:
            agg_data = pd.concat([agg_data, agg_total])

    return agg_data


def agg_dfs(dfs: List[pd.DataFrame],
            agg_func: Callable = np.nanmean
            ) -> pd.DataFrame:
    """
    aggregate a list of identically shaped dataframes elementwise with agg_func
    index and columns must agree across dfs
    """
    if len(dfs) == 0:
        raise ValueError("dfs is empty")
    pd_data = pd.concat([df.stack() for df in dfs], axis=1)
    pd_avg = pd_data.apply(lambda x: agg_func(x.to_numpy()), axis=1)
    avg_data = pd_avg.unstack()
    return avg_data


def compute_df_desc_data(df: pd.DataFrame,
                         funcs: Optional[Dict[str, Callable]] = None,
                         axis: Literal[0, 1] = 0
                         ) -> pd.DataFrame:
    """descriptive table of df: one row (axis=0) or column (axis=1) per entry of funcs"""
    _validate_axis(axis)
    if funcs is None:  # never a mutable default argument
        funcs = {'avg': np.nanmean, 'min': np.nanmin, 'max': np.nanmax, 'last': df_last_row}
    desc_data = {}
    for key, func in funcs.items():
        desc_data[key] = _to_agg_series(agg_data=func(df, axis=axis), df=df, axis=axis, name=key)
    if axis == 0:
        desc_data = pd.DataFrame.from_dict(desc_data, orient='index')
    else:
        desc_data = pd.DataFrame.from_dict(desc_data, orient='columns')
    return desc_data
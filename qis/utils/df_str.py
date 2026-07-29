"""
formatting a frame for display: numbers to strings, and the parse back.

``df_to_str`` and ``series_to_str`` are the entry points; ``var_formats`` overrides the default
format per column, positionally when a list is passed and by column name when a dict is.
``timeseries_df_to_str`` resamples first, for a table of returns by period.

nan becomes ``EMPTY_NUM``, a single space, so a fixed-width column stays aligned.
``str_to_float`` and ``df_to_numeric`` strip thousands separators and percent signs and return
nan on what they cannot parse; '%' goes without a divide by 100, so '50%' parses back to 50.0
and the round trip is not an inverse for percent formats. ``tabulate_df`` renders aligned plain
text and is a local stand-in for the tabulate library, which qis does not depend on.

Which format a statistic gets is decided alongside its column definition in ``generic.py``.
"""
# packages
import numpy as np
import pandas as pd
from functools import partial
from pandas.api.types import is_datetime64_any_dtype as is_datetime, is_string_dtype
from typing import List, Optional, Union, Dict, Any
from qis.utils.dates import DATE_FORMAT

# Sentinel for "empty" cells in formatted output.
# Single space preserves column alignment in fixed-width tables; if you don't
# need that, '' is cleaner. Whatever you pick, use it consistently.
EMPTY_NUM = ' '


def get_fmt_str(x: Any, fill: int) -> str:
    """Right-align `x` in a field of width `fill`. Private helper for df_all_to_str."""
    return '{message: >{fill}}'.format(message=x, fill=fill)


def float_to_str(x: Any, var_format: str, is_exclude_nans: bool = True) -> str:
    """
    Format a single scalar as a string.

    Kept as a public helper for backward compatibility, but `series_to_str`
    no longer routes through this function on its hot path — it uses a
    vectorized astype(float) instead, which avoids the historical "x must
    be float not int" / numpy-scalar issues entirely.

    Accepts anything castable to float (int, float, np.integer, np.floating,
    np.bool_, Decimal, ...). NaN handling is controlled by is_exclude_nans.
    """
    try:
        xf = float(x)
    except (TypeError, ValueError) as e:
        raise ValueError(f"cannot format value of type={type(x)}: {x}") from e

    if np.isnan(xf):
        return EMPTY_NUM if is_exclude_nans else var_format.format(xf)
    return var_format.format(xf)


def str_to_float(x: Any) -> float:
    """
    Inverse of float_to_str — parse a formatted string back to float.
    Strips thousands separators and percent signs. Returns NaN on failure.
    """
    # Already-NaN floats and the empty sentinel both map to NaN.
    if isinstance(x, float) and np.isnan(x):
        return np.nan
    if isinstance(x, str) and x.strip() in ('', EMPTY_NUM.strip()):
        return np.nan
    try:
        return float(str(x).replace(',', '').replace('%', ''))
    except (ValueError, AttributeError):
        # Catches non-numeric strings; deliberately narrow — never bare except.
        return np.nan


def date_to_str(x: Any, var_format: str = DATE_FORMAT) -> str:
    """
    Format a Timestamp/date-like as a string. Empty for NaT/None/NaN.
    """
    if x is None or pd.isna(x):  # pd.isna handles NaT, np.nan, None uniformly
        return EMPTY_NUM
    return pd.Timestamp(x).strftime(var_format)


def series_to_str(ds: pd.Series,
                  var_format: str = '{:.2f}',
                  is_exclude_nans: bool = True
                  ) -> pd.Series:
    """
    Format a Series as strings.

    Three branches: string, datetime, numeric. The numeric path is now
    vectorized — astype(float) coerces ints, numpy scalars, bools, etc.
    in one shot, eliminating the per-element isinstance check that used
    to raise on int columns coming out of perf tables.
    """
    if not isinstance(ds, pd.Series):
        raise TypeError(f"expected pd.Series, got {type(ds)}")

    # String columns: pass through unchanged. Caller asked for str, already str.
    if is_string_dtype(ds):
        return ds.copy()

    # Datetime columns: per-element format (strftime is not vectorized for arbitrary fmts).
    if is_datetime(ds):
        return ds.apply(partial(date_to_str, var_format=var_format))

    # Numeric path: coerce to float once, then format.
    # `errors='ignore'` would mask real problems — let astype raise if a column
    # is genuinely non-numeric (caller should have routed it to the string branch).
    numeric = ds.astype(float)

    # Vectorized format via .map. We can't use a true ufunc because var_format
    # is a Python format string, but .map is still 5–10× faster than .apply
    # with a lambda closure on large frames.
    if is_exclude_nans:
        out = numeric.map(lambda v: EMPTY_NUM if np.isnan(v) else var_format.format(v))
    else:
        out = numeric.map(var_format.format)

    # Preserve original index and name.
    out.index = ds.index
    out.name = ds.name
    return out


def series_to_date_str(ds: pd.Series,
                       var_format: str = DATE_FORMAT,
                       is_exclude_nans: bool = True
                       ) -> pd.Series:
    """
    Format a Series as date strings. Thin wrapper around date_to_str.
    """
    if not isinstance(ds, pd.Series):
        raise TypeError(f"expected pd.Series, got {type(ds)}")
    out = ds.apply(partial(date_to_str, var_format=var_format))
    if is_exclude_nans:
        # date_to_str already returns EMPTY_NUM for NaT, but belt-and-braces
        # in case someone passes a pre-formatted string Series with real NaNs.
        out = out.where(out.notna(), EMPTY_NUM)
    return out


def series_to_numeric(ds: pd.Series) -> np.ndarray:
    """
    Convert a Series of mixed/string/numeric content to a float ndarray.
    Used by df_to_numeric for parsing formatted tables back into numbers.
    """
    if pd.api.types.is_numeric_dtype(ds):
        return ds.to_numpy(dtype=float)
    # Object/string dtype: parse each cell. The old `isinstance(ds.dtypes, object)`
    # check was always True (everything inherits from object) — this is the fix.
    return ds.apply(str_to_float).to_numpy(dtype=float)


def df_to_numeric(df: pd.DataFrame) -> np.ndarray:
    """Convert a DataFrame of formatted strings back to a 2-D float ndarray."""
    return df.apply(series_to_numeric).to_numpy()


def df_to_str(df: pd.DataFrame,
              var_format: Optional[str] = '{:.2f}',
              var_formats: Union[List[Optional[str]], Dict[str, str], None] = None,
              is_exclude_nans: bool = True
              ) -> pd.DataFrame:
    """
    Format a DataFrame as strings, with per-column format overrides.

    var_formats can be:
      - None: every column uses var_format
      - list: one format per column, matched positionally (must match df.columns length)
      - dict: {column_name: format}; columns not in dict fall back to var_format
              (or are skipped if var_format is None)
    """
    # Resolve var_formats into a per-column list, then format each column.
    if var_formats is None:
        formats = [var_format] * len(df.columns)
    elif isinstance(var_formats, list):
        if len(var_formats) != len(df.columns):
            raise ValueError(
                f"var_formats length {len(var_formats)} != df.columns length {len(df.columns)}"
            )
        formats = var_formats
    elif isinstance(var_formats, dict):
        # Dict gets merged with var_format default. Missing columns get None
        # (i.e. not formatted) only if var_format is also None.
        formats = [var_formats.get(col, var_format) for col in df.columns]
    else:
        raise TypeError(f"var_formats must be list, dict, or None; got {type(var_formats)}")

    df = df.copy()
    for column, fmt in zip(df.columns, formats):
        if fmt is not None:
            df[column] = series_to_str(ds=df[column], var_format=fmt, is_exclude_nans=is_exclude_nans)
    return df


def timeseries_df_to_str(df: pd.DataFrame,
                         freq: Optional[str] = 'QE',
                         date_format: str = '%b-%y',
                         var_format: str = '{:.0%}',
                         var_formats: Optional[List[str]] = None,
                         transpose: bool = True
                         ) -> pd.DataFrame:
    """
    Format a time-series DataFrame as a string table — typically for display
    of resampled returns by date column.
    """
    df = df.copy()  # explicit; resample returns a new frame but be safe
    if freq is not None:
        df = df.resample(freq).last()
    df.index = df.index.strftime(date_format)
    if transpose:
        df = df.T
    return df_to_str(df, var_format=var_format, var_formats=var_formats)


def df_with_ci_to_str(df: pd.DataFrame,
                      df_ci: pd.DataFrame,
                      var_format: str = '{:.2f}',
                      is_exclude_nans: bool = True,
                      sep: str = u"\u00B1"
                      ) -> pd.DataFrame:
    """
    Format point estimates with confidence intervals: "1.23 ± 0.45".
    """
    df_out = pd.DataFrame(index=df.index, columns=df.columns)
    for column in df.columns:
        val = series_to_str(ds=df[column], var_format=var_format, is_exclude_nans=is_exclude_nans)
        ci = series_to_str(ds=df_ci[column], var_format=var_format, is_exclude_nans=is_exclude_nans)
        df_out[column] = join_str_series(ds1=val, ds2=ci, sep=sep)
    return df_out


def join_str_series(ds1: pd.Series, ds2: pd.Series, sep: str = u"\u00B1") -> pd.Series:
    """
    Element-wise concatenate two string Series with a separator.

    Returns a Series (not ndarray) preserving ds1's index, so it can be
    safely assigned to a DataFrame column without positional surprises.
    Reindexes ds2 to ds1's index to avoid silent zip mismatches.
    """
    ds2_aligned = ds2.reindex(ds1.index)
    return ds1.astype(str) + sep + ds2_aligned.astype(str)


def df_all_to_str(df: pd.DataFrame, index_name: str = '') -> str:
    """
    Render an entire DataFrame as a single aligned plain-text string.
    Used for log dumps and copy-paste-friendly output.
    """
    df = df.copy()  # we mutate index.name and reset_index, don't touch caller's df
    df.index.name = index_name
    df = df.reset_index()
    col_lens = df.astype(str).agg(lambda x: x.str.len()).max()

    pad = 10  # spacing between columns
    fmts = {}
    for idx, c_len in col_lens.items():
        # Handle MultiIndex column tuples and plain string labels uniformly.
        if isinstance(idx, tuple):
            lab_len = max(len(str(x)) for x in idx)
        else:
            lab_len = len(str(idx))
        fill = max(lab_len, c_len) + pad - 1
        fmts[idx] = partial(get_fmt_str, fill=fill)

    df_str = df.apply(fmts)
    return tabulate_df(df_str, showindex=False, floatfmt='.2f', headers=df.columns)


def series_values_to_str(ds: pd.Series, include_index: bool = True) -> str:
    """Flatten a Series into a comma-separated string for logging."""
    if include_index:
        return ', '.join(f"{k}: {v}" for k, v in ds.items())
    return ', '.join(str(v) for v in ds.values)


def df_index_to_str(df: pd.DataFrame,
                    freq: str = 'QE',
                    date_format: str = 'Q%q-%y'  # renamed from data_str for clarity
                    ) -> pd.DataFrame:
    """
    Convert a DatetimeIndex to a formatted period string index (e.g. 'Q1-25').
    """
    df = df.copy()
    df.index = pd.PeriodIndex(pd.to_datetime(df.index).date, freq=freq).strftime(date_format)
    return df


def idx_to_alphabet(idx: int = 1, capitalise: bool = True) -> str:
    """
    Map a 1-based index to an alphabet character (1 -> A, 26 -> Z).
    Excel-style multi-letter (AA, AB, …) for idx > 26.
    """
    if idx < 1:
        raise ValueError(f"idx must be >= 1, got {idx}")
    base = ord('A') if capitalise else ord('a')
    chars = []
    n = idx
    while n > 0:
        n, rem = divmod(n - 1, 26)
        chars.append(chr(base + rem))
    return ''.join(reversed(chars))


def tabulate_df(df: pd.DataFrame,
                showindex: bool = False,
                floatfmt: str = '.2f',
                headers: Union[List[str], pd.Index, None] = None
                ) -> str:
    """
    Render a DataFrame as an aligned plain-text table.
    Drop-in replacement for the `tabulate` library covering the common case;
    keeps qis dependency-free.
    """
    if headers is None:
        headers = list(df.columns)
    else:
        headers = list(headers)

    # Build rows as list of string lists.
    rows = []
    for idx, row in df.iterrows():
        row_strs = [str(idx)] if showindex else []
        for val in row:
            if isinstance(val, float):
                row_strs.append(format(val, floatfmt))
            else:
                row_strs.append(str(val))
        rows.append(row_strs)

    if showindex:
        headers = [str(df.index.name or '')] + headers

    # Compute column widths from header + all rows.
    n_cols = len(headers)
    col_widths = [len(h) for h in headers]
    for row in rows:
        for j in range(n_cols):
            col_widths[j] = max(col_widths[j], len(row[j]))

    # Right-justify everything. (Could be parameterized per-column for
    # left-aligned string columns, but right-align is the common case for
    # numeric tables.)
    header_line = '  '.join(h.rjust(col_widths[j]) for j, h in enumerate(headers))
    sep_line = '  '.join('-' * w for w in col_widths)
    data_lines = ['  '.join(row[j].rjust(col_widths[j]) for j in range(n_cols)) for row in rows]

    return '\n'.join([header_line, sep_line] + data_lines)

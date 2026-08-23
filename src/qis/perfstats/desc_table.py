"""
descriptive statistics per column of a panel, formatted for display.

``compute_desc_table`` transposes the panel - time by ticker in, ticker by statistic out - and
``DescTableType`` selects what is reported: mean and standard deviation first, though
``AVG_WITH_POSITIVE_PROB`` and ``SKEW_KURTOSIS`` drop both again, then skewness, kurtosis, a
normality p-value, quantiles, the median, the positive share, or the percentile rank of the last
value. Values come back as formatted strings because the table renderer takes them directly.

``annualize_vol`` scales the standard deviation by the square root of the factor inferred from
the index via ``infer_annualisation_factor_from_df``, reporting ``STD_AN`` rather than ``STD``.
``is_add_tstat`` divides the mean by that volatility, nan where the mean is not positive; with
``annualize_vol`` off ``an_factor`` is 1.0, so neither side is annualised. Risk-adjusted
statistics are ``perf_stats.py``.
"""
# packages
import numpy as np
import pandas as pd
from typing import Union
from scipy.stats import skew, kurtosis, percentileofscore, normaltest
from enum import Enum

# qis
from qis.utils.annualisation import infer_annualisation_factor_from_df
from qis.perfstats.config import PerfStat


class DescTableType(Enum):
    NONE = 0
    SHORT = 1
    AVG_WITH_POSITIVE_PROB = 2
    WITH_POSITIVE_PROB = 3
    WITH_KURTOSIS = 4
    WITH_NORMAL_PVAL = 5
    WITH_SCORE = 6
    EXTENSIVE = 7
    SKEW_KURTOSIS = 8
    WITH_MEDIAN = 9


def _compute_positive_probability(data: np.ndarray) -> np.ndarray:
    """Compute each column's positive share over its observed values.

    Args:
        data: Two-dimensional numerical observations arranged by row and column.

    Returns:
        Positive-observation probability for each column, or NaN when a column has no
        observations.
    """
    # Missing returns are absent observations, not non-positive outcomes.
    observed_counts = np.sum(np.logical_not(np.isnan(data)), axis=0)
    positive_counts = np.sum(np.greater(data, 0.0), axis=0)
    return np.divide(
        positive_counts,
        observed_counts,
        out=np.full_like(positive_counts, np.nan, dtype=float),
        where=np.greater(observed_counts, 0),
    )


def compute_desc_table(df: Union[pd.DataFrame, pd.Series],
                       desc_table_type: DescTableType = DescTableType.SHORT,
                       var_format: str = '{:.2f}',
                       annualize_vol: bool = False,
                       is_add_tstat: bool = False,
                       norm_variable_display_type: str = '{:.1f}',  # for t-stsat
                       **kwargs
                       ) -> pd.DataFrame:
    """
    descriptive statistics per column, formatted for display.

    Transposes the panel: the input is time by ticker, the output is ticker by statistic.
    Values are returned as formatted strings, not numbers, because this feeds the table
    renderer directly — use the underlying statistic functions if the numbers are wanted.
    Columns may contain nans; statistics are computed on the available observations.
    Positive probabilities divide positive returns by non-missing observations in each column;
    zero returns are observed and non-positive.

    Args:
        df: returns panel, index is time and columns are tickers; a Series is treated as one
            column named after it
        desc_table_type: which set of statistics to report; positive-probability modes use each
            column's non-missing observation count as the denominator
        var_format: format applied to the statistics
        annualize_vol: report volatility per annum rather than per period; reduced modes omit
            the volatility column selected by this convention
        is_add_tstat: add the t-statistic of the mean
        norm_variable_display_type: format applied to the t-statistic

    Returns:
        table indexed by ticker, one column per statistic, values as strings

    Raises:
        TypeError: if ``df`` is neither pd.DataFrame nor pd.Series
    """
    if isinstance(df, pd.DataFrame):
        descriptive_table = pd.DataFrame(index=df.columns)
    elif isinstance(df, pd.Series):
        descriptive_table = pd.DataFrame(index=[df.name])
        df = df.to_frame()
    else:
        raise TypeError(f"unsupported data type = {type(df)}")

    data_np = df.to_numpy()
    mean = np.nanmean(data_np, axis=0)
    std = np.nanstd(data_np, ddof=1, axis=0)

    descriptive_table[PerfStat.AVG.to_str()] = [var_format.format(x) for x in mean]

    if annualize_vol:
        an_factor = infer_annualisation_factor_from_df(data=df)
        vol = std * np.sqrt(an_factor)
        volatility_column = PerfStat.STD_AN.value.name
    else:
        an_factor = 1.0
        vol = std
        volatility_column = PerfStat.STD.value.name
    # Keep the selected label so reduced modes remove the column they actually created.
    descriptive_table[volatility_column] = [var_format.format(x) for x in vol]

    if is_add_tstat:
        an_mean = an_factor * mean
        # NumPy 2.x: explicit out= so masked positions are deterministic nan, not uninitialized memory.
        # NB: original mask is `an_mean > 0`, not `vol > 0` — preserved; revisit if the intent was
        # actually to guard div-by-zero on vol rather than to filter on mean sign.
        tstats = np.divide(
            an_mean, vol,
            out=np.full_like(an_mean, np.nan, dtype=float),
            where=np.greater(an_mean, 0.0),
        )
        descriptive_table[PerfStat.T_STAT.to_str()] = [norm_variable_display_type.format(x) for x in tstats]

    nan_policy = 'omit'  # skip nans
    if desc_table_type == desc_table_type.SHORT:
        pass

    elif desc_table_type == desc_table_type.AVG_WITH_POSITIVE_PROB:
        # Remove the setup columns before reporting the reduced positive-only schema.
        descriptive_table = descriptive_table.drop(
            [PerfStat.AVG.value.name, volatility_column], axis=1)

        prob = _compute_positive_probability(data=data_np)
        descriptive_table[PerfStat.POSITIVE.to_str(short=True, short_n=True)] = ['{:.1%}'.format(x) for x in prob]

    elif desc_table_type == desc_table_type.WITH_POSITIVE_PROB:
        prob = _compute_positive_probability(data=data_np)
        descriptive_table[PerfStat.POSITIVE.to_str(short=True, short_n=True)] = ['{:.1%}'.format(x) for x in prob]

    elif desc_table_type == desc_table_type.WITH_KURTOSIS:
        descriptive_table[PerfStat.SKEWNESS.to_str(short=True, short_n=True)] = [norm_variable_display_type.format(x) for x in skew(data_np, axis=0, nan_policy=nan_policy)]
        descriptive_table[PerfStat.KURTOSIS.to_str(short=True, short_n=True)] = [norm_variable_display_type.format(x) for x in kurtosis(data_np, axis=0, nan_policy=nan_policy)]

    elif desc_table_type == desc_table_type.WITH_NORMAL_PVAL:
        descriptive_table[PerfStat.SKEWNESS.to_str(short=True, short_n=True)] = [norm_variable_display_type.format(x) for x in skew(data_np, axis=0, nan_policy=nan_policy)]
        descriptive_table[PerfStat.KURTOSIS.to_str(short=True, short_n=True)] = [norm_variable_display_type.format(x) for x in kurtosis(data_np, axis=0, nan_policy=nan_policy)]
        k2, ps = normaltest(a=data_np, axis=0, nan_policy='omit')
        descriptive_table[PerfStat.NORMTEST.to_str(short=True, short_n=True)] = ['{:.2f}'.format(x) for x in ps]

    elif desc_table_type == desc_table_type.SKEW_KURTOSIS:
        # Remove the setup columns before reporting the reduced moment-only schema.
        descriptive_table = descriptive_table.drop(
            [PerfStat.AVG.value.name, volatility_column], axis=1)
        descriptive_table[PerfStat.SKEWNESS.to_str(short=True, short_n=True)] = [norm_variable_display_type.format(x) for x in skew(data_np, axis=0, nan_policy=nan_policy)]
        descriptive_table[PerfStat.KURTOSIS.to_str(short=True, short_n=True)] = [norm_variable_display_type.format(x) for x in kurtosis(data_np, axis=0, nan_policy=nan_policy)]
    elif desc_table_type == desc_table_type.WITH_SCORE:
        column_data = [df[column].dropna() for column in df.columns]
        percentiles = [percentileofscore(a=x, score=x.iloc[-1], kind='rank') for x in column_data]
        descriptive_table[PerfStat.LAST.to_str()] = [var_format.format(x.iloc[-1]) for x in column_data]
        descriptive_table[PerfStat.RANK.to_str()] = ['{:.0%}'.format(0.01*x) for x in percentiles]

    elif desc_table_type == desc_table_type.EXTENSIVE:
        descriptive_table[PerfStat.SKEWNESS.to_str(short=True, short_n=True)] \
            = [norm_variable_display_type.format(x) for x in skew(df.values, axis=0, nan_policy=nan_policy)]
        descriptive_table[PerfStat.KURTOSIS.to_str(short=True, short_n=True)] \
            = [norm_variable_display_type.format(x) for x in kurtosis(df.values, axis=0, nan_policy=nan_policy)]
        descriptive_table[PerfStat.MIN.to_str()] \
            = [var_format.format(x) for x in np.nanmin(df.values, axis=0)]
        descriptive_table[PerfStat.QUANT_M_1STD.to_str(short=True)]\
            = [var_format.format(x) for x in np.nanquantile(df.values, q=0.16, axis=0)]
        descriptive_table[PerfStat.MEDIAN.to_str(short=True)] \
            = [var_format.format(x) for x in np.nanmedian(df.values, axis=0)]
        descriptive_table[PerfStat.QUANT_P1_STD.to_str(short=True)] \
            = [var_format.format(x) for x in np.nanquantile(df.values, q=0.84, axis=0)]
        descriptive_table[PerfStat.MAX.to_str()] \
            = [var_format.format(x) for x in np.nanmax(df.values, axis=0)]

    elif desc_table_type == desc_table_type.WITH_MEDIAN:
        descriptive_table[PerfStat.MEDIAN.to_str(short=True)] \
            = [var_format.format(x) for x in np.nanmedian(df.values, axis=0)]
        descriptive_table[PerfStat.SKEWNESS.to_str(short=True, short_n=True)] \
            = [norm_variable_display_type.format(x) for x in skew(df.values, axis=0, nan_policy=nan_policy)]
        descriptive_table[PerfStat.KURTOSIS.to_str(short=True, short_n=True)] \
            = [norm_variable_display_type.format(x) for x in kurtosis(df.values, axis=0, nan_policy=nan_policy)]

    else:
        raise TypeError(f"desc_table_type={desc_table_type} is not implemented")

    return descriptive_table

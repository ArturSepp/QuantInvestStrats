"""
descriptive statistics per column of a panel, formatted for display.

``compute_desc_table`` transposes the panel - time by ticker in, ticker by statistic out - and
``DescTableType`` selects what is reported: mean and standard deviation first, though
``AVG_WITH_POSITIVE_PROB`` and ``SKEW_KURTOSIS`` drop both again, then skewness, kurtosis, a
normality p-value, quantiles, the median, the positive share, or the percentile rank of the last
value. Values come back as formatted strings because the table renderer takes them directly.

``annualize_vol`` scales the standard deviation by the square root of the factor inferred from
the index via ``infer_annualisation_factor_from_df``, reporting ``STD_AN`` rather than ``STD``.
``is_add_tstat`` reports the signed sample mean divided by its standard error, independently of
whether the volatility column is annualised. Risk-adjusted statistics are ``perf_stats.py``.
"""
# packages
import numpy as np
import pandas as pd
from typing import Callable, Union
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


def _reduce_observed(
        data: np.ndarray,
        reduction: Callable[[np.ndarray], np.ndarray],
        minimum_observations: int = 1) -> np.ndarray:
    """Apply a reduction only to columns meeting its sample-size requirement.

    Args:
        data: Two-dimensional numerical observations arranged by row and column.
        reduction: Column-wise reduction returning one value per supplied column.
        minimum_observations: Required number of non-missing values in each column.

    Returns:
        Reduction values in input-column order, with NaN for undersized columns.
    """
    # Select columns per statistic so undersized samples never reach warning-producing reducers.
    observation_counts = np.sum(np.logical_not(np.isnan(data)), axis=0)
    eligible_columns = np.greater_equal(observation_counts, minimum_observations)
    values = np.full(data.shape[1], np.nan, dtype=float)
    if np.any(eligible_columns):
        values[eligible_columns] = reduction(data[:, eligible_columns])
    return values


def compute_sample_mean_tstat(
        mean: np.ndarray,
        sample_std: np.ndarray,
        observation_counts: np.ndarray) -> np.ndarray:
    """Compute signed one-sample t-statistics from column summaries.

    Args:
        mean: Sample mean for each column.
        sample_std: Sample standard deviation with ``ddof=1`` for each column.
        observation_counts: Non-missing sample size for each column.

    Returns:
        Mean divided by its standard error, with NaN for undersized or zero-volatility samples.
    """
    # Sample inference uses observed counts and is independent of display annualization.
    eligible_columns = (
        np.greater_equal(observation_counts, 2)
        & np.greater(sample_std, 0.0)
        & np.isfinite(sample_std)
    )
    return np.divide(
        mean * np.sqrt(observation_counts),
        sample_std,
        out=np.full_like(mean, np.nan, dtype=float),
        where=eligible_columns,
    )


def _add_moment_columns(
        descriptive_table: pd.DataFrame,
        data: np.ndarray,
        value_format: str) -> None:
    """Add formatted skewness and kurtosis for each observed input column.

    Args:
        descriptive_table: Output table to which the two moment columns are added.
        data: Two-dimensional numerical observations arranged by row and column.
        value_format: Display format applied to each moment.
    """
    # Two observations define the displayed moments; smaller samples remain explicitly missing.
    skews = _reduce_observed(
        data, lambda values: skew(values, axis=0, nan_policy='omit'),
        minimum_observations=2)
    kurts = _reduce_observed(
        data, lambda values: kurtosis(values, axis=0, nan_policy='omit'),
        minimum_observations=2)
    descriptive_table[PerfStat.SKEWNESS.value.short] = [
        value_format.format(x) for x in skews]
    descriptive_table[PerfStat.KURTOSIS.value.short_n] = [
        value_format.format(x) for x in kurts]


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
    Columns may contain nans or ``pd.NA`` in nullable numeric dtypes; both are treated as missing,
    and statistics are computed on the available observations.
    Dated columns without observations remain in the table with formatted missing statistics
    and do not emit reduction warnings.
    Statistics whose columns do not meet their minimum sample sizes also remain formatted as
    missing: sample standard deviation, skewness, and kurtosis require two observations, while
    the normality p-value requires 20 observations.
    Positive probabilities divide positive returns by non-missing observations in each column;
    zero returns are observed and non-positive.

    Args:
        df: returns panel, index is time and columns are tickers; a Series is treated as one
            column named after it; repeated DataFrame column labels are retained and calculated
            independently
        desc_table_type: which set of statistics to report; positive-probability modes use each
            column's non-missing observation count as the denominator
        var_format: format applied to the statistics
        annualize_vol: report volatility per annum rather than per period; reduced modes omit
            the volatility column selected by this convention
        is_add_tstat: add the signed sample mean divided by its standard error; samples with fewer
            than two observations or non-positive sample volatility remain formatted as missing
        norm_variable_display_type: format applied to the t-statistic

    Returns:
        table indexed by ticker, one column per statistic, values as strings

    Raises:
        TypeError: if ``df`` is neither pd.DataFrame nor pd.Series
        ValueError: if ``df`` contains no observations
    """
    if isinstance(df, pd.DataFrame):
        descriptive_table = pd.DataFrame(index=df.columns)
    elif isinstance(df, pd.Series):
        descriptive_table = pd.DataFrame(index=[df.name])
        df = df.to_frame()
    else:
        raise TypeError(f"unsupported data type = {type(df)}")

    # Reject zero rows uniformly before reducers or annualization produce incidental outcomes.
    if df.index.empty:
        raise ValueError("data must contain at least one observation")

    # Normalize nullable pandas values so numerical reducers receive np.nan rather than pd.NA.
    data_np = df.to_numpy(dtype=float, na_value=np.nan)
    observation_counts = np.sum(np.logical_not(np.isnan(data_np)), axis=0)
    # Skip all-missing dated columns so undefined base statistics remain NaN without warnings.
    mean = _reduce_observed(data_np, lambda values: np.nanmean(values, axis=0))
    std = _reduce_observed(
        data_np,
        lambda values: np.nanstd(values, ddof=1, axis=0),
        minimum_observations=2)

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
        # Keep inference signed and sample-based even when volatility is displayed annually.
        tstats = compute_sample_mean_tstat(
            mean=mean,
            sample_std=std,
            observation_counts=observation_counts,
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
        # Evaluate moments only for observed columns so SciPy does not warn for empty samples.
        _add_moment_columns(descriptive_table, data_np, norm_variable_display_type)

    elif desc_table_type == desc_table_type.WITH_NORMAL_PVAL:
        # Apply moments and normality only where a sample exists, retaining NaN elsewhere.
        _add_moment_columns(descriptive_table, data_np, norm_variable_display_type)
        # Require SciPy's warning-free accuracy threshold, not only its eight-value hard minimum.
        ps = _reduce_observed(
            data_np,
            lambda values: normaltest(a=values, axis=0, nan_policy=nan_policy)[1],
            minimum_observations=20)
        descriptive_table[PerfStat.NORMTEST.value.short_n] = [
            '{:.2f}'.format(x) for x in ps]

    elif desc_table_type == desc_table_type.SKEW_KURTOSIS:
        # Remove setup columns and leave unobserved moments undefined in the reduced schema.
        descriptive_table = descriptive_table.drop(
            [PerfStat.AVG.value.name, volatility_column], axis=1)
        _add_moment_columns(descriptive_table, data_np, norm_variable_display_type)
    elif desc_table_type == desc_table_type.WITH_SCORE:
        # Iterate physical columns so repeated labels are scored independently.
        column_data = [column.dropna() for _, column in df.items()]
        # A dated but unobserved history has neither a last value nor a percentile rank.
        last_values = [x.iloc[-1] if not x.empty else np.nan for x in column_data]
        percentiles = [
            percentileofscore(a=x, score=last_value, kind='rank') if not x.empty else np.nan
            for x, last_value in zip(column_data, last_values)
        ]
        descriptive_table[PerfStat.LAST.value.name] = [var_format.format(x) for x in last_values]
        descriptive_table[PerfStat.RANK.to_str()] = ['{:.0%}'.format(0.01*x) for x in percentiles]

    elif desc_table_type == desc_table_type.EXTENSIVE:
        # Apply each extended reduction only to columns containing observations.
        _add_moment_columns(descriptive_table, data_np, norm_variable_display_type)
        minimums = _reduce_observed(data_np, lambda values: np.nanmin(values, axis=0))
        lower_quantiles = _reduce_observed(
            data_np, lambda values: np.nanquantile(values, q=0.16, axis=0))
        medians = _reduce_observed(data_np, lambda values: np.nanmedian(values, axis=0))
        upper_quantiles = _reduce_observed(
            data_np, lambda values: np.nanquantile(values, q=0.84, axis=0))
        maximums = _reduce_observed(data_np, lambda values: np.nanmax(values, axis=0))
        descriptive_table[PerfStat.MIN.to_str()] \
            = [var_format.format(x) for x in minimums]
        descriptive_table[PerfStat.QUANT_M_1STD.to_str(short=True)]\
            = [var_format.format(x) for x in lower_quantiles]
        descriptive_table[PerfStat.MEDIAN.to_str(short=True)] \
            = [var_format.format(x) for x in medians]
        descriptive_table[PerfStat.QUANT_P1_STD.to_str(short=True)] \
            = [var_format.format(x) for x in upper_quantiles]
        descriptive_table[PerfStat.MAX.to_str()] \
            = [var_format.format(x) for x in maximums]

    elif desc_table_type == desc_table_type.WITH_MEDIAN:
        # Leave unobserved medians and moments as NaN without invoking warning-producing reducers.
        medians = _reduce_observed(data_np, lambda values: np.nanmedian(values, axis=0))
        descriptive_table[PerfStat.MEDIAN.to_str(short=True)] \
            = [var_format.format(x) for x in medians]
        _add_moment_columns(descriptive_table, data_np, norm_variable_display_type)

    else:
        raise TypeError(f"desc_table_type={desc_table_type} is not implemented")

    return descriptive_table

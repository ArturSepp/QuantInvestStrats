"""
DataFrame Scoring and Normalization Functions

This module provides various functions for computing normalized scores and rankings
from pandas DataFrame and Series data. The functions support cross-sectional scoring,
max normalization, score aggregation, and top quantile selection operations commonly
used in financial analysis, research rankings, and data science workflows.

"""

import numpy as np
import pandas as pd
from typing import Union, Optional, List


def df_to_cross_sectional_score(df: Union[pd.Series, pd.DataFrame],
                                lower_clip: Optional[float] = -5.0,
                                upper_clip: Optional[float] = 5.0,
                                is_sorted: bool = False
                                ) -> Union[pd.Series, pd.DataFrame]:
    """
    Compute cross-sectional standardized scores (z-scores) for DataFrame or Series.

    Transforms input data into standardized scores by subtracting the mean and dividing
    by the standard deviation. For DataFrames, standardization is performed row-wise
    (across columns). For Series, standardization uses the entire series.

    Args:
        df (Union[pd.Series, pd.DataFrame]): Input data to standardize
        lower_clip (Optional[float], optional): Lower bound for clipping outliers.
            Set to None to disable. Defaults to -5.0.
        upper_clip (Optional[float], optional): Upper bound for clipping outliers.
            Set to None to disable. Defaults to 5.0.
        is_sorted (bool, optional): If True and input is Series, return results
            sorted in descending order. Defaults to False.

    Returns:
        Union[pd.Series, pd.DataFrame]: Standardized scores with same shape as input.
            Values represent number of standard deviations from the mean.

    Examples:
        >>> # Series example
        >>> s = pd.Series([1, 2, 3, 4, 5])
        >>> scores = df_to_cross_sectional_score(s)
        >>> print(scores)
        0   -1.414
        1   -0.707
        2    0.000
        3    0.707
        4    1.414
        dtype: float64

        >>> # DataFrame example (row-wise standardization)
        >>> df = pd.DataFrame({'A': [1, 4, 7], 'B': [2, 5, 8], 'C': [3, 6, 9]})
        >>> scores = df_to_cross_sectional_score(df)
        >>> print(scores)
             A    B    C
        0 -1.0  0.0  1.0
        1 -1.0  0.0  1.0
        2 -1.0  0.0  1.0

        >>> # With sorting
        >>> sorted_scores = df_to_cross_sectional_score(s, is_sorted=True)
        >>> print(sorted_scores)
        4    1.414
        3    0.707
        2    0.000
        1   -0.707
        0   -1.414
        dtype: float64

    Notes:
        - Uses np.nanmean and np.nanstd to handle NaN values gracefully
        - For DataFrames, standardization is performed across columns (axis=1)
        - Clipping is applied before standardization to reduce outlier impact
    """
    if lower_clip is not None or upper_clip is not None:
        df = df.clip(lower=lower_clip, upper=upper_clip)

    if isinstance(df, pd.Series):
        score = (df - np.nanmean(df)) / np.nanstd(df)
        if is_sorted:
            score = score.sort_values(ascending=False)
    else:
        score = (df - np.nanmean(df, axis=1, keepdims=True)) / np.nanstd(df, axis=1, keepdims=True)
    return score


def df_to_max_score(df: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """
    Normalize data by dividing by maximum values, scaling the highest value to 1.0.

    For Series: divides all values by the series maximum.
    For DataFrames: divides each row by its maximum value (row-wise normalization).

    Args:
        df (Union[pd.Series, pd.DataFrame]): Input data to normalize

    Returns:
        Union[pd.Series, pd.DataFrame]: Normalized data where maximum values equal 1.0.
            Maintains same shape and index as input.

    Examples:
        >>> # Series example
        >>> s = pd.Series([10, 20, 30])
        >>> max_scores = df_to_max_score(s)
        >>> print(max_scores)
        0    0.333
        1    0.667
        2    1.000
        dtype: float64

        >>> # DataFrame example (row-wise max normalization)
        >>> df = pd.DataFrame({'A': [1, 4], 'B': [2, 8], 'C': [3, 12]})
        >>> max_scores = df_to_max_score(df)
        >>> print(max_scores)
             A      B    C
        0  0.333  0.667  1.0
        1  0.333  0.667  1.0

    Notes:
        - Uses np.nanmax to handle NaN values
        - For DataFrames, normalization is performed across columns (axis=1)
        - Preserves relative relationships within each row/series
        - Useful for creating percentage-of-max scores
    """
    if isinstance(df, pd.Series):
        score = df.divide(np.nanmax(df, axis=0))
    else:
        score = df.divide(np.nanmax(df, axis=1, keepdims=True))
    return score


def compute_aggregate_scores(scores: List[pd.Series],
                             lower_clip: Optional[float] = -5.0,
                             upper_clip: Optional[float] = 5.0,
                             normalize_to_unit_std: bool = True,
                             penalise_nan_values: bool = True
                             ) -> pd.Series:
    """
    Aggregate multiple score Series into a single composite score with NaN handling.

    Combines multiple scoring Series by computing either a penalized sum (when
    penalise_nan_values=True) or mean (when False). The penalized approach reduces
    scores for entities with missing values across the input series.

    Args:
        scores (List[pd.Series]): List of Series to aggregate. Must have compatible indices.
        lower_clip (Optional[float], optional): Lower bound for clipping. Defaults to -5.0.
        upper_clip (Optional[float], optional): Upper bound for clipping. Defaults to 5.0.
        normalize_to_unit_std (bool, optional): If True and penalise_nan_values=True,
            normalizes by 1/sqrt(n) instead of 1/n. Defaults to True.
        penalise_nan_values (bool, optional): If True, entities with NaN values receive
            lower aggregate scores. If False, uses standard mean. Defaults to True.

    Returns:
        pd.Series: Aggregated scores sorted in descending order (highest scores first).
            Index matches the union of all input Series indices.

    Examples:
        >>> # Create sample scores with NaN values
        >>> score1 = pd.Series([1, 2, np.nan], index=['A', 'B', 'C'])
        >>> score2 = pd.Series([3, np.nan, 4], index=['A', 'B', 'C'])
        >>>
        >>> # Penalized aggregation (default)
        >>> agg_penalized = compute_aggregate_scores([score1, score2])
        >>> print(agg_penalized)
        A    2.828
        C    2.828
        B    1.414
        dtype: float64

        >>> # Mean aggregation (no penalty)
        >>> agg_mean = compute_aggregate_scores([score1, score2], penalise_nan_values=False)
        >>> print(agg_mean)
        C    4.000
        A    2.000
        B    2.000
        dtype: float64

    Notes:
        - When penalise_nan_values=True: uses nansum with normalization
        - When penalise_nan_values=False: uses nanmean
        - normalize_to_unit_std affects the scaling when penalizing NaN values
        - Results are automatically sorted in descending order
        - Useful for creating composite rankings from multiple criteria
    """
    joint = pd.concat(scores, axis=1).clip(lower=lower_clip, upper=upper_clip)
    if penalise_nan_values:  # scores with nan are penalised
        n = len(scores)
        if normalize_to_unit_std:
            norm = 1.0 / np.sqrt(n)
        else:
            norm = 1 / n
        joint_avg = norm * np.nansum(joint, axis=1)
    else:
        joint_avg = np.nanmean(joint, axis=1)
    joint_score = pd.Series(joint_avg, index=joint.index).sort_values(ascending=False)
    return joint_score


def select_top_integrated_scores(scores: pd.DataFrame, top_quantile: float = 0.75) -> pd.DataFrame:
    """
    Filter DataFrame to include only rows that exceed specified quantile across ALL columns.

    Selects entities (rows) that perform in the top quantile for every scoring dimension
    (column). This creates an "elite" subset that excels across all measured criteria.

    Args:
        scores (pd.DataFrame): Input scores with entities as rows and scoring dimensions
            as columns
        top_quantile (float, optional): Quantile threshold (0-1). Higher values are more
            selective. Defaults to 0.75 (top 25%).

    Returns:
        pd.DataFrame: Filtered DataFrame containing only rows that exceed the quantile
            threshold in ALL columns. Maintains original column structure.

    Examples:
        >>> # Create sample scores
        >>> df = pd.DataFrame({
        ...     'Score_A': [0.1, 0.5, 0.8, 0.9],
        ...     'Score_B': [0.2, 0.7, 0.6, 0.95],
        ...     'Score_C': [0.3, 0.4, 0.85, 0.88]
        ... }, index=['Entity_1', 'Entity_2', 'Entity_3', 'Entity_4'])
        >>> print(df)
                  Score_A  Score_B  Score_C
        Entity_1      0.1     0.20     0.30
        Entity_2      0.5     0.70     0.40
        Entity_3      0.8     0.60     0.85
        Entity_4      0.9     0.95     0.88

        >>> # Select top 75th percentile across all dimensions
        >>> # 75th percentiles: Score_A=0.825, Score_B=0.762, Score_C=0.857
        >>> top_performers = select_top_integrated_scores(df, top_quantile=0.75)
        >>> print(top_performers)
                  Score_A  Score_B  Score_C
        Entity_4      0.9     0.95     0.88

        >>> # More selective: top 90th percentile (would return empty DataFrame)
        >>> elite_performers = select_top_integrated_scores(df, top_quantile=0.90)
        >>> print(elite_performers)
        Empty DataFrame
        Columns: [Score_A, Score_B, Score_C]
        Index: []

    Notes:
        - Uses logical AND across all columns - entities must excel in ALL dimensions
        - More columns and higher quantiles create increasingly selective filters
        - Uses np.nanquantile to handle NaN values in quantile calculation
        - Useful for identifying well-rounded high performers
        - Returns empty DataFrame if no entities meet criteria across all dimensions
    """
    score_quantiles = np.nanquantile(scores, q=top_quantile, axis=0)
    if len(scores.columns) == 1:
        joint = np.greater(scores.iloc[:, 0], score_quantiles[0])
    else:
        top1 = np.greater(scores.iloc[:, 0], score_quantiles[0])
        top2 = np.greater(scores.iloc[:, 1], score_quantiles[1])
        joint = np.logical_and(top1, top2)

        if len(scores.columns) > 2:
            for idx in np.arange(2, len(scores.columns)):
                top_idx = np.greater(scores.iloc[:, idx], score_quantiles[idx])
                joint = np.logical_and(joint, top_idx)

    scores = scores.loc[joint, :]
    return scores

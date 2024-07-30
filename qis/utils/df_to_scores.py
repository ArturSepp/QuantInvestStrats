"""
various df functions to create scores of df data
"""
import numpy as np
import pandas as pd
from typing import Union, Optional, List


def df_to_cross_sectional_score(df: Union[pd.Series, pd.DataFrame],
                                lower_clip: Optional[float] = -5.0,
                                upper_clip: Optional[float] = 5.0
                                ) -> Union[pd.Series, pd.DataFrame]:
    """
    compute cross sectional score
    """
    if lower_clip is not None or upper_clip is not None:
        df = df.clip(lower=lower_clip, upper=upper_clip)

    if isinstance(df, pd.Series):
        score = (df - np.nanmean(df)) / np.nanstd(df)
    else:
        score = (df - np.nanmean(df, axis=1, keepdims=True)) / np.nanstd(df, axis=1, keepdims=True)
    return score


def df_to_max_score(df: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """
    normalized rows by cross-sectional max: max element = 1.0
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
    aggregate list scores
    nan values are penalised
    normalize_to_unit_std: avg of scores is unit std
    """
    joint = pd.concat(scores, axis=1).clip(lower=lower_clip, upper=upper_clip)
    if penalise_nan_values: # scores with nan are penalised
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

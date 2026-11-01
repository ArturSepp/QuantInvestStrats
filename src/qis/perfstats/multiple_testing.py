"""Holm and Benjamini-Yekutieli corrections for an explicitly supplied test family.

Holm controls the family-wise error rate (FWER); BHY uses the Benjamini-Yekutieli
harmonic correction to control the false discovery rate (FDR) under arbitrary dependence.
Both require valid marginal p-values and a family specified before inspecting results.
For sorted p-values p_(i), Holm takes cumulative maxima of (M-i+1)*p_(i), while BHY
takes reverse cumulative minima of M*H_M*p_(i)/i, with H_M=sum(1/j, j=1,...,M).
Adjusted values are capped at one and returned in the supplied hypothesis order.

P-values and error targets are dimensionless. This module adjusts supplied p-values;
it does not estimate returns, reconstruct omitted research trials, or adjust intervals.
"""

# packages
import numpy as np
import pandas as pd
from enum import Enum
from numbers import Real
from statsmodels.stats.multitest import multipletests


class MultipleTestingMethod(str, Enum):
    """Supported explicit-family error-control procedures.

    Attributes:
        HOLM: Holm step-down FWER control, mapped to statsmodels ``'holm'``.
        BHY: Benjamini-Yekutieli FDR control, mapped to ``'fdr_by'`` with the
            harmonic dependence correction; this is not Benjamini-Hochberg.
    """

    HOLM = 'holm'
    BHY = 'fdr_by'


def _valid_hypothesis_label(label: object) -> bool:
    """Accept nonmissing scalar or tuple labels, rejecting empty tuple/string labels."""
    if isinstance(label, tuple):
        return bool(label) and all(_valid_hypothesis_label(part) for part in label)
    if isinstance(label, (str, bytes)):
        return bool(label.strip())
    if not pd.api.types.is_scalar(label):
        return False
    missing = pd.isna(label)
    return isinstance(missing, (bool, np.bool_)) and not missing


def adjust_pvalues(
        pvalues: pd.Series,
        family_id: str,
        method: MultipleTestingMethod = MultipleTestingMethod.HOLM,
        significance_level: float = 0.05,
) -> pd.DataFrame:
    """Adjust one complete, named family of marginal p-values in its supplied order.

    The family must include every hypothesis to which the claim applies, including
    unsuccessful variants. Invalid entries reject the entire family rather than reduce
    its size. The default controls FWER at 5%; selecting BHY interprets the level as an
    FDR target instead. Rejection is ``pvalue_adjusted <= significance_level``.

    Args:
        pvalues: Nonempty real-numeric Series of finite values in [0, 1], indexed by
            unique nonmissing, nonblank hypothesis labels. Numeric and tuple/MultiIndex
            labels are supported. Nullable numeric dtypes are accepted only without
            missing values; booleans, complex values and numeric strings are rejected.
        family_id: Nonblank identifier for the predeclared family; recorded unchanged.
        method: A MultipleTestingMethod member. BHY includes the harmonic correction;
            plain strings and other procedures are not accepted.
        significance_level: Finite error-rate target strictly between zero and one.

    Returns:
        A new DataFrame with ``pvalue_raw``, ``pvalue_adjusted`` and boolean ``reject``
        columns, preserving the original index and order. Its ``attrs`` record
        ``family_id``, ordered ``family_members`` (tuple), ``n_hypotheses``, ``method``
        (enum), ``significance_level`` and ``error_control`` (FWER or FDR). Adjusted
        p-values are not posterior probabilities that individual hypotheses are false.

    Raises:
        ValueError: If the family, labels, values, method or error target are invalid.
    """
    if not isinstance(pvalues, pd.Series):
        raise ValueError(f'pvalues must be a pandas Series, got {type(pvalues).__name__}')
    if pvalues.empty:
        raise ValueError('pvalues must contain a nonempty hypothesis family')
    if not isinstance(family_id, str) or not family_id.strip():
        raise ValueError(f'family_id must be a nonblank string, got {family_id!r}')
    if not isinstance(method, MultipleTestingMethod):
        raise ValueError(f'method must be a MultipleTestingMethod member, got {method!r}')
    if (isinstance(significance_level, (bool, np.bool_))
            or not isinstance(significance_level, Real)):
        raise ValueError(f'significance_level must be a real number, got {significance_level!r}')
    try:
        level = float(significance_level)
    except (OverflowError, TypeError, ValueError) as exception:
        raise ValueError(f'invalid significance_level, got {significance_level!r}') from exception
    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError(
            f'significance_level must be finite and strictly between zero and one, '
            f'got {significance_level!r}'
        )
    invalid_labels = [label for label in pvalues.index if not _valid_hypothesis_label(label)]
    if invalid_labels:
        raise ValueError(
            f'hypothesis labels must be nonmissing and nonempty, got {invalid_labels!r}'
        )
    try:
        unique = pvalues.index.is_unique
    except TypeError as exception:
        raise ValueError('hypothesis labels must be hashable') from exception
    if not unique:
        duplicates = pvalues.index[pvalues.index.duplicated(keep=False)].tolist()
        raise ValueError(f'hypothesis labels must be unique, got duplicates {duplicates!r}')
    if (not pd.api.types.is_numeric_dtype(pvalues.dtype)
            or pd.api.types.is_complex_dtype(pvalues.dtype)
            or pd.api.types.is_bool_dtype(pvalues.dtype)):
        raise ValueError(
            'pvalues must be real numeric values, not boolean, complex or strings; '
            f'hypotheses: {pvalues.index.tolist()!r}'
        )
    values = pvalues.to_numpy(dtype=float, na_value=np.nan)
    invalid = ~np.isfinite(values) | (values < 0.0) | (values > 1.0)
    if invalid.any():
        labels = pvalues.index[invalid].tolist()
        raise ValueError(f'pvalues must be finite and in [0, 1]; invalid hypotheses: {labels!r}')

    _, adjusted, _, _ = multipletests(
        values, alpha=level, method=method.value, is_sorted=False, returnsorted=False,
    )
    result = pd.DataFrame({
        'pvalue_raw': values,
        'pvalue_adjusted': adjusted,
        'reject': adjusted <= level,
    }, index=pvalues.index.copy())
    result.attrs = dict(
        family_id=family_id,
        family_members=tuple(pvalues.index),
        n_hypotheses=len(pvalues),
        method=method,
        significance_level=level,
        error_control='FWER' if method is MultipleTestingMethod.HOLM else 'FDR',
    )
    return result

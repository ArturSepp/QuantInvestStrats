"""Shared static accounting partitions for grouped portfolio analytics.

Ordinary labels remain a Series. Explicit matrix rows override those labels;
omitted rows retain their categorical membership. No exposure is inferred.
"""
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from qis.utils import df_groups as dfg


def _align_partition(group_loadings: pd.DataFrame, assets: pd.Index) -> pd.DataFrame:
    """Validate a complete nonnegative partition without normalising any row."""
    if not isinstance(group_loadings, pd.DataFrame):
        raise TypeError("lookthrough_matrix must be a pandas DataFrame")
    if not group_loadings.index.is_unique or not group_loadings.columns.is_unique:
        raise ValueError("Look-through instrument identifiers and group names must be unique")
    if not assets.is_unique or group_loadings.index.hasnans:
        raise ValueError("Look-through instrument identifiers must be unique and nonmissing")
    if group_loadings.columns.empty or group_loadings.columns.hasnans:
        raise ValueError("Look-through requires named exposure groups")
    # Validate supplied rows even when a particular portfolio uses only a subset.
    values = group_loadings.astype(float).to_numpy()
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("Look-through fractions must be finite and nonnegative")
    invalid = ~np.isclose(values.sum(axis=1), 1.0, atol=1e-10, rtol=0.0)
    if invalid.any():
        raise ValueError("Look-through rows must sum to one; invalid instruments: "
                         f"{group_loadings.index[invalid].tolist()}")
    missing = assets.difference(group_loadings.index)
    if len(missing):
        raise ValueError(f"Missing look-through mappings for {missing.tolist()}")
    return group_loadings.reindex(assets).astype(float)


def resolve_group_partition(
    group_data: Optional[pd.Series],
    assets: pd.Index,
    lookthrough_matrix: Optional[pd.DataFrame] = None,
    group_order: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Resolve labels and explicit fractional overrides into one accounting partition.

    Args:
        group_data: Ordinary instrument labels. Required for every asset without
            an explicit matrix row; must be a Series, never a matrix.
        assets: Unique instrument identifiers required by this calculation.
        lookthrough_matrix: Optional static instrument-by-sleeve overrides.
            Every supplied row is finite, nonnegative and sums to one. Superset
            rows are allowed for reports comparing different portfolio universes.
        group_order: Preferred group order; remaining used groups are appended.

    Returns:
        An instrument-by-group DataFrame. Inputs are never mutated.

    Raises:
        TypeError: If labels or the matrix have the wrong container type.
        ValueError: For duplicate identifiers, missing membership or invalid fractions.
    """
    assets = pd.Index(assets)
    if not assets.is_unique or assets.hasnans:
        raise ValueError("Reporting instrument identifiers must be unique and nonmissing")
    if group_data is not None:
        if not isinstance(group_data, pd.Series):
            raise TypeError("group_data must be a pandas Series; use lookthrough_matrix")
        if not group_data.index.is_unique:
            raise ValueError("Group instrument identifiers must be unique")
    if lookthrough_matrix is None:
        matrix = pd.DataFrame(index=pd.Index([]))
    else:
        matrix = _align_partition(lookthrough_matrix, lookthrough_matrix.index)
    overrides = assets.intersection(matrix.index, sort=False)
    fallback = assets[~assets.isin(overrides)]
    labels = (pd.Series(index=fallback, dtype=object) if group_data is None
              else group_data.reindex(fallback))
    if labels.isna().any():
        missing = labels.index[labels.isna()].tolist()
        raise ValueError(f"Missing group labels or look-through mappings for {missing}")
    groups = list(matrix.columns)
    groups.extend(g for g in pd.unique(labels) if g not in groups)
    if group_order is not None:
        preferred = [g for g in dict.fromkeys(group_order) if g in groups]
        groups = preferred + [g for g in groups if g not in preferred]
    result = pd.DataFrame(0.0, index=assets, columns=groups)
    if len(overrides):
        result.loc[overrides, matrix.columns] = matrix.loc[overrides].to_numpy()
    for group in pd.unique(labels):
        result.loc[labels.index[labels.eq(group)], group] = 1.0
    return result


def aggregate_grouped_data(
    df,
    group_data,
    group_order=None,
    total_column=None,
    agg_func=np.nansum,
    *,
    lookthrough_matrix=None,
):
    """Sum contributions by the shared partition, preserving the legacy None path.

    Args:
        lookthrough_matrix: Optional static instrument-by-sleeve fractions. Supplied rows
            are finite, nonnegative and sum to one; omitted rows use group_data.
            None preserves categorical reporting. This is an accounting split.
        group_data: Ordinary instrument labels as a Series; defaults to portfolio metadata
            where available. Fractional memberships belong in lookthrough_matrix.
    """
    if group_data is not None and not isinstance(group_data, pd.Series):
        raise TypeError("group_data must be a pandas Series; use lookthrough_matrix")
    if lookthrough_matrix is None:
        return dfg.agg_df_by_groups_ax1(
            df=df, group_data=group_data, group_order=group_order,
            total_column=total_column, agg_func=agg_func)
    partition = resolve_group_partition(group_data, df.columns, lookthrough_matrix, group_order)
    if total_column is not None and total_column in partition.columns:
        raise ValueError("Total label cannot also be a reporting group")
    result = df.fillna(0.0).dot(partition)
    if total_column is not None:
        result.insert(0, total_column, df.sum(axis=1))
    return result


def split_grouped_data(
    df,
    group_data,
    group_order=None,
    total_column=None,
    *,
    lookthrough_matrix=None,
):
    """Return each group's instrument contributions, splitting only supplied rows.

    Args:
        lookthrough_matrix: Optional static instrument-by-sleeve fractions. Supplied rows
            are finite, nonnegative and sum to one; omitted rows use group_data.
            None preserves categorical reporting. This is an accounting split.
        group_data: Ordinary instrument labels as a Series; defaults to portfolio metadata
            where available. Fractional memberships belong in lookthrough_matrix.
    """
    if group_data is not None and not isinstance(group_data, pd.Series):
        raise TypeError("group_data must be a pandas Series; use lookthrough_matrix")
    if lookthrough_matrix is None:
        return dfg.split_df_by_groups(
            df=df, group_data=group_data, group_order=group_order, total_column=total_column)
    partition = resolve_group_partition(group_data, df.columns, lookthrough_matrix, group_order)
    if total_column is not None and total_column in partition.columns:
        raise ValueError("Total label cannot also be a reporting group")
    result = {}
    for group in partition:
        fractions = partition[group]
        members = fractions[fractions.ne(0.0)]
        result[group] = df.loc[:, members.index].mul(members, axis=1)
    if total_column is not None:
        result = {total_column: df.copy(), **result}
    return result


def compute_lookthrough_exposures(
    weights: pd.DataFrame,
    group_loadings: Optional[pd.DataFrame] = None,
    *,
    group_data: Optional[pd.Series] = None,
    lookthrough_matrix: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Split weights or contributions using explicit fractions and ordinary labels.

    Args:
        weights: Date-by-instrument weights or arithmetic return contributions.
            Missing cells mean inactive positions and are treated as zero.
        group_loadings: Legacy alias for a complete lookthrough_matrix; cannot be
            combined with lookthrough_matrix. Prefer the latter in new callers.
        group_data: Ordinary Series labels for rows omitted from the matrix.
        lookthrough_matrix: Optional static nonnegative row-sum-one overrides.

    Returns:
        Date-by-sleeve exposures retaining input row sums and dates.

    Raises:
        ValueError: For invalid mappings, conflicting aliases or infinite inputs.
        TypeError: If group_data is not a Series.
    """
    if group_loadings is not None:
        if lookthrough_matrix is not None:
            raise ValueError("Pass lookthrough_matrix or legacy group_loadings, not both")
        lookthrough_matrix = group_loadings
    loadings = resolve_group_partition(group_data, weights.columns, lookthrough_matrix)
    data = weights.fillna(0.0).astype(float)
    if not np.isfinite(data.to_numpy()).all():
        raise ValueError("Look-through inputs must be finite")
    return data.dot(loadings)


def resolve_reporting_inputs(portfolios, group_data=None, lookthrough_matrix=None):
    """Resolve one shared partition for portfolios with different instrument universes.

    Args:
        lookthrough_matrix: Optional static instrument-by-sleeve fractions. Supplied rows
            are finite, nonnegative and sum to one; omitted rows use group_data.
            None preserves categorical reporting. This is an accounting split.
        group_data: Ordinary instrument labels as a Series; defaults to portfolio metadata
            where available. Fractional memberships belong in lookthrough_matrix.
    """
    if group_data is not None and not isinstance(group_data, pd.Series):
        raise TypeError("group_data must be a pandas Series; use lookthrough_matrix")
    if lookthrough_matrix is None:
        return group_data, None
    assets = pd.Index([])
    labels = pd.Series(dtype=object)
    for portfolio in portfolios:
        assets = assets.union(portfolio.weights.columns, sort=False)
        current = portfolio.group_data
        if group_data is None:
            shared = labels.index.intersection(current.index)
            conflict = shared[labels.loc[shared].ne(current.loc[shared])]
            conflict = conflict.difference(lookthrough_matrix.index)
            if len(conflict):
                raise ValueError("Conflicting portfolio group labels; supply group_data for "
                                 f"{conflict.tolist()}")
            labels = labels.combine_first(current)
    if group_data is not None:
        labels = group_data
    return labels, resolve_group_partition(labels, assets, lookthrough_matrix)

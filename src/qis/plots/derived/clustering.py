"""Composite dendrogram and membership views of already fitted asset clusters."""
from collections.abc import Mapping
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qis.plots.dendrogram import plot_dendrogram, _validate_tree
from qis.plots.table import plot_df_table
from qis.plots.utils import get_table_lines_for_group_data


def plot_clusters(
    clusters: Mapping[str, pd.Series],
    linkages: Mapping[str, np.ndarray],
    cutoffs: Mapping[str, float],
    figsize: tuple[float, float] = (14, 10),
    *,
    axes: Optional[Mapping[str, plt.Axes]] = None,
    table_ax: Optional[plt.Axes] = None,
    titles: Optional[Mapping[str, str]] = None,
    display_names: Optional[Mapping] = None,
    fontsize: float = 10,
    show_distance: bool = False,
    table_title: str = "Cluster IDs",
    table_kwargs: Optional[dict] = None,
) -> tuple[pd.Series, plt.Figure]:
    """Render supplied trees and membership with automatic or caller-owned axes.

    The aggregate ordering preserves OP's contract: concatenate in reverse linkage-key
    order, sort by cadence-prefixed cluster ID, then reverse that sorted Series.
    Display aliases do not change the asset IDs or ordering in the returned Series.
    Any positive number of disjoint groups/cadences is supported; no fit is performed.

    Args:
        clusters: Cadence/group to asset-indexed membership Series in linkage leaf order.
        linkages: Matching group to full SciPy linkage array. A singleton uses shape (0, 4).
        cutoffs: Matching group to finite nonnegative merge-distance cutoff.
        figsize: Figure size when axes are created automatically.
        axes: Optional group-to-axis mapping, with exactly the input groups.
        table_ax: Membership-table axis, required together with axes.
        titles: Optional group headings; defaults use Monthly/Quarterly for ME/QE.
        display_names: Optional asset-ID-to-display-name mapping, used only for rendering.
        fontsize: Leaf and table font size.
        show_distance: Display merge-distance ticks.
        table_title: Heading above the membership table.
        table_kwargs: Additional qis table options, including colour and column widths.

    Returns:
        Cadence-prefixed membership Series with original asset IDs, and its figure.

    Raises:
        ValueError: If groups, trees, asset identities, cutoffs or supplied axes disagree.
    """
    keys = set(clusters)
    if not keys or set(linkages) != keys or set(cutoffs) != keys:
        raise ValueError("clusters, linkages and cutoffs require the same nonempty groups")
    order = list(reversed(linkages))
    for group, members in clusters.items():
        if not members.index.is_unique or members.isna().any():
            raise ValueError("Membership requires unique asset IDs and nonmissing cluster IDs")
        _validate_tree(linkages[group], members.index)
        if not np.isfinite(cutoffs[group]) or cutoffs[group] < 0:
            raise ValueError("Cluster cutoffs must be finite nonnegative distances")
    identities = pd.concat([clusters[group] for group in order])
    if not identities.index.is_unique:
        raise ValueError("Assets must belong to exactly one cadence/group")
    if (axes is None) != (table_ax is None):
        raise ValueError("Supply both axes and table_ax, or neither")
    if axes is not None:
        if set(axes) != keys:
            raise ValueError("axes must cover exactly the supplied groups")
        fig = table_ax.figure
        all_axes = [axes[group] for group in order] + [table_ax]
        if len({id(ax) for ax in all_axes}) != len(all_axes):
            raise ValueError("Each group and the membership table need distinct axes")
        if any(ax.figure is not fig for ax in all_axes):
            raise ValueError("All supplied axes must belong to one figure")
    else:
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        grid = fig.add_gridspec(len(order), 3, wspace=.1,
                               height_ratios=[max(len(clusters[g]), 3) for g in order])
        axes = {group: fig.add_subplot(grid[i, :2]) for i, group in enumerate(order)}
        table_ax = fig.add_subplot(grid[:, 2])
    titles = titles or {}
    aliases = display_names if display_names is not None else {}
    labelled = []
    for group in order:
        members = clusters[group]
        labels = [aliases.get(asset, str(asset)) for asset in members.index]
        title = titles.get(group, {"ME": "Monthly", "QE": "Quarterly"}.get(group, str(group)))
        plot_dendrogram(linkages[group], labels, cutoff=cutoffs[group], ax=axes[group],
                        title=title, fontsize=fontsize, show_distance=show_distance)
        labelled.append(members.map(lambda value: f"{group}-{value}"))
    aggregate = pd.concat(labelled).sort_values()
    aggregate = aggregate.reindex(index=aggregate.index[::-1])
    table = aggregate.rename(index=aliases).to_frame(name="Cluster ID")
    options = dict(index_column_name="Instrument", fontsize=fontsize, title=table_title,
                   rows_edge_lines=get_table_lines_for_group_data(aggregate))
    options.update(table_kwargs or {})
    plot_df_table(table, ax=table_ax, **options)
    return aggregate, fig

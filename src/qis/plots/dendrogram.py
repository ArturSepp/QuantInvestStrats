"""Single-axis rendering of a supplied hierarchical clustering tree.

This module renders an existing SciPy linkage; it never estimates dependence or clusters.
Leaf labels must follow the observation order used to construct that linkage.
"""
from collections.abc import Sequence
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hierarchy


def _validate_tree(linkage: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    """Check the full tree against its ordered leaves before creating any artists."""
    tree = np.asarray(linkage, dtype=float)
    if tree.size == 0:
        tree = tree.reshape(0, 4)
    if not len(labels) or tree.shape != (len(labels) - 1, 4):
        raise ValueError("A full linkage requires N-1 rows, four columns and N leaf labels")
    if not np.isfinite(tree).all():
        raise ValueError("Linkage values must be finite")
    if len(tree):
        hierarchy.is_valid_linkage(tree, throw=True, name="linkage")
    return tree


def plot_dendrogram(
    linkage: np.ndarray,
    labels: Sequence[str],
    cutoff: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    orientation: str = "right",
    title: Optional[str] = None,
    fontsize: float = 10,
    show_distance: bool = True,
    **kwargs,
) -> plt.Axes:
    """Plot an existing full linkage with optional cluster-cutoff annotation.

    Args:
        linkage: SciPy linkage array, with N-1 rows and four columns.
        labels: N display labels in the original observation order.
        cutoff: Nonnegative merge distance for branch colouring and a black cut line.
        ax: Destination axis; create a figure when omitted.
        orientation: SciPy orientation: right, left, top or bottom.
        title: Optional axis heading.
        fontsize: Leaf-label font size.
        show_distance: Show merge-distance tick labels.
        **kwargs: Additional SciPy dendrogram styling options. Tree truncation and
            label replacement are excluded to preserve the full supplied leaf identity.

    Returns:
        The axis containing the dendrogram.

    Raises:
        ValueError: If the tree, cutoff, orientation or identity-changing options are invalid.
    """
    labels = list(labels)
    tree = _validate_tree(linkage, labels)
    if cutoff is not None and (not np.isfinite(cutoff) or cutoff < 0):
        raise ValueError("cutoff must be a finite nonnegative distance")
    if orientation not in ("right", "left", "top", "bottom"):
        raise ValueError("Unknown dendrogram orientation")
    reserved = {"Z", "labels", "ax", "orientation", "no_plot", "truncate_mode",
                "leaf_label_func", "no_labels", "color_threshold"}
    if reserved.intersection(kwargs):
        raise ValueError("Options cannot replace leaves, truncate or override the cutoff")
    if ax is None:
        _, ax = plt.subplots()
    horizontal = orientation in ("left", "right")
    if len(tree):
        hierarchy.dendrogram(tree, labels=labels, orientation=orientation,
                             color_threshold=cutoff, ax=ax, **kwargs)
    else:
        ax.text(.02, .5, str(labels[0]), transform=ax.transAxes,
                fontsize=fontsize, va="center")
        ax.set_yticks([])
        ax.set_xticks([])
    if cutoff is not None:
        if horizontal:
            ax.axvline(cutoff, color="k")
        else:
            ax.axhline(cutoff, color="k")
    if title is not None:
        ax.set_title(title)
    ax.tick_params(axis="y" if horizontal else "x", labelsize=fontsize)
    ax.tick_params(axis="x" if horizontal else "y",
                   **({"labelbottom": show_distance} if horizontal
                      else {"labelleft": show_distance}))
    return ax

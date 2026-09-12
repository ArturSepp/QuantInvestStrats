"""Identity, topology and axes contracts for reusable fitted-cluster plots."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import dendrogram

from qis.plots.dendrogram import plot_dendrogram
from qis.plots.derived.clustering import plot_clusters


@pytest.fixture(autouse=True)
def close_figures():
    """Close all figures allocated by each plotting contract."""
    yield
    plt.close("all")


def inputs():
    """Two disjoint groups with deliberate nonalphabetical leaf order."""
    clusters = {"ME": pd.Series([2, 1, 1], index=["C", "A", "B"]),
                "QE": pd.Series([1], index=["D"])}
    linkages = {"ME": np.array([[1., 2., .2, 2.], [0., 3., .9, 3.]]),
                "QE": np.empty((0, 4))}
    return clusters, linkages, {"ME": .5, "QE": .1}


@pytest.mark.parametrize("orientation", ["right", "left", "top", "bottom"])
def test_primitive_matches_scipy_leaf_order_merge_heights_and_cutoff(orientation):
    """The artist coordinates agree with the independent SciPy no-plot representation."""
    tree = inputs()[1]["ME"]
    labels = ["C", "A", "B"]
    before = tree.copy()
    ref = dendrogram(tree, labels=labels, no_plot=True)
    ax = plot_dendrogram(tree, labels, cutoff=.5, orientation=orientation)
    horizontal = orientation in ("left", "right")
    ticks = ax.get_yticklabels() if horizontal else ax.get_xticklabels()
    assert [tick.get_text() for tick in ticks] == ref["ivl"]
    heights = sorted(tuple(segment[:, 0 if horizontal else 1])
                     for collection in ax.collections for segment in collection.get_segments())
    assert heights == sorted(tuple(row) for row in ref["dcoord"])
    line = ax.lines[-1]
    np.testing.assert_array_equal(line.get_xdata() if horizontal else line.get_ydata(), [.5, .5])
    np.testing.assert_array_equal(tree, before)


def test_singleton_and_no_cutoff_render_without_inventing_merges():
    """A lone member is labelled with no tree or cutoff line."""
    ax = plot_dendrogram(np.empty((0, 4)), ["Only member"])
    assert len(ax.collections) == len(ax.lines) == 0
    assert ax.texts[0].get_text() == "Only member"


@pytest.mark.parametrize("tree,labels,options", [
    (np.zeros((1, 4)), ["one"], {}),
    (np.array([[0., 1., np.nan, 2.]]), ["a", "b"], {}),
    (np.array([[0., 1., .5, 2.]]), ["a", "b"], {"cutoff": -1}),
    (np.array([[0., 1., .5, 2.]]), ["a", "b"], {"orientation": "diagonal"}),
    (np.array([[0., 1., .5, 2.]]), ["a", "b"], {"truncate_mode": "lastp"}),
    (np.empty((0, 4)), [], {}),
])
def test_invalid_primitive_inputs_fail_before_creating_figure(tree, labels, options):
    """Reject incomplete or altered identity contracts without leaving empty figures."""
    before = plt.get_fignums()
    with pytest.raises(ValueError):
        plot_dendrogram(tree, labels, **options)
    assert plt.get_fignums() == before


def test_aliases_only_change_display_and_caller_axes_remain_in_place():
    """Underlying asset IDs and legacy aggregate ordering survive report-friendly names."""
    clusters, linkages, cutoffs = inputs()
    fig = plt.figure()
    axes = {"ME": fig.add_axes([.1, .55, .5, .3]), "QE": fig.add_axes([.1, .1, .5, .3])}
    table_ax = fig.add_axes([.7, .1, .25, .8])
    positions = [ax.get_position().bounds for ax in fig.axes]
    aggregate, actual = plot_clusters(
        clusters, linkages, cutoffs, axes=axes, table_ax=table_ax,
        display_names={"A": "Alpha", "B": "Beta", "C": "Charlie", "D": "Delta"},
        titles={"ME": "Liquid assets", "QE": "Quarterly assets"},
    )
    assert actual is fig
    assert [ax.get_position().bounds for ax in fig.axes] == positions
    expected = pd.Series(["QE-1", "ME-2", "ME-1", "ME-1"], index=["D", "C", "B", "A"])
    pd.testing.assert_series_equal(aggregate, expected)
    assert [tick.get_text() for tick in axes["ME"].get_yticklabels()] == [
        "Charlie", "Alpha", "Beta"]
    assert axes["ME"].get_title() == "Liquid assets"
    assert axes["QE"].texts[0].get_text() == "Delta"
    assert "Alpha" in [cell.get_text().get_text()
                       for table in table_ax.tables for cell in table.get_celld().values()]


def test_three_groups_get_independent_axes_and_correct_single_cadence_title():
    """QIS generalises cadence counts while reading headings from keys."""
    clusters, linkages, cutoffs = inputs()
    clusters["YE"], linkages["YE"], cutoffs["YE"] = pd.Series([1], index=["E"]), [], 0.
    result, fig = plot_clusters(clusters, linkages, cutoffs)
    assert len(fig.axes) == 4
    assert len(result) == 5
    _, single = plot_clusters({"QE": clusters["QE"]}, {"QE": []}, {"QE": .1})
    assert single.axes[0].get_title() == "Quarterly"


@pytest.mark.parametrize("problem", ["groups", "duplicates", "empty", "cutoff", "tree", "missing"])
def test_invalid_composite_inputs_fail_before_rendering(problem):
    """Cadence/identity mismatch cannot silently relabel a plausible-looking tree."""
    clusters, linkages, cutoffs = inputs()
    if problem == "groups":
        cutoffs.pop("ME")
    elif problem == "duplicates":
        clusters["QE"].index = ["A"]
    elif problem == "empty":
        clusters, linkages, cutoffs = {}, {}, {}
    elif problem == "cutoff":
        cutoffs["ME"] = float("inf")
    elif problem == "tree":
        linkages["ME"] = np.empty((0, 4))
    else:
        clusters["ME"].iloc[0] = np.nan
    before = plt.get_fignums()
    with pytest.raises(ValueError):
        plot_clusters(clusters, linkages, cutoffs)
    assert plt.get_fignums() == before


@pytest.mark.parametrize("problem", ["partial", "keys", "same_axis", "different_figure"])
def test_axes_contract_rejects_partial_or_conflicting_destinations(problem):
    """A composite never moves axes between figures or draws two groups on one axis."""
    clusters, linkages, cutoffs = inputs()
    fig, axs = plt.subplots(1, 3)
    axes, table_ax = {"ME": axs[0], "QE": axs[1]}, axs[2]
    if problem == "partial":
        table_ax = None
    elif problem == "keys":
        axes.pop("QE")
    elif problem == "same_axis":
        axes["ME"] = table_ax
    else:
        _, table_ax = plt.subplots()
    with pytest.raises(ValueError):
        plot_clusters(clusters, linkages, cutoffs, axes=axes, table_ax=table_ax)

"""Offline examples of single-axis and composite fitted-tree plots.

Run: python -m examples.plots.cluster_dendrograms --output-dir <local-output-directory>
The frozen QIS synthetic universe supplies all observations; no vendor data is needed.
This example fits illustrative trees with SciPy. Production callers pass their saved
estimator linkages and memberships instead.
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster

import qis
from qis.datasets import generate_synthetic_universe


def make_cluster_inputs():
    """Build two illustrative trees from disjoint synthetic monthly-return panels."""
    universe = generate_synthetic_universe()
    prices = universe.prices.ffill()
    returns = qis.to_returns(prices, freq="ME", is_log_returns=True,
                             is_first_zero=False).dropna()
    groups = {"Group A": returns.columns[:5], "Group B": returns.columns[5:]}
    clusters, linkages, cutoffs = {}, {}, {}
    for group, assets in groups.items():
        observations = returns[assets].to_numpy().T
        observations = (observations - observations.mean(axis=1, keepdims=True))
        observations /= observations.std(axis=1, keepdims=True)
        tree = linkage(observations, method="ward")
        cutoff = .6 * tree[:, 2].max()
        clusters[group] = pd.Series(fcluster(tree, cutoff, criterion="distance"), index=assets)
        linkages[group], cutoffs[group] = tree, cutoff
    return clusters, linkages, cutoffs


def main(output_dir=None):
    """Draw a standalone tree and a report page with caller-supplied axes."""
    clusters, linkages, cutoffs = make_cluster_inputs()
    ax = qis.plot_dendrogram(linkages["Group A"], clusters["Group A"].index,
                             cutoff=cutoffs["Group A"], title="Standalone dendrogram")
    standalone = ax.figure
    fig = plt.figure(figsize=(14, 10), layout="constrained")
    grid = fig.add_gridspec(2, 3)
    axes = {group: fig.add_subplot(grid[i, :2]) for i, group in enumerate(clusters)}
    table_ax = fig.add_subplot(grid[:, 2])
    membership, _ = qis.plot_clusters(
        clusters, linkages, cutoffs, axes=axes, table_ax=table_ax,
        titles={group: f"{group}: illustrative Ward tree" for group in clusters},
        show_distance=True, table_title="Fitted membership",
    )
    fig.suptitle("Synthetic asset clustering: reusable QIS plots")
    print(f"{len(membership)} assets across {len(clusters)} supplied trees")
    if output_dir is None:
        plt.show()
    else:
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        standalone.savefig(folder / "standalone_dendrogram.png", dpi=120)
        fig.savefig(folder / "cluster_dendrograms.png", dpi=120)
        membership.to_csv(folder / "cluster_membership.csv")
        plt.close(standalone)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    main(parser.parse_args().output_dir)

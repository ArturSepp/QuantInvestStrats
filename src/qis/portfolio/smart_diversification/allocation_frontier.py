"""
the coverage-floor exhibit: stacked portfolios and a solved allocation frontier.

``plot_overlay_allocation_frontier`` draws, in the coordinates of Bear contribution and Sharpe
ratio, the benchmark, the stacked portfolios of the benchmark plus one overlay each, and the
optimal stacked portfolios along a coverage floor: Figure 4 of Sepp and Kastenholz (2026), whose
replication draws it with this function. The module performs no estimation or optimisation. The
statistics come from ``qis.regimes.compute_regime_premium_table`` on stacked-portfolio returns,
and the allocations from the maximum-Sharpe program with a Bear coverage floor, the paper's
program (11), which optimalportfolios solves. All statistics must share one sample, one benchmark
classification and one Sharpe convention. Frontier rows are joined in the order supplied,
including repeated points where the floor is slack.
"""
from typing import Any, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _validate_coordinates(table: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    """Reject ambiguous labels, missing coordinates and non-finite plotted values."""
    if not table.index.is_unique or not table.columns.is_unique:
        raise ValueError(f'{name} must have unique row and column labels.')
    if table.empty or any(column not in table for column in columns):
        raise ValueError(f'{name} must contain nonempty {list(columns)} coordinates.')
    if not np.isfinite(table.loc[:, list(columns)].to_numpy(dtype=float)).all():
        raise ValueError(f'{name} coordinates must be finite.')


def plot_overlay_allocation_frontier(
        portfolio_stats: pd.DataFrame,
        frontier_stats: Optional[pd.DataFrame] = None,
        groups: Optional[pd.Series] = None,
        benchmark: Optional[str] = None,
        group_styles: Optional[Mapping[str, Mapping[str, Any]]] = None,
        highlights: Optional[Mapping[str, Mapping[str, Any]]] = None,
        annotations: Optional[Sequence[str]] = None,
        label_offsets: Optional[Mapping[str, Tuple[float, float]]] = None,
        x_column: str = 'bear_sharpe',
        y_column: str = 'sharpe',
        frontier_label: str = 'Optimal stacked portfolios',
        frontier_color: str = '#009E73',
        xlabel: str = 'Bear Sharpe contribution of the stacked portfolio',
        ylabel: str = 'Arithmetic excess Sharpe ratio of the stacked portfolio',
        title: Optional[str] = None,
        legend_loc: Optional[str] = 'upper center',
        figsize: Tuple[float, float] = (8.6, 5.2),
        ax: Optional[plt.Axes] = None,
        bbox_to_anchor: Optional[Tuple[float, float]] = (0.5, -0.14),
        ncols: int = 2,
) -> Optional[plt.Figure]:
    """Stacked portfolios by group and an optional solved coverage-floor frontier.

    Every point is a portfolio, not a standalone overlay: the benchmark itself, or the benchmark
    at weight one plus an overlay at the chosen budget, the stacked portfolio of Sepp and
    Kastenholz (2026, Definition 3). Build the stacked returns ``r_B + w r_i`` first and pass
    them, with the benchmark, through ``qis.regimes.compute_regime_premium_table``, whose
    ``bear_sharpe`` and ``sharpe`` columns are the default coordinates. Compute the frontier rows
    the same way from the solved allocations. Weights and optimiser objects are not needed, and
    the function does not fit, sort, round or recompute coordinates.

    Args:
        portfolio_stats: statistics of the benchmark and the stacked portfolios, indexed by
            unique portfolio names
        frontier_stats: statistics of the solved portfolios in the order of the floor; None
            draws only the points. Repeated coordinate pairs are kept, as where the floor is slack
        groups: group label by portfolio name, covering every point of ``portfolio_stats``; None
            puts the points in one group and the benchmark in its own
        benchmark: name of the benchmark row, whose x coordinate sets the dashed vertical line
        group_styles: matplotlib scatter options by group, such as color, marker, s and label;
            groups without options use the matplotlib colour cycle
        highlights: scatter options by portfolio name for selected allocations, drawn once apart
            from their groups and named in the legend
        annotations: names of the points to label; None labels every point that is not
            highlighted, and an empty sequence none
        label_offsets: label offsets in points by portfolio name
        x_column: column of the x coordinate in both tables, the Bear contribution by default
        y_column: column of the y coordinate in both tables, the Sharpe ratio by default
        frontier_label: legend label of the frontier
        frontier_color: colour of the frontier line
        xlabel: x-axis label; state the Sharpe convention when it matters
        ylabel: y-axis label
        title: optional title
        legend_loc: legend location; None draws no legend. The default puts the legend's upper
            centre below the axes, in two columns without a frame
        figsize: size of a new figure
        ax: axis to draw on; None creates a figure
        bbox_to_anchor: legend anchor in axes coordinates; None uses ``legend_loc`` alone
        ncols: number of legend columns

    Returns:
        the new figure, or None when ``ax`` is given

    Raises:
        ValueError: if the coordinates, the row labels, the groups or a selected name are invalid
    """
    if x_column == y_column:
        raise ValueError('x_column and y_column must differ.')
    columns = (x_column, y_column)
    _validate_coordinates(portfolio_stats, columns, 'portfolio_stats')
    if frontier_stats is not None:
        _validate_coordinates(frontier_stats, columns, 'frontier_stats')
    if benchmark is not None and benchmark not in portfolio_stats.index:
        raise ValueError('benchmark must name a portfolio_stats row.')
    highlights = highlights or {}
    if any(name not in portfolio_stats.index for name in highlights):
        raise ValueError('Every highlight must name a portfolio_stats row.')
    if groups is None:
        groups = pd.Series('Portfolios', index=portfolio_stats.index)
        if benchmark is not None:
            groups.loc[benchmark] = 'Benchmark'
    elif not groups.index.is_unique:
        raise ValueError('groups must have unique portfolio labels.')
    else:
        groups = groups.reindex(portfolio_stats.index)
    if groups.isna().any():
        raise ValueError('groups must cover every portfolio_stats row.')
    labels = (portfolio_stats.index.difference(list(highlights), sort=False).tolist()
              if annotations is None else list(annotations))
    if any(name not in portfolio_stats.index for name in labels):
        raise ValueError('Every annotation must name a portfolio_stats row.')

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    else:
        fig = None
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    markers = ('o', '^', 'v', 's', 'D', 'P')
    ordinary = ~portfolio_stats.index.isin(list(highlights))
    for position, group in enumerate(pd.unique(groups)):
        selected = portfolio_stats.loc[ordinary & groups.eq(group).to_numpy()]
        if selected.empty:
            continue
        style = dict(s=64, color=colors[position % len(colors)],
                     marker=markers[position % len(markers)], edgecolor='white',
                     linewidth=0.8, label=str(group), zorder=3)
        if group == 'Benchmark':
            style.update(color='black', marker='s')
        style.update((group_styles or {}).get(group, {}))
        ax.scatter(selected[x_column], selected[y_column], **style)
    for name in labels:
        row = portfolio_stats.loc[name]
        ax.annotate(str(name), (row[x_column], row[y_column]), textcoords='offset points',
                    xytext=(label_offsets or {}).get(name, (6, 3)),
                    fontsize=8, color='#333333')
    if frontier_stats is not None:
        ax.plot(frontier_stats[x_column], frontier_stats[y_column], color=frontier_color,
                ls=':', lw=1.8, marker='.', label=frontier_label, zorder=3)
    for name, options in highlights.items():
        row = portfolio_stats.loc[name]
        style = dict(color=frontier_color, marker='*', s=210, edgecolor='black',
                     label=str(name), zorder=5)
        style.update(options)
        ax.scatter(row[x_column], row[y_column], **style)
    if benchmark is not None:
        ax.axvline(float(portfolio_stats.loc[benchmark, x_column]),
                   color='#999999', ls='--', lw=1.0, zorder=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if legend_loc is not None:
        ax.legend(loc=legend_loc, bbox_to_anchor=bbox_to_anchor, ncols=ncols,
                  frameon=False, fontsize=9)
    return fig

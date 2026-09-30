"""Plot supplied overlay-stack statistics and solved allocation frontiers.

This module performs no estimation or optimisation. Statistics must describe the same
sample, benchmark regimes and Sharpe convention. Frontier rows are joined in supplied
policy order, including repeated points where a constraint is slack.
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
    """Draw grouped portfolio points and an optional solved coverage-floor frontier.

    Pass tables produced by qis.regimes.compute_regime_premium_table or equivalent
    precomputed coordinates. A fund point must describe the core plus that fund at the
    chosen budget, rather than the standalone fund. Weights and optimiser objects are
    not required. The function does not fit, sort, round or recompute coordinates.

    Args:
        portfolio_stats: statistics indexed by unique portfolio names.
        frontier_stats: solved portfolios in policy order; None draws only the points.
            Repeated coordinate pairs are retained, for example when the floor is slack.
        groups: group labels indexed by portfolio name; must cover every plotted point.
            None groups the points together and labels the benchmark separately.
        benchmark: portfolio name whose x coordinate sets the vertical reference line.
        group_styles: per-group matplotlib scatter options, such as color, marker, s
            and label. Omitted groups use the matplotlib colour cycle.
        highlights: per-portfolio scatter options for selected allocations. These points
            are drawn once, separately from their groups, and labelled in the legend.
        annotations: names to annotate; None annotates all non-highlighted points.
            An empty sequence suppresses annotations.
        label_offsets: text offsets in points keyed by portfolio name.
        x_column: coordinate column in both tables, defaulting to the Bear contribution.
        y_column: coordinate column in both tables, defaulting to total Sharpe.
        frontier_label: legend label for the connected frontier.
        frontier_color: colour of the frontier line.
        xlabel: horizontal axis label, including the applicable convention if needed.
        ylabel: vertical axis label.
        title: optional plot title.
        legend_loc: legend location; None suppresses the legend. The default places
            its upper centre below the axes, with two columns and no frame.
        figsize: size of a new figure.
        ax: axis to draw on; None creates a figure.
        bbox_to_anchor: legend anchor in axes coordinates; None uses legend_loc alone.
        ncols: number of legend columns.

    Returns:
        The new figure, or None when an axis is supplied.

    Raises:
        ValueError: if coordinates, row labels, groups or selected names are invalid.
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

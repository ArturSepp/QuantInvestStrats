"""
regime premium exhibits drawn from the tables of ``qis.regimes``.

``plot_regime_sharpe_decomposition`` draws, for each row of a premium table, the additive regime
contributions to the Sharpe ratio as one horizontal stacked bar, positive contributions right of
zero and negative ones left, with a tick at the total Sharpe ratio and a diamond at the null of the
lowest-bucket contribution. The distance from the diamond to the end of the Bear segment is the
convexity premium. ``plot_regime_beta_profiles`` draws the regime betas of groups of assets: the
group mean per regime, a band over the members' range, optional error bars of the mean
bootstrap standard error, and a dotted line at the group's mean total beta, where every regime beta
sits under the Gaussian null.

Both functions take the output of ``compute_regime_premium_table`` and ``compute_regime_betas``
rather than prices, so the numbers on the figure are the numbers in the table. The regime
colours default to those of ``BenchmarkReturnsQuantilesRegime``.
"""
# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Sequence, Tuple
# qis
from qis.perfstats.regime_classifier import BenchmarkReturnsQuantilesRegime
from qis.plots.utils import set_title


def _regimes_of_table(table: pd.DataFrame, regime_ids: Optional[Sequence[str]]) -> List[str]:
    """Regime ids whose ``<id>_sharpe`` contribution columns the table carries."""
    if regime_ids is not None:
        return [str(x) for x in regime_ids]
    default = ['Bear', 'Normal', 'Bull']
    if all(f"{x.lower()}_sharpe" in table.columns for x in default):
        return default
    n = 1
    while f"q{n}_sharpe" in table.columns:
        n += 1
    if n == 1:
        raise ValueError(f"no regime contribution columns in {list(table.columns)}")
    return [f"Q{i}" for i in range(1, n)]


def _default_regime_colors(regime_ids: Sequence[str]) -> Dict[str, str]:
    """The classifier's colours for the regime ids, by bucket position."""
    n_buckets = len(regime_ids)
    classifier = (BenchmarkReturnsQuantilesRegime() if n_buckets == 3
                  else BenchmarkReturnsQuantilesRegime(q=n_buckets))
    return dict(zip(regime_ids, classifier.regime_ids_colors.values()))


def plot_regime_sharpe_decomposition(table: pd.DataFrame,
                                     regime_ids: Optional[Sequence[str]] = None,
                                     null_column: Optional[str] = None,
                                     total_column: str = 'sharpe',
                                     regime_colors: Optional[Dict[str, str]] = None,
                                     var_format: str = '{:.2f}',
                                     min_label_width: float = 0.12,
                                     row_separators: Optional[Sequence[int]] = None,
                                     xlabel: str = 'Sharpe ratio contribution (annualised)',
                                     legend_loc: Optional[str] = 'lower right',
                                     title: Optional[str] = None,
                                     fontsize: int = 10,
                                     figsize: Tuple[float, float] = (11.0, 6.2),
                                     ax: Optional[plt.Axes] = None
                                     ) -> Optional[plt.Figure]:
    """Horizontal stacked regime contributions per row, with the total and the tail null.

    Args:
        table: a premium table, one row per asset, as ``compute_regime_premium_table`` returns;
            rows are drawn top to bottom in table order
        regime_ids: regimes to stack, in order; None reads Bear, Normal, Bull or Q1 to Qn off
            the ``<id>_sharpe`` columns
        null_column: column of the null marked by a diamond; None uses ``null_<first id>_sharpe``
            when present and draws no diamond otherwise
        total_column: column of the total Sharpe ratio, marked by a tick
        regime_colors: colour per regime id; None uses the classifier's colours
        var_format: format of the segment labels
        min_label_width: narrowest segment that carries its value
        row_separators: positions after which a horizontal rule separates groups of rows
        xlabel: x-axis label
        legend_loc: legend location; None hides the legend
        title: axis title
        fontsize: font size of the labels and legend
        figsize: size of a new figure
        ax: axis to draw on; None creates a figure

    Returns:
        the new figure, or None when ``ax`` is given

    Raises:
        ValueError: if the table has no regime contribution columns
    """
    regime_ids = _regimes_of_table(table, regime_ids)
    colors = regime_colors or _default_regime_colors(regime_ids)
    if null_column is None:
        candidate = f"null_{regime_ids[0].lower()}_sharpe"
        null_column = candidate if candidate in table.columns else None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None
    n = len(table.index)
    y = np.arange(n)[::-1]  # the first row at the top
    for row, (_, values) in enumerate(table.iterrows()):
        positive, negative = 0.0, 0.0
        for regime in regime_ids:
            v = float(values[f"{regime.lower()}_sharpe"])
            left = positive if v >= 0.0 else negative + v
            ax.barh(y[row], abs(v), left=left, height=0.62, color=colors[regime], edgecolor='white',
                    lw=0.5, label=f"{regime} contribution" if row == 0 else None, zorder=2)
            if abs(v) >= min_label_width:
                ax.text(left + abs(v) / 2.0, y[row], var_format.format(v), ha='center', va='center',
                        fontsize=fontsize - 2, zorder=4)
            if v >= 0.0:
                positive += v
            else:
                negative += v
        total = float(values[total_column])
        ax.plot([total, total], [y[row] - 0.38, y[row] + 0.38], color='black', lw=2.0,
                label='Total Sharpe ratio' if row == 0 else None, zorder=5)
        if null_column is not None:
            ax.scatter(float(values[null_column]), y[row], marker='D', s=42, facecolor='white',
                       edgecolor='black', lw=1.2, zorder=6,
                       label=f"Null of the {regime_ids[0]} contribution" if row == 0 else None)
    for boundary in row_separators or ():
        ax.axhline(y[0] - boundary - 0.5, color='grey', lw=0.6, zorder=1)
    ax.axvline(0.0, color='grey', lw=1.0, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([str(x) for x in table.index], fontsize=fontsize)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.grid(axis='x', color='#DDDDDD', lw=0.6, zorder=0)
    if legend_loc is not None:
        ax.legend(loc=legend_loc, frameon=False, fontsize=fontsize - 1)
    if title is not None:
        set_title(ax=ax, title=title, fontsize=fontsize + 2)
    return fig


def plot_regime_beta_profiles(betas: pd.DataFrame,
                              groups: pd.Series,
                              se: Optional[pd.DataFrame] = None,
                              regime_ids: Sequence[str] = ('Bear', 'Normal', 'Bull'),
                              group_colors: Optional[Dict[str, str]] = None,
                              group_labels: Optional[Dict[str, str]] = None,
                              markers: Sequence[str] = ('o', '^', 'v', 's', 'D', 'P'),
                              xlabel: Optional[str] = 'Benchmark regime',
                              ylabel: str = 'Regime beta on the benchmark',
                              legend_loc: Optional[str] = 'lower right',
                              title: Optional[str] = None,
                              fontsize: int = 10,
                              figsize: Tuple[float, float] = (7.2, 4.4),
                              ax: Optional[plt.Axes] = None
                              ) -> Optional[plt.Figure]:
    """Group-mean regime betas with the members' range, mean standard errors and total beta.

    Args:
        betas: one row per asset with ``beta_<id>`` per regime and ``beta_total``, as
            ``compute_regime_betas`` returns
        groups: group of each asset, indexed like ``betas``; groups are drawn in order of first
            appearance
        se: optional standard errors with ``beta_<id>_se`` columns, as
            ``compute_regime_betas_bootstrap`` returns; the error bars are the group mean
        regime_ids: regimes on the x axis, in order
        group_colors: colour per group; None uses the matplotlib cycle
        group_labels: legend label per group; None uses the group names
        markers: marker per group, in group order
        xlabel: x-axis label
        ylabel: y-axis label
        legend_loc: legend location; None hides the legend
        title: axis title
        fontsize: font size of the labels and legend
        figsize: size of a new figure
        ax: axis to draw on; None creates a figure

    Returns:
        the new figure, or None when ``ax`` is given
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None
    columns = [f"beta_{regime.lower()}" for regime in regime_ids]
    x = np.arange(len(regime_ids), dtype=float)
    group_names = list(dict.fromkeys(groups.reindex(betas.index).dropna()))
    cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0'])
    offsets = np.linspace(-0.06, 0.06, len(group_names)) if len(group_names) > 1 else [0.0]
    for i, group in enumerate(group_names):
        color = (group_colors or {}).get(group, cycle[i % len(cycle)])
        members = groups.index[groups == group].intersection(betas.index)
        values = betas.loc[members, columns].astype(float)
        mean = values.mean(axis=0).to_numpy()
        ax.fill_between(x + offsets[i], values.min(axis=0).to_numpy(),
                        values.max(axis=0).to_numpy(), color=color, alpha=0.10, lw=0)
        yerr = None
        if se is not None:
            se_columns = [f"{c}_se" for c in columns]
            yerr = se.loc[members, se_columns].astype(float).mean(axis=0).to_numpy()
        ax.errorbar(x + offsets[i], mean, yerr=yerr, color=color, marker=markers[i % len(markers)],
                    ms=8, lw=2.0, capsize=4, label=(group_labels or {}).get(group, group), zorder=3)
        if 'beta_total' in betas.columns:
            ax.axhline(float(betas.loc[members, 'beta_total'].astype(float).mean()), color=color,
                       ls=':', lw=1.2, zorder=2)
    ax.axhline(0.0, color='grey', lw=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(list(regime_ids), fontsize=fontsize)
    ax.set_xlim(-0.4, len(regime_ids) - 0.6)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if legend_loc is not None:
        ax.legend(loc=legend_loc, frameon=False, fontsize=fontsize - 1)
    if title is not None:
        set_title(ax=ax, title=title, fontsize=fontsize + 2)
    return fig

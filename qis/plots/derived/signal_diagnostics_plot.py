"""
Visualisation of cross-sectional signal predictive diagnostics.

Renders the output of `qis.perfstats.signal_diagnostics.estimate_signal_diagnostics`:

    plot_signal_diagnostics_scatter — per-horizon pooled scatter of cross-
        sectional return vs lagged signal, OLS fit overlay.

    plot_signal_diagnostics_bars — per-horizon bar chart of β by group with
        t-stat annotations, plus a pooled bar.

    plot_signal_diagnostics — composite two-row figure for a strategy
        factsheet: scatters on top, group bars on the bottom, one column per
        horizon.

Follows the qis convention: each plotting function accepts an optional `ax`
so it can be embedded in a larger factsheet layout, and returns the Figure
(or None when plotting into a supplied axis).
"""
# built-in
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

# qis
from qis.perfstats.signal_diagnostics import (
    SignalDiagnosticsColumns,
    SignalDiagnosticsResult,
)


# ───────────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────────


def _significance_marker(t_stat: float) -> str:
    """ASCII significance marker for a t-statistic (two-sided)."""
    if not np.isfinite(t_stat):
        return ''
    abs_t = abs(t_stat)
    if abs_t >= 2.58:
        return '***'
    if abs_t >= 1.96:
        return '**'
    if abs_t >= 1.65:
        return '*'
    return ''


def _default_group_colors(group_order: List[str]) -> Dict[str, str]:
    """Tab10/Tab20-derived colour cycle for groups, deterministic by order."""
    cmap_names = ['tab10', 'tab20']
    cmaps = [plt.get_cmap(name) for name in cmap_names]
    colors: Dict[str, str] = {}
    for i, g in enumerate(group_order):
        cmap = cmaps[i // 10 % len(cmaps)]
        rgba = cmap(i % 10)
        # Hex format for stable comparison / serialisation
        colors[g] = '#{:02x}{:02x}{:02x}'.format(
            int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255),
        )
    return colors


# ───────────────────────────────────────────────────────────────────────────────
# Per-horizon scatter
# ───────────────────────────────────────────────────────────────────────────────


def plot_signal_diagnostics_scatter(
        result: SignalDiagnosticsResult,
        horizon: str,
        group_colors: Optional[Dict[str, str]] = None,
        title: Optional[str] = None,
        xlabel: str = 'Signal at t-1',
        ylabel: str = r'Vol-normalised cross-sectional return  $\tilde y$',
        scatter_size: float = 8.0,
        scatter_alpha: float = 0.40,
        fit_linewidth: float = 2.0,
        legend_loc: Optional[str] = 'lower right',
        ax: Optional[plt.Subplot] = None,
        **kwargs,
) -> Optional[plt.Figure]:
    """Pooled cross-sectional scatter for one horizon, asset-level points
    coloured by group, with OLS line annotated by β / t-stat / IC.

    Args:
        result: output of estimate_signal_diagnostics.
        horizon: label of the horizon to plot (must exist in
            result.horizon_labels).
        group_colors: dict mapping group label -> hex colour. If None, a
            default Tab10 colour cycle is used over result.group_order.
        title: figure title. If None, an auto-generated title is used.
        xlabel, ylabel: axis labels.
        scatter_size, scatter_alpha: marker styling.
        fit_linewidth: width of the OLS line.
        legend_loc: legend location; pass None to suppress the legend.
        ax: optional matplotlib axis. New figure created when None.
        **kwargs: ignored, accepted for compatibility.

    Returns:
        The Figure when `ax` is None, otherwise None.
    """
    if horizon not in result.pairs:
        raise ValueError(f"horizon {horizon!r} not in result.horizon_labels = "
                         f"{result.horizon_labels}")

    pairs = result.pairs[horizon]
    stats = result.pooled_universe.loc[horizon]

    fig: Optional[plt.Figure] = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    if group_colors is None:
        group_colors = _default_group_colors(result.group_order or
                                             sorted(pairs['group'].dropna().unique().tolist()))

    # Scatter by group (or single colour when no grouping)
    if result.group_order:
        for g in result.group_order:
            sub = pairs[pairs['group'] == g]
            if len(sub) == 0:
                continue
            ax.scatter(sub['z'], sub['r_norm_univ'],
                       s=scatter_size, alpha=scatter_alpha,
                       color=group_colors.get(g, '#888888'),
                       edgecolor='none', label=g)
    else:
        ax.scatter(pairs['z'], pairs['r_norm_univ'],
                   s=scatter_size, alpha=scatter_alpha,
                   color='#444', edgecolor='none')

    # OLS fit line through the data range
    beta = stats[SignalDiagnosticsColumns.BETA.value]
    if np.isfinite(beta) and len(pairs) > 0:
        z_min, z_max = float(pairs['z'].min()), float(pairs['z'].max())
        z_grid = np.linspace(z_min, z_max, 100)
        ax.plot(z_grid, beta * z_grid, color='black', lw=fit_linewidth)

    ax.axhline(0, color='grey', lw=0.5, ls=':')
    ax.axvline(0, color='grey', lw=0.5, ls=':')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)

    # Title with stat summary
    t_stat = stats[SignalDiagnosticsColumns.T_STAT.value]
    ic_p = stats[SignalDiagnosticsColumns.IC_PEARSON.value]
    n_obs = stats[SignalDiagnosticsColumns.N.value]
    sig = _significance_marker(t_stat)
    if title is None:
        title = (f"{horizon} horizon{(' ' + sig) if sig else ''}\n"
                 f"β = {beta:+.4f}   t = {t_stat:+.2f}   IC = {ic_p:+.4f}   "
                 f"n = {int(n_obs):,}")
    ax.set_title(title, fontsize=11)

    if result.group_order and legend_loc is not None:
        ax.legend(loc=legend_loc, fontsize=7, framealpha=0.95, ncol=2)

    return fig


# ───────────────────────────────────────────────────────────────────────────────
# Per-horizon bar chart
# ───────────────────────────────────────────────────────────────────────────────


def plot_signal_diagnostics_bars(
        result: SignalDiagnosticsResult,
        horizon: str,
        group_colors: Optional[Dict[str, str]] = None,
        title: Optional[str] = None,
        ylabel: str = r'β  (per unit of signal)',
        pooled_label: str = 'POOLED',
        pooled_color: str = '#444444',
        annotate_t_stat: bool = True,
        ax: Optional[plt.Subplot] = None,
        **kwargs,
) -> Optional[plt.Figure]:
    """Per-group β bar chart for one horizon, with pooled β as the right-most bar.

    Args:
        result: output of estimate_signal_diagnostics.
        horizon: label of the horizon to plot.
        group_colors: dict mapping group label -> hex colour.
        title: figure title; auto-generated when None.
        ylabel: y-axis label.
        pooled_label: label for the pooled bar.
        pooled_color: colour for the pooled bar.
        annotate_t_stat: place 't = ...' annotations above each bar.
        ax: optional matplotlib axis.
        **kwargs: ignored.

    Returns:
        The Figure when `ax` is None, otherwise None.
    """
    if horizon not in result.pooled_universe.index:
        raise ValueError(f"horizon {horizon!r} not in result.horizon_labels = "
                         f"{result.horizon_labels}")

    fig: Optional[plt.Figure] = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    if group_colors is None:
        group_colors = _default_group_colors(result.group_order)

    labels: List[str] = []
    betas: List[float] = []
    ts: List[float] = []
    cols: List[str] = []

    if (horizon, ) in result.per_group.index.droplevel('group').unique().to_list() or True:
        pass  # placeholder; the actual selection follows

    if horizon in result.per_group.index.get_level_values('horizon'):
        sub = result.per_group.loc[horizon]
        for g in result.group_order:
            if g not in sub.index:
                continue
            row = sub.loc[g]
            labels.append(g)
            betas.append(float(row[SignalDiagnosticsColumns.BETA.value]))
            ts.append(float(row[SignalDiagnosticsColumns.T_STAT.value]))
            cols.append(group_colors.get(g, '#888888'))

    # Pooled bar at the end
    pooled = result.pooled_universe.loc[horizon]
    labels.append(pooled_label)
    betas.append(float(pooled[SignalDiagnosticsColumns.BETA.value]))
    ts.append(float(pooled[SignalDiagnosticsColumns.T_STAT.value]))
    cols.append(pooled_color)

    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, betas, color=cols, edgecolor='black')
    ax.axhline(0, color='black', lw=0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, axis='y')

    if annotate_t_stat:
        for bar, t_stat in zip(bars, ts):
            height = bar.get_height()
            if not np.isfinite(height):
                continue
            sig = _significance_marker(t_stat)
            y_off = abs(height) * 0.04 + 0.005
            y_pos = height + y_off if height >= 0 else height - y_off
            va = 'bottom' if height >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f"t={t_stat:+.2f}{(' ' + sig) if sig else ''}",
                    ha='center', va=va, fontsize=8)

    if title is None:
        title = f"{horizon} horizon — β by group"
    ax.set_title(title, fontsize=11)
    return fig


# ───────────────────────────────────────────────────────────────────────────────
# Composite multi-horizon figure
# ───────────────────────────────────────────────────────────────────────────────


def plot_signal_diagnostics(
        result: SignalDiagnosticsResult,
        figsize: Tuple[float, float] = (20, 11),
        group_colors: Optional[Dict[str, str]] = None,
        title: Optional[str] = None,
) -> plt.Figure:
    """Two-row diagnostic figure: pooled scatters on top, per-group bars on the bottom,
    one column per horizon.

    Suitable for embedding as a single signal-quality page in a strategy
    factsheet.

    Args:
        result: output of estimate_signal_diagnostics.
        figsize: figure dimensions in inches.
        group_colors: dict mapping group label -> hex colour.
        title: figure-level title; auto-generated when None.

    Returns:
        Matplotlib Figure.
    """
    horizons = result.horizon_labels
    if not horizons:
        raise ValueError("result.horizon_labels is empty — nothing to plot")

    if group_colors is None:
        group_colors = _default_group_colors(result.group_order)

    n_cols = len(horizons)
    show_bars_row = bool(result.group_order)
    n_rows = 2 if show_bars_row else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                             constrained_layout=True, squeeze=False)

    # Row 0: scatters
    for j, h in enumerate(horizons):
        plot_signal_diagnostics_scatter(
            result=result, horizon=h,
            group_colors=group_colors,
            ax=axes[0, j],
            legend_loc='lower right' if j == 0 else None,
        )

    # Row 1: per-group bar charts
    if show_bars_row:
        # Pre-compute global y limits across horizons for visual comparability
        max_abs_beta = 0.0
        for h in horizons:
            if h in result.per_group.index.get_level_values('horizon'):
                vals = result.per_group.loc[h, SignalDiagnosticsColumns.BETA.value]
                if len(vals):
                    max_abs_beta = max(max_abs_beta, float(vals.abs().max()))
            pooled_beta = result.pooled_universe.loc[h,
                                                     SignalDiagnosticsColumns.BETA.value]
            if np.isfinite(pooled_beta):
                max_abs_beta = max(max_abs_beta, abs(pooled_beta))
        y_pad = max(max_abs_beta * 1.35, 0.10)

        for j, h in enumerate(horizons):
            plot_signal_diagnostics_bars(
                result=result, horizon=h,
                group_colors=group_colors,
                ax=axes[1, j],
            )
            axes[1, j].set_ylim(-y_pad, y_pad)

    # Figure title
    if title is None:
        date_part = ''
        if result.start_date is not None and result.end_date is not None:
            date_part = (f"  ({result.start_date.strftime('%b-%Y')} → "
                         f"{result.end_date.strftime('%b-%Y')})")
        title = (r"Cross-sectional signal predictive regression  $\tilde y_{i,t,t+h} = "
                 r"\beta \cdot z_{i,t-1} + \varepsilon$  (no intercept)" + date_part)
    fig.suptitle(title, fontsize=12, y=1.02)

    return fig

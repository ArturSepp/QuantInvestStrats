"""
Visualisation of cross-sectional signal predictive diagnostics.

Renders the output of ``qis.perfstats.signal_diagnostics.estimate_signal_diagnostics``.

Three plotting functions:

    plot_signal_diagnostics_boxplot — per-horizon quantile-bucket boxplot of
        the cross-sectional return distribution conditional on the signal
        score; the boxplot replaces an earlier scatter-based view that
        produced 5+ MB PDFs with thousands of overlaid points. Pooled
        regression β / t-stat / IC are summarised in the title.

    plot_signal_diagnostics_group_boxplot — per-horizon distribution of
        per-asset β values, one box per group, with the per-group
        regression t-stat annotated on each box. Visually consistent
        with the conditional-return boxplots above.

    plot_signal_diagnostics — composite two-row figure for a strategy
        factsheet: per-horizon return-conditional boxplots on top,
        per-group β boxplots on the bottom.

Follows the qis convention: each plotting function accepts an optional
``ax`` so it can be embedded in a larger factsheet layout, and returns
the Figure (or None when plotting into a supplied axis).
"""
# built-in
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Sequence, Tuple, Union

# qis
from qis.perfstats.signal_diagnostics import (
    SignalDiagnosticsColumns,
    SignalDiagnosticsResult,
    estimate_signal_diagnostics,
)
from qis.plots.boxplot import df_boxplot_by_classification_var, plot_box


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
        colors[g] = '#{:02x}{:02x}{:02x}'.format(
            int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255),
        )
    return colors


def _compute_per_asset_betas_from_pairs(
        pairs: pd.DataFrame,
        min_obs_per_asset: int,
) -> pd.DataFrame:
    """Per-asset no-intercept β on (z, r_norm_univ) from a pairs DataFrame.

    Inlined here to keep the plotting layer free of cross-module imports
    on a small bit of arithmetic. Matches qis.compute_per_asset_betas
    in semantics.

    Args:
        pairs: long-format DataFrame with at least the columns
            ``asset``, ``group``, ``z``, ``r_norm_univ``.
        min_obs_per_asset: Minimum (z, r) pair count per asset required
            to report a β. Assets with fewer observations are dropped.

    Returns:
        DataFrame indexed positionally with columns
        ``['asset', 'group', 'beta', 'n']``.
    """
    rows: List[Dict[str, object]] = []
    for asset, sub in pairs.groupby('asset', sort=False):
        sub = sub.dropna(subset=['z', 'r_norm_univ'])
        n = len(sub)
        if n < min_obs_per_asset:
            continue
        z = sub['z'].to_numpy(dtype=float)
        y = sub['r_norm_univ'].to_numpy(dtype=float)
        zz = float(z @ z)
        if zz <= 0:
            continue
        beta = float(z @ y) / zz
        group = sub['group'].iloc[0] if 'group' in sub.columns else None
        rows.append({'asset': asset, 'group': group, 'beta': beta, 'n': n})
    return pd.DataFrame(rows)


# ───────────────────────────────────────────────────────────────────────────────
# Per-horizon conditional-return boxplot
# ───────────────────────────────────────────────────────────────────────────────


def plot_signal_diagnostics_boxplot(
        result: SignalDiagnosticsResult,
        horizon: str,
        num_buckets: int = 10,
        title: Optional[str] = None,
        xlabel: str = 'Signal at t-1 (quantile buckets)',
        ylabel: str = r'Vol-normalised return  $\tilde y$',
        ax: Optional[plt.Subplot] = None,
        **kwargs,
) -> Optional[plt.Figure]:
    """Distribution of cross-sectional returns conditional on the signal.

    Buckets the signal score ``z`` into ``num_buckets`` quantile groups
    and renders the conditional return distribution as a boxplot. Each
    box reads "at this signal level, here's the distribution of
    realised forward returns across assets and dates". A monotone
    upward trend in box medians is the visual signature of a working
    signal; the title carries the pooled β/t/IC for the precise number.

    Replaces the earlier scatter-based view: visually equivalent
    information at a fraction of the file size (one box per bucket vs
    several thousand alpha-blended points).

    Args:
        result: Output of ``estimate_signal_diagnostics``.
        horizon: Label of the horizon to plot (must exist in
            ``result.horizon_labels``).
        num_buckets: Number of quantile buckets for the signal axis.
            Default 10 (deciles).
        title: Figure title. Auto-generated with β/t/IC when None.
        xlabel, ylabel: Axis labels.
        ax: Optional matplotlib axis. New figure created when None.
        **kwargs: Forwarded to ``df_boxplot_by_classification_var``.

    Returns:
        The Figure when ``ax`` is None, otherwise None.
    """
    if horizon not in result.pairs:
        raise ValueError(f"horizon {horizon!r} not in result.horizon_labels = "
                         f"{result.horizon_labels}")

    pairs = result.pairs[horizon][['z', 'r_norm_univ']].dropna()
    stats = result.pooled_universe.loc[horizon]

    fig: Optional[plt.Figure] = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Title with stat summary
    beta = stats[SignalDiagnosticsColumns.BETA.value]
    t_stat = stats[SignalDiagnosticsColumns.T_STAT.value]
    ic_p = stats[SignalDiagnosticsColumns.IC_PEARSON.value]
    n_obs = stats[SignalDiagnosticsColumns.N.value]
    sig = _significance_marker(t_stat)
    if title is None:
        title = (f"{horizon} horizon{(' ' + sig) if sig else ''}\n"
                 f"β = {beta:+.4f}   t = {t_stat:+.2f}   IC = {ic_p:+.4f}   "
                 f"n = {int(n_obs):,}")

    if len(pairs) == 0:
        ax.text(0.5, 0.5, 'No (z, r) pairs', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title(title, fontsize=11)
        return fig

    # Adaptive bucket count — pd.qcut raises on duplicate bin edges,
    # which happens routinely on signal panels with mass at zero
    # (sparse alphas, ranked signals with ties). Reduce the requested
    # bucket count until the empirical signal quantiles are unique.
    # Lower bound of 3 buckets to keep the plot interpretable; below
    # that we fall back to plain boxplot of returns (no signal buckets).
    z_arr = pairs['z'].to_numpy(dtype=float)
    n_buckets_use = num_buckets
    while n_buckets_use >= 3:
        edges = np.quantile(z_arr, np.linspace(0, 1, n_buckets_use + 1))
        if len(np.unique(edges)) == len(edges):
            break
        n_buckets_use -= 1
    if n_buckets_use < 3:
        # Fallback — too much mass on a single value to bucket usefully.
        # Show a single box of all returns; annotate the degeneracy.
        ax.boxplot(pairs['r_norm_univ'].to_numpy(dtype=float),
                   showfliers=False, medianprops=dict(linewidth=1.5))
        ax.set_xticks([1])
        ax.set_xticklabels(['all (signal degenerate)'])
        ax.axhline(0, color='grey', lw=0.5, ls=':')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, axis='y')
        ax.set_title(title, fontsize=11)
        return fig

    df_boxplot_by_classification_var(
        df=pairs,
        x='z',
        y='r_norm_univ',
        num_buckets=n_buckets_use,
        title=title,
        is_add_xlabel=False,
        xvar_format='{:+.2f}',
        yvar_format='{:+.3f}',
        medianline=True,
        showfliers=False,
        ax=ax,
        **kwargs,
    )
    ax.axhline(0, color='grey', lw=0.5, ls=':')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, axis='y')
    # Quantile-bucket labels are long ("[-1.31, -0.86)" style); rotate
    # so they don't overlap across adjacent boxes.
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45)
        lbl.set_horizontalalignment('right')
        lbl.set_fontsize(7)
    return fig


# ───────────────────────────────────────────────────────────────────────────────
# Per-horizon per-group β boxplot
# ───────────────────────────────────────────────────────────────────────────────


def plot_signal_diagnostics_group_boxplot(
        result: SignalDiagnosticsResult,
        horizon: str,
        min_obs_per_asset: int = 12,
        group_colors: Optional[Dict[str, str]] = None,
        title: Optional[str] = None,
        ylabel: str = r'$\beta$ per asset (no-intercept)',
        annotate_t_stat: bool = True,
        ax: Optional[plt.Subplot] = None,
        **kwargs,
) -> Optional[plt.Figure]:
    """Per-group β distribution as boxplots, with regression t-stat annotation.

    For each group, computes one β per asset (no-intercept regression of
    ``r_norm_univ`` on ``z`` from ``result.pairs[horizon]``) and plots
    the cross-asset β distribution as a box. The per-group regression
    t-statistic from ``result.per_group`` is annotated above each box,
    making this visually equivalent to the bar-chart view but using the
    same boxplot aesthetic as the row above.

    Args:
        result: Output of ``estimate_signal_diagnostics``.
        horizon: Horizon to plot.
        min_obs_per_asset: Minimum observations per asset to include
            in the per-asset β computation. Default 12.
        group_colors: Optional palette; defaults to Tab10.
        title: Figure title; auto-generated when None.
        ylabel: Y-axis label.
        annotate_t_stat: Annotate "t=...sig" above each box. Default True.
        ax: Optional axis.
        **kwargs: Ignored (kept for caller-API stability).

    Returns:
        Figure when ``ax`` is None, otherwise None.
    """
    if horizon not in result.pairs:
        raise ValueError(f"horizon {horizon!r} not in result.horizon_labels = "
                         f"{result.horizon_labels}")

    fig: Optional[plt.Figure] = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    if title is None:
        title = f"{horizon} horizon — β by group"
    ax.set_title(title, fontsize=11)

    pairs = result.pairs[horizon]
    if pairs is None or len(pairs) == 0 or 'group' not in pairs.columns:
        ax.text(0.5, 0.5, 'No grouped pairs at this horizon',
                ha='center', va='center', transform=ax.transAxes)
        return fig

    per_asset = _compute_per_asset_betas_from_pairs(pairs, min_obs_per_asset)
    per_asset = per_asset[per_asset['group'].notna()]
    if per_asset.empty:
        ax.text(0.5, 0.5,
                f'No assets meet min_obs_per_asset={min_obs_per_asset}',
                ha='center', va='center', transform=ax.transAxes)
        return fig

    # Order groups by result.group_order (filter to those actually present)
    group_order = [g for g in (result.group_order or [])
                   if g in per_asset['group'].values]
    if not group_order:
        group_order = sorted(per_asset['group'].dropna().unique().tolist())

    if group_colors is None:
        group_colors = _default_group_colors(group_order)

    colors = [group_colors.get(g, '#888888') for g in group_order]

    plot_box(
        df=per_asset[['group', 'beta']],
        x='group',
        y='beta',
        labels=group_order,
        colors=colors,
        xlabel=None,
        ylabel=ylabel,
        title=title,
        yvar_format='{:+.3f}',
        showfliers=False,
        showmedians=True,
        add_zero_line=True,
        x_rotation=45,
        legend_loc=None,
        ax=ax,
    )

    # Annotate per-group regression t-stat above each box. Pulls from
    # result.per_group rather than recomputing — the t-stat there is
    # the within-group pooled regression which is the canonical
    # "significance" number for that group.
    if annotate_t_stat and horizon in result.per_group.index.get_level_values('horizon'):
        per_group_h = result.per_group.loc[horizon]
        y_min, y_max = ax.get_ylim()
        y_span = y_max - y_min
        y_pad = y_span * 0.03 if y_span > 0 else 0.005
        for i, g in enumerate(group_order):
            if g not in per_group_h.index:
                continue
            t_stat = float(per_group_h.loc[g, SignalDiagnosticsColumns.T_STAT.value])
            sig = _significance_marker(t_stat)
            sub = per_asset[per_asset['group'] == g]['beta']
            if len(sub) == 0:
                continue
            # Position label just above each box's top whisker
            top = float(np.nanpercentile(sub, 75))
            iqr = top - float(np.nanpercentile(sub, 25))
            anchor = top + 1.5 * iqr if iqr > 0 else top
            anchor = min(anchor + y_pad, y_max - y_pad)
            ax.text(i, anchor,
                    f"t={t_stat:+.2f}{(' ' + sig) if sig else ''}",
                    ha='center', va='bottom', fontsize=8)
    return fig


# ───────────────────────────────────────────────────────────────────────────────
# Composite multi-horizon figure
# ───────────────────────────────────────────────────────────────────────────────


def plot_signal_diagnostics(
        result: SignalDiagnosticsResult,
        figsize: Tuple[float, float] = (20, 11),
        group_colors: Optional[Dict[str, str]] = None,
        title: Optional[str] = None,
        num_buckets: int = 10,
        min_obs_per_asset: int = 12,
) -> plt.Figure:
    """Two-row diagnostic figure for a signal panel.

    Row 0: Per-horizon conditional-return boxplot — distribution of
        forward returns at each signal-quantile bucket. Pooled β/t/IC
        annotated in each panel title.
    Row 1: Per-horizon per-group β boxplot — distribution of per-asset
        β by group, with the per-group regression t-stat annotated.

    Both rows share the boxplot aesthetic; the bottom row is omitted
    when ``result.group_order`` is empty.

    Args:
        result: Output of ``estimate_signal_diagnostics``.
        figsize: Figure dimensions.
        group_colors: Optional group → hex palette.
        title: Figure-level title; auto-generated when None.
        num_buckets: Number of quantile buckets for row 0. Default 10.
        min_obs_per_asset: Per-asset β minimum-obs threshold for row 1.

    Returns:
        Matplotlib Figure.
    """
    horizons = result.horizon_labels
    if not horizons:
        raise ValueError("result.horizon_labels is empty — nothing to plot")

    if group_colors is None:
        group_colors = _default_group_colors(result.group_order)

    n_cols = len(horizons)
    show_bottom_row = bool(result.group_order)
    n_rows = 2 if show_bottom_row else 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    # Row 0: conditional-return boxplots
    for j, h in enumerate(horizons):
        plot_signal_diagnostics_boxplot(
            result=result,
            horizon=h,
            num_buckets=num_buckets,
            ax=axes[0, j],
        )

    # Row 1: per-group β boxplots
    if show_bottom_row:
        for j, h in enumerate(horizons):
            plot_signal_diagnostics_group_boxplot(
                result=result,
                horizon=h,
                min_obs_per_asset=min_obs_per_asset,
                group_colors=group_colors,
                ax=axes[1, j],
            )
        # Match y-limits across the row for visual comparability
        ymins, ymaxs = [], []
        for j in range(n_cols):
            yl = axes[1, j].get_ylim()
            ymins.append(yl[0]); ymaxs.append(yl[1])
        if ymins:
            y_lim = (min(ymins), max(ymaxs))
            for j in range(n_cols):
                axes[1, j].set_ylim(*y_lim)

    # Figure title — placed with explicit y so the suptitle doesn't
    # overlap row-0 subplot titles. constrained_layout was creating
    # this overlap; we drop it and use tight_layout + subplots_adjust
    # to reserve top margin.
    if title is None:
        date_part = ''
        if result.start_date is not None and result.end_date is not None:
            date_part = (f"  ({result.start_date.strftime('%b-%Y')} → "
                         f"{result.end_date.strftime('%b-%Y')})")
        title = (r"Cross-sectional signal predictive regression  $\tilde y_{i,t,t+h} = "
                 r"\beta \cdot z_{i,t-1} + \varepsilon$  (no intercept)" + date_part)

    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    fig.suptitle(title, fontsize=12, y=0.97)
    return fig


# ───────────────────────────────────────────────────────────────────────────────
# Compute+plot convenience wrappers
# ───────────────────────────────────────────────────────────────────────────────
#
# These two functions handle the common "I have a per-frequency returns
# dict and a single signal panel — give me the composite or the beta
# boxplot" call pattern. They were previously thin wrappers in
# optimalportfolios.alphas; moving them to qis removes a layer for any
# caller whose signal is already a DataFrame (i.e. doesn't need
# AlphasData resolution). Callers with AlphasData should resolve to a
# DataFrame first via the helper in optimalportfolios.alphas.

def plot_signal_diagnostics_for_returns(
        asset_returns_dict: Dict[str, pd.DataFrame],
        signal: pd.DataFrame,
        *,
        horizons: Sequence[Union[int, str]] = (1, 2, 3, 6),
        group_data: Optional[pd.Series] = None,
        group_order: Optional[Sequence[str]] = None,
        is_log_returns: bool = True,
        figsize: Tuple[float, float] = (22, 12),
        title: Optional[str] = None,
        group_colors: Optional[Dict[str, str]] = None,
        num_buckets: int = 10,
        min_obs_per_asset: int = 12,
) -> plt.Figure:
    """Run the predictive regression and render the composite figure.

    One-shot "compute+plot" path for callers that have a per-frequency
    returns dict and a single signal DataFrame. Equivalent to
    ``estimate_signal_diagnostics`` followed by
    ``plot_signal_diagnostics``, with the regression result discarded
    after rendering.

    For AlphasData-keyed callers (multi-component signal containers
    used elsewhere in the alpha-aggregator world), resolve the
    DataFrame first in caller code, then pass it here. This keeps qis
    free of any AlphasData dependency.

    Args:
        asset_returns_dict: Per-frequency returns dict.
        signal: Signal panel (DataFrame indexed by date, columns by asset).
        horizons: Forward-return horizons.
        group_data: Optional asset → group-label Series.
        group_order: Explicit group ordering for the per-group panel.
        is_log_returns: True if ``asset_returns_dict`` contains log returns.
        figsize: Figure dimensions in inches.
        title: Figure-level title. Auto-generated when None.
        group_colors: Optional palette mapping.
        num_buckets: Quantile buckets for row-0 conditional-return boxplots.
        min_obs_per_asset: Minimum observations per asset for the
            row-1 per-asset β computation.

    Returns:
        Matplotlib Figure.
    """
    result = estimate_signal_diagnostics(
        asset_returns_dict=asset_returns_dict,
        signal=signal,
        horizons=horizons,
        group_data=group_data,
        group_order=group_order,
        is_log_returns=is_log_returns,
    )
    return plot_signal_diagnostics(
        result=result,
        figsize=figsize,
        title=title,
        group_colors=group_colors,
        num_buckets=num_buckets,
        min_obs_per_asset=min_obs_per_asset,
    )


def plot_signal_diagnostics_beta_boxplot(
        asset_returns_dict: Dict[str, pd.DataFrame],
        signal: pd.DataFrame,
        *,
        horizons: Sequence[Union[int, str]] = (1, 2, 3, 6),
        group_data: Optional[pd.Series] = None,
        is_log_returns: bool = True,
        min_obs_per_asset: int = 12,
        hue: Optional[str] = 'asset_freq',
        hue_display_label: str = 'Asset returns sampling',
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (12, 7),
) -> plt.Figure:
    """Boxplot of per-asset β across horizons.

    For each horizon, estimates one β per asset (no-intercept regression
    on that asset's (z, r_norm_univ) pairs) and visualises the
    cross-asset distribution as boxes. Boxes are optionally coloured by
    the asset's native cadence (``asset_freq`` column on the underlying
    pairs frame); the legend exposes this with a user-facing label
    rather than the internal column name.

    Args:
        asset_returns_dict: Per-frequency returns dict.
        signal: Signal panel.
        horizons: Forward-return horizons.
        group_data: Optional asset → group-label Series. Doesn't affect
            the boxplot but populates the ``group`` column on the
            underlying per-asset table.
        is_log_returns: True if returns dict contains log returns.
        min_obs_per_asset: Minimum observations per asset per horizon.
        hue: Column to use for box colouring. Default ``'asset_freq'``;
            also supports ``'group'``. Pass ``None`` to disable.
        hue_display_label: User-facing legend title for ``hue``.
            Default ``'Asset returns sampling'``.
        title: Figure title. Auto-generated when None.
        figsize: Figure dimensions in inches.

    Returns:
        Matplotlib Figure.
    """
    from qis.perfstats.signal_diagnostics import compute_per_asset_betas

    result = estimate_signal_diagnostics(
        asset_returns_dict=asset_returns_dict,
        signal=signal,
        horizons=horizons,
        group_data=group_data,
        is_log_returns=is_log_returns,
    )
    df = compute_per_asset_betas(
        result=result, min_obs_per_asset=min_obs_per_asset,
    )

    if title is None:
        title = (f"Per-asset β by horizon — pooled signal "
                 f"(min {min_obs_per_asset} obs)")

    fig, ax = plt.subplots(figsize=figsize)
    if df.empty:
        ax.text(0.5, 0.5,
                f"No assets meet min_obs_per_asset={min_obs_per_asset}",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return fig

    # qis.plot_box's internal palette helper does df.groupby(x).mean()
    # on the WHOLE frame — projecting to the columns plot_box actually
    # consumes avoids pandas trying to "average" the asset string col.
    # Rename hue column to the display label so the legend reads
    # naturally.
    use_hue = hue
    if use_hue is not None:
        if use_hue not in df.columns or df[use_hue].nunique() <= 1:
            use_hue = None

    df_plot = df[['horizon', 'beta']].copy()
    plot_hue: Optional[str] = None
    if use_hue is not None:
        plot_hue = hue_display_label
        df_plot[plot_hue] = df[use_hue].values

    plot_box(
        df=df_plot,
        x='horizon',
        y='beta',
        hue=plot_hue,
        xlabel='Horizon (native cadence units)',
        ylabel=r'$\beta$ (per unit of signal)',
        title=title,
        showmedians=True,
        add_zero_line=True,
        yvar_format='{:+.3f}',
        legend_loc='upper right',
        add_hue_to_legend_title=True,
        ax=ax,
    )
    return fig
"""The strategy factsheet page for realised drawdowns and benchmark-event returns."""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd

from qis.perfstats.drawdown_events import compute_benchmark_drawdown_events
from qis.perfstats.perf_stats import compute_drawdowns_stats_table, compute_max_current_drawdown
from qis.plots.derived.drawdowns import _plot_ranked_drawdown_paths


def _message(ax, message):
    """Keep the panel heading when its sample is unavailable."""
    ax.set_axis_off()
    ax.text(0.5, 0.5, message, ha='center', va='center', transform=ax.transAxes, fontsize=9)


def _plot_event_bars(ax, events, point, labels, colors):
    """Plot signed paired returns on the two observed endpoints for each event."""
    ax.axvline(0, color='0.55', linewidth=0.7)
    rows = np.arange(len(events))
    for key, offset, color in zip(('strategy', 'benchmark'), (-0.18, 0.18), colors):
        values = events[f'{key}_{point}_return']
        observed = values.notna()
        bars = ax.barh(rows[observed] + offset, values[observed], height=0.32, color=color,
                       label=key.capitalize())
        ax.bar_label(bars, labels=[f'{value:+.1%}' for value in values[observed]],
                     padding=2, fontsize=7, color=color)
        for ypos in rows[~observed] + offset:
            ax.annotate('n/a', (0, ypos), xytext=(3, 0), textcoords='offset points',
                        va='center', fontsize=7, color=color)
    ax.set_yticks(rows, labels, fontsize=7)
    ax.set_ylim(len(events) - 0.5, -0.5)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.tick_params(axis='x', labelsize=7)
    ax.grid(axis='x', alpha=0.2)
    ax.set_axisbelow(True)


def plot_drawdown_analysis(strategy, benchmark, min_depth, named_windows, figsize):
    """Render one page on the two original observation grids within the report period."""
    limit = 4
    colors = ('#24597b', '#ad582c')
    strategy = strategy.sort_index().dropna()
    benchmark = benchmark.sort_index().dropna()
    strategy_name = str(strategy.name or 'Strategy')
    benchmark_name = str(benchmark.name or 'Benchmark')
    has_strategy = len(strategy) > 0 and np.isfinite(strategy).all() and (strategy > 0).all()
    if not has_strategy:
        strategy = strategy.iloc[:0]
    if benchmark.empty or not np.isfinite(benchmark).all() or (benchmark <= 0).any():
        reason = 'Benchmark events require observed positive NAVs in the reporting period.'
    else:
        events = compute_benchmark_drawdown_events(strategy, benchmark, min_depth, named_windows)
        reason = '' if not events.empty else f'No benchmark event deeper than {min_depth:.1%}.'

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    heights = [1, 2, 0.45] if reason else [0.80, 1.35, 1.30, 1.25]
    grid = fig.add_gridspec(len(heights), 2, height_ratios=heights, hspace=0.10)
    table_ax = fig.add_subplot(grid[0, :])
    paths_ax = fig.add_subplot(grid[1, :])
    fig.suptitle(f'Drawdown analysis\nStrategy: {strategy_name}    |    '
                 f'Benchmark: {benchmark_name}', fontsize=13, fontweight='bold')
    if not has_strategy:
        _message(table_ax, 'Strategy drawdowns require observed positive NAVs.')
        _message(paths_ax, 'No strategy episodes to display.')
    else:
        maximum, current = compute_max_current_drawdown(strategy)
        episodes = compute_drawdowns_stats_table(strategy, freq=None, max_num=limit)
        summary = f'Strategy episodes    |    Maximum {maximum:.1%}    Current {current:.1%}'
        table_ax.set_title(summary, fontsize=10, loc='left')
        table_ax.set_axis_off()
        if episodes.empty:
            _message(table_ax, 'No decline below a running peak in the reporting period.')
            _message(paths_ax, 'No strategy episodes to display.')
        else:
            rows = [[f'{e.start:%d %b %y}', f'{e.trough:%d %b %y}', f'{e.max_dd:.1%}',
                     f'{e.end:%d %b %y}' if e.is_recovered else 'Open',
                     int(e.days_to_trough), int(e.days_recovery) if e.is_recovered else '—']
                    for e in episodes.itertuples()]
            table = table_ax.table(cellText=rows, colLabels=[
                'Peak', 'Trough', 'Depth', 'Recovery', 'Peak to trough\n(observations)',
                'Trough to recovery\n(observations)'], cellLoc='center', bbox=[0, 0.05, 1, 0.85])
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            for (row, _), cell in table.get_celld().items():
                cell.set_edgecolor('white')
                cell.set_facecolor('#e4ecf1' if row == 0 else ('#f3f6f8' if row % 2 else 'white'))
            labels = [f"{e.start:%d %b %y}{'' if e.is_recovered else ' (open)'}"
                      for e in episodes.itertuples()]
            _plot_ranked_drawdown_paths(
                strategy, episodes, freq=None, ax=paths_ax, title='Strategy paths from each peak',
                fontsize=8, legend_labels=labels, highlight_ongoing=True, var_format='{:.1%}',
                legend_loc='upper center', bbox_to_anchor=(0.5, -0.17), ncols=len(labels),
                y_limits=(None, None))
            paths_ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))

    if reason:
        notice = fig.add_subplot(grid[2, :])
        notice.set_title('Benchmark drawdowns', fontsize=10)
        _message(notice, reason)
        return fig

    loss_ax = fig.add_subplot(grid[2, 0])
    recovery_ax = fig.add_subplot(grid[2, 1])
    event_ax = fig.add_subplot(grid[3, :])
    displayed = events.head(limit)
    labels = []
    for e in displayed.itertuples():
        recovery = f'{e.recovery:%d %b %y}' if pd.notna(e.recovery) else 'Open'
        label = f'{e.peak:%d %b %y} – {e.trough:%d %b %y}\nRecovery: {recovery}'
        labels.append(f'{e.name}\n{label}' if e.name else label)
    _plot_event_bars(loss_ax, displayed, 'trough', labels, colors)
    _plot_event_bars(recovery_ax, displayed, 'recovery', [''] * len(displayed), colors)
    values = displayed.filter(like='_return').to_numpy(dtype=float)
    lo, hi = min(0, np.nanmin(values)), max(0, np.nanmax(values))
    margin = 0.22 * (hi - lo)
    for ax in (loss_ax, recovery_ax):
        ax.set_xlim(lo - margin, hi + margin)
    coverage = f'{len(displayed)} of {len(events)} benchmark declines deeper than {min_depth:.1%}'
    loss_ax.set_title(f'{coverage}\nPeak to trough', fontsize=9)
    recovery_ax.set_title('Same benchmark episodes\nPeak to recovery', fontsize=9)
    recovery_ax.legend(loc='best', fontsize=7)

    worst = displayed.iloc[0]
    ref = benchmark.loc[worst.peak:worst.end]
    nav = strategy.reindex(ref.index)
    for series, label, color in ((nav, strategy_name, colors[0]), (ref, benchmark_name, colors[1])):
        if pd.notna(series.iloc[0]):
            event_ax.plot(series.index, 100 * series / series.iloc[0], label=label,
                          color=color, linewidth=1.4, marker='.' if len(ref) < 20 else None)
    event_ax.axvline(worst.trough, color='0.5', linestyle=':', linewidth=1)
    event_ax.set_ylabel('NAV at peak = 100', fontsize=8)
    event_ax.legend(fontsize=8)
    if len(ref) <= 7:
        event_ax.set_xticks(ref.index, ref.index.strftime('%d %b %y'), fontsize=7)
    else:
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        event_ax.xaxis.set_major_locator(locator)
        event_ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    event_ax.tick_params(labelsize=8)
    event_ax.grid(alpha=0.2)
    end_label = 'open at' if pd.isna(worst.recovery) else 'recovered'
    event_ax.set_title(f'Deepest benchmark event    |    {worst.peak:%d %b %Y} to '
                       f'{worst.trough:%d %b %Y}    |    {end_label} {worst.end:%d %b %Y}',
                       fontsize=9)
    if pd.isna(nav.iloc[0]):
        event_ax.text(0.01, 0.03, 'Strategy NAV unavailable at the benchmark peak.',
                      transform=event_ax.transAxes, fontsize=8)
    return fig

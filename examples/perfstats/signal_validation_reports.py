"""Validate low-beta and momentum scores on ten synthetic assets, entirely offline.

Display the figures with `python -m examples.perfstats.signal_validation_reports`.
Call `run_local(Locals.EXPORT, output_path='reports')` to save them.

The PDF shows both the matched forward-return reports and QIS's native predictive
regressions. The ZIP contains their full tables, including dates, sample membership
counts, weights and inference statuses.
"""
from enum import Enum
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from examples.perfstats.forward_signal_report import dated_ic_for_lookback, synthetic_inputs
from qis.local_path import get_output_path
from qis.perfstats.forward_return_targets import compute_forward_return_targets
from qis.perfstats.signal_diagnostics import (
    compute_ic_timeseries, compute_per_asset_betas, estimate_ic_ir, estimate_signal_diagnostics,
)
from qis.perfstats.signal_ic import compute_signal_ic_summary
from qis.plots.derived.signal_diagnostics_plot import (
    plot_signal_diagnostics, plot_signal_diagnostics_beta_boxplot,
)


HORIZONS = (1, 2, 3, 6)
SIGNALS = {'low_beta': 'Low beta', 'momentum': 'Momentum'}
TARGETS = {'raw': 'Raw', 'market_adjusted': 'Market-adjusted',
           'factor_adjusted': 'Market + bond adjusted'}
COLORS = ('#254c77', '#b45520', '#007a72', '#784889')


class Locals(Enum):
    SHOW = 1
    EXPORT = 2


def horizon_report(common, ewma_span, warmup_periods, *, horizons=HORIZONS, targets=TARGETS,
                   unit='quarters', universe='Ten synthetic assets'):
    """Plot the mean rank IC and the within-date change from raw, on identical assets."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey='row')
    for (signal, label), column in zip(SIGNALS.items(), axes.T):
        for ax, metric, lower, upper in (
                (column[0], 'mean_ic', 'ic_lower', 'ic_upper'),
                (column[1], 'mean_ic_minus_reference', 'difference_lower', 'difference_upper')):
            for i, (target, target_label) in enumerate(targets.items()):
                if metric == 'mean_ic_minus_reference' and target == 'raw':
                    continue
                rows = common.query("signal == @signal and target == @target").sort_values(
                    'horizon')
                x = rows.horizon + (i - 1) * 0.09
                ax.plot(x, rows[metric], 'o-', color=COLORS[i], label=target_label)
                ax.vlines(x, rows[lower], rows[upper], color=COLORS[i], linewidth=1.6)
            ax.axhline(0, color='0.5', linewidth=0.8)
            ax.grid(axis='y', alpha=0.22)
        column[0].set_title(label)
        column[1].set(xlabel=f'Forward horizon ({unit})', xticks=horizons)
    axes[0, 0].set_ylabel('Mean rank IC')
    axes[1, 0].set_ylabel('Adjusted IC minus raw IC')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', ncol=3,
               bbox_to_anchor=(0.5, 0.915), frameon=False)
    fig.suptitle('How much predictive power survives factor adjustment?', fontsize=18)
    as_of = common.as_of.iloc[0]
    fig.text(0.5, 0.925, f'{universe} through {as_of:%B %Y}  |  '
             f'Common sample, EWMA span {ewma_span:g} valid dates', ha='center', fontsize=11)
    fig.text(0.5, 0.025, 'Bars show pointwise 95% HAC intervals. '
             f'Inference requires {warmup_periods} effective dates and estimable variance.',
             ha='center', fontsize=10)
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.12, top=0.82, hspace=0.18, wspace=0.09)
    return fig


def sensitivity_report(summary, lookbacks, ewma_span, rolling_span, *, horizons=HORIZONS,
                       unit='quarters', target='factor_adjusted', lookback_labels=None):
    """Compare endpoint weighting and the momentum formation window."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True, layout='constrained')
    adjusted = summary.query("sample == 'common' and target == @target")
    policies = {'equal': 'Full history', 'rolling': f'Rolling {rolling_span}',
                'ewma': f'EWMA {ewma_span:g}'}
    for ax, (signal, label) in zip(axes[:2], SIGNALS.items()):
        for (policy, name), color in zip(policies.items(), COLORS):
            rows = adjusted.query("signal == @signal and weighting == @policy").sort_values(
                'horizon')
            ax.plot(rows.horizon, rows.mean_ic, 'o-', color=color, label=name)
        ax.set_title(f'{label} by weighting')
    sweep = lookbacks.query(
        "sample == 'common' and target == @target and signal == 'momentum'")
    for (lookback, rows), color in zip(sweep.groupby('lookback'), COLORS):
        rows = rows.sort_values('horizon')
        label = lookback_labels[lookback] if lookback_labels else f'{lookback}Q lookback'
        axes[2].plot(rows.horizon, rows.mean_ic, 'o-', color=color, label=label)
    axes[2].set_title(f'Momentum lookback, EWMA {ewma_span:g}')
    for ax in axes:
        ax.set(xlabel=f'Forward horizon ({unit})', xticks=horizons)
        ax.axhline(0, color='0.5', lw=0.8)
        ax.grid(axis='y', alpha=0.22)
        ax.legend(frameon=False)
    axes[0].set_ylabel('Mean rank IC on two-factor returns')
    fig.suptitle('Sensitivity to history and momentum lookback', fontsize=17)
    return fig


def dated_report(dated):
    """Show quarterly realised rank IC and the effect of missing loadings on breadth."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout='constrained')
    common = dated.query("horizon == 1 and sample == 'common' and target == 'factor_adjusted'")
    for (signal, label), color in zip(SIGNALS.items(), COLORS):
        rows = common[common.signal == signal]
        axes[0].plot(rows.realisation_date, rows.ic, color=color, label=label, lw=1.25)
    axes[0].set(title='One-quarter rank IC on two-factor returns',
                ylabel='Realised IC', ylim=(-1.05, 1.05))
    axes[0].axhline(0, color='0.5', lw=0.8)
    raw = dated.query("signal == 'low_beta' and horizon == 1 and target == 'raw'")
    for sample, label, color in [('available_raw', 'Available raw', COLORS[0]),
                                 ('common', 'Common to all targets', COLORS[2])]:
        rows = raw[raw['sample'] == sample]
        axes[1].step(rows.realisation_date, rows.n_assets, where='post', label=label, color=color)
    axes[1].set(title='Eligible assets in each comparison', ylabel='Assets',
                xlabel='Realisation date')
    for ax in axes:
        ax.legend(frameon=False, loc='lower right', ncol=2)
        ax.grid(axis='y', alpha=0.22)
    fig.suptitle('When did the signal work, and on how many assets?', fontsize=17)
    return fig


def native_reports(quarterly, low_beta):
    """Fit through-origin slopes on cross-sectionally standardised scores and returns."""
    returns = {'QE-DEC': quarterly}
    groups = pd.Series(np.where(low_beta >= low_beta.median(), 'Lower beta', 'Higher beta'),
                       index=low_beta.index)
    scores = {'low_beta': pd.DataFrame([low_beta] * len(quarterly), index=quarterly.index),
              'momentum': quarterly}
    scores = {name: frame.copy() for name, frame in scores.items()}
    for frame in scores.values():
        frame.iloc[:4] = np.nan
    return native_score_reports(returns, scores, groups, HORIZONS)


def native_score_reports(returns, scores, groups, horizons):
    """Plot and tabulate native raw-return diagnostics for dated asset score panels."""
    figures, tables = [], {}
    for signal, frame in scores.items():
        z = frame.sub(frame.mean(axis=1), axis=0).div(frame.std(axis=1), axis=0)
        result = estimate_signal_diagnostics(returns, z, group_data=groups, horizons=horizons)
        tables[f'{signal}_pooled'] = result.pooled_universe.reset_index()
        tables[f'{signal}_groups'] = result.per_group.reset_index()
        tables[f'{signal}_assets'] = compute_per_asset_betas(result)
        tables[f'{signal}_pairs'] = pd.concat(
            [pairs.assign(horizon=h) for h, pairs in result.pairs.items()], ignore_index=True)
        for method in ('spearman', 'pearson'):
            by_date = compute_ic_timeseries(result, method=method)
            tables[f'{signal}_{method}_ic'] = pd.concat(
                [values.rename_axis('date').reset_index().assign(horizon=h)
                 for h, values in by_date.items()], ignore_index=True)
            tables[f'{signal}_{method}_ir'] = estimate_ic_ir(result, method=method).reset_index()
        label = SIGNALS[signal]
        figures.append(plot_signal_diagnostics(
            result, num_buckets=5, figsize=(17, 8),
            title=f'{label} | Standardised raw returns by score bucket and slopes by group'))
        figures.append(plot_signal_diagnostics_beta_boxplot(
            returns, z, horizons=horizons, hue=None,
            title=f'{label} | Distribution of per-asset predictive slopes at each horizon'))
    return figures, tables


def stability_report(tables, *, horizon_label='Qtrs', grid='quarterly'):
    """Display the full-history IC information ratios, t-statistics and hit rates."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for ax, (signal, label) in zip(axes, SIGNALS.items()):
        rows = []
        for method, name in [('spearman', 'Rank'), ('pearson', 'Pearson')]:
            for row in tables[f'{signal}_{method}_ir'].itertuples(index=False):
                rows.append([name, row.horizon, f'{row.n_dates:.0f}', f'{row.mean_IC:.3f}',
                             f'{row.IC_IR:.2f}', f'{row.IC_IR_an:.2f}', f'{row.t_stat:.2f}',
                             f'{row.hit_rate:.0%}'])
        table = ax.table(cellText=rows, colLabels=['IC', horizon_label, 'Dates', 'Mean IC',
                         'IC-IR', 'Ann. IR', 't-stat', 'Hit rate'], cellLoc='center',
                         bbox=[0, 0.07, 1, 0.83], edges='horizontal')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        ax.set_title(label)
        ax.axis('off')
    fig.suptitle(f'How stable are the raw-return ICs?\nFull history on the {grid} grid',
                 fontsize=16)
    return fig


def run_local(local: Locals, *, output_path=None, ewma_span: float = 24,
              rolling_span: int = 24, warmup_periods: int = 12) -> None:
    """Show all pages or export the PDF and CSV archive under the QIS output path.

    Args:
        local: SHOW opens the figures. EXPORT writes the two files.
        output_path: Output directory override for EXPORT. Defaults to QIS OUTPUT_PATH.
        ewma_span: Decay span in valid observations at the given forward horizon.
        rolling_span: Number of valid observations required for the rolling comparison.
        warmup_periods: Minimum effective dates for a pointwise HAC interval.
    """
    if not isinstance(local, Locals):
        raise ValueError('local must be a Locals member')
    dates, observations, adjustments, quarterly, low_beta = synthetic_inputs()
    targets = compute_forward_return_targets(dates, observations, horizons=HORIZONS,
                                             adjustments=adjustments)
    dated = dated_ic_for_lookback(targets, quarterly, low_beta, lookback=1)
    settings = dict(as_of=dates[-1], warmup_periods=warmup_periods)
    full = compute_signal_ic_summary(dated, **settings)
    rolling = compute_signal_ic_summary(dated, rolling_span=rolling_span, **settings)
    ewma = compute_signal_ic_summary(dated, ewma_span=ewma_span, **settings)
    summary = pd.concat([report.summary for report in (full, rolling, ewma)], ignore_index=True)
    sweep = [ewma.summary.assign(lookback=1)]
    for lookback in (2, 4):
        scores = dated_ic_for_lookback(targets, quarterly, low_beta, lookback)
        sweep.append(compute_signal_ic_summary(scores, ewma_span=ewma_span, **settings)
                     .summary.assign(lookback=lookback))
    lookbacks = pd.concat(sweep, ignore_index=True)
    diagnostics, tables = native_reports(quarterly, low_beta)
    figures = [horizon_report(ewma.summary.query("sample == 'common'"), ewma_span, warmup_periods),
               sensitivity_report(summary, lookbacks, ewma_span, rolling_span),
               dated_report(ewma.dated), stability_report(tables), *diagnostics]
    try:
        if local is Locals.SHOW:
            plt.show()
            return
        path = Path(get_output_path(output_path))
        with PdfPages(path / 'signal_validation.pdf') as pdf:
            for fig in figures:
                pdf.savefig(fig)
        tables.update(forward_targets=targets, matched_summary=summary,
                      dated_full=full.dated, dated_rolling=rolling.dated, dated_ewma=ewma.dated,
                      lookback_summary=lookbacks)
        with ZipFile(path / 'signal_validation_tables.zip', 'w', ZIP_DEFLATED) as archive:
            for name, table in tables.items():
                archive.writestr(f'{name}.csv', table.to_csv(index=False))
        print(f'Wrote {len(figures)} report pages and {len(tables)} tables to {path}')
    finally:
        for fig in figures:
            plt.close(fig)


if __name__ == '__main__':
    run_local(local=Locals.SHOW)

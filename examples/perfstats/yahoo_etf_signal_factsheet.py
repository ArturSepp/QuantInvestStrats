"""Export the full QIS signal validation factsheet for the Yahoo SPY/TLT ETF comparison.

Run python -m examples.perfstats.yahoo_etf_signal_factsheet to use the configured output directory.
Pass output_dir or prices_csv to run_example to override the destination or replay saved closes.
"""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd

from examples.perfstats.signal_validation_reports import (
    SIGNALS, horizon_report, native_score_reports, sensitivity_report, stability_report,
)
from examples.perfstats.yahoo_etf_signal_ic import (
    ASSETS, BENCHMARKS, FIRST_DECISION, HORIZONS, LAST_MONTH, dated_ics, formation_panel,
    load_prices, matched_membership,
)
from qis.local_path import get_output_path
from qis.perfstats.signal_ic import compute_signal_ic_summary
from qis.perfstats.signal_ic_attribution import compute_beta_reuse_ic, decompose_rank_ic


TARGETS = {'raw': 'Raw', 'residual': 'SPY/TLT residual'}
COLORS = {'raw': '#305375', 'residual': '#a44d21'}
GROUPS = pd.Series({'IWM': 'Equity', 'EFA': 'Equity', 'EEM': 'Equity', 'VNQ': 'Equity',
                    'LQD': 'Credit', 'HYG': 'Credit', 'GLD': 'Commodity', 'DBC': 'Commodity'})
DOCS = 'https://quantinveststrats.readthedocs.io/en/latest/signal_diagnostics.html'


def overview(common, ewma_span):
    """Show the exact matched ICs and specify the scores, targets and investment universe."""
    fig = plt.figure(figsize=(15, 9))
    fig.text(.07, .86, 'ETF signal validation', fontsize=33, weight='bold', color='#183952')
    fig.text(.07, .805, 'Low beta and 12–1 momentum   |   Forward returns, January 2011 to '
             'December 2025', fontsize=16, color='#305375')
    values = common.set_index(['horizon', 'signal', 'target'])
    rows = []
    for h in HORIZONS:
        scope = values.loc[(h, 'low_beta', 'raw')]
        ics = [f'{values.loc[(h, s, t), "mean_ic"]:+.4f}' for s in SIGNALS for t in TARGETS]
        rows.append([f'{h} month' if h == 1 else f'{h} months', f'{scope.n_valid:.0f}',
                     f'{scope.mean_n_assets:.0f}', *ics])
    ax = fig.add_axes([.07, .475, .86, .265])
    ax.axis('off')
    headings = ['Horizon', 'Valid dates', 'ETFs', 'Low beta\nraw', 'Low beta\nresidual',
                'Momentum\nraw', 'Momentum\nresidual']
    table = ax.table(cellText=rows, colLabels=headings, cellLoc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('white')
        cell.set_facecolor('#254b66' if r == 0 else ('#eaf1f4' if r % 2 else '#f6f8f9'))
        if r == 0:
            cell.set_text_props(color='white', weight='bold')
        elif c in (4, 6):
            cell.set_text_props(color=COLORS['residual'], weight='bold')
    fig.text(.07, .40, 'Ranked ETFs', fontsize=15, weight='bold', color='#183952')
    fig.text(.07, .355, 'IWM  ·  EFA  ·  EEM  ·  LQD  ·  HYG  ·  GLD  ·  DBC  ·  VNQ',
             fontsize=16)
    fig.text(.07, .272, 'How to read the comparison', fontsize=15, weight='bold', color='#183952')
    lines = [
        'Low beta ranks lower signed SPY loadings higher, conditional on TLT. Momentum ranks '
        'log price changes from 12 months ago to one month ago.',
        'Raw is the forward log total return. Residual subtracts the realised SPY and TLT '
        'log returns times their joint betas held fixed at formation.',
        f'Betas use a {ewma_span:g}-month EWMA. Mean Spearman IC uses EWMA span {ewma_span:g} '
        'valid dates per horizon. Both targets use the same names and dates.',
    ]
    for y, line in zip((.222, .169, .116), lines):
        fig.text(.07, y, line, fontsize=12)
    return fig


def history_and_coverage(dated, members):
    """Show monthly realised ICs for both targets and exact breadth at every horizon."""
    fig = plt.figure(figsize=(15, 9), layout='constrained')
    grid = fig.add_gridspec(2, 2, width_ratios=(2.5, 1.0))
    for row, (signal, label) in enumerate(SIGNALS.items()):
        ax = fig.add_subplot(grid[row, 0])
        series = dated.query("sample == 'common' and signal == @signal and horizon == 1")
        for name, target in TARGETS.items():
            points = series.loc[series.target.eq(name)].sort_values('realisation_date')
            ax.plot(points.realisation_date, points.ic, lw=.95, color=COLORS[name], label=target)
        ax.set(title=f'{label}  |  1-month realised rank IC', ylabel='Spearman IC',
               ylim=(-1.05, 1.05))
        ax.axhline(0, color='.55', lw=.8)
        ax.grid(axis='y', alpha=.2)
        ax.legend(ncol=2, loc='lower right', framealpha=.95)
    ax = fig.add_subplot(grid[:, 1])
    ax.axis('off')
    ax.text(.5, .82, 'Matched windows per ETF', ha='center', weight='bold',
            transform=ax.transAxes, fontsize=14)
    ax.table(cellText=members.reset_index().to_numpy(), colLabels=['ETF', '1M', '3M', '6M', '12M'],
             cellLoc='center', bbox=[.03, .24, .94, .52], edges='horizontal').set_fontsize(12)
    fig.suptitle('Realised rank IC and ETF coverage', fontsize=20)
    return fig


def rank_attribution_tables(panel, report):
    """Use the common-sample weights and matched assets underlying the headline table."""
    panels = panel.loc[panel.eligible].copy()
    fields = [*SIGNALS, *TARGETS]
    panels[fields] = panels[fields].where(np.isfinite(panels[fields]).all(axis=1), axis=0)
    components, contributions = [], []
    for (h, signal, target), rows in report.dated.query("sample == 'common'").groupby(
            ['horizon', 'signal', 'target']):
        window = panels.loc[panels.horizon.eq(h)]
        wide = window.pivot(index='realisation_date', columns='asset', values=[signal, target])
        weights = rows.set_index('realisation_date').weight
        assets = decompose_rank_ic(wide[signal], wide[target], weights)
        sums = assets[['persistent', 'within']].sum(min_count=1)
        components.append(dict(horizon=h, signal=signal, target=target,
                               n_valid=int(weights.gt(0).sum()), total=sums.sum(min_count=1),
                               persistent=sums.persistent, within=sums.within))
        contributions.append(assets.reset_index().assign(horizon=h, signal=signal, target=target))
    return pd.DataFrame(components), pd.concat(contributions, ignore_index=True)


def diagnostic_table(ax, frame, bbox):
    """Draw a numeric factsheet table with a distinct header."""
    table = ax.table(cellText=frame.values, colLabels=frame.columns, cellLoc='center', bbox=bbox)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    for (r, _), cell in table.get_celld().items():
        cell.set_edgecolor('white')
        cell.set_facecolor('#254b66' if r == 0 else ('#eaf1f4' if r % 2 else '#f6f8f9'))
        if r == 0:
            cell.set_text_props(color='white', weight='bold')


def reuse_page(report, ewma_span):
    """Show the paired effect of sharing the fitted beta with the forward hedge."""
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.axis('off')
    fig.subplots_adjust(left=.065, right=.935, top=.94, bottom=.06)
    ax.text(0, .96, 'Does reusing the score beta inflate low-beta IC?',
            fontsize=23, weight='bold', color='#183952', transform=ax.transAxes)
    ax.text(0, .89, f'Joint SPY/TLT fits on alternate historical months  |  '
            f'EWMA span {ewma_span:g} months for betas and {ewma_span:g} valid dates for IC',
            fontsize=13, transform=ax.transAxes)
    for leg, label, top in [('score_factor', 'Switch only the SPY hedge beta', .78),
                            ('joint', 'Switch both SPY and TLT hedge betas', .41)]:
        data = report.summary.query('hedge == @leg').set_index(['horizon', 'target'])
        values = []
        for h in HORIZONS:
            same, disjoint = data.loc[(h, 'same')], data.loc[(h, 'disjoint')]
            bounds = (f'[{same.difference_lower:+.3f}, {same.difference_upper:+.3f}]'
                      if pd.notna(same.difference_lower) else 'Unavailable')
            values.append([h, same.n_paired_valid, f'{same.mean_ic:+.3f}',
                           f'{disjoint.mean_ic:+.3f}', f'{same.mean_ic_minus_reference:+.3f}',
                           bounds])
        ax.text(0, top, label, transform=ax.transAxes, fontsize=15, weight='bold')
        headings = ['Months', 'Paired dates', 'Same-fit IC', 'Disjoint-fit IC',
                    'Same − disjoint', '95% HAC interval']
        diagnostic_table(ax, pd.DataFrame(values, columns=headings), [0, top-.29, 1, .25])
    ax.text(0, .018, 'Hold each fold’s score fixed, switch the hedge fit, then reverse the '
            'folds. Each holding window contributes one averaged pair.', transform=ax.transAxes,
            fontsize=12)
    return fig


def attribution_page(summary, ewma_span):
    """Show persistent and within-ETF components of both raw and residual mean rank IC."""
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.axis('off')
    fig.subplots_adjust(left=.065, right=.935, top=.94, bottom=.06)
    ax.text(0, .96, 'Persistent ETF ranks and within-ETF rank movements',
            fontsize=23, weight='bold', color='#183952', transform=ax.transAxes)
    ax.text(0, .88, f'Mean Spearman IC  |  EWMA span {ewma_span:g} valid dates at each horizon',
            fontsize=14, transform=ax.transAxes)
    headings = ['Months', 'Raw IC', 'Persistent', 'Within ETF', 'Residual IC', 'Persistent',
                'Within ETF']
    for (signal, label), top in zip(SIGNALS.items(), [.77, .41]):
        ax.text(0, top, label, fontsize=15, weight='bold', transform=ax.transAxes)
        data = summary.query('signal == @signal').set_index(['horizon', 'target'])
        values = [[h, *[f'{data.loc[(h, t), c]:+.4f}' for t in TARGETS
                        for c in ('total', 'persistent', 'within')]] for h in HORIZONS]
        diagnostic_table(ax, pd.DataFrame(values, columns=headings), [0, top-.29, 1, .25])
    ax.text(0, .018, 'Persistent measures alignment of each ETF’s average score and return '
            'ranks. Within ETF measures their co-movement around those averages.',
            fontsize=12, transform=ax.transAxes)
    return fig


def export_factsheet(daily, output_dir, *, ewma_span=36, rolling_span=36, warmup_periods=12):
    """Export the matched IC analyses, native raw-return diagnostics and underlying tables."""
    panel, log_prices = formation_panel(daily, ewma_span=ewma_span)
    members = matched_membership(panel)
    dated = dated_ics(panel)
    settings = dict(as_of=pd.Timestamp(LAST_MONTH), warmup_periods=warmup_periods)
    ewma = compute_signal_ic_summary(dated, ewma_span=ewma_span, **settings)
    full = compute_signal_ic_summary(dated, **settings)
    rolling = compute_signal_ic_summary(dated, rolling_span=rolling_span, **settings)
    summary = pd.concat([r.summary for r in (full, rolling, ewma)], ignore_index=True)
    common = ewma.summary.query("sample == 'common'")
    sweep = [ewma.summary.assign(lookback=12)]
    for lag in (6, 18):
        score = (log_prices[ASSETS].shift(1) - log_prices[ASSETS].shift(lag)).stack()
        score.index.names = ['formation_date', 'asset']
        variant = panel.drop(columns='momentum').join(score.rename('momentum'),
                                                      on=['formation_date', 'asset'])
        report = compute_signal_ic_summary(dated_ics(variant), ewma_span=ewma_span, **settings)
        sweep.append(report.summary.assign(lookback=lag))
    lookbacks = pd.concat(sweep, ignore_index=True)
    monthly = log_prices.loc[FIRST_DECISION:, ASSETS].diff().iloc[1:]
    formations = panel.loc[panel.horizon.eq(1)].set_index(['formation_date', 'asset'])
    decisions = log_prices.loc[FIRST_DECISION:].index
    scores = {s: formations[s].unstack().reindex(index=decisions, columns=ASSETS) for s in SIGNALS}
    native, tables = native_score_reports({'ME': monthly}, scores, GROUPS, HORIZONS)
    increments = log_prices.diff().iloc[1:]
    reuse, folds = compute_beta_reuse_ic(
        increments[ASSETS], increments[BENCHMARKS], decisions, score_factor='SPY',
        horizons=HORIZONS, beta_ewma_span=ewma_span, ewma_span=ewma_span,
        warmup_periods=warmup_periods)
    attribution, per_etf = rank_attribution_tables(panel, ewma)
    pages = [
        (overview(common, ewma_span), 'Yahoo Finance adjusted closes, December 2007 to December '
         '2025. First formation 31 December 2010. SPY and TLT supply the two benchmark returns.'),
        (horizon_report(common, ewma_span, warmup_periods, horizons=HORIZONS, targets=TARGETS,
                        unit='months', universe='Eight ETFs'),
         'Paired change is residual rank IC minus raw rank IC on the same ETFs at each date. '
         'Intervals use Bartlett HAC with up to three lags.'),
        (reuse_page(reuse, ewma_span),
         'The SPY-only switch holds TLT at its full-history formation estimate. Pointwise 95% '
         'intervals use paired differences, Bartlett HAC (up to three lags) and the '
         'report warmup.'),
        (attribution_page(attribution, ewma_span),
         'Total IC = persistent + within ETF before rounding. Each asset’s average rank is '
         'estimated over the reported sample. These components are descriptive.'),
        (history_and_coverage(ewma.dated, members),
         'Each plotted point is one realised cross-sectional correlation. Counts require both '
         'scores and both forward targets to be finite in the same window.'),
        (sensitivity_report(summary, lookbacks, ewma_span, rolling_span, horizons=HORIZONS,
                            unit='months', target='residual',
                            lookback_labels={k: f'{k}–1 months' for k in (6, 12, 18)}),
         f'Rolling {rolling_span} requires {rolling_span} valid dates. Lookback variants skip '
         'the latest month and use the same reporting dates and frozen benchmark betas.'),
        (stability_report(tables, horizon_label='Months', grid='monthly'),
         'Equal-weight raw ICs. IC-IR = mean / sample standard deviation. Annualised IR '
         'multiplies by sqrt(12 / horizon). Hit rate is the fraction of positive dated ICs.'),
    ]
    for i, fig in enumerate(native):
        label = list(SIGNALS.values())[i // 2]
        fig.set_size_inches(15, 9)
        if i % 2 == 0:
            fig.suptitle(f'{label} | Raw-return regressions by score bucket and group', fontsize=19)
            fig.subplots_adjust(left=.065, right=.985, top=.83, bottom=.12, wspace=.41, hspace=.68)
            for ax in fig.axes[:len(HORIZONS)]:
                title = ax.get_title().replace(' horizon', 'M').replace('   IC =', '\nIC =')
                ax.set_title(title, color='#183952', fontsize=11)
            for ax in fig.axes[len(HORIZONS):]:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%+.2f'))
                ax.set_title(ax.get_title().replace(' horizon', 'M'), color='#183952', fontsize=11)
        else:
            ax = fig.axes[0]
            ax.set_title(f'{label} | Distribution of per-ETF raw-return slopes',
                         fontsize=19, color='#183952', pad=14)
            ax.set_xlabel('Forward horizon (months)')
        note = ('Through-origin slope and pooled Pearson IC use standardised scores and raw '
                'returns. Group stars mark |t| of 1.65 (*), 1.96 (**) and 2.58 (***).'
                if i % 2 == 0 else
                'Each ETF supplies one through-origin slope on standardised scores and raw '
                'returns per horizon. QIS requires at least 12 paired observations per ETF.')
        pages.append((fig, note))
    tables.update(forward_targets_and_scores=panel, matched_windows=members.reset_index(),
                  matched_summary=summary, dated_full=full.dated, dated_rolling=rolling.dated,
                  dated_ewma=ewma.dated, lookback_summary=lookbacks,
                  beta_reuse_summary=reuse.summary, beta_reuse_dated=reuse.dated,
                  beta_reuse_folds=folds, rank_attribution=attribution,
                  rank_attribution_by_etf=per_etf,
                  etf_groups=GROUPS.rename_axis('asset').rename('group').reset_index())
    out = Path(get_output_path(output_dir))
    out.mkdir(parents=True, exist_ok=True)
    pdf_path = out / 'etf_signal_validation.pdf'
    archive_path = out / 'etf_signal_validation_tables.zip'
    try:
        with PdfPages(pdf_path, metadata={'Title': 'ETF signal validation | SPY and TLT'}) as pdf:
            for number, (fig, note) in enumerate(pages, start=1):
                fig.set_size_inches(15, 9)
                layout = fig.get_layout_engine()
                if isinstance(layout, ConstrainedLayoutEngine):
                    layout.set(rect=(.035, .02, .93, .95))
                fig.text(.04, -.043, note, fontsize=9, color='#334854')
                fig.text(.04, -.071, 'QIS signal diagnostics', fontsize=9,
                         color='#305375', url=(DOCS + '#beta-reuse-and-rank-attribution'
                                              if number in (3, 4) else DOCS))
                fig.text(.96, -.071, f'{number} / {len(pages)}', ha='right', fontsize=9)
                pdf.savefig(fig, bbox_inches=Bbox.from_extents(0, -.78, 15, 9))
        with ZipFile(archive_path, 'w', ZIP_DEFLATED) as archive:
            for name, frame in tables.items():
                archive.writestr(f'{name}.csv', frame.to_csv(index=False))
    finally:
        for fig, _ in pages:
            plt.close(fig)
    return pdf_path, archive_path


def run_example(output_dir: str | Path | None = None, *,
                prices_csv: str | Path | None = None) -> None:
    """Download Yahoo adjusted closes or replay a local snapshot, then write the factsheet."""
    for path in export_factsheet(load_prices(prices_csv), output_dir):
        print(path)


if __name__ == '__main__':
    run_example()

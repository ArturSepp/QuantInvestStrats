"""Preserve reviewed unhedged evidence and explicitly recheck authorised private inputs.

Offline generation validates public aggregate records and frozen PNGs only. The
private command rechecks two FX inputs on the recent derived panel, not vendor
acquisition or undistributed full histories. Index-implied FX is a diagnostic.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools.docs_analytics.hedged_index_case_study import anchored_nav

ROOT = Path(__file__).resolve().parents[2]
RECENT = '2021-Sep2026'


def produce(spec):
    """Validate public aggregates and return reviewed previews without vendor access."""
    parameters = spec['parameters']
    statistics = pd.DataFrame(parameters['statistics'])
    pairs = pd.DataFrame(parameters['pairs'])
    currency = pd.DataFrame(parameters['currency_diagnostics'])
    tickers = parameters['observed_tickers']
    expected = {(ticker, window) for ticker in tickers for window in parameters['windows']}
    observed = list(zip(statistics['Observed_ticker'], statistics['Window']))
    if len(tickers) != 22 or len(set(tickers)) != 22:
        raise ValueError('Expected 22 unique observed tickers')
    if set(observed) != expected or len(observed) != len(expected):
        raise ValueError('Incomplete or duplicate ticker/window statistics')
    if len(pairs) != 22 or set(pairs['Observed_ticker']) != set(tickers):
        raise ValueError('Incomplete or duplicate matched index pairs')
    registered = set(zip(pairs['Base_ticker'], pairs['Observed_ticker']))
    families = dict(zip(pairs['Base_ticker'], pairs['Family']))
    for row in pairs.to_dict('records'):
        anchor = (row['Index_FX_anchor_base'], row['Index_FX_anchor_target'])
        if (anchor not in registered or row['Base_ticker'] == anchor[0]
                or row['Observed_ticker'] == anchor[1]
                or families[anchor[0]] == row['Family']):
            raise ValueError('Circular or unregistered held-out FX anchor')
    for row in statistics.to_dict('records'):
        identity = pairs.loc[pairs['Observed_ticker'].eq(row['Observed_ticker'])].iloc[0]
        if any(row[key] != identity[key] for key in pairs.columns):
            raise ValueError('Statistic identity differs from its matched pair')
        if len(pd.date_range(row['Start'], row['End'], freq='ME')) != row['Months']:
            raise ValueError('Noncontiguous full-history sample')
    numeric = statistics.select_dtypes(include='number')
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError('Non-finite empirical summary')
    if not (statistics['R_squared'].between(0.0, 1.0)
            & statistics['Index_FX_holdout_R_squared'].between(0.0, 1.0)).all():
        raise ValueError('Invalid empirical R-squared')
    if not ((statistics['Months'] >= 12)
            & (statistics['RMSE_monthly_bp'] >= statistics['MAE_monthly_bp'])
            & (statistics['MAE_monthly_bp'] >= 0.0)
            & (statistics['Tracking_error_pa_bp'] >= 0.0)
            & (statistics['Index_FX_holdout_TE_pa_bp'] >= 0.0)
            & (statistics['Index_FX_holdout_MAE_monthly_bp'] >= 0.0)
            & (statistics['Missing_internal_months'] == 0)).all():
        raise ValueError('Invalid sample size or empirical error bounds')
    recent = statistics.loc[statistics['Window'].eq(RECENT)]
    if not ((recent['Months'] == 69) & recent['Start'].eq('2021-01-31')
            & recent['End'].eq('2026-09-30')).all():
        raise ValueError('Recent comparison must use 69 complete monthly observations')
    # Aggregate integrity identity, not a replacement tracking-error implementation.
    n = statistics['Months']
    np.testing.assert_allclose(
        statistics['RMSE_monthly_bp'] ** 2,
        statistics['Tracking_error_pa_bp'] ** 2 / 12.0 * (n - 1.0) / n
        + (statistics['Mean_error_pa_bp'] / 12.0) ** 2, atol=1e-9, rtol=1e-10,
    )
    np.testing.assert_allclose(
        statistics['CAGR_difference_bp'],
        10000.0 * (statistics['QIS_return_pa'] - statistics['Observed_return_pa']),
        atol=1e-9, rtol=1e-10,
    )
    if (len(currency) != 3 or set(currency['Currency']) != {'CHF', 'EUR', 'GBP'}
            or not (currency['Months'] == 69).all()
            or not (currency['Bond_families'] == 6).all()
            or not np.isfinite(currency.select_dtypes(include='number')).all().all()
            or not (currency['Implied_FX_cross_family_std_monthly_bp'] >= 0.0).all()):
        raise ValueError('Invalid currency diagnostic aggregates')
    figures = {}
    for item in parameters['frozen_images']:
        path = ROOT / 'docs/images' / item['filename']
        if path.parent != ROOT / 'docs/images' or path.suffix != '.png':
            raise ValueError('Unsafe frozen preview path')
        if path.name in figures:
            raise ValueError('Duplicate frozen preview filename')
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'Frozen preview hash mismatch: {path.name}')
        figures[path.name] = path
    expected_images = {
        f'unhedged_index_case_study_{ticker.split()[0].lower()}.png' for ticker in tickers}
    if set(figures) != expected_images:
        raise ValueError('Incomplete frozen preview set')
    return {
        'figures': figures,
        'tables': {'statistics': statistics, 'pairs': pairs, 'currency_diagnostics': currency},
        'checks': dict.fromkeys(spec['checks'], True),
        'parameters': parameters,
        'conventions': spec['conventions'],
        'summary': {
            'evidence': 'Frozen observed-data aggregates; integrity, not an offline refit',
            'pairs': 22, 'windows': parameters['windows'],
            'FX_inputs': ['Existing generic spots', 'Different-family index-implied FX'],
            'raw_vendor_data_distributed': False, 'independent_WMR_replication': False,
            'numerical_review': parameters['numerical_review'],
        },
    }


def plot_comparison(frame, row, output):
    """Plot both FX reconstructions against observed returns using QIS scatterplots."""
    import matplotlib.pyplot as plt
    import qis

    figure, axes = plt.subplots(1, 3, figsize=(16.0, 5.2))
    figure.subplots_adjust(left=0.055, right=0.98, bottom=0.22, top=0.80, wspace=0.35)
    for position, label, description in (
            (0, 'QIS', 'QIS with existing FX inputs'),
            (1, 'Index FX holdout', 'QIS with held-out index-implied FX')):
        qis.plot_scatter(
            frame[['Observed', label]], x='Observed', y=label, order=1, full_sample_order=1,
            fit_intercept=True, add_45line=True, align_axis=True,
            add_universe_model_label=True, markersize=25,
            xlabel='Observed unhedged monthly return', ylabel=description,
            xvar_format='{:.1%}', yvar_format='{:.1%}', fontsize=10, ax=axes[position],
        )
    axes[0].get_legend().get_texts()[0].set_text(
        f"OLS: beta={row['Beta']:.6f}, intercept={row['Intercept_monthly_bp']:.3f} bp\n"
        f"R-squared={row['R_squared']:.8f}")
    axes[1].get_legend().get_texts()[0].set_text(
        f"R-squared={row['Index_FX_holdout_R_squared']:.8f}\n"
        f"TE={row['Index_FX_holdout_TE_pa_bp']:.3f} bp p.a.")
    nav = anchored_nav(frame[['Observed', 'QIS', 'Index FX holdout']])
    relative = nav[['QIS', 'Index FX holdout']].div(nav['Observed'], axis=0).sub(1)
    relative.columns = ['Existing FX', 'Held-out index FX']
    qis.plot_time_series(
        relative, legend_stats=qis.LegendStats.NONE, var_format='{:.3%}',
        ylabel='Compounded relative return', x_date_freq='YE', date_format='%Y',
        linewidth=1.7, title='Cumulative replication difference', fontsize=10, ax=axes[2],
    )
    figure.suptitle(f"{row['Family']} unhedged in {row['Reference_ccy']}",
                   fontsize=15, y=0.98)
    figure.text(0.5, 0.87, f"{row['Base_ticker']} to {row['Observed_ticker']} | "
                f"n={row['Months']} months", ha='center', fontsize=10)
    figure.text(
        0.5, 0.08, f"R-squared {row['R_squared']:.8f} | "
        f"TE {row['Tracking_error_pa_bp']:.3f} bp p.a. | "
        f"MAE {row['MAE_monthly_bp']:.3f} bp/month | "
        f"CAGR difference {row['CAGR_difference_bp']:+.3f} bp p.a.\n"
        "January 2021-September 2026; simple total returns; current-period realised spot; "
        "hedge ratio 0; no cash deduction.", ha='center', fontsize=10,
    )
    filename = f"unhedged_index_case_study_{row['Observed_ticker'].split()[0].lower()}.png"
    figure.savefig(output / filename, dpi=150, facecolor='white')
    plt.close(figure)


def refit(source_csv, spec, output_dir):
    """Recheck both FX methods on a hash-matched private 69-month derived panel."""
    from scipy.stats import linregress
    import qis
    from tools.docs_analytics.run import output_boundary

    if hashlib.sha256(Path(source_csv).read_bytes()).hexdigest() != (
            spec['parameters']['input_artifacts']['monthly_comparison.csv']['sha256']):
        raise ValueError('Private input differs from the frozen study snapshot')
    output = output_boundary(Path(output_dir))
    panel = pd.read_csv(source_csv, parse_dates=['Date'])
    tickers = spec['parameters']['observed_tickers']
    if set(panel['Observed_ticker']) != set(tickers):
        raise ValueError('Unexpected private ticker coverage')
    output.mkdir(parents=True)
    recorded = pd.DataFrame(spec['parameters']['statistics'])
    recorded = recorded.loc[recorded['Window'].eq(RECENT)].set_index('Observed_ticker')
    frames = {ticker: panel.loc[panel['Observed_ticker'].eq(ticker)].set_index('Date')
              .sort_index() for ticker in tickers}
    grid = pd.date_range('2021-01-31', '2026-09-30', freq='ME')
    required = ['Observed', 'QIS', 'Base', 'FX', 'Independent', 'Level translation',
                'Zero rates', 'Cash lag 0', 'Log to simple', 'Index FX holdout']
    for ticker, frame in frames.items():
        if not frame.index.equals(grid) or not np.isfinite(frame[required]).all().all():
            raise ValueError(f'Incomplete or duplicate private observations: {ticker}')
    rows = []
    for ticker, frame in frames.items():
        reference = recorded.loc[ticker]
        anchor = frames[reference['Index_FX_anchor_target']]
        if (ticker == reference['Index_FX_anchor_target']
                or reference['Base_ticker'] == reference['Index_FX_anchor_base']):
            raise ValueError('Circular private FX anchor')
        anchor_fx_return = (1 + anchor['Observed']) / (1 + anchor['Base']) - 1
        np.testing.assert_allclose(frame['QIS'],
                                   (1 + frame['Base']) * (1 + frame['FX']) - 1, atol=1e-12)
        np.testing.assert_allclose(frame['Index FX holdout'],
                                   (1 + frame['Base']) * (1 + anchor_fx_return) - 1,
                                   atol=1e-12)
        for label in ('Independent', 'Level translation', 'Zero rates',
                      'Cash lag 0', 'Log to simple'):
            np.testing.assert_allclose(frame['QIS'], frame[label], atol=1e-12)
        # Reconstruct both FX paths via FxRatesData from the derived monthly panel.
        base_nav = anchored_nav(frame['Base'].to_frame())['Base']
        for label, fx_return in (('QIS', frame['FX']), ('Index FX holdout', anchor_fx_return)):
            cross_nav = anchored_nav(fx_return.to_frame('cross'))['cross']
            spots = pd.DataFrame({'USD': 1.0, reference['Reference_ccy']: 1 / cross_nav})
            fx = qis.FxRatesData(fx_spots=spots, domestic_rates=spots * 0.0)
            _, reproduced = fx.compute_performance_of_local_ccy_asset_in_reference_ccy(
                asset_price_local_ccy=base_nav, hedge_ratio=0.0, local_ccy='USD',
                reference_ccy=reference['Reference_ccy'], freq='ME',
                is_excess_returns=False, is_log_returns=False, cash_rate_lag=1,
            )
            np.testing.assert_allclose(frame[label], reproduced.loc[grid], atol=1e-12)
        nav = anchored_nav(frame[['Observed', 'QIS', 'Index FX holdout']])
        pa = dict(zip(nav.columns, np.atleast_1d(qis.compute_pa_return(nav))))
        values = {'Months': len(frame), 'Observed_return_pa': pa['Observed'],
                  'QIS_return_pa': pa['QIS'],
                  'CAGR_difference_bp': (pa['QIS'] - pa['Observed']) * 10000,
                  'Index_FX_holdout_CAGR_difference_bp': (
                      pa['Index FX holdout'] - pa['Observed']) * 10000}
        for label, prefix in (('QIS', ''), ('Index FX holdout', 'Index_FX_holdout_')):
            fit = linregress(frame['Observed'], frame[label])
            error = frame[label] - frame['Observed']
            te, _ = qis.compute_te_ir_errors(error.to_frame('error'))
            np.testing.assert_allclose(te.iloc[0], error.std(ddof=1) * np.sqrt(12), atol=1e-12)
            values[prefix + 'R_squared'] = fit.rvalue ** 2
            values[prefix + ('Tracking_error_pa_bp' if not prefix else 'TE_pa_bp')] = (
                te.iloc[0] * 10000)
            values[prefix + 'MAE_monthly_bp'] = error.abs().mean() * 10000
            if not prefix:
                values.update(Beta=fit.slope, Intercept_monthly_bp=fit.intercept * 10000,
                              RMSE_monthly_bp=np.sqrt(np.mean(error ** 2)) * 10000,
                              Mean_error_pa_bp=error.mean() * 120000)
        for label, value in values.items():
            np.testing.assert_allclose(value, reference[label], atol=1e-9, rtol=1e-10)
        row = dict(reference, **values, Observed_ticker=ticker)
        rows.append(row)
        plot_comparison(frame, row, output)
    result = pd.DataFrame(rows)
    result.to_csv(output / 'statistics.csv', index=False, float_format='%.17g')
    return result


def main():
    """Run an explicit private recheck without changing frozen public previews."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-csv', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'tools/docs_analytics/manifest.json').read_text('utf-8'))
    result = refit(args.source_csv, manifest['producers']['unhedged_index_case_study'],
                   args.output_dir)
    print(f'Verified {len(result)} recent-period pairs for both FX inputs')


if __name__ == '__main__':
    main()

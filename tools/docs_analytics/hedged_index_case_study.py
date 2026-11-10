"""Preserve reviewed hedged-index evidence and recheck a supplied private monthly panel.

The unattended producer uses public aggregates and allowlisted PNGs only. The explicit
private command rechecks the recent comparison panel, not undistributed full histories
or Bloomberg acquisition. Original FX-payoff verification is recorded in the manifest.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RECENT = '2021–Sep2026'


def produce(spec):
    """Validate frozen public aggregates and return reviewed preview paths.

    Args:
        spec: Registered empirical producer configuration.

    Returns:
        Figures, aggregate tables, integrity checks and dated study provenance.
    """
    parameters = spec['parameters']
    statistics = pd.DataFrame(parameters['statistics'])
    tickers = parameters['observed_tickers']
    expected = {(ticker, window) for ticker in tickers for window in parameters['windows']}
    observed = list(zip(statistics['Observed_ticker'], statistics['Window']))
    if set(observed) != expected or len(observed) != len(expected):
        raise ValueError('Incomplete or duplicate ticker/window statistics')
    numeric = statistics.select_dtypes(include='number')
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError('Non-finite empirical summary')
    if not statistics['R_squared'].between(0.0, 1.0).all():
        raise ValueError('Invalid empirical R-squared')
    if not ((statistics['Months'] >= 12)
            & (statistics['RMSE_monthly_bp'] >= statistics['MAE_monthly_bp'])
            & (statistics['MAE_monthly_bp'] >= 0.0)
            & (statistics['Tracking_error_pa_bp'] >= 0.0)).all():
        raise ValueError('Invalid sample size or empirical error bounds')
    recent = statistics.loc[statistics['Window'].eq(RECENT)]
    if not ((recent['Months'] == 69) & (recent['Missing_internal_months'] == 0)
            & recent['Start'].eq('2021-01-31') & recent['End'].eq('2026-09-30')).all():
        raise ValueError('Recent comparison must use 69 complete monthly observations')
    # Integrity identity for af=12 and sample TE with ddof=1, not another TE implementation.
    n = statistics['Months']
    np.testing.assert_allclose(
        statistics['RMSE_monthly_bp'] ** 2,
        statistics['Tracking_error_pa_bp'] ** 2 / 12.0 * (n - 1.0) / n
        + (statistics['Mean_error_pa_bp'] / 12.0) ** 2,
        atol=1e-9, rtol=1e-10,
    )
    np.testing.assert_allclose(
        statistics['CAGR_difference_bp'],
        10000.0 * (statistics['QIS_return_pa'] - statistics['Observed_return_pa']),
        atol=1e-9, rtol=1e-10,
    )
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
        f'hedged_index_case_study_{ticker.split()[0].lower()}.png' for ticker in tickers}
    if set(figures) != expected_images:
        raise ValueError('Incomplete frozen preview set')
    return {
        'figures': figures,
        'tables': {'statistics': statistics},
        'checks': dict.fromkeys(spec['checks'], True),
        'parameters': parameters,
        'conventions': spec['conventions'],
        'summary': {
            'evidence': 'Frozen observed-data aggregates; integrity, not an offline refit',
            'pairs': len(tickers), 'windows': parameters['windows'],
            'raw_vendor_data_distributed': False,
            'numerical_review': parameters['numerical_review'],
        },
    }


def anchored_nav(returns):
    """Keep the first earned return by prepending the preceding month-end NAV anchor."""
    import qis

    anchor = pd.DataFrame(0.0, columns=returns.columns,
                          index=[returns.index.min() - pd.offsets.MonthEnd(1)])
    return qis.returns_to_nav(pd.concat([anchor, returns]), init_period=None,
                              is_log_returns=False, ffill_between_nans=False)


def plot_comparison(frame, row, output):
    """Show monthly fit and accumulated drift through canonical QIS plotting functions."""
    import matplotlib.pyplot as plt
    import qis

    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    figure.subplots_adjust(left=0.08, right=0.97, bottom=0.22, top=0.80, wspace=0.28)
    qis.plot_scatter(
        frame[['Observed', 'QIS']], x='Observed', y='QIS', order=1, full_sample_order=1,
        fit_intercept=True, add_45line=True, align_axis=True, add_universe_model_label=True,
        markersize=25, xlabel='Observed hedged monthly return',
        ylabel='QIS reconstructed monthly return', xvar_format='{:.1%}',
        yvar_format='{:.1%}', fontsize=10, ax=axes[0],
    )
    axes[0].get_legend().get_texts()[0].set_text(
        f"OLS: β={row['Beta']:.4f}, intercept={row['Intercept_monthly_bp']:.2f} bp\n"
        f"R²={row['R_squared']:.5f}")
    nav = anchored_nav(frame[['Observed', 'QIS']])
    relative = (nav['QIS'] / nav['Observed'] - 1.0).to_frame('QIS / observed − 1')
    qis.plot_time_series(
        relative, legend_stats=qis.LegendStats.NONE, var_format='{:.2%}',
        ylabel='Compounded relative return', x_date_freq='YE', date_format='%Y',
        linewidth=1.7, title='Cumulative replication difference', fontsize=10, ax=axes[1],
    )
    figure.suptitle(f"{row['Family']} hedged to {row['Reference_ccy']}", fontsize=15, y=0.98)
    figure.text(0.5, 0.87,
                f"{row['Base_ticker']} → {row['Observed_ticker']} | n={row['Months']} months",
                ha='center', fontsize=10)
    figure.text(
        0.5, 0.08,
        f"R² {row['R_squared']:.5f} | TE {row['Tracking_error_pa_bp']:.1f} bp p.a. | "
        f"MAE {row['MAE_monthly_bp']:.1f} bp/month | "
        f"CAGR difference {row['CAGR_difference_bp']:+.1f} bp p.a.\n"
        '2021–September 2026; simple total returns; monthly opening-principal hedge; '
        'rate-implied carry; no excess-cash subtraction.', ha='center', fontsize=10,
    )
    filename = f"hedged_index_case_study_{row['Observed_ticker'].split()[0].lower()}.png"
    figure.savefig(output / filename, dpi=150, facecolor='white')
    plt.close(figure)


def refit(source_csv, spec, output_dir):
    """Recheck recent recorded fits and plots using an authorised private derived panel.

    Args:
        source_csv: Hash-matched monthly comparison snapshot; no acquisition occurs.
        spec: Frozen empirical-study registry entry.
        output_dir: Fresh C-local directory for checks and optional replacement previews.

    Returns:
        Independently checked recent-period statistics; full histories remain frozen.
    """
    from scipy.stats import linregress
    import qis
    from tools.docs_analytics.run import output_boundary

    if hashlib.sha256(Path(source_csv).read_bytes()).hexdigest() != (
            spec['parameters']['input_artifacts']['monthly_comparison.csv']['sha256']):
        raise ValueError('Private input differs from the frozen study snapshot')
    output = output_boundary(Path(output_dir))
    output.mkdir(parents=True)
    panel = pd.read_csv(source_csv, parse_dates=['Date'])
    recorded = pd.DataFrame(spec['parameters']['statistics'])
    recorded = recorded.loc[recorded['Window'].eq(RECENT)].set_index('Observed_ticker')
    rows = []
    for ticker in spec['parameters']['observed_tickers']:
        frame = panel.loc[panel['Observed_ticker'].eq(ticker)].set_index('Date').sort_index()
        if frame.index.has_duplicates or len(frame) != 69:
            raise ValueError(f'Duplicate dates or incomplete recent panel: {ticker}')
        grid = pd.date_range('2021-01-31', '2026-09-30', freq='ME')
        if not frame.index.equals(grid):
            raise ValueError(f'Unexpected observation dates: {ticker}')
        required = ['Observed', 'QIS', 'Base index', 'Spot interaction', 'QIS carry']
        if not np.isfinite(frame[required].to_numpy()).all():
            raise ValueError(f'Incomplete private observations: {ticker}')
        np.testing.assert_allclose(
            frame['QIS'], frame['Base index'] + frame['Spot interaction'] + frame['QIS carry'],
            atol=1e-12, rtol=1e-11,
        )
        fit = linregress(frame['Observed'], frame['QIS'])
        error = frame['QIS'] - frame['Observed']
        te, _ = qis.compute_te_ir_errors(error.to_frame('error'))
        np.testing.assert_allclose(te.iloc[0], error.std(ddof=1) * np.sqrt(12.0), atol=1e-12)
        nav = anchored_nav(frame[['Observed', 'QIS']])
        pa = dict(zip(nav.columns, np.atleast_1d(qis.compute_pa_return(nav))))
        values = {
            'Months': len(frame), 'R_squared': fit.rvalue ** 2, 'Beta': fit.slope,
            'Intercept_monthly_bp': fit.intercept * 10000.0,
            'MAE_monthly_bp': error.abs().mean() * 10000.0,
            'RMSE_monthly_bp': np.sqrt(np.mean(error ** 2)) * 10000.0,
            'Tracking_error_pa_bp': te.iloc[0] * 10000.0,
            'Mean_error_pa_bp': error.mean() * 120000.0,
            'Observed_return_pa': pa['Observed'], 'QIS_return_pa': pa['QIS'],
            'CAGR_difference_bp': (pa['QIS'] - pa['Observed']) * 10000.0,
        }
        reference = recorded.loc[ticker]
        for label, value in values.items():
            np.testing.assert_allclose(value, reference[label], atol=1e-9, rtol=1e-10)
        row = dict(reference, **values, Observed_ticker=ticker)
        rows.append(row)
        plot_comparison(frame, row, output)
    result = pd.DataFrame(rows)
    result.to_csv(output / 'statistics.csv', index=False, float_format='%.17g')
    return result



def recompute(returns_csv, spots_csv, rates_csv, pairs_csv, output_dir,
              cutoff="2026-09-30"):
    """Validate current ACT/365F payoffs against supplied, unfilled provider returns.

    Inputs are private CSV panels: monthly SIMPLE total returns by ticker, USD-per-local
    spots, annual-decimal rates, and a pair catalogue with Base_ticker, Observed_ticker,
    Local_ccy, Reference_ccy, Family and Match_type. No vendor acquisition occurs.
    The current QIS API constructs the payoff; independently priced forward delivery
    checks it. Both recent and full available windows retain the observed-data mask.
    """
    import qis
    from scipy.stats import linregress
    from tools.docs_analytics.run import output_boundary

    output = output_boundary(Path(output_dir))
    output.mkdir(parents=True)
    def read_panel(path):
        """Keep raw observations, rejecting unordered or duplicate timestamps."""
        panel = pd.read_csv(path, index_col=0, parse_dates=True)
        if not panel.index.is_unique or not panel.index.is_monotonic_increasing:
            raise ValueError(f"Unordered or duplicate dates: {path}")
        return panel.loc[:cutoff]
    raw, spots, rates = map(read_panel, (returns_csv, spots_csv, rates_csv))
    pairs = pd.read_csv(pairs_csv)
    fx = qis.FxRatesData(spots, rates)
    rows, monthly = [], []
    max_payoff_error = 0.
    for identity in pairs.to_dict('records'):
        base, target = identity['Base_ticker'], identity['Observed_ticker']
        local, reference = identity['Local_ccy'], identity['Reference_ccy']
        native = raw[base].dropna()
        grid = pd.date_range(native.index.min(), native.index.max(), freq='ME')
        if not native.index.equals(grid):
            raise ValueError(f"Base return history must have a complete monthly grid: {base}")
        prices = anchored_nav(native.to_frame())
        actual = qis.compute_fx_returns(fx, qis.FxAssetReturnsRequest(
            prices=prices, local_ccys=local, reference_ccy=reference,
            return_type=qis.FxReturnType.REFERENCE_TOTAL,
            coordinate=qis.FxReturnCoordinate.SIMPLE, hedge_ratios=1.0,
            frequency='ME', zero_return_to_nan=False))['ME'].values.iloc[:, 0]
        dates = prices.index
        marks = spots.reindex(dates, method='ffill')
        quotes = rates.reindex(dates, method='ffill')
        tau = dates.to_series().diff().dt.total_seconds() / (365 * 86400.)
        cross = marks[local] / marks[reference]
        gross_spot = cross / cross.shift(1)
        delivery = (1 + quotes[reference].shift(1) * tau) / (1 + quotes[local].shift(1) * tau)
        independent = (1 + native) * gross_spot + delivery - gross_spot - 1
        frame = pd.DataFrame({'Observed': raw[target], 'QIS': actual,
            'Base index': native, 'No carry': native * gross_spot,
            'Independent': independent, 'Spot interaction': native * (gross_spot - 1),
            'QIS carry': delivery - 1}).loc['1999-01-31':cutoff]
        check = frame[['QIS', 'Independent']].dropna()
        np.testing.assert_allclose(check.QIS, check.Independent, atol=1e-12, rtol=1e-11)
        max_payoff_error = max(max_payoff_error, float((check.QIS-check.Independent).abs().max()))
        for window in ('Full available', RECENT):
            paired = frame.loc['2021-01-31' if window == RECENT else frame.index.min():].dropna()
            fit = linregress(paired.Observed, paired.QIS)
            error = paired.QIS - paired.Observed
            te, _ = qis.compute_te_ir_errors(error.to_frame('error'))
            np.testing.assert_allclose(te.iloc[0], error.std(ddof=1)*np.sqrt(12), atol=1e-12)
            nav = anchored_nav(paired[['Observed', 'QIS']])
            pa = dict(zip(nav.columns, np.atleast_1d(qis.compute_pa_return(nav))))
            row = dict(identity, Window=window, Start=str(paired.index.min().date()),
                End=str(paired.index.max().date()), Months=len(paired),
                R_squared=fit.rvalue**2, Beta=fit.slope,
                Intercept_monthly_bp=fit.intercept*10000,
                MAE_monthly_bp=error.abs().mean()*10000,
                RMSE_monthly_bp=np.sqrt(np.mean(error**2))*10000,
                Tracking_error_pa_bp=te.iloc[0]*10000,
                Mean_error_pa_bp=error.mean()*120000,
                Observed_return_pa=pa['Observed'], QIS_return_pa=pa['QIS'],
                CAGR_difference_bp=(pa['QIS']-pa['Observed'])*10000,
                Missing_internal_months=len(pd.date_range(
                    paired.index.min(), paired.index.max(), freq='ME'))-len(paired))
            rows.append(row)
            if window == RECENT:
                plot_comparison(paired, row, output)
                monthly.append(paired.assign(Observed_ticker=target).rename_axis('Date').reset_index())
    result = pd.DataFrame(rows)
    result.to_csv(output/'statistics.csv', index=False, float_format='%.17g')
    pd.concat(monthly, ignore_index=True).to_csv(output/'monthly_comparison.csv', index=False,
                                               float_format='%.17g')
    provenance = {'cash_convention': 'ACT/365F; opening quotes; monthly principal resets',
        'cutoff': cutoff, 'max_independent_payoff_error': max_payoff_error,
        'input_sha256': {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                         for p in (returns_csv, spots_csv, rates_csv, pairs_csv)},
        'qis_fx_source_sha256': hashlib.sha256(b''.join(p.read_bytes() for p in
            sorted((Path(qis.__file__).parent/'fx').rglob('*.py')))).hexdigest()}
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n', encoding='utf-8')
    return result


def main():
    """Run an explicit private-input recheck without changing the public frozen study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-csv', type=Path)
    parser.add_argument('--returns-csv', type=Path)
    parser.add_argument('--spots-csv', type=Path)
    parser.add_argument('--rates-csv', type=Path)
    parser.add_argument('--pairs-csv', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'tools/docs_analytics/manifest.json').read_text('utf-8'))
    if args.returns_csv is not None:
        if not all((args.spots_csv, args.rates_csv, args.pairs_csv)):
            parser.error('Recompute requires spots, rates and pairs CSVs')
        result = recompute(args.returns_csv, args.spots_csv, args.rates_csv,
                           args.pairs_csv, args.output_dir)
        print(result.to_string(index=False))
        return
    if args.source_csv is None:
        parser.error('Supply --source-csv to recheck or --returns-csv to recompute')
    result = refit(args.source_csv, manifest['producers']['hedged_index_case_study'],
                   args.output_dir)
    print(f'Verified {len(result)} recent-period pairs against frozen statistics')


if __name__ == '__main__':
    main()

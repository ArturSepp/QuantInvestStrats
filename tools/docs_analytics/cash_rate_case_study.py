"""Preserve reviewed cash-rate evidence; optionally refit from a private return panel.

Only aggregate statistics and registered preview PNGs are public inputs. The default
producer copies those frozen figures, rather than claiming to regenerate vendor data.
The explicit refresh command needs a privately supplied CSV and never acquires data.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
METHODS = {'Lag=0': 'end_1_12', 'Lag=1': 'start_1_12', 'Midpoint': 'midpoint_1_12'}


def produce(spec):
    """Validate and return frozen image paths and the published aggregate table."""
    parameters = spec['parameters']
    statistics = pd.DataFrame(parameters['statistics'])
    currencies = parameters['currencies']
    expected = {(ccy, method) for ccy in currencies for method in METHODS}
    observed = list(zip(statistics['Currency'], statistics['Method']))
    if set(observed) != expected or len(observed) != len(expected):
        raise ValueError('Incomplete or duplicate currency/method statistics')
    numeric = statistics.select_dtypes(include='number')
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError('Non-finite empirical summary')
    if not statistics['R_squared'].between(0.0, 1.0).all():
        raise ValueError('Invalid empirical R-squared')
    if not (statistics['RMSE_bp'] >= statistics['MAE_bp']).all():
        raise ValueError('RMSE must not be below MAE')
    figures = {}
    for item in parameters['frozen_images']:
        path = ROOT / 'docs/images' / item['filename']
        if path.parent != ROOT / 'docs/images' or path.suffix != '.png':
            raise ValueError('Unsafe frozen preview path')
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'Frozen preview hash mismatch: {path.name}')
        figures[path.name] = path
    return {
        'figures': figures,
        'tables': {'statistics': statistics},
        'checks': dict.fromkeys(spec['checks'], True),
        'parameters': parameters,
        'conventions': spec['conventions'],
        'summary': {
            'evidence': 'Frozen observed-data study; aggregate integrity, not an offline refit',
            'currencies': currencies,
            'raw_vendor_data_distributed': False,
            'numerical_review': parameters['numerical_review'],
        },
    }


def refit(source_csv, spec, output_dir):
    """Reproduce plots and independently check recorded statistics from private inputs."""
    import matplotlib.pyplot as plt
    from scipy.stats import linregress
    import qis
    from tools.docs_analytics.run import output_boundary
    from tools.docs_analytics.style import save_figure

    output = output_boundary(Path(output_dir))
    expected_hash = spec['parameters']['input_artifacts'][
        'cash_rate_monthly_comparison_csv']['sha256']
    if hashlib.sha256(Path(source_csv).read_bytes()).hexdigest() != expected_hash:
        raise ValueError('Private input differs from the frozen study snapshot')
    output.mkdir(parents=True)
    panel = pd.read_csv(source_csv, parse_dates=['Date'])
    params = spec['parameters']
    panel = panel.loc[panel['Date'].between(params['start'], params['end'])]
    recorded = pd.DataFrame(params['statistics']).set_index(['Currency', 'Method'])
    rows = []
    for currency in params['currencies']:
        data = panel.loc[panel['Currency'].eq(currency)].dropna(
            subset=['JPM', *METHODS.values()])
        if data['Date'].duplicated().any():
            raise ValueError(f'Duplicate observation dates for {currency}')
        plot_rows = []
        estimates = {
            'Lag=0': data['rate_end_pa'].to_numpy() / 12.0,
            'Lag=1': data['rate_start_pa'].to_numpy() / 12.0,
            'Midpoint': (data['rate_start_pa'].to_numpy()
                         + data['rate_end_pa'].to_numpy()) / 24.0,
        }
        for method, column in METHODS.items():
            x, y = data['JPM'].to_numpy(), estimates[method]
            np.testing.assert_allclose(y, data[column], atol=1e-14, rtol=1e-12)
            fit = linregress(x, y)
            error = (y - x) * 10000.0
            row = {
                'Currency': currency, 'Ticker': data['Ticker'].iloc[0], 'Method': method,
                'N': len(data), 'R_squared': fit.rvalue ** 2, 'OLS_slope': fit.slope,
                'OLS_intercept_bp': fit.intercept * 10000.0,
                'Bias_bp': error.mean(), 'MAE_bp': np.abs(error).mean(),
                'RMSE_bp': np.sqrt(np.mean(error ** 2)),
            }
            reference = recorded.loc[(currency, method)]
            for label, value in row.items():
                if label in ('Currency', 'Method'):
                    continue
                if isinstance(value, str):
                    assert value == reference[label]
                else:
                    np.testing.assert_allclose(value, reference[label], atol=1e-9, rtol=1e-10)
            rows.append(row)
            plot_rows.append(pd.DataFrame({
                'JPM': x, 'Estimate': y,
                'Method': f'{method}: R²={row["R_squared"]:.3f}, MAE={row["MAE_bp"]:.2f} bp',
            }))
        figure, ax = plt.subplots(figsize=(8, 7))
        figure.subplots_adjust(left=0.14, right=0.97, bottom=0.20, top=0.85)
        qis.plot_scatter(
            pd.concat(plot_rows, ignore_index=True), x='JPM', y='Estimate', hue='Method',
            order=1, full_sample_order=1, fit_intercept=True, add_hue_model_label=False,
            add_universe_model_label=False, add_45line=True, align_axis=True,
            xlabel='JPM monthly cash return', ylabel='Estimated monthly cash return',
            xvar_format='{:.2%}', yvar_format='{:.2%}', fontsize=11, markersize=28,
            colors=['#D55E00', '#0072B2', '#009E73'], ax=ax,
        )
        figure.suptitle(f'{currency} cash return timing comparison', fontsize=17, y=0.98)
        ax.set_title(f'{data["Ticker"].iloc[0]} | January 2021–September 2026 | n={len(data)}',
                     fontsize=11, pad=12)
        figure.text(0.5, 0.08, 'Lag=0: end rate/12. Lag=1: previous end rate/12.\n'
                    'Midpoint: mean of both. Dashed diagonal: exact agreement.\n'
                    'Observed data; OLS with intercept; missing observations excluded.',
                    ha='center', fontsize=10)
        save_figure(figure, output / f'cash_rate_case_study_{currency.lower()}.png')
        plt.close(figure)
    result = pd.DataFrame(rows)
    result.to_csv(output / 'statistics.csv', index=False, float_format='%.17g')
    return result


def main():
    """Run an explicit private-input refit, leaving the public frozen bundle unchanged."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-csv', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'tools/docs_analytics/manifest.json').read_text('utf-8'))
    result = refit(args.source_csv, manifest['producers']['cash_rate_case_study'], args.output_dir)
    print(f'Verified {len(result)} currency/method fits against frozen statistics')


if __name__ == '__main__':
    main()

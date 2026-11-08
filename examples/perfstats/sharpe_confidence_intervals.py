"""Single-strategy Sharpe uncertainty: IID-normal chart and HAC comparison, offline.

Run ``python -m examples.perfstats.sharpe_confidence_intervals``. CHART displays the
figure and prints results without writes; EXPORT also saves PNG, CSVs and metadata
through qis.local_path (explicit output_path/settings_path or portable YAML defaults).

The chart holds the observed arithmetic Sharpe at 2 for each hypothetical record
length. Prefixes of the frozen synthetic SEQ_US returns are separately standardised
and shifted to achieve that value, not presented as one evolving realised strategy.
No simulation, strategy search, bootstrap or calibration is run. The second table
compares IID-normal, HAC(0) and HAC(10) on the same unmodified synthetic returns.
Cash is zero and returns are gross; intervals are pointwise normal approximations.
All new analytics are explicit source-module imports, not root API exports.
"""
from dataclasses import asdict
from enum import Enum
import hashlib
from importlib.metadata import version
import inspect
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qis import local_path
from qis.datasets.synthetic import generate_synthetic_prices
from qis.perfstats.sharpe_inference import SharpeInference, sharpe_ratio_test


AF = 252.0  # Explicit daily convention for reproducing the post, not QIS's B-grid default.


class Locals(Enum):
    """Display only, or also export through the shared YAML path resolver."""

    CHART = 1
    EXPORT = 2


def example_returns():
    """Fixed synthetic daily simple returns; the source generator is not changed."""
    prices = generate_synthetic_prices(end='2009-12-31', apply_quirks=False)
    return prices['SEQ_US'].pct_change(fill_method=None).iloc[1:]


def fixed_sharpe_returns(returns, n_obs, sharpe=2.0):
    """Construct a hypothetical complete record with a prescribed ddof=0 sample Sharpe."""
    sample = returns.iloc[:n_obs]
    if len(sample) != n_obs:
        raise ValueError('not enough fixed fixture observations')
    standardized = (sample - sample.mean()) / sample.std(ddof=0)
    return 0.01 * (standardized + sharpe / np.sqrt(AF))


def build_example():
    """Return the chart and its full-precision supporting tables without writing files."""
    returns = example_returns()
    counts = sorted(set(range(32, 1009, 8)) | {126, 252, 504, 756, 1008})
    rows = []
    for n_obs in counts:
        result = sharpe_ratio_test(
            fixed_sharpe_returns(returns, n_obs), method=SharpeInference.IID_NORMAL, af=AF,
        )
        rows.append(dict(years=n_obs / AF, **asdict(result)))
    intervals = pd.DataFrame(rows).set_index('n_obs')
    comparisons = {}
    for label, method, lags in (
            ('IID-normal', SharpeInference.IID_NORMAL, None),
            ('HAC(0)', SharpeInference.HAC, 0), ('HAC(10)', SharpeInference.HAC, 10)):
        comparisons[label] = asdict(sharpe_ratio_test(
            returns, method=method, af=AF, n_lags=lags,
        ))
    comparison = pd.DataFrame.from_dict(comparisons, orient='index')

    with plt.rc_context({'font.size': 10}):
        figure, axis = plt.subplots(figsize=(8.3, 7.4))
        figure.subplots_adjust(left=0.11, right=0.96, bottom=0.28, top=0.77)
        figure.text(0.11, 0.94, 'A Sharpe of 2 over 6 months\ncould still be zero edge',
                    fontsize=19, weight='bold', va='top')
        figure.text(0.11, 0.825, 'Pointwise 95% CI for an observed Sharpe of 2', color='0.3')
        years = intervals.years.to_numpy()
        axis.set_facecolor('#f7f7f5')
        axis.fill_between(years, intervals.ci_lower, intervals.ci_upper,
                          color='#cbdff2', alpha=0.85)
        axis.plot(years, intervals.ci_lower, color='#6b9fd0', linewidth=1.2)
        axis.plot(years, intervals.ci_upper, color='#6b9fd0', linewidth=1.2)
        axis.axhline(2.0, color='#1878c5', linewidth=2)
        axis.axhline(0.0, color='0.4', linestyle='--', linewidth=1)
        half = intervals.loc[126]
        axis.vlines(0.5, half.ci_lower, half.ci_upper, color='0.25', linewidth=1.4)
        axis.scatter([0.5, 0.5], [half.ci_lower, half.ci_upper],
                     color='#1878c5', edgecolors='white', zorder=4)
        axis.annotate(f'6 months: 95% CI\n{half.ci_lower:.1f} to {half.ci_upper:.1f}',
                      xy=(0.5, half.ci_upper), xytext=(0.7, 4.9),
                      arrowprops=dict(arrowstyle='-', color='0.35'))
        # Linear interpolation of saved bounds: no duplicate inference implementation.
        crossing = float(np.interp(0.0, intervals.ci_lower, years))
        axis.scatter([crossing], [0], color='#1878c5', edgecolors='white', zorder=4)
        axis.annotate('About 1 year: the lower\nbound first clears zero',
                      xy=(crossing, 0), xytext=(1.35, -1.65),
                      arrowprops=dict(arrowstyle='-', color='0.35'))
        axis.text(3.98, 2.15, 'Observed Sharpe 2', ha='right')
        axis.text(3.98, 0.15, 'Sharpe 0 (zero mean excess return)', ha='right', fontsize=9)
        axis.set(xlim=(0.12, 4.05), ylim=(-2, 6), yticks=[-2, 0, 2, 4, 6],
                 xticks=[0.5, 1, 2, 3, 4], xticklabels=['6 mo', '1 yr', '2 yrs', '3 yrs', '4 yrs'],
                 xlabel='Length of hypothetical track record',
                 ylabel='Annualised arithmetic Sharpe')
        axis.grid(axis='y', color='white')
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_visible(False)
        figure.text(0.11, 0.19, 'Track length for a fixed observed Sharpe to exclude zero',
                    weight='bold')
        figure.text(0.11, 0.15,
                    'Sharpe 2: ~1 year     Sharpe 1: ~4 years     Sharpe 0.5: ~15 years')
        figure.text(0.11, 0.065,
                    'IID-normal plug-in approximation, af=252, ddof=0; not a power guarantee.\n'
                    'Synthetic inputs; each record is rescaled separately to keep Sharpe=2.\n'
                    'No simulation or strategy-selection correction. Source: QIS; Lo (2002).',
                    fontsize=8, color='0.35')
    return figure, intervals, comparison


def run_local(local: Locals, *, output_path=None, settings_path=None):
    """Display the example; only EXPORT resolves output paths or writes files."""
    if not isinstance(local, Locals):
        raise ValueError('local must be a Locals member')
    figure, intervals, comparison = build_example()
    print('Pointwise arithmetic-Sharpe intervals; no finite-sample coverage guarantee.')
    print(intervals.loc[[126, 252, 504, 1008], ['years', 'sr_an', 'se_an', 'ci_lower', 'ci_upper']])
    print('\nSame unmodified synthetic return sample, different variance assumptions:')
    print(comparison[['sr_an', 'se_an', 'ci_lower', 'ci_upper', 'p_value', 'n_obs']])
    if local is Locals.EXPORT:
        output = Path(local_path.get_output_path(output_path, settings_path=settings_path))
        output = output / 'sharpe_confidence_intervals'
        output.mkdir(parents=True, exist_ok=True)
        intervals.to_csv(output / 'track_length.csv', float_format='%.17g')
        comparison.to_csv(output / 'inference_comparison.csv', float_format='%.17g')
        figure.savefig(output / 'sharpe_confidence_intervals.png', dpi=160)
        inputs = example_returns()
        metadata = dict(
            fixture='frozen synthetic SEQ_US; apply_quirks=False; default fixture seed',
            sample_start=str(inputs.index[0]), sample_end=str(inputs.index[-1]),
            fixture_sha256=hashlib.sha256(inputs.to_csv().encode()).hexdigest(),
            distribution_version=version('qis'),
            implementation_sha256=hashlib.sha256(
                inspect.getsource(sharpe_ratio_test).encode()).hexdigest(),
            af=AF, ddof=0,
            chart_method='IID_NORMAL', confidence_level=0.95, observed_sharpe=2.0,
            return_basis='simple; zero cash assumed; gross',
            construction='each prefix standardized and shifted separately; not one evolving record',
            inference='pointwise asymptotic normal; not bootstrap, power or calibration evidence',
        )
        (output / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    plt.show()
    return figure, intervals, comparison


if __name__ == '__main__':
    run_local(local=Locals.CHART)

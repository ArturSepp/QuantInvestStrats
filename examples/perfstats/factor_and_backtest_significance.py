"""Significance of means, alphas and Sharpe differences for a declared family of strategies.

Run ``python -m examples.perfstats.factor_and_backtest_significance`` from the checkout. The
default ANALYTIC branch prints the family tables and the Bartlett-versus-prewhitened comparison
and shows one figure; BOOTSTRAP adds the stationary-bootstrap families; EXPORT also writes the
tables and the figure to YAML ``OUTPUT_PATH`` or an explicit ``output_path``.

Sample: the nine instruments of the frozen synthetic panel that report over its whole monthly
history (``SEQ_EM`` starts late and is left out): 251 month-end simple returns from 2005-02 to
2025-12, with the synthetic 60/40 benchmark. Returns are gross and cash is assumed zero; the
tests subtract no funding or costs. ``SAL_HF`` and ``SAL_PE`` carry appraisal smoothing and
have the largest fitted rho; even there the prewhitened-to-Bartlett standard-error ratio stays
within a few percent at L=3, because the monthly scores of this panel are nearly white.

Analytics demonstrated: ``compute_mean_significance`` and ``compute_alpha_significance`` with
``InterceptInference.BARTLETT_HAC`` and ``PREWHITENED_AR1``; ``compute_sharpe_significance``
against the fixed benchmark; ``MultipleTestingMethod.HOLM`` and ``BHY`` through the family
functions and directly through ``adjust_pvalues``; ``estimate_prewhitened_intercepts`` for the
standard-error comparison; ``compute_bootstrap_mean_significance`` and
``compute_bootstrap_alpha_significance`` sharing one index matrix. Mean/alpha displays use
af=12; paired Sharpe displays use sqrt(12). These are full-sample descriptive tests on a
synthetic panel. The family interfaces are source-only, not in ``qis.__all__``, and their
finite-sample size and coverage are not established.
"""

# packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from enum import Enum
from pathlib import Path
from typing import Dict, Tuple, Union
# qis
import qis
from qis import local_path
from qis.datasets.synthetic import generate_synthetic_universe
from qis.perfstats.bootstrap_significance import (
    compute_bootstrap_alpha_significance, compute_bootstrap_mean_significance,
)
from qis.perfstats.multiple_testing import MultipleTestingMethod, adjust_pvalues
from qis.perfstats.significance import (
    InterceptInference, SignificanceResult, compute_alpha_significance,
    compute_mean_significance, compute_sharpe_significance,
)
from qis.utils.hac_intercept_prewhitening import estimate_prewhitened_intercepts


FACTORS = ['SEQ_US', 'SBD_TSY']  # the benchmark constituents double as the alpha factors
BOOTSTRAP_DRAWS = 199  # 1/(B+1) = 0.005 reaches Holm's first rank 0.05/9 for nine members
METHODS = (MultipleTestingMethod.HOLM, MultipleTestingMethod.BHY)
INFERENCES = (InterceptInference.BARTLETT_HAC, InterceptInference.PREWHITENED_AR1)
SETTINGS = dict(
    return_basis='fractional simple total returns; zero cash assumed',
    cost_basis='gross; no fees or transaction costs',
    sampling_frequency='ME', hac_lags=3, confidence_level=0.95, significance_level=0.05, af=12,
)


class Locals(Enum):
    """Analytic families with the figure, plus bootstrap families, plus file export."""

    ANALYTIC = 1
    BOOTSTRAP = 2
    EXPORT = 3


def load_example_returns() -> Tuple[pd.DataFrame, pd.Series]:
    """Return month-end simple returns of the full-history instruments and the 60/40 benchmark."""
    universe = generate_synthetic_universe()
    returns = qis.to_returns(universe.prices, freq='ME', is_log_returns=False, drop_first=True)
    benchmark = qis.to_returns(universe.benchmark_prices, freq='ME', is_log_returns=False,
                               drop_first=True)
    return returns.dropna(axis=1), benchmark.iloc[:, 0]


def run_families(returns: pd.DataFrame, benchmark: pd.Series) -> Dict[str, SignificanceResult]:
    """Fix the three families, then run both corrections and both intercept inferences.

    Args:
        returns: Complete monthly simple returns, one candidate per column.
        benchmark: Fixed benchmark returns on the same dates for the paired Sharpe family.

    Returns:
        Results keyed ``mean_<inference>_<method>``, ``alpha_<inference>_<method>`` and
        ``sharpe_<method>``. Each estimand is its own family; Holm and BHY are alternative
        error targets, not a menu to pick the smaller adjusted p-value from.
    """
    responses = returns.drop(columns=FACTORS)
    factors = returns[FACTORS]
    results: Dict[str, SignificanceResult] = {}
    for method in METHODS:
        for inference in INFERENCES:
            key = f'{inference.value}_{method.name}'
            results[f'mean_{key}'] = compute_mean_significance(
                returns, family_id='synthetic_means', method=method, inference=inference,
                **SETTINGS,
            )
            results[f'alpha_{key}'] = compute_alpha_significance(
                responses, factors, family_id='synthetic_alphas', method=method,
                inference=inference, **SETTINGS,
            )
        results[f'sharpe_{method.name}'] = compute_sharpe_significance(
            returns, benchmark, family_id='synthetic_sharpe_pairs', method=method, **SETTINGS,
        )
    return results


def run_bootstrap_families(returns: pd.DataFrame) -> Dict[str, SignificanceResult]:
    """Null-centred stationary bootstrap for means and alphas, one index matrix for all four.

    Args:
        returns: Complete monthly simple returns, one candidate per column.

    Returns:
        Results keyed ``bootstrap_mean_<inference>`` and ``bootstrap_alpha_<inference>`` under
        Holm. The first call generates the (T, B) indices with the qis defaults; the others reuse
        that exact matrix, so every family is resampled on the same draws.
    """
    responses = returns.drop(columns=FACTORS)
    factors = returns[FACTORS]
    first = compute_bootstrap_mean_significance(
        returns, family_id='synthetic_bootstrap_means', num_samples=BOOTSTRAP_DRAWS,
        include_diagnostics=True, inference=InterceptInference.BARTLETT_HAC, **SETTINGS,
    )
    indices = first.metadata['bootstrap_indices']
    results = {'bootstrap_mean_bartlett_hac': first}
    results['bootstrap_mean_prewhitened_ar1'] = compute_bootstrap_mean_significance(
        returns, family_id='synthetic_bootstrap_means', bootstrapped_indices=indices,
        inference=InterceptInference.PREWHITENED_AR1, **SETTINGS,
    )
    for inference in INFERENCES:
        results[f'bootstrap_alpha_{inference.value}'] = compute_bootstrap_alpha_significance(
            responses, factors, family_id='synthetic_bootstrap_alphas',
            bootstrapped_indices=indices, inference=inference, **SETTINGS,
        )
    return results


def compare_inference(results: Dict[str, SignificanceResult], family: str) -> pd.DataFrame:
    """Place Bartlett and prewhitened inference for one family side by side.

    Args:
        results: Output of run_families.
        family: 'mean' or 'alpha'.

    Returns:
        Per-member annualised estimate, both standard errors and their ratio, fitted rho, raw
        p-values under both inferences, and the Holm/BHY adjusted p-values of each.
    """
    bartlett = results[f'{family}_bartlett_hac_HOLM']
    prewhitened = results[f'{family}_prewhitened_ar1_HOLM']
    table = pd.DataFrame({
        'estimate': bartlett.table['estimate'],
        'se_bartlett': bartlett.table['hac_se'],
        'se_prewhitened': prewhitened.table['hac_se'],
        'rho': prewhitened.table['rho'],
        'p_bartlett': bartlett.table['pvalue_raw'],
        'holm_bartlett': bartlett.table['pvalue_adjusted'],
        'bhy_bartlett': results[f'{family}_bartlett_hac_BHY'].table['pvalue_adjusted'],
        'p_prewhitened': prewhitened.table['pvalue_raw'],
        'holm_prewhitened': prewhitened.table['pvalue_adjusted'],
        'bhy_prewhitened': results[f'{family}_prewhitened_ar1_BHY'].table['pvalue_adjusted'],
    })
    table.insert(3, 'se_ratio', table['se_prewhitened'] / table['se_bartlett'])
    return table


def direct_calls(returns: pd.DataFrame, results: Dict[str, SignificanceResult]) -> pd.DataFrame:
    """Reproduce two family columns with the underlying analytics called directly.

    Args:
        returns: Complete monthly simple returns.
        results: Output of run_families.

    Returns:
        The mean family's Holm-adjusted p-values from adjust_pvalues and the per-period
        prewhitened standard errors from estimate_prewhitened_intercepts, which equal the
        family function's columns (the family function annualises the SE by af).
    """
    means = results['mean_bartlett_hac_HOLM']
    corrected = adjust_pvalues(means.table['pvalue_raw'], family_id='synthetic_means')
    np.testing.assert_allclose(corrected['pvalue_adjusted'], means.table['pvalue_adjusted'])
    filtered = estimate_prewhitened_intercepts(returns, hac_lags=SETTINGS['hac_lags'])
    prewhitened = results['mean_prewhitened_ar1_HOLM']
    np.testing.assert_allclose(SETTINGS['af'] * filtered.table['standard_error'],
                               prewhitened.table['hac_se'])
    np.testing.assert_allclose(filtered.table['rho'], prewhitened.table['rho'])
    return pd.DataFrame({'holm_direct': corrected['pvalue_adjusted'],
                         'se_prewhitened_per_period': filtered.table['standard_error'],
                         'rho': filtered.table['rho'],
                         'innovations': filtered.innovations.shape[0]})


def plot_family_summary(mean_table: pd.DataFrame, alpha_table: pd.DataFrame) -> plt.Figure:
    """Draw the mean family's p-values and the prewhitened-to-Bartlett SE ratios.

    Args:
        mean_table: compare_inference(results, 'mean').
        alpha_table: compare_inference(results, 'alpha').

    Returns:
        A two-panel figure: raw and corrected p-values per member with the 5% line, and the
        standard-error ratio per member for means and alphas.
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5), tight_layout=True)
    qis.set_suptitle(fig, 'Synthetic monthly panel, 2005-02 to 2025-12: family p-values and '
                          'the effect of AR(1) prewhitening on standard errors (L=3)')
    pvalues = mean_table[['p_bartlett', 'holm_bartlett', 'bhy_bartlett', 'p_prewhitened']]
    pvalues.columns = ['raw, Bartlett', 'Holm, Bartlett', 'BHY, Bartlett', 'raw, prewhitened']
    qis.plot_bars(pvalues, stacked=False, title='Mean family: raw and corrected p-values',
                  yvar_format='{:.2f}', legend_loc='upper left', x_rotation=45, ax=axs[0])
    axs[0].axhline(SETTINGS['significance_level'], color='black', linestyle='--', linewidth=1)
    ratios = pd.concat([mean_table['se_ratio'].rename('means'),
                        alpha_table['se_ratio'].rename('alphas')], axis=1)
    qis.plot_bars(ratios, stacked=False, title='Prewhitened / Bartlett standard error',
                  yvar_format='{:.2f}', legend_loc='upper left', x_rotation=45,
                  y_limits=(0.9, 1.1), ax=axs[1])
    axs[1].axhline(1.0, color='black', linestyle='--', linewidth=1)
    for position, (name, rho) in enumerate(mean_table['rho'].items()):
        axs[1].annotate(f'rho={rho:.2f}'.replace('-0.00', '0.00'),
                        (position, ratios.loc[name].max()),
                        textcoords='offset points', xytext=(0, 4), ha='center', fontsize=8)
    return fig


def run_local(
        local: Locals,
        output_path: Union[str, Path, None] = None,
) -> Dict[str, SignificanceResult]:
    """Print the walkthrough, show the figure and optionally export tables and figure.

    Args:
        local: ANALYTIC prints the families and comparisons; BOOTSTRAP adds four bootstrap
            families; EXPORT adds the bootstrap and writes everything.
        output_path: Explicit output-directory override. Otherwise EXPORT uses YAML
            OUTPUT_PATH. Supplying an override with another branch also requests export.

    Returns:
        Every SignificanceResult by name; inspect both .table and .metadata, not only reject.

    Raises:
        ValueError: An unknown local branch or invalid analytics/path configuration.
    """
    if not isinstance(local, Locals):
        raise ValueError(f'Unknown example branch: {local!r}')
    returns, benchmark = load_example_returns()
    results = run_families(returns, benchmark)
    sample = results['mean_bartlett_hac_HOLM'].metadata
    print(f"Common sample: {sample['sample_start']:%Y-%m-%d} to {sample['sample_end']:%Y-%m-%d}; "
          f"{sample['nobs']} monthly simple returns; members {list(returns.columns)}.")
    print('Zero cash assumed; gross returns; three monthly HAC lags; af=12 for display.')
    print('Holm targets FWER; BHY targets FDR. Intervals are unadjusted/pointwise normal.')
    comparisons = {family: compare_inference(results, family) for family in ('mean', 'alpha')}
    for family, table in comparisons.items():
        print(f'\n{family} family: Bartlett versus AR(1)-prewhitened intercept inference')
        print(table.to_string(float_format=lambda value: f'{value:.4g}'))
    print('\nsharpe family: candidate minus benchmark arithmetic Sharpe, sqrt(12) display')
    sharpe = results['sharpe_HOLM'].table.copy()
    sharpe['pvalue_adjusted_bhy'] = results['sharpe_BHY'].table['pvalue_adjusted']
    print(sharpe.to_string(float_format=lambda value: f'{value:.4g}'))
    print('\nDirect calls reproduce the family columns:')
    print(direct_calls(returns, results).to_string(float_format=lambda value: f'{value:.4g}'))

    if local in (Locals.BOOTSTRAP, Locals.EXPORT):
        bootstrap = run_bootstrap_families(returns)
        results.update(bootstrap)
        meta = bootstrap['bootstrap_mean_bartlett_hac'].metadata
        print(f"\nBootstrap: B={meta['num_samples']}, expected block {meta['block_size']}, "
              f"seed {meta['seed']}, minimum p={meta['minimum_pvalue']:.3g}, one index matrix "
              f"(SHA256 {meta['indices_sha256'][:12]}...) reused by every family.")
        for name, result in bootstrap.items():
            columns = ['estimate', 'pvalue_hac', 'pvalue_raw', 'pvalue_adjusted', 'reject',
                       'bootstrap_exceedances']
            print(f"\n{name}: Holm on bootstrap p-values; pvalue_hac is the normal p-value")
            print(result.table[columns].to_string(float_format=lambda value: f'{value:.4g}'))

    fig = plot_family_summary(comparisons['mean'], comparisons['alpha'])
    if local is Locals.EXPORT or output_path is not None:
        output = Path(local_path.get_output_path(output_path)) / 'factor_and_backtest_significance'
        output.mkdir(parents=True, exist_ok=True)
        for name, result in results.items():
            result.table.to_csv(output / f'{name}.csv', float_format='%.17g')
        for family, table in comparisons.items():
            table.to_csv(output / f'{family}_inference_comparison.csv', float_format='%.17g')
        qis.save_fig(fig, file_name='family_summary', local_path=str(output))
        print(f'\nTables and figure saved to {output}')
    plt.show()
    return results


if __name__ == '__main__':
    run_local(local=Locals.ANALYTIC)

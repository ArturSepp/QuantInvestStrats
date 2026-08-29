"""Illustrate model-layer attribution on a simulated four-layer model, offline and seeded.

The simulation builds monthly log returns for a benchmark ``B``, a risk layer ``R`` (beta 1.05,
alpha 1% per year), a standalone signal sleeve ``A`` (beta 1.00, alpha 3% per year), and a full
model ``F`` that runs at beta 0.85, keeps all of the risk-layer alpha and 60% of the signal alpha,
and carries its own residual. A net NAV subtracts a proportional trading cost. Residuals are AR(1)
so that the Bartlett HAC intervals differ from OLS intervals.

The example prints the regression table and the annualised components, checks the three
identities of the method (linearity of the integration coefficients, bar heights equal to OLS
alphas, invariance of alphas and intervals to an excess-return definition of the sleeve), reports
how the interval half-widths move when the lag count follows the Newey-West rule instead of the
default three, and draws the return bridge with 95% HAC(3) whiskers on the three alpha bars.

Run from the repository root: ``python -m examples.perfstats.model_layer_attribution_simulated``.
"""
# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from typing import Optional
# qis
import qis
from qis.perfstats.model_layer_attribution import (
    ALPHA_AN_CI_LOW_COLUMN,
    ALPHA_AN_CI_HIGH_COLUMN,
    ALPHA_HAC_SE_COLUMN,
)
from qis.utils.regression import newey_west_lag_rule

FREQ = 'ME'
PERIODS_PER_YEAR = 12.0
START, END = '2005-12-31', '2025-12-31'  # 240 monthly log returns
SEED = 169

BENCHMARK_MEAN, BENCHMARK_VOL = 0.06, 0.10  # per year
RISK_ALPHA, RISK_BETA, RISK_RESIDUAL_VOL = 0.01, 1.05, 0.02  # per year
SIGNAL_ALPHA, SIGNAL_BETA, SIGNAL_RESIDUAL_VOL = 0.03, 1.00, 0.04  # per year
FULL_BETA, FULL_RISK_SHARE, FULL_SIGNAL_SHARE, FULL_RESIDUAL_VOL = 0.85, 1.0, 0.6, 0.02
RESIDUAL_AR1 = 0.3
TRADING_COST = 0.0015  # per year, proportional drag on the net NAV


def simulate_ar1(n_periods: int, rho: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Return an AR(1) series with autocorrelation ``rho`` and innovation volatility ``sigma``."""
    innovations = sigma * rng.standard_normal(n_periods)
    series = np.zeros(n_periods)
    for t in range(1, n_periods):
        series[t] = rho * series[t - 1] + innovations[t]
    return series


def simulate_layer_navs(seed: int = SEED) -> dict[str, pd.Series]:
    """Simulate monthly log returns of the four layers and the net model, and return their NAVs."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, END, freq=FREQ)
    n_periods = len(dates) - 1
    monthly_vol = 1.0 / np.sqrt(PERIODS_PER_YEAR)
    r_b = (
        BENCHMARK_MEAN / PERIODS_PER_YEAR
        + BENCHMARK_VOL * monthly_vol * rng.standard_normal(n_periods)
    )
    risk_residual = simulate_ar1(n_periods, RESIDUAL_AR1, RISK_RESIDUAL_VOL * monthly_vol, rng)
    signal_residual = simulate_ar1(n_periods, RESIDUAL_AR1, SIGNAL_RESIDUAL_VOL * monthly_vol, rng)
    full_residual = simulate_ar1(n_periods, RESIDUAL_AR1, FULL_RESIDUAL_VOL * monthly_vol, rng)
    r_r = RISK_ALPHA / PERIODS_PER_YEAR + RISK_BETA * r_b + risk_residual
    r_a = SIGNAL_ALPHA / PERIODS_PER_YEAR + SIGNAL_BETA * r_b + signal_residual
    r_f = (FULL_BETA * r_b
           + FULL_RISK_SHARE * (r_r - RISK_BETA * r_b)
           + FULL_SIGNAL_SHARE * (r_a - SIGNAL_BETA * r_b)
           + full_residual)
    r_f_net = r_f - TRADING_COST / PERIODS_PER_YEAR

    def to_nav(log_returns: np.ndarray, name: str) -> pd.Series:
        return pd.Series(
            np.exp(np.concatenate([[0.0], np.cumsum(log_returns)])),
            index=dates,
            name=name,
        )

    return dict(benchmark_nav=to_nav(r_b, 'Benchmark'),
                risk_layer_nav=to_nav(r_r, 'Risk Layer'),
                alpha_layer_nav=to_nav(r_a, 'Alpha Layer'),
                full_model_nav=to_nav(r_f, 'Full Model'),
                full_model_net_nav=to_nav(r_f_net, 'Full Model Net'))


def check_identities(attribution: qis.ModelLayerAlphaBetaAttribution,
                     navs: dict[str, pd.Series]
                     ) -> None:
    """Verify linearity, bar heights, and the excess-basis invariance, and print the residuals."""
    table = attribution.regression_table
    alpha, beta = qis.PerfStat.ALPHA.to_str(), qis.PerfStat.BETA.to_str()
    an_alpha = qis.PerfStat.ALPHA_AN.to_str()
    linearity_beta = table.loc['Integration', beta] - (
        table.loc['Full Model', beta]
        - table.loc['Risk Layer', beta]
        - table.loc['Alpha Layer', beta]
    )
    linearity_alpha = table.loc['Integration', alpha] - (
        table.loc['Full Model', alpha]
        - table.loc['Risk Layer', alpha]
        - table.loc['Alpha Layer', alpha]
    )
    bars = attribution.annualised_components
    bar_heights = max(abs(bars['Risk Layer Alpha'] - table.loc['Risk Layer', an_alpha]),
                      abs(bars['Alpha Layer Alpha'] - table.loc['Alpha Layer', an_alpha]),
                      abs(bars['Integration Alpha'] - table.loc['Integration', an_alpha]))
    # excess basis: the sleeve NAV divided by the benchmark NAV has log return r_A - r_B
    excess_navs = dict(navs)
    excess_navs['alpha_layer_nav'] = (
        navs['alpha_layer_nav'] / navs['benchmark_nav']
    ).rename('Alpha Layer')
    excess = qis.compute_model_layer_alpha_beta_attribution(
        freq=FREQ, **excess_navs
    ).regression_table
    invariant_columns = [
        alpha,
        an_alpha,
        ALPHA_HAC_SE_COLUMN,
        ALPHA_AN_CI_LOW_COLUMN,
        ALPHA_AN_CI_HIGH_COLUMN,
        qis.PerfStat.ALPHA_PVALUE.to_str(),
    ]
    invariance = np.abs(excess[invariant_columns] - table[invariant_columns]).max().max()
    beta_shift_sleeve = excess.loc['Alpha Layer', beta] - table.loc['Alpha Layer', beta]
    beta_shift_integration = excess.loc['Integration', beta] - table.loc['Integration', beta]
    print(f'linearity: beta residual {linearity_beta:.1e}, alpha residual {linearity_alpha:.1e}')
    print(f'bar heights minus annualised OLS alphas: {bar_heights:.1e}')
    print(f'excess basis: max change in alphas, SEs, CIs, p-values {invariance:.1e}; '
          f'beta shifts sleeve {beta_shift_sleeve:+.6f}, integration {beta_shift_integration:+.6f}')
    assert abs(linearity_beta) < 1e-12 and abs(linearity_alpha) < 1e-12
    assert bar_heights < 1e-12
    assert invariance < 1e-10
    assert abs(beta_shift_sleeve + 1.0) < 1e-10
    assert abs(beta_shift_integration - 1.0) < 1e-10


def report_lag_rule_half_widths(attribution: qis.ModelLayerAlphaBetaAttribution,
                                navs: dict[str, pd.Series]
                                ) -> None:
    """Print the alpha interval half-widths at the default lag count and at the Newey-West rule."""
    n_obs = len(attribution.periodic_returns)
    rule_lags = newey_west_lag_rule(nobs=n_obs)
    rule = qis.compute_model_layer_alpha_beta_attribution(freq=FREQ, hac_lags=rule_lags, **navs)
    layers = ['Risk Layer', 'Alpha Layer', 'Integration', 'Full Model']
    default_table = attribution.regression_table.loc[layers]
    rule_table = rule.regression_table.loc[layers]
    half_widths = pd.DataFrame(
        {
            f'HAC({attribution.hac_lags}) bp': 0.5
            * 1e4
            * (
                default_table[ALPHA_AN_CI_HIGH_COLUMN]
                - default_table[ALPHA_AN_CI_LOW_COLUMN]
            ),
            f'HAC({rule.hac_lags}) bp': 0.5
            * 1e4
            * (
                rule_table[ALPHA_AN_CI_HIGH_COLUMN]
                - rule_table[ALPHA_AN_CI_LOW_COLUMN]
            ),
        }
    )
    print(
        'annualised alpha interval half-widths, default lags versus '
        f'Newey-West rule at T = {n_obs}:'
    )
    print(half_widths.round(1).to_string())


def plot_return_bridge(attribution: qis.ModelLayerAlphaBetaAttribution,
                       ax: Optional[plt.Axes] = None
                       ) -> plt.Figure:
    """Draw the annualised log-return bridge with HAC whiskers on the three alpha bars."""
    values = attribution.annualised_components
    table = attribution.regression_table
    beta = qis.PerfStat.BETA.to_str()
    steps = [
        ('Benchmark', values['Benchmark Return'], None, False),
        (
            f'Systematic\nbeta = {table.loc["Full Model", beta]:.2f}',
            values['Systematic Return'],
            None,
            False,
        ),
        (
            f'+ Risk-layer\nalpha\nbeta = {table.loc["Risk Layer", beta]:.2f}',
            values['Risk Layer Alpha'],
            'Risk Layer',
            True,
        ),
        (
            f'+ Signal\nalpha\nbeta = {table.loc["Alpha Layer", beta]:.2f}',
            values['Alpha Layer Alpha'],
            'Alpha Layer',
            True,
        ),
        (
            f'+ Integration\nalpha\nbeta = {table.loc["Integration", beta]:.2f}',
            values['Integration Alpha'],
            'Integration',
            True,
        ),
        ('Trading-cost\ndrag', values['Trading Cost Drag'], None, True),
        ('Full model\nnet', values['Full Model Net Return'], None, False),
    ]
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(11, 6))
    else:
        fig = ax.figure
    level = 0.0
    for position, (label, value, layer, is_step) in enumerate(steps):
        bottom = level if is_step else 0.0
        colour = '#3B5F7A' if not is_step else ('#8A6F3D' if value >= 0.0 else '#B45B5B')
        ax.bar(position, value, bottom=bottom, width=0.6, color=colour)
        top = bottom + value
        if layer is not None:
            ci_low = table.loc[layer, ALPHA_AN_CI_LOW_COLUMN]
            ci_high = table.loc[layer, ALPHA_AN_CI_HIGH_COLUMN]
            ax.errorbar(position, top, yerr=[[value - ci_low], [ci_high - value]], fmt='none',
                        ecolor='black', elinewidth=1.2, capsize=5)
            annotation_level = bottom + (ci_high if value >= 0.0 else ci_low)
        else:
            annotation_level = top
        ax.annotate(
            f'{value:+.1%}' if is_step else f'{value:.1%}',
            xy=(position, annotation_level),
            xytext=(0, 5 if value >= 0.0 else -14),
            textcoords='offset points',
            ha='center',
            fontsize=10,
        )
        level = top if is_step else value
        if position == 0:
            level = 0.0
    ax.set_xticks(range(len(steps)), [step[0] for step in steps], fontsize=9)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel('Annualised mean log return')
    ax.set_title(
        'Model-layer attribution on simulated layers: '
        'whiskers are 95% Bartlett HAC(3) intervals'
    )
    ax.axhline(0.0, color='black', linewidth=0.8)
    ax.spines[['top', 'right']].set_visible(False)
    return fig


def run_example(save_figure_path: Optional[str] = None) -> qis.ModelLayerAlphaBetaAttribution:
    """Run the simulated attribution and draw the return bridge."""
    navs = simulate_layer_navs()
    attribution = qis.compute_model_layer_alpha_beta_attribution(freq=FREQ, **navs)
    pd.set_option('display.width', 200)
    first_date = attribution.periodic_returns.index[0]
    last_date = attribution.periodic_returns.index[-1]
    print(
        f'common sample: {len(attribution.periodic_returns)} '
        f'{attribution.freq} log returns, {first_date:%b %Y} to {last_date:%b %Y}; '
        f'HAC lags {attribution.hac_lags}, '
        f'confidence level {attribution.confidence_level:.0%}'
    )
    print('regression table (alpha and its bounds annualised, beta and SE periodic):')
    print(attribution.regression_table.round(5).to_string())
    print('annualised components (bar heights):')
    print((100.0 * attribution.annualised_components).round(2).to_string())
    check_identities(attribution=attribution, navs=navs)
    report_lag_rule_half_widths(attribution=attribution, navs=navs)
    fig = plot_return_bridge(attribution=attribution)
    if save_figure_path is not None:
        fig.savefig(save_figure_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    return attribution


if __name__ == '__main__':
    run_example()

"""Illustrate model-layer attribution on a simulated four-layer model, offline and seeded.

The simulation builds monthly log returns for a benchmark ``B``, a risk layer ``R`` (beta 1.05,
alpha 1% per year), a signal layer ``S`` (beta 1.00, alpha 3% per year), and a full
model ``F`` that runs at beta 0.85, keeps all of the risk-layer alpha and 60% of the signal-layer
alpha, and carries its own residual. A net NAV subtracts a proportional trading cost. Residuals
are AR(1) so that the Bartlett HAC intervals differ from OLS intervals.

The example prints the regression table and the annualised components, checks the three
identities of the method (linearity of the integration coefficients, bar heights equal to OLS
alphas, invariance of alphas and intervals to an excess-return definition of the signal layer),
reports how the interval half-widths move when the lag count follows the Newey-West rule
instead of the default three, and draws three exhibits: the return bridge, additive cumulative
alpha, and a controlled two-feature Shapley sensitivity analysis with 95% HAC(3) intervals.

Run from the repository root: ``python -m examples.perfstats.model_layer_attribution_simulated``.
"""
# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from pathlib import Path
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

BETA_SPAN_FEATURE = 'beta_span'
SIGNAL_HORIZON_FEATURE = 'signal_horizon'
FEATURE_LABELS = {
    BETA_SPAN_FEATURE: 'Beta-estimation span x2',
    SIGNAL_HORIZON_FEATURE: 'Signal horizon x2',
}
REPORT_BACKGROUND = '#F7F5EF'
REPORT_TEXT = '#23313B'
REPORT_GRID = '#D8D4CB'
TOTAL_ALPHA_COLOR = '#126B52'
RISK_ALPHA_COLOR = '#5B9A91'
SIGNAL_ALPHA_COLOR = '#D99D1E'
INTEGRATION_ALPHA_COLOR = '#8A6F3D'
FEATURE_COLORS = {
    BETA_SPAN_FEATURE: '#3B5F7A',
    SIGNAL_HORIZON_FEATURE: SIGNAL_ALPHA_COLOR,
}


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
    r_s = SIGNAL_ALPHA / PERIODS_PER_YEAR + SIGNAL_BETA * r_b + signal_residual
    r_f = (FULL_BETA * r_b
           + FULL_RISK_SHARE * (r_r - RISK_BETA * r_b)
           + FULL_SIGNAL_SHARE * (r_s - SIGNAL_BETA * r_b)
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
                signal_layer_nav=to_nav(r_s, 'Signal Layer'),
                full_model_nav=to_nav(r_f, 'Full Model'),
                full_model_net_nav=to_nav(r_f_net, 'Full Model Net'))


def _orthogonal_feature_return(
        annual_mean: float,
        annual_vol: float,
        benchmark_returns: np.ndarray,
        rng: np.random.Generator,
) -> np.ndarray:
    """Simulate a centred AR(1) feature effect orthogonal to the benchmark in sample."""
    residual = simulate_ar1(
        n_periods=len(benchmark_returns),
        rho=RESIDUAL_AR1,
        sigma=annual_vol / np.sqrt(PERIODS_PER_YEAR),
        rng=rng,
    )
    residual = residual - residual.mean()
    benchmark_centred = benchmark_returns - benchmark_returns.mean()
    projection = np.dot(residual, benchmark_centred) / np.dot(
        benchmark_centred,
        benchmark_centred,
    )
    residual = residual - projection * benchmark_centred
    return annual_mean / PERIODS_PER_YEAR + residual


def simulate_feature_scenarios(
        navs: dict[str, pd.Series],
        seed: int = SEED + 1,
) -> dict[frozenset[str], qis.ModelLayerNavs]:
    """Create the complete two-feature experiment used by the Shapley illustration."""
    rng = np.random.default_rng(seed)
    dates = navs['benchmark_nav'].index
    layer_fields = (
        'risk_layer_nav',
        'signal_layer_nav',
        'full_model_nav',
        'full_model_net_nav',
    )
    base_returns = {
        field: np.log(navs[field]).diff().dropna().to_numpy()
        for field in layer_fields
    }
    benchmark_returns = np.log(navs['benchmark_nav']).diff().dropna().to_numpy()

    def build_effects(
            annual_means: dict[str, float],
            annual_vols: dict[str, float],
            annual_cost: float,
    ) -> dict[str, np.ndarray]:
        """Simulate one feature's layer effects and its incremental implementation cost."""
        effects = {
            field: _orthogonal_feature_return(
                annual_mean=annual_means[field],
                annual_vol=annual_vols[field],
                benchmark_returns=benchmark_returns,
                rng=rng,
            )
            for field in ('risk_layer_nav', 'signal_layer_nav', 'full_model_nav')
        }
        effects['full_model_net_nav'] = (
            effects['full_model_nav'] - annual_cost / PERIODS_PER_YEAR
        )
        return effects

    beta_effects = build_effects(
        annual_means={
            'risk_layer_nav': 0.0015,
            'signal_layer_nav': -0.0005,
            'full_model_nav': 0.0025,
        },
        annual_vols={
            'risk_layer_nav': 0.006,
            'signal_layer_nav': 0.005,
            'full_model_nav': 0.009,
        },
        annual_cost=0.0003,
    )
    signal_effects = build_effects(
        annual_means={
            'risk_layer_nav': 0.0,
            'signal_layer_nav': 0.0045,
            'full_model_nav': 0.0020,
        },
        annual_vols={
            'risk_layer_nav': 0.004,
            'signal_layer_nav': 0.010,
            'full_model_nav': 0.009,
        },
        annual_cost=0.0005,
    )
    interaction_effects = build_effects(
        annual_means={
            'risk_layer_nav': 0.0004,
            'signal_layer_nav': 0.0006,
            'full_model_nav': 0.0012,
        },
        annual_vols={
            'risk_layer_nav': 0.003,
            'signal_layer_nav': 0.004,
            'full_model_nav': 0.005,
        },
        annual_cost=0.0001,
    )

    def to_nav(log_returns: np.ndarray, name: str) -> pd.Series:
        """Convert one scenario's monthly log returns to a unit-initialised NAV."""
        return pd.Series(
            np.exp(np.concatenate([[0.0], np.cumsum(log_returns)])),
            index=dates,
            name=name,
        )

    coalitions = (
        frozenset(),
        frozenset({BETA_SPAN_FEATURE}),
        frozenset({SIGNAL_HORIZON_FEATURE}),
        frozenset({BETA_SPAN_FEATURE, SIGNAL_HORIZON_FEATURE}),
    )
    scenarios: dict[frozenset[str], qis.ModelLayerNavs] = {}
    for coalition in coalitions:
        scenario_returns = {
            field: values.copy() for field, values in base_returns.items()
        }
        if BETA_SPAN_FEATURE in coalition:
            scenario_returns = {
                field: values + beta_effects[field]
                for field, values in scenario_returns.items()
            }
        if SIGNAL_HORIZON_FEATURE in coalition:
            scenario_returns = {
                field: values + signal_effects[field]
                for field, values in scenario_returns.items()
            }
        if len(coalition) == 2:
            scenario_returns = {
                field: values + interaction_effects[field]
                for field, values in scenario_returns.items()
            }
        label = '+'.join(sorted(coalition)) or 'production'
        scenarios[coalition] = qis.ModelLayerNavs(
            benchmark_nav=navs['benchmark_nav'],
            risk_layer_nav=to_nav(
                scenario_returns['risk_layer_nav'], f'{label} Risk Layer'
            ),
            signal_layer_nav=to_nav(
                scenario_returns['signal_layer_nav'], f'{label} Signal Layer'
            ),
            full_model_nav=to_nav(
                scenario_returns['full_model_nav'], f'{label} Full Model'
            ),
            full_model_net_nav=to_nav(
                scenario_returns['full_model_net_nav'], f'{label} Full Model Net'
            ),
        )
    return scenarios


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
        - table.loc['Signal Layer', beta]
    )
    linearity_alpha = table.loc['Integration', alpha] - (
        table.loc['Full Model', alpha]
        - table.loc['Risk Layer', alpha]
        - table.loc['Signal Layer', alpha]
    )
    bars = attribution.annualised_components
    bar_heights = max(abs(bars['Risk Layer Alpha'] - table.loc['Risk Layer', an_alpha]),
                      abs(bars['Signal Layer Alpha'] - table.loc['Signal Layer', an_alpha]),
                      abs(bars['Integration Alpha'] - table.loc['Integration', an_alpha]))
    # excess basis: the signal-layer NAV divided by the benchmark NAV has log return r_S - r_B
    excess_navs = dict(navs)
    excess_navs['signal_layer_nav'] = (
        navs['signal_layer_nav'] / navs['benchmark_nav']
    ).rename('Signal Layer')
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
    beta_shift_signal_layer = excess.loc['Signal Layer', beta] - table.loc['Signal Layer', beta]
    beta_shift_integration = excess.loc['Integration', beta] - table.loc['Integration', beta]
    print(f'linearity: beta residual {linearity_beta:.1e}, alpha residual {linearity_alpha:.1e}')
    print(f'bar heights minus annualised OLS alphas: {bar_heights:.1e}')
    print(
        f'excess basis: max change in alphas, SEs, CIs, p-values {invariance:.1e}; '
        f'beta shifts signal layer {beta_shift_signal_layer:+.6f}, '
        f'integration {beta_shift_integration:+.6f}'
    )
    assert abs(linearity_beta) < 1e-12 and abs(linearity_alpha) < 1e-12
    assert bar_heights < 1e-12
    assert invariance < 1e-10
    assert abs(beta_shift_signal_layer + 1.0) < 1e-10
    assert abs(beta_shift_integration - 1.0) < 1e-10


def report_lag_rule_half_widths(attribution: qis.ModelLayerAlphaBetaAttribution,
                                navs: dict[str, pd.Series]
                                ) -> None:
    """Print the alpha interval half-widths at the default lag count and at the Newey-West rule."""
    n_obs = len(attribution.periodic_returns)
    rule_lags = newey_west_lag_rule(nobs=n_obs)
    rule = qis.compute_model_layer_alpha_beta_attribution(freq=FREQ, hac_lags=rule_lags, **navs)
    layers = ['Risk Layer', 'Signal Layer', 'Integration', 'Full Model']
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
            f'+ Signal-layer\nalpha\nbeta = {table.loc["Signal Layer", beta]:.2f}',
            values['Signal Layer Alpha'],
            'Signal Layer',
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


def _new_exhibit(title: str, subtitle: str) -> tuple[plt.Figure, plt.Axes]:
    """Create one presentation-style attribution exhibit."""
    fig, ax = plt.subplots(1, 1, figsize=(11, 7))
    fig.patch.set_facecolor(REPORT_BACKGROUND)
    ax.set_facecolor(REPORT_BACKGROUND)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.80, bottom=0.20)
    fig.text(
        0.10,
        0.935,
        title,
        color=REPORT_TEXT,
        fontsize=20,
        fontweight='bold',
        ha='left',
    )
    fig.text(0.10, 0.885, subtitle, color='#5F6F78', fontsize=12, ha='left')
    return fig, ax


def _style_exhibit_axes(ax: plt.Axes) -> None:
    """Apply the common documentation-exhibit axis styling."""
    ax.grid(axis='y', color=REPORT_GRID, linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(colors=REPORT_TEXT)
    ax.xaxis.label.set_color(REPORT_TEXT)
    ax.yaxis.label.set_color(REPORT_TEXT)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_color(REPORT_GRID)


def plot_cumulative_alpha(
        attribution: qis.ModelLayerAlphaBetaAttribution,
) -> plt.Figure:
    """Draw additive cumulative alpha paths from the exact periodic bridge components."""
    alpha_returns = attribution.component_returns.loc[:, [
        'Risk Layer Alpha',
        'Signal Layer Alpha',
        'Integration Alpha',
    ]].rename(columns={
        'Risk Layer Alpha': 'Risk-layer alpha',
        'Signal Layer Alpha': 'Signal-layer alpha',
        'Integration Alpha': 'Integration alpha',
    })
    alpha_returns.insert(0, 'Total model alpha', alpha_returns.sum(axis=1))
    initial_date = alpha_returns.index[0] - pd.tseries.frequencies.to_offset(
        attribution.freq
    )
    alpha_paths = pd.concat([
        pd.DataFrame(0.0, index=[initial_date], columns=alpha_returns.columns),
        alpha_returns.cumsum(),
    ])
    np.testing.assert_allclose(
        alpha_paths['Total model alpha'],
        alpha_paths[
            ['Risk-layer alpha', 'Signal-layer alpha', 'Integration alpha']
        ].sum(axis=1),
        atol=1.0e-12,
        rtol=0.0,
    )

    fig, ax = _new_exhibit(
        title='Cumulative model-layer alpha contributions',
        subtitle='Additive full-sample attribution | cumulative monthly log returns, start = 0%',
    )
    qis.plot_time_series(
        df=alpha_paths,
        ax=ax,
        colors=[
            TOTAL_ALPHA_COLOR,
            RISK_ALPHA_COLOR,
            SIGNAL_ALPHA_COLOR,
            INTEGRATION_ALPHA_COLOR,
        ],
        linewidth=1.8,
        x_date_freq='4YE',
        date_format='%Y',
        legend_stats=qis.LegendStats.NONE,
        legend_loc='upper left',
        ylabel='Cumulative alpha contribution',
        var_format='{:.1%}',
        text_weight='normal',
    )
    ax.lines[0].set_linewidth(2.6)
    ax.axhline(0.0, color=REPORT_TEXT, linewidth=0.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    _style_exhibit_axes(ax=ax)
    fig.text(
        0.10,
        0.055,
        'Paths cumulatively sum the beta-adjusted monthly log-return components estimated '
        'with full-sample OLS betas.\nTotal model alpha equals risk-layer + signal-layer + '
        'integration alpha at every date; the paths are explanatory contributions, not NAVs.',
        color='#65747C',
        fontsize=9,
        ha='left',
    )
    return fig


def check_feature_design(
        decomposition: qis.ModelFeatureAlphaBetaAttribution,
) -> None:
    """Check the Shapley estimates against the simulation's independently derived means."""
    expected = pd.DataFrame(
        {
            'Annualised Full Model Net Return': [0.00275, 0.00205],
            'Full Model Alpha': [0.00310, 0.00260],
            'Risk Layer Alpha': [0.00170, 0.00020],
            'Signal Layer Alpha': [-0.00020, 0.00480],
            'Integration Alpha': [0.00160, -0.00240],
        },
        index=[BETA_SPAN_FEATURE, SIGNAL_HORIZON_FEATURE],
    )
    actual = decomposition.summary.loc['Shapley', expected.columns]
    np.testing.assert_allclose(actual, expected, atol=1.0e-10, rtol=0.0)
    np.testing.assert_allclose(
        actual['Full Model Alpha'],
        actual[['Risk Layer Alpha', 'Signal Layer Alpha', 'Integration Alpha']].sum(axis=1),
        atol=1.0e-12,
        rtol=0.0,
    )
    assert float(decomposition.identity_errors.max()) < 1.0e-12
    print('two-feature Shapley effects (annualised percentage points):')
    print((100.0 * actual).round(3).to_string())


def plot_feature_decomposition(
        decomposition: qis.ModelFeatureAlphaBetaAttribution,
) -> plt.Figure:
    """Draw grouped Shapley net-return and layer-alpha effects with HAC intervals."""
    summary = decomposition.summary.loc['Shapley']
    metric_specs = (
        ('Annualised Full Model Net Return', 'Total net\nreturn change'),
        ('Full Model Alpha', 'Total alpha\nchange'),
        ('Risk Layer Alpha', 'Risk-layer\nalpha'),
        ('Signal Layer Alpha', 'Signal-layer\nalpha'),
        ('Integration Alpha', 'Integration\nalpha'),
    )
    x_positions = np.arange(len(metric_specs), dtype=float)
    width = 0.34
    fig, ax = _new_exhibit(
        title='Two-feature model sensitivity',
        subtitle=(
            'Annualised net-return and OLS alpha effects | monthly log returns | '
            'order-independent Shapley allocation'
        ),
    )
    for feature_index, feature in enumerate(
            (BETA_SPAN_FEATURE, SIGNAL_HORIZON_FEATURE)
    ):
        offset = (-0.5 if feature_index == 0 else 0.5) * width
        positions = x_positions + offset
        values = np.array([
            float(summary.loc[feature, metric]) for metric, _ in metric_specs
        ])
        ci_low = np.array([
            float(summary.loc[feature, f'{metric} CI Low'])
            for metric, _ in metric_specs
        ])
        ci_high = np.array([
            float(summary.loc[feature, f'{metric} CI High'])
            for metric, _ in metric_specs
        ])
        interval_midpoints = 0.5 * (ci_low + ci_high)
        np.testing.assert_allclose(values, interval_midpoints, atol=1.0e-12, rtol=0.0)
        ax.bar(
            positions,
            values,
            width=width * 0.88,
            color=FEATURE_COLORS[feature],
            label=FEATURE_LABELS[feature],
            zorder=3,
        )
        ax.errorbar(
            positions,
            values,
            yerr=np.vstack([values - ci_low, ci_high - values]),
            fmt='none',
            color='black',
            ecolor='black',
            elinewidth=1.3,
            capsize=4.0,
            zorder=5,
        )
        ax.scatter(
            positions,
            interval_midpoints,
            s=24.0,
            color='black',
            edgecolors=REPORT_BACKGROUND,
            linewidths=0.8,
            zorder=6,
        )
        for position, value, low, high in zip(positions, values, ci_low, ci_high):
            label_anchor = high if value >= 0.0 else low
            ax.annotate(
                f'{value:+.2%}',
                xy=(position, label_anchor),
                xytext=(0, 6 if value >= 0.0 else -8),
                textcoords='offset points',
                ha='center',
                va='bottom' if value >= 0.0 else 'top',
                color=REPORT_TEXT,
                fontsize=8.5,
                fontweight='bold',
            )
    ax.set_xticks(x_positions, [label for _, label in metric_specs])
    ax.axhline(0.0, color=REPORT_TEXT, linewidth=0.8)
    ax.set_ylabel('Annualised effect')
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(frameon=False, ncols=2, loc='upper center')
    ax.margins(x=0.08, y=0.28)
    _style_exhibit_axes(ax=ax)
    fig.text(
        0.10,
        0.045,
        f'Whiskers are {decomposition.confidence_level:.0%} Bartlett HAC('
        f'{decomposition.hac_lags}) intervals: intercept-only for net-return changes and '
        'benchmark OLS for alpha.\nEach feature receives half of the two-feature interaction; '
        'total alpha is the exact sum of risk, signal, and integration alpha.',
        color='#65747C',
        fontsize=9,
        ha='left',
    )
    return fig


def run_example(
        save_figure_path: Optional[str] = None,
        output_dir: Optional[Path] = None,
) -> qis.ModelLayerAlphaBetaAttribution:
    """Run the seeded layer and feature attributions and draw all three exhibits.

    Args:
        save_figure_path: Optional legacy destination for the return-bridge figure.
        output_dir: Optional directory for all three documentation figures.

    Returns:
        The base model-layer attribution used by the bridge and cumulative-alpha exhibits.
    """
    navs = simulate_layer_navs()
    attribution = qis.compute_model_layer_alpha_beta_attribution(freq=FREQ, **navs)
    feature_decomposition = qis.compute_model_feature_alpha_beta_attribution(
        scenario_layer_navs=simulate_feature_scenarios(navs=navs),
        freq=FREQ,
    )
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
    check_feature_design(decomposition=feature_decomposition)
    bridge_figure = plot_return_bridge(attribution=attribution)
    cumulative_figure = plot_cumulative_alpha(attribution=attribution)
    feature_figure = plot_feature_decomposition(
        decomposition=feature_decomposition
    )
    if save_figure_path is not None:
        bridge_figure.savefig(save_figure_path, dpi=150, bbox_inches='tight')
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        documentation_figures = {
            'model_layer_attribution_simulated.png': bridge_figure,
            'model_layer_attribution_cumulative_alpha_simulated.png': cumulative_figure,
            'model_feature_attribution_simulated.png': feature_figure,
        }
        for file_name, figure in documentation_figures.items():
            figure.savefig(output_dir.joinpath(file_name), dpi=150, bbox_inches='tight')
    if save_figure_path is None and output_dir is None:
        plt.show()
    else:
        for figure in (bridge_figure, cumulative_figure, feature_figure):
            plt.close(figure)
    return attribution


if __name__ == '__main__':
    run_example()

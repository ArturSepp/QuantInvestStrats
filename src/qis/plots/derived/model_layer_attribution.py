"""Plots for current and rolling EWMA model-layer attribution.

The numerical work is deliberately outside this module.  Return components, current EWMA
regressions, Bartlett-HAC intervals and effective sample sizes come from
``ModelLayerEwmaRegressionAttribution``. Common-denominator Sharpe contributions come from
``compute_model_layer_ewma_sharpe_contributions``. The functions here only validate those
labelled outputs and render them.

Both bridges use the same ordered layers: benchmark is shown as a reference, systematic return
is the first model stage, and risk-layer, signal-layer and integration effects are added in that
order. Sharpe contributions divide each model return component by one common full-model EWMA
volatility, so they remain additive and retain the signs of the corresponding alphas.
"""

from __future__ import annotations

import textwrap
from collections.abc import Mapping
from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

if TYPE_CHECKING:
    from qis.perfstats.model_layer_attribution import ModelLayerEwmaRegressionAttribution


_DEFAULT_COLORS = {
    'Benchmark': '#8E9A9E',
    'Systematic': '#3B5F7A',
    'Risk Layer': '#5B9A91',
    'Signal Layer': '#D99D1E',
    'Integration': '#8A6F3D',
    'Trading Cost Drag': '#B45B5B',
    'Full Model Gross': '#126B52',
    'Full Model Net': '#126B52',
}
_NEGATIVE_COLOR = '#B45B5B'
_TEXT_COLOR = '#23313B'
_GRID_COLOR = '#D8D4CB'
_BRIDGE_WIDTH = 0.62


def _updated_mapping(
        defaults: Mapping[str, str],
        updates: Optional[Mapping[str, str]],
        name: str,
) -> dict[str, str]:
    """Return validated display overrides applied to a copy of ``defaults``."""
    output = dict(defaults)
    if updates is None:
        return output
    unknown = sorted(set(updates).difference(output))
    if unknown:
        raise ValueError(f'{name} contains unsupported keys {unknown!r}')
    output.update(updates)
    return output


def _display_labels(
        model_name: str,
        benchmark_label: str,
        labels: Optional[Mapping[str, str]],
) -> dict[str, str]:
    """Build semantic bridge labels before beta annotations are appended."""
    defaults = {
        'Benchmark': benchmark_label,
        'Systematic': 'Systematic\nEWMA return',
        'Risk Layer': '+ Risk-layer\nEWMA alpha',
        'Signal Layer': '+ Signal-layer\nEWMA alpha',
        'Integration': '+ Integration\nEWMA alpha',
        'Trading Cost Drag': 'Trading-cost\ndrag',
        'Full Model Gross': f'{model_name}\ngross',
        'Full Model Net': f'{model_name}\nnet',
    }
    return _updated_mapping(defaults=defaults, updates=labels, name='labels')


def _validate_attribution(
        attribution: ModelLayerEwmaRegressionAttribution,
) -> None:
    """Validate the public attribution object and the labelled values used by both plots."""
    from qis.perfstats.model_layer_attribution import ModelLayerEwmaRegressionAttribution

    if not isinstance(attribution, ModelLayerEwmaRegressionAttribution):
        raise TypeError(
            'attribution must be ModelLayerEwmaRegressionAttribution, got '
            f'{type(attribution)!r}'
        )
    if attribution.annualised_components.empty:
        raise ValueError('attribution.annualised_components must not be empty')
    if attribution.regression_table.empty:
        raise ValueError('attribution.regression_table must not be empty')
    if attribution.periodic_returns.empty or attribution.component_returns.empty:
        raise ValueError('attribution return histories must not be empty')
    if attribution.span <= 0:
        raise ValueError(f'attribution.span must be positive, got {attribution.span!r}')
    if attribution.hac_lags < 0:
        raise ValueError(
            f'attribution.hac_lags must be non-negative, got {attribution.hac_lags!r}'
        )
    if not 0.0 < attribution.confidence_level < 1.0:
        raise ValueError(
            'attribution.confidence_level must be between zero and one, got '
            f'{attribution.confidence_level!r}'
        )
    if not np.isfinite(attribution.effective_nobs) or attribution.effective_nobs <= 0.0:
        raise ValueError(
            'attribution.effective_nobs must be finite and positive, got '
            f'{attribution.effective_nobs!r}'
        )


def _get_beta(attribution: ModelLayerEwmaRegressionAttribution, layer: str) -> float:
    """Return one finite current EWMA regression beta."""
    from qis.perfstats.config import PerfStat

    beta_column = PerfStat.BETA.to_str()
    try:
        beta = float(attribution.regression_table.loc[layer, beta_column])
    except KeyError as exception:
        raise ValueError(
            f'attribution regression table is missing beta for {layer!r}'
        ) from exception
    if not np.isfinite(beta):
        raise ValueError(f'attribution beta for {layer!r} must be finite, got {beta!r}')
    return beta


def _get_r_squared(attribution: ModelLayerEwmaRegressionAttribution, layer: str) -> float:
    """Return one finite current EWMA regression R-squared statistic."""
    from qis.perfstats.config import PerfStat

    r_squared_column = PerfStat.R2.to_str()
    try:
        r_squared = float(attribution.regression_table.loc[layer, r_squared_column])
    except KeyError as exception:
        raise ValueError(
            f'attribution regression table is missing R-squared for {layer!r}'
        ) from exception
    if not np.isfinite(r_squared):
        raise ValueError(
            f'attribution R-squared for {layer!r} must be finite, got {r_squared!r}'
        )
    return r_squared


def _beta_label(label: str, beta: float) -> str:
    """Append the current beta estimate as the third label row."""
    return f'{label}\n$\\hat{{\\beta}}$ = {beta:.2f}'


def _regression_label(label: str, beta: float, r_squared: float) -> str:
    """Append current beta and R-squared estimates below an alpha-bar label."""
    return f'{_beta_label(label=label, beta=beta)}\n$R^2$ = {r_squared:.2f}'


def _ewma_span_label(attribution: ModelLayerEwmaRegressionAttribution) -> str:
    """Return a frequency-aware adjective such as ``36-month`` or ``12-quarter``."""
    frequency = str(attribution.freq).upper().split('-', maxsplit=1)[0]
    if frequency in {'M', 'ME', 'MS', 'BM', 'BME', 'BMS', 'CBM', 'CBME', 'CBMS'}:
        unit = 'month'
    elif frequency in {'Q', 'QE', 'QS', 'BQ', 'BQE', 'BQS'}:
        unit = 'quarter'
    elif frequency in {'Y', 'YE', 'YS', 'A', 'AS', 'BA', 'BAS', 'BY', 'BYE', 'BYS'}:
        unit = 'year'
    elif frequency == 'W':
        unit = 'week'
    elif frequency in {'D', 'B', 'C'}:
        unit = 'day'
    else:
        unit = 'period'
    return f'{attribution.span}-{unit}'


def _new_axes(
        ax: Optional[plt.Axes],
        detailed_mode: bool,
) -> tuple[Optional[Figure], plt.Axes]:
    """Create the standard bridge canvas, or reuse the supplied axis."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(11.0, 7.0))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        if detailed_mode:
            fig.subplots_adjust(left=0.09, right=0.98, top=0.82, bottom=0.30)
        else:
            fig.subplots_adjust(left=0.09, right=0.98, top=0.96, bottom=0.18)
        return fig, ax
    return None, ax


def _style_axes(ax: plt.Axes) -> None:
    """Apply restrained, report-neutral styling shared by the two bridges."""
    ax.set_facecolor('white')
    ax.grid(axis='y', color=_GRID_COLOR, linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(colors=_TEXT_COLOR)
    ax.xaxis.label.set_color(_TEXT_COLOR)
    ax.yaxis.label.set_color(_TEXT_COLOR)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_color(_GRID_COLOR)


def _add_details(
        ax: plt.Axes,
        title: str,
        subtitle: str,
        note: str,
        detailed_mode: bool,
) -> None:
    """Add title, subtitle and methodology note only in detailed mode."""
    if not detailed_mode:
        return
    wrapped_title = textwrap.fill(
        title,
        width=62,
        break_long_words=False,
        break_on_hyphens=False,
    )
    wrapped_note = textwrap.fill(
        note,
        width=115,
        break_long_words=False,
        break_on_hyphens=False,
    )
    ax.set_title(
        wrapped_title,
        loc='left',
        color=_TEXT_COLOR,
        fontsize=18,
        fontweight='bold',
        pad=34,
    )
    ax.text(
        0.0,
        1.025,
        subtitle,
        transform=ax.transAxes,
        color='#5F6F78',
        fontsize=11,
        ha='left',
        va='bottom',
    )
    ax.text(
        0.0,
        -0.25,
        wrapped_note,
        transform=ax.transAxes,
        color='#5F6F78',
        fontsize=9,
        ha='left',
        va='top',
    )


def _draw_bar(
        ax: plt.Axes,
        position: int,
        start: float,
        contribution: float,
        color: str,
        is_contribution: bool,
        value_format: str,
        confidence_interval: Optional[tuple[float, float]] = None,
) -> None:
    """Draw one floating or absolute bridge bar and its optional translated interval."""
    end = start + contribution
    ax.bar(
        position,
        abs(contribution),
        bottom=min(start, end),
        width=_BRIDGE_WIDTH,
        color=color if contribution >= 0.0 else _NEGATIVE_COLOR,
    )
    annotation_level = end
    if confidence_interval is not None:
        ci_low, ci_high = confidence_interval
        midpoint = 0.5 * (ci_low + ci_high)
        if not np.isclose(contribution, midpoint, atol=1.0e-10, rtol=0.0):
            raise RuntimeError(
                f'alpha bar height {contribution:.6e} differs from its HAC interval midpoint '
                f'{midpoint:.6e}'
            )
        lower_error = contribution - ci_low
        upper_error = ci_high - contribution
        if lower_error < 0.0 or upper_error < 0.0:
            raise RuntimeError('alpha estimate falls outside its HAC confidence interval')
        ax.errorbar(
            position,
            end,
            yerr=np.array([[lower_error], [upper_error]]),
            fmt='none',
            ecolor='black',
            elinewidth=1.2,
            capsize=5.0,
            capthick=1.2,
            zorder=5,
        )
        ax.scatter(position, end, s=20.0, color='black', zorder=6)
        annotation_level = start + (ci_high if contribution >= 0.0 else ci_low)
    label = value_format.format(contribution)
    if is_contribution and not label.startswith('-'):
        label = f'+{label}'
    ax.annotate(
        label,
        xy=(position, annotation_level),
        xytext=(0, 5 if contribution >= 0.0 else -14),
        textcoords='offset points',
        ha='center',
        va='bottom' if contribution >= 0.0 else 'top',
        color=_TEXT_COLOR,
        fontsize=10.5,
        fontweight='bold',
    )


def _draw_split_endpoint(
        ax: plt.Axes,
        position: int,
        systematic_return: float,
        cost_drag: Optional[float],
        total_alpha: float,
        total_return: float,
        systematic_color: str,
        cost_color: str,
        alpha_color: str,
        confidence_interval: tuple[float, float],
) -> None:
    """Draw systematic, optional cost and gross alpha segments with translated inference."""
    cost_value = 0.0 if cost_drag is None else float(cost_drag)
    if not np.isclose(
            systematic_return + cost_value + total_alpha,
            total_return,
            atol=1.0e-12,
            rtol=0.0,
    ):
        raise RuntimeError('split endpoint does not reconstruct the full-model return')
    ci_low, ci_high = confidence_interval
    midpoint = 0.5 * (ci_low + ci_high)
    if not np.isclose(total_alpha, midpoint, atol=1.0e-10, rtol=0.0):
        raise RuntimeError(
            f'total alpha {total_alpha:.6e} differs from its HAC interval midpoint '
            f'{midpoint:.6e}'
        )
    lower_error = total_alpha - ci_low
    upper_error = ci_high - total_alpha
    if lower_error < 0.0 or upper_error < 0.0:
        raise RuntimeError('total alpha falls outside its HAC confidence interval')

    segments = [(0.0, systematic_return, systematic_color, 3)]
    if cost_drag is not None:
        segments.append((systematic_return, cost_value, cost_color, 8))
    alpha_start = systematic_return + cost_value
    segments.append((alpha_start, total_alpha, alpha_color, 4))
    for start, contribution, color, zorder in segments:
        end = start + contribution
        ax.bar(
            position,
            abs(contribution),
            bottom=min(start, end),
            width=_BRIDGE_WIDTH,
            color=color if contribution >= 0.0 else _NEGATIVE_COLOR,
            zorder=zorder,
        )
    alpha_label = f'{total_alpha:.1%}'
    if not alpha_label.startswith('-'):
        alpha_label = f'+{alpha_label}'
    alpha_fill_color = alpha_color if total_alpha >= 0.0 else _NEGATIVE_COLOR
    ax.text(
        position,
        0.5 * systematic_return,
        f'Systematic\n{systematic_return:.1%}',
        ha='center',
        va='center',
        color='white',
        fontsize=8.5,
        fontweight='bold',
        zorder=4,
    )
    if cost_drag is not None:
        ax.annotate(
            f'Cost\n{cost_value:.1%}',
            xy=(position, systematic_return + 0.5 * cost_value),
            xytext=(28, -2),
            textcoords='offset points',
            ha='left',
            va='center',
            color=cost_color,
            fontsize=8.0,
            fontweight='bold',
            arrowprops={'arrowstyle': '-', 'color': cost_color, 'linewidth': 0.9},
            annotation_clip=False,
            zorder=8,
        )
    ax.text(
        position,
        alpha_start + 0.5 * total_alpha,
        f'Total alpha (gross)\n{alpha_label}',
        ha='center',
        va='center',
        color='white',
        fontsize=8.5,
        fontweight='bold',
        bbox={
            'boxstyle': 'square,pad=0.1',
            'facecolor': alpha_fill_color,
            'edgecolor': 'none',
        },
        zorder=7,
    )
    ax.errorbar(
        position,
        total_return,
        yerr=np.array([[lower_error], [upper_error]]),
        fmt='none',
        ecolor='black',
        elinewidth=1.2,
        capsize=5.0,
        capthick=1.2,
        zorder=9,
    )
    ax.scatter(position, total_return, s=20.0, color='black', zorder=10)
    annotation_level = alpha_start + (
        ci_high if total_alpha >= 0.0 else ci_low
    )
    ax.annotate(
        f'{total_return:.1%}',
        xy=(position, annotation_level),
        xytext=(0, 5 if total_alpha >= 0.0 else -14),
        textcoords='offset points',
        ha='center',
        va='bottom' if total_alpha >= 0.0 else 'top',
        color=_TEXT_COLOR,
        fontsize=10.5,
        fontweight='bold',
    )


def _draw_connectors(
        ax: plt.Axes,
        connectors: list[tuple[int, int, float]],
) -> None:
    """Draw dashed horizontal connectors between adjacent bridge steps."""
    for first, second, level in connectors:
        ax.plot(
            [first + _BRIDGE_WIDTH / 2.0, second - _BRIDGE_WIDTH / 2.0],
            [level, level],
            color='#A7ABA8',
            linewidth=1.0,
            linestyle='--',
        )


def _current_stage_sharpes(
        attribution: ModelLayerEwmaRegressionAttribution,
) -> pd.Series:
    """Return the latest labelled stage Sharpes from the canonical QIS computation."""
    computed = _compute_ewma_stage_sharpes(attribution, norm_type=2)
    if isinstance(computed, pd.DataFrame):
        if computed.empty:
            raise ValueError('EWMA stage Sharpe history must not be empty')
        sharpes = computed.iloc[-1]
    elif isinstance(computed, pd.Series):
        sharpes = computed
    else:
        raise TypeError(
            'compute_model_layer_ewma_stage_sharpes must return a Series or DataFrame, got '
            f'{type(computed)!r}'
        )
    aliases = {
        'Benchmark': ('Benchmark', 'Static Benchmark'),
        'Systematic': ('Systematic',),
        'Risk Layer': ('Risk Layer',),
        'Signal Layer': ('Signal Layer',),
        'Full Model Gross': ('Full Model Gross', 'Full Model'),
        'Full Model Net': ('Full Model Net',),
    }
    normalised: dict[str, float] = {}
    for stage, candidates in aliases.items():
        candidate = next((name for name in candidates if name in sharpes.index), None)
        if candidate is not None:
            normalised[stage] = float(sharpes[candidate])
    required = ['Benchmark', 'Systematic', 'Risk Layer', 'Signal Layer', 'Full Model Gross']
    missing = [stage for stage in required if stage not in normalised]
    if missing:
        raise ValueError(f'EWMA stage Sharpes are missing {missing!r}')
    output = pd.Series(normalised, dtype=float)
    if not np.isfinite(output.to_numpy(dtype=float)).all():
        raise ValueError('EWMA stage Sharpes must all be finite')
    return output


def _compute_ewma_stage_sharpes(
        attribution: ModelLayerEwmaRegressionAttribution,
        norm_type: int,
) -> pd.DataFrame:
    """Import and call the numerical Sharpe API lazily to avoid package import cycles."""
    from qis.perfstats.model_layer_attribution import compute_model_layer_ewma_stage_sharpes

    return compute_model_layer_ewma_stage_sharpes(attribution, norm_type=norm_type)


def _compute_ewma_sharpe_contributions(
        attribution: ModelLayerEwmaRegressionAttribution,
) -> pd.Series:
    """Call the numerical common-denominator Sharpe API without creating an import cycle."""
    from qis.perfstats.model_layer_attribution import (
        compute_model_layer_ewma_sharpe_contributions,
    )

    return compute_model_layer_ewma_sharpe_contributions(attribution=attribution)


def _compute_rolling_ewma_regression_alpha(
        attribution: ModelLayerEwmaRegressionAttribution,
) -> pd.DataFrame:
    """Call the numerical rolling EWMA-WLS alpha API without creating an import cycle."""
    from qis.perfstats.model_layer_attribution import (
        compute_model_layer_rolling_ewma_regression_alpha,
    )

    return compute_model_layer_rolling_ewma_regression_alpha(attribution=attribution)


def plot_model_layer_ewma_return_bridge(
        attribution: ModelLayerEwmaRegressionAttribution,
        model_name: str = 'Model',
        benchmark_label: str = 'Benchmark',
        labels: Optional[Mapping[str, str]] = None,
        colors: Optional[Mapping[str, str]] = None,
        detailed_mode: bool = True,
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
) -> Optional[Figure]:
    """Plot the current annualised EWMA return bridge with Bartlett-HAC intervals.

    The final endpoint is split into gross systematic return, optional realised cost drag and
    gross total alpha. The gross-alpha interval is translated by the preceding segments, so its
    midpoint is the displayed gross or net total.

    Args:
        attribution: Current EWMA regression attribution computed by QIS.
        model_name: Model name used in the title and endpoint labels.
        benchmark_label: Display label for the benchmark reference bar.
        labels: Optional overrides keyed by ``Benchmark``, ``Systematic``, ``Risk Layer``,
            ``Signal Layer``, ``Integration``, ``Trading Cost Drag``, ``Full Model Gross`` or
            ``Full Model Net``.
        colors: Optional colour overrides using the same semantic keys as ``labels``.
        detailed_mode: Whether to draw the title, subtitle and methodology note.
        title: Detailed-mode title. None uses a frequency-aware EWMA-span title.
        ax: Existing axis. None creates a new report-sized figure.

    Returns:
        The created figure, or None when drawing on a supplied axis.

    Raises:
        TypeError: If ``attribution`` or a computed Sharpe container has an unsupported type.
        ValueError: If required labelled inputs or estimator settings are missing or invalid.
        RuntimeError: If the return bridge, net-cost identity, or HAC midpoint does not reconcile.
    """
    from qis.perfstats.config import PerfStat
    from qis.perfstats.model_layer_attribution import (
        ALPHA_AN_CI_HIGH_COLUMN,
        ALPHA_AN_CI_LOW_COLUMN,
    )

    _validate_attribution(attribution=attribution)
    display_labels = _display_labels(
        model_name=model_name,
        benchmark_label=benchmark_label,
        labels=labels,
    )
    display_colors = _updated_mapping(
        defaults=_DEFAULT_COLORS,
        updates=colors,
        name='colors',
    )
    values = attribution.annualised_components
    required_components = [
        'Benchmark Return',
        'Systematic Return',
        'Risk Layer Alpha',
        'Signal Layer Alpha',
        'Integration Alpha',
        'Full Model Return',
    ]
    missing = [component for component in required_components if component not in values.index]
    if missing:
        raise ValueError(f'annualised components are missing {missing!r}')
    selected = values.loc[required_components].astype(float)
    if not np.isfinite(selected.to_numpy()).all():
        raise ValueError('annualised bridge components must all be finite')

    benchmark_return = float(values['Benchmark Return'])
    systematic_return = float(values['Systematic Return'])
    risk_alpha = float(values['Risk Layer Alpha'])
    signal_alpha = float(values['Signal Layer Alpha'])
    integration_alpha = float(values['Integration Alpha'])
    full_return = float(values['Full Model Return'])
    reconstructed = systematic_return + risk_alpha + signal_alpha + integration_alpha
    if not np.isclose(reconstructed, full_return, atol=1.0e-12, rtol=0.0):
        raise RuntimeError('annualised EWMA return bridge does not reconstruct full-model return')

    has_net_return = 'Full Model Net Return' in values.index
    has_cost_drag = 'Trading Cost Drag' in values.index
    if has_net_return != has_cost_drag:
        raise ValueError(
            'Full Model Net Return and Trading Cost Drag must either both be present or absent'
        )
    cost_drag = float(values['Trading Cost Drag']) if has_net_return else None
    net_return = float(values['Full Model Net Return']) if has_net_return else None
    if has_net_return:
        if not np.isfinite([cost_drag, net_return]).all():
            raise ValueError('net return and trading-cost drag must be finite')
        if not np.isclose(full_return + cost_drag, net_return, atol=1.0e-12, rtol=0.0):
            raise RuntimeError('annualised EWMA trading-cost bridge does not reconcile')

    beta_full = _get_beta(attribution=attribution, layer='Full Model')
    beta_risk = _get_beta(attribution=attribution, layer='Risk Layer')
    beta_signal = _get_beta(attribution=attribution, layer='Signal Layer')
    beta_integration = _get_beta(attribution=attribution, layer='Integration')
    r_squared_full = _get_r_squared(attribution=attribution, layer='Full Model')
    r_squared_risk = _get_r_squared(attribution=attribution, layer='Risk Layer')
    r_squared_signal = _get_r_squared(attribution=attribution, layer='Signal Layer')
    r_squared_integration = _get_r_squared(attribution=attribution, layer='Integration')
    alpha_column = PerfStat.ALPHA_AN.to_str()
    confidence_intervals: dict[str, tuple[float, float]] = {}
    for component, layer, contribution in (
            ('Risk Layer', 'Risk Layer', risk_alpha),
            ('Signal Layer', 'Signal Layer', signal_alpha),
            ('Integration', 'Integration', integration_alpha),
    ):
        try:
            alpha = float(attribution.regression_table.loc[layer, alpha_column])
            ci_low = float(attribution.regression_table.loc[layer, ALPHA_AN_CI_LOW_COLUMN])
            ci_high = float(attribution.regression_table.loc[layer, ALPHA_AN_CI_HIGH_COLUMN])
        except KeyError as exception:
            raise ValueError(
                f'attribution regression table is missing alpha inference for {layer!r}'
            ) from exception
        if not np.isfinite([alpha, ci_low, ci_high]).all():
            raise ValueError(f'alpha inference for {layer!r} must be finite')
        if not np.isclose(contribution, alpha, atol=1.0e-10, rtol=0.0):
            raise RuntimeError(
                f'{component} bar height {contribution:.6e} differs from its EWMA alpha '
                f'{alpha:.6e}'
            )
        confidence_intervals[component] = (ci_low, ci_high)

    fig, ax = _new_axes(ax=ax, detailed_mode=detailed_mode)
    bars = [
        (0, 0.0, benchmark_return, 'Benchmark', False, None),
        (2, 0.0, systematic_return, 'Systematic', False, None),
        (
            3,
            systematic_return,
            risk_alpha,
            'Risk Layer',
            True,
            confidence_intervals['Risk Layer'],
        ),
        (
            4,
            systematic_return + risk_alpha,
            signal_alpha,
            'Signal Layer',
            True,
            confidence_intervals['Signal Layer'],
        ),
        (
            5,
            systematic_return + risk_alpha + signal_alpha,
            integration_alpha,
            'Integration',
            True,
            confidence_intervals['Integration'],
        ),
    ]
    tick_labels = [
        display_labels['Benchmark'],
        _beta_label(display_labels['Systematic'], beta_full),
        _regression_label(display_labels['Risk Layer'], beta_risk, r_squared_risk),
        _regression_label(display_labels['Signal Layer'], beta_signal, r_squared_signal),
        _regression_label(
            display_labels['Integration'], beta_integration, r_squared_integration
        ),
    ]
    if has_net_return:
        endpoint_position = 7
        endpoint_return = net_return
        bars.append((6, full_return, cost_drag, 'Trading Cost Drag', True, None))
        tick_labels.extend([
            display_labels['Trading Cost Drag'],
            _regression_label(display_labels['Full Model Net'], beta_full, r_squared_full),
        ])
    else:
        endpoint_position = 6
        endpoint_return = full_return
        tick_labels.append(
            _regression_label(display_labels['Full Model Gross'], beta_full, r_squared_full)
        )

    try:
        endpoint_alpha = float(attribution.regression_table.loc['Full Model', alpha_column])
        endpoint_ci = (
            float(attribution.regression_table.loc['Full Model', ALPHA_AN_CI_LOW_COLUMN]),
            float(attribution.regression_table.loc['Full Model', ALPHA_AN_CI_HIGH_COLUMN]),
        )
    except KeyError as exception:
        raise ValueError(
            "attribution regression table is missing endpoint inference for 'Full Model'"
        ) from exception
    if not np.isfinite([endpoint_alpha, *endpoint_ci]).all():
        raise ValueError("endpoint inference for 'Full Model' must be finite")
    endpoint_systematic = beta_full * benchmark_return
    endpoint_cost = cost_drag if has_net_return else None
    endpoint_cost_value = 0.0 if endpoint_cost is None else endpoint_cost
    if not np.isclose(
            endpoint_systematic + endpoint_alpha + endpoint_cost_value,
            endpoint_return,
            atol=1.0e-10,
            rtol=0.0,
    ):
        raise RuntimeError(
            'gross systematic return, realised costs and gross alpha do not reconstruct the '
            'endpoint return'
        )

    for position, start, contribution, key, is_contribution, interval in bars:
        _draw_bar(
            ax=ax,
            position=position,
            start=start,
            contribution=contribution,
            color=display_colors[key],
            is_contribution=is_contribution,
            value_format='{:.1%}',
            confidence_interval=interval,
        )
    _draw_split_endpoint(
        ax=ax,
        position=endpoint_position,
        systematic_return=endpoint_systematic,
        cost_drag=endpoint_cost,
        total_alpha=endpoint_alpha,
        total_return=endpoint_return,
        systematic_color=display_colors['Systematic'],
        cost_color=display_colors['Trading Cost Drag'],
        alpha_color=display_colors['Full Model Gross'],
        confidence_interval=endpoint_ci,
    )
    connectors = [
        (2, 3, systematic_return),
        (3, 4, systematic_return + risk_alpha),
        (4, 5, systematic_return + risk_alpha + signal_alpha),
    ]
    if has_net_return:
        connectors.extend([(5, 6, full_return), (6, 7, net_return)])
    else:
        connectors.append((5, 6, full_return))
    _draw_connectors(ax=ax, connectors=connectors)

    ax.axvline(1.0, color=_GRID_COLOR, linewidth=1.0, linestyle=':')
    ax.axhline(0.0, color=_TEXT_COLOR, linewidth=0.8)
    ax.set_xticks([bar[0] for bar in bars] + [endpoint_position], tick_labels)
    ax.set_ylabel('Current annualised EWMA log return', color=_TEXT_COLOR)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.margins(x=0.04, y=0.18)
    _style_axes(ax=ax)

    final_date = pd.Timestamp(attribution.periodic_returns.index[-1])
    span_label = _ewma_span_label(attribution=attribution)
    plot_title = title or f'{model_name.upper()} current {span_label} EWMA attribution'
    _add_details(
        ax=ax,
        title=plot_title,
        subtitle=(
            f'Annualised EWMA log-return contributions through {final_date:%d %b %Y} | '
            f'{attribution.confidence_level:.0%} Bartlett HAC({attribution.hac_lags})'
        ),
        note=(
            f'Black whiskers show {attribution.confidence_level:.0%} EWMA Bartlett '
            f'HAC({attribution.hac_lags}) intervals; Kish effective sample size '
            f'{attribution.effective_nobs:.1f} from {attribution.nobs} returns. '
            'Beta and R-squared labels are current EWMA regression estimates. Integration is '
            'the exact residual after systematic, risk-layer and signal-layer contributions. '
            'The final bar shows gross-model systematic return, realised cost drag and gross '
            'total alpha; its fourth whisker is the gross total-alpha interval.'
        ),
        detailed_mode=detailed_mode,
    )
    return fig


def plot_model_layer_rolling_ewma_regression_alpha(
        attribution: ModelLayerEwmaRegressionAttribution,
        model_name: str = 'Model',
        colors: Optional[Mapping[str, str]] = None,
        detailed_mode: bool = True,
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
) -> Optional[Figure]:
    """Plot expanding-prefix EWMA-WLS annualised alpha paths.

    The plotted estimates are descriptive contemporaneous regressions. They are distinct from
    lagged-beta realised alpha, and their final values exactly match the gross current EWMA
    attribution bars.

    Args:
        attribution: Current EWMA regression attribution defining the sample and estimator.
        model_name: Model name used in the title.
        colors: Optional semantic colour overrides; supported keys are documented by the return
            bridge.
        detailed_mode: Whether to draw the title, subtitle and methodology note.
        title: Detailed-mode title. None uses a frequency-aware rolling EWMA-WLS title.
        ax: Existing axis. None creates a new report-sized figure.

    Returns:
        The created figure, or None when drawing on a supplied axis.

    Raises:
        TypeError: If ``attribution`` is not a model-layer EWMA regression result.
        ValueError: If the rolling alpha output is missing a required series.
    """
    _validate_attribution(attribution=attribution)
    display_colors = _updated_mapping(
        defaults=_DEFAULT_COLORS,
        updates=colors,
        name='colors',
    )
    paths = _compute_rolling_ewma_regression_alpha(attribution=attribution)
    series_specs = (
        ('Total model alpha', 'Total Model Alpha', display_colors['Full Model Gross'], 2.6),
        ('Risk-layer alpha', 'Risk Layer Alpha', display_colors['Risk Layer'], 1.8),
        ('Signal-layer alpha', 'Signal Layer Alpha', display_colors['Signal Layer'], 1.8),
        ('Integration alpha', 'Integration Alpha', display_colors['Integration'], 1.8),
    )
    missing = [column for _, column, _, _ in series_specs if column not in paths.columns]
    if missing:
        raise ValueError(f'rolling EWMA-WLS alpha paths are missing {missing!r}')

    fig, ax = _new_axes(ax=ax, detailed_mode=detailed_mode)
    ax.axhline(0.0, color=_TEXT_COLOR, linewidth=0.8)
    for label, column, color, linewidth in series_specs:
        ax.plot(
            paths.index,
            paths[column],
            label=label,
            color=color,
            linewidth=linewidth,
        )
    ax.set_ylabel('Rolling annualised EWMA-WLS alpha', color=_TEXT_COLOR)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(frameon=False, loc='best')
    ax.margins(x=0.0, y=0.10)
    _style_axes(ax=ax)

    final_date = pd.Timestamp(paths.index[-1])
    span_label = _ewma_span_label(attribution=attribution)
    plot_title = title or (
        f'{model_name.upper()} rolling {span_label} EWMA-WLS annualised alpha'
    )
    _add_details(
        ax=ax,
        title=plot_title,
        subtitle=(
            f'Expanding-prefix {span_label} geometric regressions through '
            f'{final_date:%d %b %Y}'
        ),
        note=(
            'Each date refits the same descriptive EWMA-WLS regression on history available '
            'through that date. Total model alpha equals risk-layer alpha + signal-layer alpha '
            '+ integration alpha at every point. The final values equal the gross alpha bars in '
            'the current EWMA attribution; these are contemporaneous estimates, not lagged '
            'out-of-sample residuals.'
        ),
        detailed_mode=detailed_mode,
    )
    return fig


def plot_model_layer_ewma_sharpe_bridge(
        attribution: ModelLayerEwmaRegressionAttribution,
        model_name: str = 'Model',
        benchmark_label: str = 'Benchmark',
        labels: Optional[Mapping[str, str]] = None,
        colors: Optional[Mapping[str, str]] = None,
        detailed_mode: bool = True,
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
) -> Optional[Figure]:
    """Plot additive current EWMA log-return Sharpe contributions.

    Args:
        attribution: Current EWMA regression attribution computed by QIS.
        model_name: Model name used in the title and endpoint labels.
        benchmark_label: Display label for the benchmark reference bar.
        labels: Optional semantic label overrides; keys are documented by the return bridge.
        colors: Optional semantic colour overrides; keys are documented by the return bridge.
        detailed_mode: Whether to draw the title, subtitle and methodology note.
        title: Detailed-mode title. None uses the standard risk-and-signal title.
        ax: Existing axis. None creates a new report-sized figure.

    Returns:
        The created figure, or None when drawing on a supplied axis.

    Raises:
        TypeError: If ``attribution`` has an unsupported type.
        ValueError: If required labelled inputs or estimator settings are missing or invalid.
        RuntimeError: If the common-denominator Sharpe bridge does not reconcile.
    """
    _validate_attribution(attribution=attribution)
    display_labels = _display_labels(
        model_name=model_name,
        benchmark_label=benchmark_label,
        labels=labels,
    )
    display_colors = _updated_mapping(
        defaults=_DEFAULT_COLORS,
        updates=colors,
        name='colors',
    )
    contributions = _compute_ewma_sharpe_contributions(attribution=attribution)
    benchmark_sharpe = float(contributions['Benchmark'])
    systematic_sharpe = float(contributions['Systematic'])
    risk_contribution = float(contributions['Risk Layer'])
    signal_contribution = float(contributions['Signal Layer'])
    integration_contribution = float(contributions['Integration'])
    has_net = 'Full Model Net' in contributions.index
    cost_contribution = (
        float(contributions['Trading Cost Drag']) if has_net else None
    )
    endpoint_sharpe = float(
        contributions['Full Model Net' if has_net else 'Full Model Gross']
    )
    bridge_total = (
        systematic_sharpe
        + risk_contribution
        + signal_contribution
        + integration_contribution
    )
    if has_net:
        bridge_total += cost_contribution
    if not np.isclose(bridge_total, endpoint_sharpe, atol=1.0e-12, rtol=0.0):
        raise RuntimeError('common-denominator EWMA Sharpe bridge does not reconcile')

    beta_full = _get_beta(attribution=attribution, layer='Full Model')
    beta_risk = _get_beta(attribution=attribution, layer='Risk Layer')
    beta_signal = _get_beta(attribution=attribution, layer='Signal Layer')
    beta_integration = _get_beta(attribution=attribution, layer='Integration')
    fig, ax = _new_axes(ax=ax, detailed_mode=detailed_mode)
    bars = [
        (0, 0.0, benchmark_sharpe, 'Benchmark', False),
        (2, 0.0, systematic_sharpe, 'Systematic', False),
        (3, systematic_sharpe, risk_contribution, 'Risk Layer', True),
        (
            4,
            systematic_sharpe + risk_contribution,
            signal_contribution,
            'Signal Layer',
            True,
        ),
        (
            5,
            systematic_sharpe + risk_contribution + signal_contribution,
            integration_contribution,
            'Integration',
            True,
        ),
    ]
    tick_labels = [
        display_labels['Benchmark'],
        _beta_label(display_labels['Systematic'], beta_full),
        _beta_label(display_labels['Risk Layer'], beta_risk),
        _beta_label(display_labels['Signal Layer'], beta_signal),
        _beta_label(display_labels['Integration'], beta_integration),
    ]
    if has_net:
        gross_sharpe = (
            systematic_sharpe
            + risk_contribution
            + signal_contribution
            + integration_contribution
        )
        bars.extend([
            (6, gross_sharpe, cost_contribution, 'Trading Cost Drag', True),
            (7, 0.0, endpoint_sharpe, 'Full Model Net', False),
        ])
        tick_labels.extend([
            display_labels['Trading Cost Drag'],
            _beta_label(display_labels['Full Model Net'], beta_full),
        ])
    else:
        bars.append((6, 0.0, endpoint_sharpe, 'Full Model Gross', False))
        tick_labels.append(_beta_label(display_labels['Full Model Gross'], beta_full))

    for position, start, contribution, key, is_contribution in bars:
        _draw_bar(
            ax=ax,
            position=position,
            start=start,
            contribution=contribution,
            color=display_colors[key],
            is_contribution=is_contribution,
            value_format='{:.2f}',
        )
    connectors = [
        (2, 3, systematic_sharpe),
        (3, 4, systematic_sharpe + risk_contribution),
        (4, 5, systematic_sharpe + risk_contribution + signal_contribution),
    ]
    if has_net:
        connectors.extend([(5, 6, gross_sharpe), (6, 7, endpoint_sharpe)])
    else:
        connectors.append((5, 6, endpoint_sharpe))
    _draw_connectors(ax=ax, connectors=connectors)

    ax.axvline(1.0, color=_GRID_COLOR, linewidth=1.0, linestyle=':')
    ax.axhline(0.0, color=_TEXT_COLOR, linewidth=0.8)
    ax.set_xticks([bar[0] for bar in bars], tick_labels)
    ax.set_ylabel('Current EWMA log-return Sharpe (rf=0)', color=_TEXT_COLOR)
    ax.margins(x=0.04, y=0.18)
    _style_axes(ax=ax)

    final_date = pd.Timestamp(attribution.periodic_returns.index[-1])
    span_label = _ewma_span_label(attribution=attribution)
    plot_title = title or (
        f'Risk and signal contributions explain {model_name.upper()} Sharpe beyond the benchmark'
    )
    _add_details(
        ax=ax,
        title=plot_title,
        subtitle=(
            f'{span_label} EWMA log-return/volatility contributions through '
            f'{final_date:%d %b %Y} | zero risk-free rate'
        ),
        note=(
            'Static Benchmark return is divided by benchmark EWMA volatility. Systematic, risk, '
            'signal, integration and realised cost returns are each divided by one common '
            'full-model endpoint EWMA volatility, so their contributions add exactly to the '
            'displayed model Sharpe and retain the signs of their return components.'
        ),
        detailed_mode=detailed_mode,
    )
    return fig

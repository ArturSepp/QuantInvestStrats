"""Plots for current EWMA model-layer return and Sharpe attribution.

The numerical work is deliberately outside this module.  Return components, current EWMA
regressions, Bartlett-HAC intervals and effective sample sizes come from
``ModelLayerEwmaRegressionAttribution``.  Sequential Sharpe levels come from
``compute_model_layer_ewma_stage_sharpes``.  The functions here only validate those labelled
outputs and render them as waterfall bridges.

Both bridges use the same ordered layers: benchmark is shown as a reference, systematic return
is the first model stage, and risk-layer, signal-layer and integration effects are added in that
order.  Sharpe increments are therefore sequential differences, not standalone component
Sharpes.  When a net model is present, gross performance is the level reached after integration;
only the trading-cost step and the final net endpoint are drawn after it.
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


def _beta_label(label: str, beta: float) -> str:
    """Append the current beta estimate as the third label row."""
    return f'{label}\n$\\hat{{\\beta}}$ = {beta:.2f}'


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
        if detailed_mode:
            fig.subplots_adjust(left=0.09, right=0.98, top=0.82, bottom=0.30)
        else:
            fig.subplots_adjust(left=0.09, right=0.98, top=0.96, bottom=0.18)
        return fig, ax
    return None, ax


def _style_axes(ax: plt.Axes) -> None:
    """Apply restrained, report-neutral styling shared by the two bridges."""
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
        _beta_label(display_labels['Risk Layer'], beta_risk),
        _beta_label(display_labels['Signal Layer'], beta_signal),
        _beta_label(display_labels['Integration'], beta_integration),
    ]
    if has_net_return:
        endpoint_beta = (
            _get_beta(attribution=attribution, layer='Full Model Net')
            if 'Full Model Net' in attribution.regression_table.index
            else beta_full
        )
        bars.extend([
            (6, full_return, cost_drag, 'Trading Cost Drag', True, None),
            (7, 0.0, net_return, 'Full Model Net', False, None),
        ])
        tick_labels.extend([
            display_labels['Trading Cost Drag'],
            _beta_label(display_labels['Full Model Net'], endpoint_beta),
        ])
    else:
        bars.append((6, 0.0, full_return, 'Full Model Gross', False, None))
        tick_labels.append(_beta_label(display_labels['Full Model Gross'], beta_full))

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
    ax.set_xticks([bar[0] for bar in bars], tick_labels)
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
            f'HAC({attribution.hac_lags}) intervals; effective observations '
            f'{attribution.effective_nobs:.1f}. '
            'Beta labels are current EWMA regression estimates. Integration is the exact '
            'residual after systematic, risk-layer and signal-layer contributions.'
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
    """Plot sequential current EWMA log-return Sharpe contributions.

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
        TypeError: If ``attribution`` or the QIS stage-Sharpe output has an unsupported type.
        ValueError: If required labelled inputs or estimator settings are missing or invalid.
        RuntimeError: If the sequential Sharpe bridge does not reconcile.
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
    sharpes = _current_stage_sharpes(attribution=attribution)
    benchmark_sharpe = float(sharpes['Benchmark'])
    systematic_sharpe = float(sharpes['Systematic'])
    risk_delta = float(sharpes['Risk Layer'] - systematic_sharpe)
    signal_delta = float(sharpes['Signal Layer'] - sharpes['Risk Layer'])
    gross_sharpe = float(sharpes['Full Model Gross'])
    integration_delta = float(gross_sharpe - sharpes['Signal Layer'])
    has_net = 'Full Model Net' in sharpes.index
    cost_delta = float(sharpes['Full Model Net'] - gross_sharpe) if has_net else None
    endpoint_sharpe = float(sharpes['Full Model Net']) if has_net else gross_sharpe
    bridge_total = systematic_sharpe + risk_delta + signal_delta + integration_delta
    if has_net:
        bridge_total += cost_delta
    if not np.isclose(bridge_total, endpoint_sharpe, atol=1.0e-12, rtol=0.0):
        raise RuntimeError('sequential EWMA Sharpe bridge does not reconcile')

    beta_full = _get_beta(attribution=attribution, layer='Full Model')
    beta_risk = _get_beta(attribution=attribution, layer='Risk Layer')
    beta_signal = _get_beta(attribution=attribution, layer='Signal Layer')
    beta_integration = _get_beta(attribution=attribution, layer='Integration')
    fig, ax = _new_axes(ax=ax, detailed_mode=detailed_mode)
    bars = [
        (0, 0.0, benchmark_sharpe, 'Benchmark', False),
        (2, 0.0, systematic_sharpe, 'Systematic', False),
        (3, systematic_sharpe, risk_delta, 'Risk Layer', True),
        (4, systematic_sharpe + risk_delta, signal_delta, 'Signal Layer', True),
        (
            5,
            systematic_sharpe + risk_delta + signal_delta,
            integration_delta,
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
        endpoint_beta = (
            _get_beta(attribution=attribution, layer='Full Model Net')
            if 'Full Model Net' in attribution.regression_table.index
            else beta_full
        )
        bars.extend([
            (6, gross_sharpe, cost_delta, 'Trading Cost Drag', True),
            (7, 0.0, endpoint_sharpe, 'Full Model Net', False),
        ])
        tick_labels.extend([
            display_labels['Trading Cost Drag'],
            _beta_label(display_labels['Full Model Net'], endpoint_beta),
        ])
    else:
        bars.append((6, 0.0, gross_sharpe, 'Full Model Gross', False))
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
        (3, 4, systematic_sharpe + risk_delta),
        (4, 5, systematic_sharpe + risk_delta + signal_delta),
    ]
    if has_net:
        connectors.extend([(5, 6, gross_sharpe), (6, 7, endpoint_sharpe)])
    else:
        connectors.append((5, 6, gross_sharpe))
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
            f'{span_label} EWMA log-return Sharpe through '
            f'{final_date:%d %b %Y} | norm_type=2, zero risk-free rate'
        ),
        note=(
            'Sharpe contributions are sequential arithmetic differences after adding each '
            'log-return component; they are not standalone component Sharpe ratios. '
            'All stages use the same EWMA span and centred-variance convention.'
        ),
        detailed_mode=detailed_mode,
    )
    return fig

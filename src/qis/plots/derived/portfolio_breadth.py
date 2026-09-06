"""Plots for portfolio breadth, allocation efficiency and concentration.

The numerical definitions live in :mod:`qis.portfolio.attribution.portfolio_breadth`. This
module only selects the labelled output columns, compares their latest observations and converts
current
absolute capital and Euler risk-contribution shares into cumulative concentration curves.

Three views are intentionally separate. ``plot_portfolio_breadth_history`` shows how one
portfolio uses its opportunity set through time. ``plot_portfolio_breadth_current_comparison``
compares the latest breadth of several layers. ``plot_portfolio_breadth_concentration`` shows
how many ranked positions account for the current capital and risk allocations. None of the
figures estimates active breadth or relates breadth to realised alpha.
"""

from __future__ import annotations

import textwrap
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, PercentFormatter

from qis.plots.bars import plot_bars
from qis.plots.lineplot import plot_line
from qis.plots.time_series import plot_time_series
from qis.plots.utils import LegendStats

if TYPE_CHECKING:
    from qis.portfolio.attribution.portfolio_breadth import PortfolioBreadthResult


_TEXT_COLOR = '#23313B'
_SUBDUED_TEXT_COLOR = '#5F6F78'
_GRID_COLOR = '#D8D4CB'
_CAPITAL_CURVE = 'Capital allocation'
_RISK_CURVE = 'Risk allocation'


def _metric_groups() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the canonical count and efficiency columns without duplicating their labels."""
    from qis.portfolio.attribution.portfolio_breadth import (
        COUNT_COLUMNS,
        EFFICIENCY_COLUMNS,
    )

    return COUNT_COLUMNS, EFFICIENCY_COLUMNS


def _default_colors() -> dict[str, str]:
    """Return an Okabe-Ito-derived semantic palette shared by all breadth plots."""
    from qis.portfolio.attribution.portfolio_breadth import (
        CAPITAL_UTILISATION,
        EFFECTIVE_CAPITAL_COUNT,
        EFFECTIVE_RISK_COUNT,
        EFFECTIVE_UNIVERSE_COUNT,
        INVESTABLE_COUNT,
        INVESTED_COUNT,
        RISK_BREADTH_EFFICIENCY,
        SELECTION_COVERAGE,
        SIZING_EVENNESS,
    )

    return {
        INVESTABLE_COUNT: '#0072B2',
        INVESTED_COUNT: '#56B4E9',
        EFFECTIVE_UNIVERSE_COUNT: '#CC79A7',
        EFFECTIVE_CAPITAL_COUNT: '#E69F00',
        EFFECTIVE_RISK_COUNT: '#009E73',
        SELECTION_COVERAGE: '#0072B2',
        SIZING_EVENNESS: '#E69F00',
        CAPITAL_UTILISATION: '#009E73',
        RISK_BREADTH_EFFICIENCY: '#CC79A7',
        _CAPITAL_CURVE: '#0072B2',
        _RISK_CURVE: '#D55E00',
    }


def _updated_colors(colors: Optional[Mapping[str, str]]) -> dict[str, str]:
    """Apply validated caller overrides to the breadth palette."""
    output = _default_colors()
    if colors is None:
        return output
    unknown = sorted(set(colors).difference(output))
    if unknown:
        raise ValueError(f'colors contains unsupported keys {unknown!r}')
    output.update(colors)
    return output


def _validate_result(result: PortfolioBreadthResult) -> None:
    """Validate the numerical result contract needed by the plotting layer."""
    from qis.portfolio.attribution.portfolio_breadth import PortfolioBreadthResult

    if not isinstance(result, PortfolioBreadthResult):
        raise TypeError(
            f'result must be PortfolioBreadthResult, got {type(result)!r}'
        )
    if result.metrics.empty:
        raise ValueError('result.metrics must not be empty')
    count_columns, efficiency_columns = _metric_groups()
    missing = [
        column for column in (*count_columns, *efficiency_columns)
        if column not in result.metrics.columns
    ]
    if missing:
        raise ValueError(f'result.metrics is missing required columns {missing!r}')


def _new_two_panel_axes(
        axs: Optional[Sequence[plt.Axes]],
        *,
        sharex: bool,
) -> tuple[Optional[Figure], tuple[plt.Axes, plt.Axes]]:
    """Create a white two-panel canvas or validate and reuse caller-owned axes."""
    if axs is None:
        fig, axes_array = plt.subplots(
            2,
            1,
            figsize=(11.0, 8.0),
            sharex=sharex,
            gridspec_kw={'height_ratios': (1.12, 1.0)},
        )
        axes = (axes_array[0], axes_array[1])
        fig.patch.set_facecolor('white')
        return fig, axes
    if len(axs) != 2:
        raise ValueError(f'axs must contain two axes, got {len(axs)}')
    axes = (axs[0], axs[1])
    if axes[0].figure is not axes[1].figure:
        raise ValueError('axs must belong to the same figure')
    return None, axes


def _style_axis(ax: plt.Axes) -> None:
    """Apply the report-neutral white background and restrained grid."""
    ax.set_facecolor('white')
    ax.grid(axis='y', color=_GRID_COLOR, linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(colors=_TEXT_COLOR)
    ax.xaxis.label.set_color(_TEXT_COLOR)
    ax.yaxis.label.set_color(_TEXT_COLOR)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_color(_GRID_COLOR)


def _add_figure_details(
        fig: Figure,
        *,
        title: str,
        note: str,
        detailed_mode: bool,
) -> None:
    """Add the figure title and methodology note when detailed output is requested."""
    if not detailed_mode:
        return
    fig.suptitle(
        textwrap.fill(title, width=72, break_long_words=False, break_on_hyphens=False),
        x=0.08,
        y=0.98,
        ha='left',
        va='top',
        color=_TEXT_COLOR,
        fontsize=17,
        fontweight='bold',
    )
    fig.text(
        0.08,
        0.015,
        textwrap.fill(note, width=130, break_long_words=False, break_on_hyphens=False),
        ha='left',
        va='bottom',
        color=_SUBDUED_TEXT_COLOR,
        fontsize=9,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.14, hspace=0.30)


def _latest_metrics(
        results: Mapping[str, PortfolioBreadthResult],
) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    """Return one latest metric row per named layer in mapping order."""
    if not results:
        raise ValueError('results must not be empty')
    rows: dict[str, pd.Series] = {}
    dates: list[pd.Timestamp] = []
    for name, result in results.items():
        if not isinstance(name, str) or not name:
            raise ValueError('results keys must be non-empty layer names')
        _validate_result(result=result)
        rows[name] = result.metrics.iloc[-1]
        dates.append(pd.Timestamp(result.metrics.index[-1]))
    return pd.DataFrame.from_dict(rows, orient='index'), dates


def _date_note(dates: Sequence[pd.Timestamp]) -> str:
    """Describe one shared latest date or the range of layer-specific latest dates."""
    first = min(dates)
    last = max(dates)
    if first == last:
        return f'Latest allocation date: {last:%d %b %Y}.'
    return f'Latest allocation dates range from {first:%d %b %Y} to {last:%d %b %Y}.'


def plot_portfolio_breadth_history(
        result: PortfolioBreadthResult,
        *,
        title: str = 'Portfolio breadth through time',
        detailed_mode: bool = True,
        colors: Optional[Mapping[str, str]] = None,
        x_date_freq: str = 'YE',
        date_format: str = '%b-%y',
        fontsize: int = 10,
        axs: Optional[Sequence[plt.Axes]] = None,
) -> Optional[Figure]:
    """Plot one portfolio's asset counts and breadth-efficiency ratios through time.

    Args:
        result: Numerical breadth result produced by ``compute_portfolio_breadth``.
        title: Figure-level title shown in detailed mode.
        detailed_mode: Whether to draw titles and the methodology note.
        colors: Optional semantic colour overrides keyed by metric label.
        x_date_freq: Tick frequency passed to the QIS time-series plotter.
        date_format: Date-label format passed to the QIS time-series plotter.
        fontsize: Base font size for axes and legends.
        axs: Two caller-owned axes, counts first and efficiencies second. None creates a figure.

    Returns:
        The figure drawn on, or None when ``axs`` was supplied.
    """
    _validate_result(result=result)
    count_columns, efficiency_columns = _metric_groups()
    palette = _updated_colors(colors=colors)
    fig, axes = _new_two_panel_axes(axs=axs, sharex=True)
    target_fig = fig if fig is not None else axes[0].figure

    plot_time_series(
        df=result.counts,
        colors=[palette[column] for column in count_columns],
        linewidth=2.0,
        x_date_freq=x_date_freq,
        date_format=date_format,
        legend_stats=LegendStats.AVG_LAST,
        legend_loc='upper left',
        var_format='{:,.1f}',
        ylabel='Number of assets',
        title='Breadth in number of assets' if detailed_mode else None,
        title_color=_TEXT_COLOR,
        fontsize=fontsize,
        framealpha=0.95,
        facecolor='white',
        ax=axes[0],
    )
    plot_time_series(
        df=result.efficiency,
        colors=[palette[column] for column in efficiency_columns],
        linewidth=2.0,
        x_date_freq=x_date_freq,
        date_format=date_format,
        legend_stats=LegendStats.AVG_LAST,
        legend_loc='upper left',
        var_format='{:.0%}',
        ylabel='Efficiency',
        y_limits=(0.0, 1.05),
        title='Breadth efficiency' if detailed_mode else None,
        title_color=_TEXT_COLOR,
        fontsize=fontsize,
        framealpha=0.95,
        facecolor='white',
        ax=axes[1],
    )
    # Seaborn creates confidence-band collections for wide-form line plots by default.
    # Breadth metrics are deterministic paths, so those bands have no statistical meaning.
    for axis in axes:
        for collection in tuple(axis.collections):
            collection.remove()
    axes[0].tick_params(axis='x', labelbottom=False)
    for ax in axes:
        _style_axis(ax=ax)
    _add_figure_details(
        fig=target_fig,
        title=title,
        note=(
            'Capital breadth uses absolute target weights; risk breadth uses absolute Euler '
            'risk-contribution shares. Positions with absolute weight at or below '
            f'{result.position_threshold:.2%} are not counted as invested.'
        ),
        detailed_mode=detailed_mode,
    )
    return fig


def plot_portfolio_breadth_current_comparison(
        results: Mapping[str, PortfolioBreadthResult],
        *,
        title: str = 'Current portfolio breadth by layer',
        detailed_mode: bool = True,
        colors: Optional[Mapping[str, str]] = None,
        fontsize: int = 10,
        axs: Optional[Sequence[plt.Axes]] = None,
) -> Optional[Figure]:
    """Compare the latest breadth counts and efficiencies of named portfolio layers.

    Args:
        results: Ordered mapping from display name to numerical breadth result.
        title: Figure-level title shown in detailed mode.
        detailed_mode: Whether to draw titles and the methodology note.
        colors: Optional semantic colour overrides keyed by metric label.
        fontsize: Base font size for axes, legends and bar annotations.
        axs: Two caller-owned axes, counts first and efficiencies second. None creates a figure.

    Returns:
        The figure drawn on, or None when ``axs`` was supplied.

    Raises:
        ValueError: If no results are supplied or a layer name is empty.
    """
    latest, dates = _latest_metrics(results=results)
    count_columns, efficiency_columns = _metric_groups()
    palette = _updated_colors(colors=colors)
    fig, axes = _new_two_panel_axes(axs=axs, sharex=False)
    target_fig = fig if fig is not None else axes[0].figure

    plot_bars(
        df=latest.loc[:, list(count_columns)],
        stacked=False,
        is_sns=False,
        colors=[palette[column] for column in count_columns],
        legend_loc='upper left',
        add_bar_values=True,
        yvar_format='{:,.1f}',
        x_rotation=0,
        ylabel='Number of assets',
        title='Current breadth in number of assets' if detailed_mode else None,
        title_color=_TEXT_COLOR,
        fontsize=fontsize,
        framealpha=0.95,
        facecolor='white',
        ax=axes[0],
    )
    plot_bars(
        df=latest.loc[:, list(efficiency_columns)],
        stacked=False,
        is_sns=False,
        colors=[palette[column] for column in efficiency_columns],
        legend_loc='upper left',
        add_bar_values=True,
        yvar_format='{:.0%}',
        x_rotation=0,
        ylabel='Efficiency',
        y_limits=(0.0, 1.05),
        title='Current breadth efficiency' if detailed_mode else None,
        title_color=_TEXT_COLOR,
        fontsize=fontsize,
        framealpha=0.95,
        facecolor='white',
        ax=axes[1],
    )
    for ax in axes:
        _style_axis(ax=ax)
    _add_figure_details(
        fig=target_fig,
        title=title,
        note=(
            f'{_date_note(dates)} Each layer is evaluated from its own latest target weights; '
            'ratios are bounded between zero and one.'
        ),
        detailed_mode=detailed_mode,
    )
    return fig


def _current_share_row(
        shares: pd.DataFrame,
        *,
        date: pd.Timestamp,
        description: str,
) -> pd.Series:
    """Return one finite, non-negative and unit-normalised share row."""
    try:
        row = shares.loc[date].astype(float)
    except KeyError as exception:
        raise ValueError(f'{description} has no row at {date:%d %b %Y}') from exception
    if (~np.isfinite(row.to_numpy())).any() or (row < 0.0).any():
        raise ValueError(f'{description} must contain finite non-negative values')
    row = row[row > 0.0]
    total = float(row.sum())
    if total <= 0.0:
        raise ValueError(f'no current {description}')
    return row / total


def _concentration_curve(shares: pd.Series) -> pd.Series:
    """Sort positive shares from largest to smallest and return their cumulative curve."""
    ranked = np.sort(shares.to_numpy(dtype=float))[::-1]
    cumulative = np.concatenate(([0.0], np.cumsum(ranked)))
    cumulative[-1] = 1.0
    return pd.Series(cumulative, index=np.arange(len(cumulative)))


def _threshold_count(curve: pd.Series, threshold: float) -> int:
    """Return the first positive rank whose cumulative share reaches ``threshold``."""
    return int(np.searchsorted(curve.to_numpy()[1:], threshold, side='left') + 1)


def _curve_label(name: str, curve: pd.Series) -> str:
    """Add the 50% and 80% concentration ranks to one curve label."""
    count_50 = _threshold_count(curve=curve, threshold=0.50)
    count_80 = _threshold_count(curve=curve, threshold=0.80)
    noun_50 = 'asset' if count_50 == 1 else 'assets'
    noun_80 = 'asset' if count_80 == 1 else 'assets'
    return (
        f'{name} (50%: {count_50} {noun_50}; '
        f'80%: {count_80} {noun_80})'
    )


def plot_portfolio_breadth_concentration(
        result: PortfolioBreadthResult,
        *,
        date: Optional[pd.Timestamp] = None,
        title: str = 'Current capital and risk concentration',
        detailed_mode: bool = True,
        colors: Optional[Mapping[str, str]] = None,
        fontsize: int = 10,
        ax: Optional[plt.Axes] = None,
) -> Optional[Figure]:
    """Plot current cumulative absolute capital and risk-contribution concentration.

    Assets are ranked independently for the two curves. The displayed 50% and 80% counts
    therefore answer how many largest positions account for each allocation, rather than which
    named instruments occupy those ranks.

    Args:
        result: Numerical breadth result produced by ``compute_portfolio_breadth``.
        date: Exact evaluation date to show. None uses the latest result date.
        title: Figure-level title shown in detailed mode.
        detailed_mode: Whether to draw the title and methodology note.
        colors: Optional semantic colour overrides. Use ``Capital allocation`` and
            ``Risk allocation`` to change the two curves.
        fontsize: Base font size for axes and legend.
        ax: Caller-owned axis. None creates a figure.

    Returns:
        The figure drawn on, or None when ``ax`` was supplied.

    Raises:
        ValueError: If the requested date is absent or current capital/risk shares are empty.
    """
    _validate_result(result=result)
    evaluation_date = (
        pd.Timestamp(result.metrics.index[-1]) if date is None else pd.Timestamp(date)
    )
    if evaluation_date not in result.metrics.index:
        raise ValueError(
            f'date {evaluation_date:%d %b %Y} is not an evaluation date in result.metrics'
        )
    capital_shares = _current_share_row(
        shares=result.absolute_weight_shares,
        date=evaluation_date,
        description='capital-allocation shares',
    )
    risk_shares = _current_share_row(
        shares=result.absolute_risk_contribution_shares,
        date=evaluation_date,
        description='risk-contribution shares',
    )
    capital_curve = _concentration_curve(shares=capital_shares)
    risk_curve = _concentration_curve(shares=risk_shares)
    capital_label = _curve_label(name=_CAPITAL_CURVE, curve=capital_curve)
    risk_label = _curve_label(name=_RISK_CURVE, curve=risk_curve)
    curves = pd.concat(
        [capital_curve.rename(capital_label), risk_curve.rename(risk_label)],
        axis=1,
    )
    palette = _updated_colors(colors=colors)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10.5, 6.5))
        fig.patch.set_facecolor('white')
    else:
        fig = None
    target_fig = fig if fig is not None else ax.figure
    plot_line(
        df=curves,
        colors=[palette[_CAPITAL_CURVE], palette[_RISK_CURVE]],
        markers=['o', 's'],
        linewidth=2.2,
        legend_loc='lower right',
        legend_stats=LegendStats.NONE,
        xvar_format='{:,.0f}',
        yvar_format='{:.0%}',
        xlabel='Number of ranked assets',
        ylabel='Cumulative allocation share',
        fontsize=fontsize,
        framealpha=0.95,
        facecolor='white',
        ax=ax,
    )
    for level in (0.50, 0.80):
        ax.axhline(
            level,
            color=_GRID_COLOR,
            linewidth=0.9,
            linestyle='--',
            zorder=0,
            label='_concentration_reference',
        )
    ax.set_xlim(0.0, float(max(len(capital_shares), len(risk_shares))))
    ax.set_ylim(0.0, 1.04)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    _style_axis(ax=ax)

    covariance_date = result.covariance_dates.loc[evaluation_date]
    covariance_note = (
        '' if pd.isna(covariance_date)
        else ' Risk contributions use the point-in-time covariance dated '
        f'{pd.Timestamp(covariance_date):%d %b %Y}.'
    )
    _add_figure_details(
        fig=target_fig,
        title=title,
        note=(
            f'Allocation date: {evaluation_date:%d %b %Y}. Curves rank absolute shares '
            f'independently and start at zero.{covariance_note}'
        ),
        detailed_mode=detailed_mode,
    )
    if detailed_mode:
        target_fig.subplots_adjust(left=0.10, right=0.98, top=0.86, bottom=0.16)
    return fig

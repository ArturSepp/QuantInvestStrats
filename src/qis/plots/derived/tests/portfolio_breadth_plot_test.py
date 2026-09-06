"""Tests for portfolio-breadth history, comparison and concentration plots."""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import qis.plots.derived.portfolio_breadth as plots  # noqa: E402
from qis.portfolio.attribution.portfolio_breadth import (  # noqa: E402
    CAPITAL_UTILISATION,
    EFFECTIVE_CAPITAL_COUNT,
    EFFECTIVE_RISK_COUNT,
    EFFECTIVE_UNIVERSE_COUNT,
    INVESTABLE_COUNT,
    INVESTED_COUNT,
    RISK_BREADTH_EFFICIENCY,
    SELECTION_COVERAGE,
    SIZING_EVENNESS,
    PortfolioBreadthResult,
)


def _result(scale: float = 1.0, with_risk: bool = True) -> PortfolioBreadthResult:
    """Return a deterministic breadth result suitable for plot assertions."""
    dates = pd.date_range('2024-03-31', periods=4, freq='QE')
    metrics = pd.DataFrame(
        {
            INVESTABLE_COUNT: scale * np.array([8.0, 9.0, 10.0, 10.0]),
            INVESTED_COUNT: scale * np.array([4.0, 5.0, 5.0, 6.0]),
            EFFECTIVE_UNIVERSE_COUNT: scale * np.array([3.2, 3.5, 3.8, 4.0]),
            EFFECTIVE_CAPITAL_COUNT: scale * np.array([3.0, 3.8, 4.1, 4.5]),
            EFFECTIVE_RISK_COUNT: scale * np.array([2.5, 3.0, 3.4, 3.6]),
            SELECTION_COVERAGE: np.array([0.50, 0.56, 0.50, 0.60]),
            SIZING_EVENNESS: np.array([0.75, 0.76, 0.82, 0.75]),
            CAPITAL_UTILISATION: np.array([0.375, 0.422, 0.410, 0.450]),
            RISK_BREADTH_EFFICIENCY: np.array([0.625, 0.600, 0.680, 0.600]),
        },
        index=dates,
    )
    availability = pd.DataFrame(
        True,
        index=dates,
        columns=['A', 'B', 'C'],
    )
    weight_shares = pd.DataFrame(
        [[0.5, 0.3, 0.2]] * len(dates),
        index=dates,
        columns=availability.columns,
    )
    risk_values = [0.7, 0.2, 0.1] if with_risk else [0.0, 0.0, 0.0]
    risk_shares = pd.DataFrame(
        [risk_values] * len(dates),
        index=dates,
        columns=availability.columns,
    )
    covariance_dates = pd.Series(dates, index=dates, name='Covariance date')
    return PortfolioBreadthResult(
        metrics=metrics,
        availability=availability,
        absolute_weight_shares=weight_shares,
        absolute_risk_contribution_shares=risk_shares,
        covariance_dates=covariance_dates,
        span=36,
        position_threshold=1.0e-4,
        covariance_source='provided',
    )


def test_breadth_history_draws_count_and_efficiency_panels_without_fill() -> None:
    """The history figure shows both metric families on white, unfilled panels."""
    result = _result()

    fig = plots.plot_portfolio_breadth_history(result=result)
    try:
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2
        counts_ax, efficiency_ax = fig.axes
        count_lines = [
            line for line in counts_ax.lines
            if len(line.get_ydata()) == len(result.metrics.index)
        ]
        efficiency_lines = [
            line for line in efficiency_ax.lines
            if len(line.get_ydata()) == len(result.metrics.index)
        ]
        assert len(count_lines) == 5
        assert len(efficiency_lines) == 4
        assert counts_ax.get_title() == 'Breadth in number of assets'
        assert efficiency_ax.get_title() == 'Breadth efficiency'
        assert fig._suptitle.get_text() == 'Portfolio breadth through time'
        assert counts_ax.get_facecolor()[:3] == (1.0, 1.0, 1.0)
        assert efficiency_ax.get_facecolor()[:3] == (1.0, 1.0, 1.0)
        assert not counts_ax.collections
        assert not efficiency_ax.collections
    finally:
        plt.close(fig)


def test_breadth_history_supports_supplied_axes_and_simple_mode() -> None:
    """Caller-owned axes are reused and simple mode suppresses explanatory text."""
    fig, axs = plt.subplots(2, 1)
    try:
        returned = plots.plot_portfolio_breadth_history(
            result=_result(),
            detailed_mode=False,
            axs=axs,
        )
        assert returned is None
        assert fig._suptitle is None
        assert all(ax.get_title() == '' for ax in axs)
        assert not fig.texts
    finally:
        plt.close(fig)


def test_current_comparison_uses_latest_metric_row_for_each_layer() -> None:
    """The layer comparison draws the latest counts and efficiencies as grouped bars."""
    results = {'Risk Layer': _result(), 'Full Model': _result(scale=1.1)}

    fig = plots.plot_portfolio_breadth_current_comparison(results=results)
    try:
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2
        counts_ax, efficiency_ax = fig.axes
        assert [tick.get_text() for tick in counts_ax.get_xticklabels()] == [
            'Risk Layer',
            'Full Model',
        ]
        assert [tick.get_text() for tick in efficiency_ax.get_xticklabels()] == [
            'Risk Layer',
            'Full Model',
        ]
        expected_counts = np.sort(
            np.concatenate([
                results[name].metrics.loc[:, [
                    INVESTABLE_COUNT,
                    INVESTED_COUNT,
                    EFFECTIVE_UNIVERSE_COUNT,
                    EFFECTIVE_CAPITAL_COUNT,
                    EFFECTIVE_RISK_COUNT,
                ]].iloc[-1].to_numpy()
                for name in results
            ])
        )
        actual_counts = np.sort(
            np.array([patch.get_height() for patch in counts_ax.patches])
        )
        np.testing.assert_allclose(actual_counts, expected_counts, atol=1.0e-12)
        assert len(efficiency_ax.patches) == 8
        assert fig._suptitle.get_text() == 'Current portfolio breadth by layer'
        assert counts_ax.get_facecolor()[:3] == (1.0, 1.0, 1.0)
        assert efficiency_ax.get_facecolor()[:3] == (1.0, 1.0, 1.0)
    finally:
        plt.close(fig)


def test_concentration_curves_rank_current_capital_and_risk_shares() -> None:
    """Current absolute capital and risk shares are sorted before accumulation."""
    result = _result()

    fig = plots.plot_portfolio_breadth_concentration(result=result)
    try:
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        curve_lines = [line for line in ax.lines if len(line.get_ydata()) == 4]
        assert len(curve_lines) == 2
        np.testing.assert_allclose(
            curve_lines[0].get_ydata(),
            [0.0, 0.5, 0.8, 1.0],
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            curve_lines[1].get_ydata(),
            [0.0, 0.7, 0.9, 1.0],
            atol=1.0e-12,
        )
        legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]
        capital_label = next(
            label for label in legend_labels if label.startswith('Capital allocation')
        )
        risk_label = next(
            label for label in legend_labels if label.startswith('Risk allocation')
        )
        assert '50%: 1 asset' in capital_label
        assert '80%: 2 assets' in capital_label
        assert '50%: 1 asset' in risk_label
        assert '80%: 2 assets' in risk_label
        assert ax.get_ylim() == pytest.approx((0.0, 1.04))
        assert ax.get_facecolor()[:3] == (1.0, 1.0, 1.0)
    finally:
        plt.close(fig)


def test_concentration_rejects_missing_current_risk_allocation() -> None:
    """A zero risk-share row cannot be presented as a risk-concentration curve."""
    with pytest.raises(ValueError, match='no current risk-contribution shares'):
        plots.plot_portfolio_breadth_concentration(result=_result(with_risk=False))
    plt.close('all')

"""Tests for generic current-EWMA model-layer bridge plots."""

from dataclasses import replace

import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PathCollection

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import qis.plots.derived.model_layer_attribution as plots  # noqa: E402
from qis.perfstats.model_layer_attribution import (  # noqa: E402
    ALPHA_AN_CI_HIGH_COLUMN,
    ALPHA_AN_CI_LOW_COLUMN,
    ModelLayerEwmaRegressionAttribution,
    compute_model_layer_ewma_regression_attribution,
)


def _nav(log_returns: np.ndarray, dates: pd.DatetimeIndex, name: str) -> pd.Series:
    """Convert deterministic monthly log returns to a unit-initialised NAV."""
    values = np.exp(np.concatenate(([0.0], np.cumsum(log_returns))))
    return pd.Series(values, index=dates, name=name)


def _attribution(with_net: bool = True) -> ModelLayerEwmaRegressionAttribution:
    """Return a non-degenerate deterministic current-EWMA attribution fixture."""
    n_returns = 96
    dates = pd.date_range('2017-12-31', periods=n_returns + 1, freq='ME')
    phase = np.arange(n_returns, dtype=float)
    benchmark = (
        0.004
        + 0.020 * np.sin(phase / 3.7)
        + 0.011 * np.cos(phase / 8.1)
    )
    risk = 0.0007 + 0.82 * benchmark + 0.004 * np.sin(phase / 2.9 + 0.4)
    signal = 0.0011 + 1.08 * benchmark + 0.006 * np.cos(phase / 4.3 + 0.7)
    full = (
        0.0018
        + 0.91 * benchmark
        + 0.003 * np.sin(phase / 5.2 + 0.3)
        + 0.002 * np.cos(phase / 3.1)
    )
    kwargs = dict(
        benchmark_nav=_nav(benchmark, dates, 'Benchmark'),
        risk_layer_nav=_nav(risk, dates, 'Risk Layer'),
        signal_layer_nav=_nav(signal, dates, 'Signal Layer'),
        full_model_nav=_nav(full, dates, 'Full Model'),
        freq='ME',
        span=24,
        hac_lags=2,
        confidence_level=0.95,
    )
    if with_net:
        kwargs['full_model_net_nav'] = _nav(full - 0.0002, dates, 'Full Model Net')
    return compute_model_layer_ewma_regression_attribution(**kwargs)


def test_return_bridge_draws_current_hac_inference_and_net_only_endpoint() -> None:
    """The detailed return bridge has black intervals, beta rows and only the net endpoint."""
    attribution = _attribution(with_net=True)
    fig = plots.plot_model_layer_ewma_return_bridge(
        attribution=attribution,
        model_name='Mac',
    )
    try:
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert ax.get_title(loc='left') == 'MAC current 24-month EWMA attribution'
        tick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
        assert len(ax.patches) == 9
        assert any('Trading-cost' in label for label in tick_labels)
        assert any('Mac\nnet' in label for label in tick_labels)
        assert all('gross' not in label.lower() for label in tick_labels)
        beta_labels = [label for label in tick_labels if r'$\hat{\beta}$' in label]
        assert len(beta_labels) == 5
        assert all(len(label.splitlines()) >= 3 for label in beta_labels)
        r2_labels = [label for label in tick_labels if r'$R^2$' in label]
        assert len(r2_labels) == 4

        midpoint_markers = [
            collection for collection in ax.collections
            if isinstance(collection, PathCollection)
        ]
        assert len(midpoint_markers) == 4
        for marker in midpoint_markers:
            np.testing.assert_allclose(marker.get_facecolor()[0, :3], np.zeros(3), atol=0.0)
        endpoint_beta = float(attribution.regression_table.loc['Full Model', 'Beta'])
        endpoint_systematic = (
            endpoint_beta * float(attribution.annualised_components['Benchmark Return'])
        )
        endpoint_alpha = float(attribution.regression_table.loc['Full Model', 'An Alpha'])
        cost_drag = float(attribution.annualised_components['Trading Cost Drag'])
        endpoint_total = float(
            attribution.annualised_components['Full Model Net Return']
        )
        np.testing.assert_allclose(
            [ax.patches[-3].get_y(), ax.patches[-3].get_height()],
            [min(0.0, endpoint_systematic), abs(endpoint_systematic)],
            atol=1.0e-12,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            [ax.patches[-2].get_y(), ax.patches[-2].get_height()],
            [
                min(endpoint_systematic, endpoint_systematic + cost_drag),
                abs(cost_drag),
            ],
            atol=1.0e-12,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            [ax.patches[-1].get_y(), ax.patches[-1].get_height()],
            [
                min(
                    endpoint_systematic + cost_drag,
                    endpoint_systematic + cost_drag + endpoint_alpha,
                ),
                abs(endpoint_alpha),
            ],
            atol=1.0e-12,
            rtol=0.0,
        )
        annotations = [text.get_text() for text in ax.texts]
        assert f'Systematic\n{endpoint_systematic:.1%}' in annotations
        assert f'Cost\n{cost_drag:.1%}' in annotations
        assert f'Total alpha (gross)\n{endpoint_alpha:+.1%}' in annotations
        assert f'{endpoint_total:.1%}' in annotations
        assert ax.patches[-2].get_zorder() > ax.patches[-1].get_zorder()
        alpha_text = next(
            text for text in ax.texts
            if text.get_text().startswith('Total alpha (gross)')
        )
        assert max(collection.get_zorder() for collection in ax.collections) > (
            alpha_text.get_zorder()
        )
        for layer in ('Risk Layer', 'Signal Layer', 'Integration', 'Full Model'):
            expected_r2 = float(attribution.regression_table.loc[layer, 'R2'])
            candidate = next(
                label for label in tick_labels
                if f'$R^2$ = {expected_r2:.2f}' in label
            )
            assert r'$\hat{\beta}$' in candidate
        assert fig.get_facecolor()[:3] == (1.0, 1.0, 1.0)
        assert ax.get_facecolor()[:3] == (1.0, 1.0, 1.0)
        detail_text = '\n'.join(text.get_text() for text in ax.texts)
        assert 'Black whiskers show 95%' in detail_text
        assert '\n' in next(
            text.get_text() for text in ax.texts
            if 'Black whiskers show 95%' in text.get_text()
        )
    finally:
        plt.close(fig)


def test_return_bridge_simple_mode_and_display_overrides() -> None:
    """Simple mode suppresses explanatory text while retaining caller labels and colours."""
    attribution = _attribution(with_net=False)
    fig = plots.plot_model_layer_ewma_return_bridge(
        attribution=attribution,
        model_name='Allocation',
        benchmark_label='Policy',
        labels={'Risk Layer': '+ Risk budget\nalpha'},
        colors={'Risk Layer': '#112233'},
        detailed_mode=False,
    )
    try:
        ax = fig.axes[0]
        assert ax.get_title(loc='left') == ''
        assert len(ax.patches) == 7
        tick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
        assert tick_labels[0] == 'Policy'
        assert any('+ Risk budget\nalpha' in label for label in tick_labels)
        assert all('Black whiskers show' not in text.get_text() for text in ax.texts)
        np.testing.assert_allclose(
            ax.patches[2].get_facecolor()[:3],
            np.array([0x11, 0x22, 0x33]) / 255.0,
            atol=1.0e-12,
        )
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    ('freq', 'expected'),
    [('ME', '24-month'), ('QE-DEC', '24-quarter'), ('W-FRI', '24-week'), ('h', '24-period')],
)
def test_ewma_span_label_is_frequency_aware(freq: str, expected: str) -> None:
    """Detailed wording must not describe non-monthly inputs as monthly observations."""
    attribution = replace(_attribution(with_net=False), freq=freq)

    assert plots._ewma_span_label(attribution=attribution) == expected


def test_return_bridge_rejects_alpha_that_is_not_the_hac_midpoint() -> None:
    """A plotted alpha cannot silently differ from the estimator behind its whisker."""
    attribution = _attribution(with_net=False)
    broken_table = attribution.regression_table.copy()
    broken_table.loc['Risk Layer', ALPHA_AN_CI_LOW_COLUMN] += 0.001
    broken_table.loc['Risk Layer', ALPHA_AN_CI_HIGH_COLUMN] += 0.001
    broken = replace(attribution, regression_table=broken_table)

    with pytest.raises(RuntimeError, match='differs from its HAC interval midpoint'):
        plots.plot_model_layer_ewma_return_bridge(
            attribution=broken,
            detailed_mode=False,
        )
    plt.close('all')


def test_sharpe_bridge_uses_common_denominator_contributions(monkeypatch) -> None:
    """The Sharpe plot delegates to additive contributions and keeps the net identity."""
    attribution = _attribution(with_net=True)
    captured: dict[str, object] = {}
    contributions = pd.Series({
        'Benchmark': 0.50,
        'Systematic': 0.45,
        'Risk Layer': 0.15,
        'Signal Layer': 0.10,
        'Integration': 0.12,
        'Trading Cost Drag': -0.04,
        'Full Model Net': 0.78,
    })

    def fake_contributions(
            attribution: ModelLayerEwmaRegressionAttribution,
    ) -> pd.Series:
        """Capture the numerical delegation and return deterministic stage levels."""
        captured['attribution'] = attribution
        return contributions

    monkeypatch.setattr(plots, '_compute_ewma_sharpe_contributions', fake_contributions)
    fig = plots.plot_model_layer_ewma_sharpe_bridge(
        attribution=attribution,
        model_name='Mac',
    )
    try:
        assert captured == {'attribution': attribution}
        ax = fig.axes[0]
        assert ax.get_title(loc='left') == (
            'Risk and signal contributions explain MAC Sharpe beyond the\nbenchmark'
        )
        assert len(ax.patches) == 7
        tick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
        assert any('Trading-cost' in label for label in tick_labels)
        assert any('Mac\nnet' in label for label in tick_labels)
        assert all('gross' not in label.lower() for label in tick_labels)
        annotations = [text.get_text() for text in ax.texts]
        assert '+0.15' in annotations
        assert '+0.10' in annotations
        assert '+0.12' in annotations
        assert '-0.04' in annotations
        assert '0.78' in annotations
    finally:
        plt.close(fig)


def test_rolling_ewma_regression_alpha_plot_ends_at_current_attribution() -> None:
    """The public rolling-alpha plot shows additive paths ending at the current WLS bars."""
    attribution = _attribution(with_net=True)
    fig = plots.plot_model_layer_rolling_ewma_regression_alpha(
        attribution=attribution,
        model_name='Rosaa',
    )
    try:
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert ax.get_title(loc='left') == (
            'ROSAA rolling 24-month EWMA-WLS annualised alpha'
        )
        assert len(ax.lines) == 5
        plotted = {line.get_label(): line.get_ydata()[-1] for line in ax.lines[1:]}
        expected = attribution.regression_table['An Alpha']
        np.testing.assert_allclose(plotted['Total model alpha'], expected['Full Model'])
        np.testing.assert_allclose(plotted['Risk-layer alpha'], expected['Risk Layer'])
        np.testing.assert_allclose(plotted['Signal-layer alpha'], expected['Signal Layer'])
        np.testing.assert_allclose(plotted['Integration alpha'], expected['Integration'])
        assert fig.get_facecolor()[:3] == (1.0, 1.0, 1.0)
        assert ax.get_facecolor()[:3] == (1.0, 1.0, 1.0)
    finally:
        plt.close(fig)


def test_bridge_functions_draw_on_supplied_axes_without_creating_a_figure() -> None:
    """Both public renderers follow the QIS convention of returning None for supplied axes."""
    attribution = _attribution(with_net=False)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    try:
        return_output = plots.plot_model_layer_ewma_return_bridge(
            attribution=attribution,
            detailed_mode=False,
            ax=axes[0],
        )
        sharpe_output = plots.plot_model_layer_ewma_sharpe_bridge(
            attribution=attribution,
            detailed_mode=False,
            ax=axes[1],
        )
        assert return_output is None
        assert sharpe_output is None
        assert len(axes[0].patches) == 7
        assert len(axes[1].patches) == 6
    finally:
        plt.close(fig)


def test_display_overrides_reject_unknown_semantic_keys() -> None:
    """Misspelled label and colour keys fail instead of being silently ignored."""
    attribution = _attribution(with_net=False)
    with pytest.raises(ValueError, match='unsupported keys'):
        plots.plot_model_layer_ewma_return_bridge(
            attribution=attribution,
            labels={'Risk layer': 'misspelled'},
        )
    plt.close('all')

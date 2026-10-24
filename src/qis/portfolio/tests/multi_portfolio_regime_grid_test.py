"""Regime estimation retains native benchmark observations on coarse report grids."""

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import qis
from qis.datasets import generate_synthetic_universe


@pytest.mark.parametrize('plot_frequency', ['ME', 'QE'])
def test_regime_shadows_keep_native_prices_on_coarse_plot_grid(
        plot_frequency: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Monthly regimes use actual monthly marks, including on a quarterly exposure panel."""
    universe = generate_synthetic_universe(start='2020-01-01', end='2025-12-31')
    benchmark = universe.benchmark_prices.iloc[:, 0]
    portfolio = qis.PortfolioData(nav=benchmark.rename('Strategy'))
    multi = qis.MultiPortfolioData([portfolio], benchmark_prices=benchmark.to_frame())
    plot_index = pd.date_range('2021-03-31', '2024-12-31', freq=plot_frequency)
    classifier = qis.BenchmarkReturnsQuantilesRegime(freq='ME')
    captured = []
    draw_shadows = qis.add_bnb_regime_shadows

    def record_prices(**kwargs) -> None:
        captured.append(kwargs['pivot_prices'])
        draw_shadows(**kwargs)

    monkeypatch.setattr(qis, 'add_bnb_regime_shadows', record_prices)
    fig, ax = plt.subplots()
    try:
        multi.add_regime_shadows(
            ax=ax, regime_benchmark=benchmark.name, index=plot_index,
            regime_classifier=classifier,
        )
        native_window = benchmark.loc['2021-03-31':'2024-12-31']
        pd.testing.assert_series_equal(captured[0], native_window)
        classified = classifier.compute_sampled_returns_with_regime_id(
            prices=captured[0], benchmark=benchmark.name,
        )
        # An independent monthly close-to-close calculation has no repeated quarterly marks.
        expected_returns = native_window.resample('ME').last().pct_change(fill_method=None)
        pd.testing.assert_series_equal(
            classified[benchmark.name].dropna(), expected_returns.dropna(), check_freq=False,
        )
        assert classified[classifier.REGIME_COLUMN].nunique() == 3
        assert ax.patches
    finally:
        plt.close(fig)

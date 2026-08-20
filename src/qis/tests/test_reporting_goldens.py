"""Structural layout regression tests for the multi-asset factsheet.

The repository does not store environment-specific baseline images. These tests instead render
the full A4 report across the reporting-frequency grid and pin its panel count and axes geometry.
Frequency labels and titles are covered by ``test_reporting_conventions.py``.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # deterministic headless rendering
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure
import pytest
import qis
from qis import ReportingFrequency
from qis.portfolio.reports.config import fetch_default_report_kwargs
from qis.portfolio.reports.multi_assets_factsheet import generate_multi_asset_factsheet


def _make_synthetic_prices(n_assets: int = 6, n_years: int = 14, seed: int = 7) -> pd.DataFrame:
    """reproducible correlated-GBM price panel (identical to test_reporting_conventions)."""
    rng = np.random.default_rng(seed)
    n = n_years * 260
    idx = pd.bdate_range(end=pd.Timestamp('2025-12-31'), periods=n)
    mu = rng.uniform(0.03, 0.12, n_assets) / 260.0
    sig = rng.uniform(0.10, 0.30, n_assets) / np.sqrt(260.0)
    rets = mu + sig * (0.5 * rng.standard_normal((n, 1)) + 0.85 * rng.standard_normal((n, n_assets)))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=idx, columns=[f'A{i}' for i in range(n_assets)])


def _multi_asset_figure(reporting_frequency: ReportingFrequency) -> Figure:
    prices = _make_synthetic_prices()
    tp = qis.get_time_period(df=prices)
    kw = fetch_default_report_kwargs(time_period=tp, reporting_frequency=reporting_frequency)
    return generate_multi_asset_factsheet(prices=prices, benchmark='A0', time_period=tp, **kw)


def _assert_figure_layout(figure: Figure) -> None:
    """the report has its expected panel grid entirely inside the figure."""
    assert len(figure.axes) == 16
    for axis in figure.axes:
        left, bottom, width, height = axis.get_position().bounds
        assert width > 0.0 and height > 0.0
        assert left >= 0.0 and bottom >= 0.0
        assert left + width <= 1.0 and bottom + height <= 1.0


@pytest.mark.parametrize('reporting_frequency', list(ReportingFrequency),
                         ids=[frequency.name.lower() for frequency in ReportingFrequency])
def test_multi_asset_report_layout(reporting_frequency: ReportingFrequency) -> None:
    figure = _multi_asset_figure(reporting_frequency)
    try:
        figure.canvas.draw()
        _assert_figure_layout(figure)
    finally:
        plt.close(figure)

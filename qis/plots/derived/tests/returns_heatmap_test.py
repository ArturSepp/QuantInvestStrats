"""Regression tests for long-history monthly-return heatmaps."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import qis.plots.derived.returns_heatmap as rhe


def _make_monthly_prices() -> pd.Series:
    index = pd.date_range('2000-01-31', '2025-12-31', freq='ME')
    returns = 0.005 + 0.015 * np.sin(np.arange(len(index)) / 5.0)
    return pd.Series(100.0 * np.cumprod(1.0 + returns), index=index, name='Strategy')


def test_returns_heatmap_limits_inverse_order_to_latest_calendar_years() -> None:
    """The summary limit retains the newest rows after full-history returns are computed."""
    fig, ax = plt.subplots()
    try:
        rhe.plot_returns_heatmap(prices=_make_monthly_prices(),
                                 is_inverse_order=True,
                                 max_years=20,
                                 ax=ax)
        labels = [label.get_text() for label in ax.get_yticklabels()]
        assert labels == [str(year) for year in range(2025, 2005, -1)]
    finally:
        plt.close(fig)


def test_ytd_colors_are_scaled_independently_from_monthly_returns() -> None:
    """Monthly and YTD magnitudes each span their own symmetric colour scale.

    The reference divides each block by its own maximum absolute value; this is independent of
    the plotting path and guards against the larger annual returns washing out monthly colours.
    """
    returns = pd.DataFrame({'Jan': [0.01, -0.02],
                            'Feb': [0.02, 0.01],
                            'YTD': [0.20, -0.10]},
                           index=['2024', '2025'])
    actual = rhe._scale_returns_heatmap_colors(returns_table=returns, ytd_name='YTD')
    expected = pd.DataFrame({'Jan': [0.5, -1.0],
                             'Feb': [1.0, 0.5],
                             'YTD': [1.0, -0.5]},
                            index=returns.index)
    pd.testing.assert_frame_equal(actual, expected)

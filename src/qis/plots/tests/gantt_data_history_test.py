"""The optional timeline backend preserves observed dates across Plotly versions."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('plotly', reason='the timeline backend requires the visualization extra')

from qis.datasets import generate_synthetic_universe  # noqa: E402
from qis.plots.derived.gantt_data_history import (  # noqa: E402
    plot_backfill_history, plot_price_history,
)


@pytest.fixture
def prices():
    """A short frozen panel with independently specified ragged endpoints."""
    data = generate_synthetic_universe(apply_quirks=False).prices.iloc[:8, :2].copy()
    data.iloc[:2, 0] = np.nan
    data.iloc[-2:, 1] = np.nan
    return data


def _intervals(figure):
    """Recover endpoints from the actual bar bases and millisecond widths."""
    result = {}
    for trace in figure.data:
        assert trace.type == 'bar'
        for task, start, duration in zip(trace.y, trace.base, trace.x):
            start = pd.Timestamp(start)
            result[(task, trace.name)] = (start, start + pd.to_timedelta(duration, unit='ms'))
    return result


def test_price_history_preserves_dates_labels_and_markers(prices):
    """Ragged data, custom descriptions and reference dates survive rendering."""
    first, second = prices.columns
    dates = prices.index
    descriptions = pd.Series({first: 'First', second: 'Second'})
    figure = plot_price_history(prices, descriptions=descriptions, resource_name='Observed',
                                dates_for_lines=[dates[0], dates[-1]])
    assert _intervals(figure) == {
        (f'{first}, First', 'Observed'): (dates[2], dates[-1]),
        (f'{second}, Second', 'Observed'): (dates[0], dates[-3]),
    }
    assert tuple(figure.layout.yaxis.categoryarray) == (f'{first}, First', f'{second}, Second')
    assert figure.layout.yaxis.autorange == 'reversed'
    assert all(trace.marker.color == 'rgb(0, 255, 100)' for trace in figure.data)
    assert [(pd.Timestamp(shape.x0), pd.Timestamp(shape.x1))
            for shape in figure.layout.shapes] == [(date, date) for date in (dates[0], dates[-1])]
    assert all(shape.yref == 'paper' and shape.y0 == 0 and shape.y1 == 1
               for shape in figure.layout.shapes)
    assert figure.to_json()


def test_backfill_history_preserves_segments_and_groups(prices):
    """Vendor, futures and cash segments share their original instrument row."""
    first, second = prices.columns
    dates = prices.index
    backfills = prices.copy()
    backfills.iloc[:, :] = 1.0
    vendor = backfills.copy()
    vendor.iloc[:5, :] = np.nan
    futures = backfills.copy()
    futures.iloc[:2, :] = np.nan
    futures[second] = np.nan
    figure = plot_backfill_history(backfills, vendor, futures, dates_for_lines=[dates[3]])
    assert _intervals(figure) == {
        (first, 'Bloomberg'): (dates[5], dates[-1]),
        (first, 'Futures Backfill'): (dates[2], dates[5]),
        (first, 'Cash Backfill'): (dates[0], dates[2]),
        (second, 'Bloomberg'): (dates[5], dates[-1]),
        (second, 'Cash Backfill'): (dates[0], dates[5]),
    }
    assert tuple(figure.layout.yaxis.categoryarray) == (first, second)
    assert {trace.name: trace.marker.color for trace in figure.data} == {
        'Bloomberg': 'rgb(0, 255, 100)',
        'Futures Backfill': 'rgb(100, 100, 100)',
        'Cash Backfill': 'rgb(220, 0, 0)',
    }
    assert len(figure.layout.shapes) == 1 and figure.layout.shapes[0].yref == 'paper'
    assert figure.to_json()

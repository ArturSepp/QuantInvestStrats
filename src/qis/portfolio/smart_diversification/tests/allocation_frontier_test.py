"""Contracts of the table-driven overlay allocation frontier."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import qis
from qis.portfolio.smart_diversification import plot_overlay_allocation_frontier


def _points():
    """Return supplied stack coordinates, without a return-history dependency."""
    return pd.DataFrame({'bear_sharpe': [-0.8, -0.4, -0.2, -0.3],
                         'sharpe': [0.5, 0.6, 0.7, 0.65]},
                        index=['Core', 'Trend', 'Defensive', 'Selected'])


def test_frontier_preserves_policy_order_and_coordinates():
    """A folded frontier is joined in policy order, retaining slack duplicates."""
    points = _points()
    frontier = pd.DataFrame({'bear_sharpe': [-0.3, -0.3, -0.1, -0.2],
                             'sharpe': [0.7, 0.7, 0.68, 0.6]},
                            index=[0.0, 0.2, 0.4, 0.6])
    original_points, original_frontier = points.copy(), frontier.copy()
    fig = plot_overlay_allocation_frontier(points, frontier, benchmark='Core')
    try:
        line = fig.axes[0].lines[0]
        np.testing.assert_array_equal(line.get_xdata(), frontier.bear_sharpe)
        np.testing.assert_array_equal(line.get_ydata(), frontier.sharpe)
        np.testing.assert_array_equal(fig.axes[0].lines[1].get_xdata(), [-0.8, -0.8])
        pd.testing.assert_frame_equal(points, original_points)
        pd.testing.assert_frame_equal(frontier, original_frontier)
    finally:
        plt.close(fig)


def test_group_alignment_highlights_and_supplied_axis():
    """Groups align by name, selected points are drawn once, and styles stay caller-owned."""
    points = _points()
    groups = pd.Series({'Selected': 'Funds', 'Defensive': 'Funds',
                        'Trend': 'Funds', 'Core': 'Core'})
    highlights = {'Selected': dict(marker='D', label='Chosen floor', color='orange')}
    fig, ax = plt.subplots()
    try:
        result = plot_overlay_allocation_frontier(
            points, groups=groups, benchmark='Core', highlights=highlights,
            group_styles={'Funds': dict(color='blue', marker='v')},
            annotations=['Trend'], label_offsets={'Trend': (3, -7)}, ax=ax, legend_loc=None)
        assert result is None
        plotted = np.vstack([item.get_offsets() for item in ax.collections])
        np.testing.assert_array_equal(plotted, points[['bear_sharpe', 'sharpe']].to_numpy())
        assert [item.get_label() for item in ax.collections] == ['Core', 'Funds', 'Chosen floor']
        assert [text.get_text() for text in ax.texts] == ['Trend']
        assert ax.texts[0].get_position() == (3, -7)
        assert highlights['Selected'] == dict(marker='D', label='Chosen floor', color='orange')
        assert ax.get_legend() is None
        assert qis.plot_overlay_allocation_frontier is plot_overlay_allocation_frontier
    finally:
        plt.close(fig)


@pytest.mark.parametrize('fault', ['missing_column', 'duplicate_names', 'nonfinite',
                                   'missing_group', 'duplicate_group', 'missing_benchmark',
                                   'missing_highlight', 'missing_annotation', 'empty_frontier',
                                   'nonfinite_frontier', 'same_columns'])
def test_invalid_coordinates_and_labels_fail_before_drawing(fault):
    """Do not silently omit malformed points or draw partially labelled figures."""
    points = _points()
    kwargs = {}
    if fault == 'missing_column':
        points = points.drop(columns='bear_sharpe')
    elif fault == 'duplicate_names':
        points.index = ['Core'] * len(points)
    elif fault == 'nonfinite':
        points.iloc[0, 0] = np.nan
    elif fault == 'missing_group':
        kwargs['groups'] = pd.Series({'Core': 'Core'})
    elif fault == 'duplicate_group':
        kwargs['groups'] = pd.Series(['a', 'b'], index=['Core', 'Core'])
    elif fault == 'missing_benchmark':
        kwargs['benchmark'] = 'Absent'
    elif fault == 'missing_highlight':
        kwargs['highlights'] = {'Absent': {}}
    elif fault == 'missing_annotation':
        kwargs['annotations'] = ['Absent']
    elif fault == 'empty_frontier':
        kwargs['frontier_stats'] = points.iloc[:0]
    elif fault == 'nonfinite_frontier':
        kwargs['frontier_stats'] = points * np.inf
    elif fault == 'same_columns':
        kwargs['y_column'] = 'bear_sharpe'
    figures = plt.get_fignums()
    with pytest.raises(ValueError):
        plot_overlay_allocation_frontier(points, **kwargs)
    assert plt.get_fignums() == figures

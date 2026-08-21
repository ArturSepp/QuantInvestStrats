import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pytest
import qis.plots.utils as put
from qis.plots.boxplot import (plot_box)


@pytest.mark.parametrize(
    ('hue_order', 'expected_n_colors'),
    [
        (None, 3),
        (['h2', 'h1', 'h0', 'missing'], 4),
    ],
    ids=['observed-hues', 'explicit-hue-order'],
)
def test_plot_box_palette_matches_hue_cardinality(
        monkeypatch: pytest.MonkeyPatch,
        hue_order: list[str] | None,
        expected_n_colors: int,
        ) -> None:
    """Size the categorical palette from the hue categories.

    Args:
        monkeypatch: Pytest fixture used to record the requested palette size.
        hue_order: Explicit hue order, or None to use the observed hue values.
        expected_n_colors: Expected number of categorical palette colors.
    """
    data = pd.DataFrame({
        'x': ['left'] * 3 + ['right'] * 3,
        'hue': ['h0', 'h1', 'h2'] * 2,
        'value': np.arange(6, dtype=float),
    })
    requested_sizes: list[int] = []
    original_get_n_colors = put.get_n_colors

    def capture_palette_size(n: int):
        requested_sizes.append(n)
        return original_get_n_colors(n=n)

    monkeypatch.setattr(put, 'get_n_colors', capture_palette_size)
    monkeypatch.setattr(sns, 'boxplot', lambda **kwargs: None)
    fig, ax = plt.subplots()
    try:
        plot_box(df=data, x='x', y='value', hue='hue', hue_order=hue_order,
                 legend_loc=None, ax=ax)
    finally:
        plt.close(fig)

    assert requested_sizes == [expected_n_colors]


def test_plot_box_palette_includes_unused_categorical_hues(
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """Keep one palette color for every declared categorical hue.

    Args:
        monkeypatch: Pytest fixture used to record the requested palette size.
    """
    data = pd.DataFrame({
        'x': ['left', 'left', 'middle', 'middle', 'right', 'right'],
        'hue': pd.Categorical(
            ['h0', 'h1', 'h0', 'h1', 'h0', 'h1'],
            categories=['h0', 'h1', 'unused'],
        ),
        'value': np.arange(6, dtype=float),
    })
    requested_sizes: list[int] = []
    original_get_n_colors = put.get_n_colors

    def capture_palette_size(n: int):
        requested_sizes.append(n)
        return original_get_n_colors(n=n)

    monkeypatch.setattr(put, 'get_n_colors', capture_palette_size)
    fig, ax = plt.subplots()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            plot_box(df=data, x='x', y='value', hue='hue', legend_loc=None, ax=ax)
        palette_warnings = [warning for warning in caught
                            if 'palette list has fewer values' in str(warning.message)]
    finally:
        plt.close(fig)

    assert requested_sizes == [3]
    assert palette_warnings == []

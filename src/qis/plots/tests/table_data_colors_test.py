"""Regression coverage for data-cell colours with and without an index column."""
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402

from qis.plots.table import plot_df_table  # noqa: E402
from qis.plots.derived.returns_heatmap import plot_sorted_periodic_returns  # noqa: E402
from qis.plots.utils import get_n_colors  # noqa: E402


@pytest.mark.parametrize('add_index_as_column', [True, False])
def test_data_colors_follow_cells_with_or_without_index(add_index_as_column):
    """Every cell uses its own colour; headers and optional row labels stay unchanged."""
    frame = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['A', 'B', 'C'],
                         index=['First', 'Second'])
    colors = [['red', 'green', 'blue'], ['cyan', 'magenta', 'yellow']]
    fig, ax = plt.subplots()
    try:
        plot_df_table(frame, add_index_as_column=add_index_as_column, data_colors=colors,
                      header_color='black', row_colors=['white'], ax=ax)
        cells = ax.tables[0].get_celld()
        first_data_column = 1 if add_index_as_column else 0
        for row in range(2):
            for column in range(3):
                cell = cells[(row + 1, column + first_data_column)]
                assert cell.get_text().get_text() == str(frame.iloc[row, column])
                np.testing.assert_allclose(cell.get_facecolor(), to_rgba(colors[row][column]))
        for column in range(3 + first_data_column):
            np.testing.assert_allclose(cells[(0, column)].get_facecolor(), to_rgba('black'))
        if add_index_as_column:
            for row in (1, 2):
                np.testing.assert_allclose(cells[(row, 0)].get_facecolor(), to_rgba('white'))
    finally:
        plt.close(fig)


def test_sorted_annual_returns_keep_each_asset_color_when_ranks_change():
    """Annual leaders rotate while each asset retains its assigned colour and return."""
    prices = pd.DataFrame(
        [[100, 100, 100], [130, 120, 110], [143, 156, 132], [171.6, 171.6, 171.6]],
        index=pd.date_range('2019-12-31', periods=4, freq='YE'), columns=['A', 'B', 'C'])
    colors = dict(zip(prices.columns, get_n_colors(n=3, is_fixed_n_colors=False)))
    expected = [ [('A', '30%'), ('B', '20%'), ('C', '10%')],
                 [('B', '30%'), ('C', '20%'), ('A', '10%')],
                 [('C', '30%'), ('A', '20%'), ('B', '10%')] ]
    fig, ax = plt.subplots()
    try:
        plot_sorted_periodic_returns(prices, freq='YE', date_format='%Y',
                                     add_total=False, ax=ax)
        cells = ax.tables[0].get_celld()
        for column, year in enumerate(expected):
            assert cells[(0, column)].get_text().get_text() == str(2020 + column)
            for rank, (asset, annual_return) in enumerate(year, start=1):
                cell = cells[(rank, column)]
                assert cell.get_text().get_text() == f'{asset}\n{annual_return}'
                np.testing.assert_allclose(cell.get_facecolor(), to_rgba(colors[asset]))
    finally:
        plt.close(fig)

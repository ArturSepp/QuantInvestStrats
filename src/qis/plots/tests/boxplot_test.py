import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pytest
from enum import Enum
import qis.utils.df_melt as dfm
import qis.plots.utils as put
from qis.plots.boxplot import (plot_box, df_boxplot_by_classification_var,
                               df_dict_boxplot_by_columns, df_boxplot_by_index,
                               df_boxplot_by_columns)



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


class LocalTests(Enum):
    RETURNS_BOXPLOT = 1
    DF_BOXPLOT = 2
    DF_BOXPLOT_INDEX = 3
    DF_WEIGHTS = 4
    DF_DICT = 5


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    from qis.tests.price_data_test import load_etf_data
    prices = load_etf_data()
    returns = prices.asfreq('QE', method='ffill').pct_change()

    if local_test == LocalTests.RETURNS_BOXPLOT:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        global_kwargs = dict(fontsize=8, linewidth=0.5, weight='normal', markersize=1)

        index_name = 'date'
        value_name = 'returns'

        box_data = dfm.melt_df_by_columns(df=returns, x_index_var_name=index_name, y_var_name=value_name)
        print(box_data)
        colors = put.compute_heatmap_colors(a=np.nanmean(returns.to_numpy(), axis=1))

        plot_box(df=box_data,
                 x=index_name,
                 y=value_name,
                 original_index=returns.index,
                 colors=colors,
                 xlabel=False,
                 ax=ax,
                 **global_kwargs)

    elif local_test == LocalTests.DF_BOXPLOT:
        # returns by the quantiles of the first variable
        var = returns.columns[0]
        df_boxplot_by_classification_var(df=returns[var].to_frame(), x=var, y=var)

    elif local_test == LocalTests.DF_BOXPLOT_INDEX:
        df_boxplot_by_index(df=returns)

    elif local_test == LocalTests.DF_WEIGHTS:
        df_boxplot_by_columns(df=prices,
                              hue_var_name='instruments',
                              y_var_name='weights',
                              ylabel='weights',
                              legend_loc=None,
                              showmedians=True,
                              add_y_median_labels=True)

    elif local_test == LocalTests.DF_DICT:
        dfs = {'alts': prices, 'bal': 0.5*prices}
        with sns.axes_style("darkgrid"):
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            df_dict_boxplot_by_columns(dfs=dfs,
                                       hue_var_name='instruments',
                                       y_var_name='weights',
                                       ylabel='weights',
                                       legend_loc='upper center',
                                       showmedians=True,
                                       add_y_median_labels=True,
                                       ncols=2,
                                       ax=ax)

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.RETURNS_BOXPLOT)

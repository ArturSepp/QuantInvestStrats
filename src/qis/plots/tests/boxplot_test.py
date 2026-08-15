import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from enum import Enum
import qis.utils.df_melt as dfm
import qis.plots.utils as put
from qis.plots.boxplot import (plot_box, df_boxplot_by_classification_var,
                               df_dict_boxplot_by_columns, df_boxplot_by_index,
                               df_boxplot_by_columns)



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

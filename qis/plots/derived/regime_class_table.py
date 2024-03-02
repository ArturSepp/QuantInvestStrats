"""
plot tables for regime classification
"""
# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from enum import Enum

# qis
import qis.utils.df_cut as dfc
import qis.utils.df_str as dfs
import qis.plots.table as ptb


def get_quantile_class_table(data: pd.DataFrame,
                             x_column: str,
                             y_column: Optional[str] = None,
                             num_buckets: int = 4,
                             y_data_lag: Optional[int] = None,
                             hue_name: str = 'hue',
                             xvar_format: str = '{:.2f}'
                             ) -> pd.DataFrame:
    scatter_data, _ = dfc.add_quantile_classification(df=data, x_column=x_column, hue_name=hue_name,
                                                      num_buckets=num_buckets,
                                                      bins=None,
                                                      xvar_format=xvar_format)
    if y_column is None:
        y_column = x_column

    if y_data_lag is not None:
        scatter_data[y_column] = scatter_data[y_column].shift(y_data_lag)
        # scatter_data = scatter_data.dropna()

    stats = scatter_data[[y_column, hue_name]].groupby(hue_name, sort=False).agg(['count', 'mean', 'std'])
    stats.columns = stats.columns.get_level_values(1)
    stats.insert(loc=0, column='freq', value=stats['count'].to_numpy() / np.sum(stats['count'].to_numpy()))
    return stats


def plot_quantile_class_table(data: pd.DataFrame,
                              x_column: str,
                              y_column: Optional[str] = None,
                              num_buckets: int = 4,
                              hue_name: str = 'hue',
                              var_format: str = '{:.2%}',
                              xvar_format: str = '{:.2f}',
                              y_data_lag: Optional[int] = None,
                              ax: plt.Subplot = None,
                              **kwargs
                              ) -> None:
    stats = get_quantile_class_table(data=data, x_column=x_column, y_column=y_column,
                                     num_buckets=num_buckets,
                                     y_data_lag=y_data_lag,
                                     hue_name=hue_name,
                                     xvar_format=xvar_format)

    # stats is freq, count, mean and std
    stats = dfs.df_to_str(stats, var_formats=['{:.0%}', '{:.0f}', var_format, var_format])

    ptb.plot_df_table(df=stats,
                      index_column_name=hue_name,
                      ax=ax,
                      **kwargs)


class UnitTests(Enum):
    QUANTILE_CLASS_TABLE = 1


def run_unit_test(unit_test: UnitTests):

    from qis.test_data import load_etf_data
    prices = load_etf_data().dropna()
    returns = prices.asfreq('QE', method='ffill').pct_change().dropna()

    if unit_test == UnitTests.QUANTILE_CLASS_TABLE:
        plot_quantile_class_table(data=returns, x_column='SPY', num_buckets=4, hue_name='quantile regime')

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.QUANTILE_CLASS_TABLE

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
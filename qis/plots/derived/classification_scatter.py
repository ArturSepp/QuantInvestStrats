"""
plot classification regression
"""
# built in
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from enum import Enum

# quis
import qis.plots.scatter as psc
from qis.utils.df_cut import add_quantile_classification


def plot_classification_scatterplot(data: pd.DataFrame,
                                    x_column: str,
                                    y_column: str,
                                    hue_name: str = 'hue',
                                    num_buckets: Optional[int] = None,
                                    bins: np.ndarray = np.array([-3.0, -1.5, 0.0, 1.5, 3.0]),
                                    order: int = 1,
                                    title: str = None,
                                    full_sample_order: int = 3,
                                    markersize: int = 10,
                                    xvar_format: str = '{:.2f}',
                                    yvar_format: str = '{:.2f}',
                                    fit_intercept: bool = False,
                                    ax: plt.Subplot = None,
                                    **kwargs
                                    ) -> None:
    """
    add bin classification using x_column
    """
    df, _ = add_quantile_classification(df=data, x_column=x_column, hue_name=hue_name, num_buckets=num_buckets, bins=bins)

    psc.plot_scatter(df=df,
                     x_column=x_column,
                     y_column=y_column,
                     hue=hue_name,
                     fit_intercept=fit_intercept,
                     title=title,
                     order=order,
                     full_sample_order=full_sample_order,
                     markersize=markersize,
                     xvar_format=xvar_format,
                     yvar_format=yvar_format,
                     add_universe_model_ci=False,
                     add_hue_model_label=True,
                     ax=ax,
                     **kwargs)


def get_data(is_random_beta: bool = True,
             n: int = 10000
             ) -> pd.DataFrame:

    x = np.random.normal(0.0, 1.0, n)
    eps = np.random.normal(0.0, 1.0, n)

    if is_random_beta:
        beta = np.random.normal(1.0, 1.0, n)*np.abs(x)
    else:
        beta = np.ones(n)

    y = beta*x + eps
    df = pd.concat([pd.Series(x, name='x'), pd.Series(y, name='y')], axis=1)
    df = df.sort_values(by='x', axis=0)

    return df


class UnitTests(Enum):
    SCATTER = 1


def run_unit_test(unit_test: UnitTests):

    np.random.seed(2)  # freeze seed

    df1 = get_data(n=100000)
    print(df1)

    if unit_test == UnitTests.SCATTER:
        plot_classification_scatterplot(data=df1, x_column='x', y_column='y')

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.SCATTER

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

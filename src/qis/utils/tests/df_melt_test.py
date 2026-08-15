
import pandas as pd
import numpy as np
from enum import Enum
from qis.utils.df_melt import melt_scatter_data_with_xvar, melt_df_by_columns



class LocalTests(Enum):
    PD_MELT = 1
    SCATTER_DATA = 2
    MELT_DF_BY_COLUMNS = 3


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """
    from qis.tests.price_data_test import load_etf_data
    prices = load_etf_data().dropna().asfreq('QE', method='ffill')
    returns = prices.pct_change()

    if local_test == LocalTests.PD_MELT:
        df = pd.DataFrame(data=[[0, 1, 2.5, 0], [2, 3, 5.0, 0], [np.nan, 1, 1, 1]],
                          index=['cat', 'dog', 'missing'],
                          columns=['weight', 'height', 'age', 'sex'])
        print(df)
        melted = pd.melt(df, value_vars=df.columns, var_name='myVarname', value_name='myValname')
        print(f"melted=\n{melted}")

        scatter_data = melt_scatter_data_with_xvar(df=df, xvar_str='age', y_column='weight_height')
        print(f"scatter_data=\n{scatter_data}")

        box_data = melt_df_by_columns(df=df, x_index_var_name='animal', hue_var_name='hue_features', y_var_name='observations')
        print(f"box_data=\n{box_data}")

    elif local_test == LocalTests.SCATTER_DATA:
        scatter_data = melt_scatter_data_with_xvar(df=returns, xvar_str='SPY')
        print(scatter_data)

    elif local_test == LocalTests.MELT_DF_BY_COLUMNS:
        box_data = melt_df_by_columns(df=returns)
        print(box_data)


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.PD_MELT)


import numpy as np
import pandas as pd
from enum import Enum
from qis.utils.df_agg import compute_df_desc_data
import qis.utils.np_ops as npo
import qis.utils.df_ops as dfo

class LocalTests(Enum):
    STACK = 1
    NAN_MEAN = 2
    TEST3 = 3
    DESC_DF = 4


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    if local_test == LocalTests.STACK:
        df = pd.DataFrame(data=[[0, 1], [2, 3]],
                          index=['cat', 'dog'],
                          columns=['weight', 'height'])
        print(df)
        stacked = df.stack()
        print(stacked)
        print(type(stacked.index))

        # melt to
        melted = pd.melt(df, value_vars=df.columns, var_name='myVarname', value_name='myValname')
        print(melted)

    elif local_test == LocalTests.NAN_MEAN:

        # 4 * 3 matrix
        a = np.array([[np.nan, np.nan, np.nan],
                      [np.nan, np.nan, 1.0],
                      [np.nan, 2.0, 2.0],
                      [2.0, 3.0, 4.0]])

        #print(f"a={a[0]}\nmean(axis=0)={nan_func_to_data(a=a[0], func=np.nanmean, axis=0)};")

        print(f"a={a}\nmean(axis=0)={npo.nan_func_to_data(a=a, func=npo.np_nanmean, axis=0)};")

        print(f"a={a}\nmean(axis=1)={npo.nan_func_to_data(a=a, func=npo.np_nanmean, axis=1)};")

        pd_a = pd.DataFrame(a)
        print(f"pd_a={pd_a};")

        lambda_mean = pd_a.apply(lambda x: x.mean(), axis=1)
        # pd_a_pd = pd_a.apply(lambda x: np.nanmean(x), axis=1)
        print(f"lambda_mean\n{lambda_mean};")

        lambda_std = pd_a.apply(lambda x: x.std(), axis=1)
        # pd_a_pd = pd_a.apply(lambda x: np.nanstd(x), axis=1)
        print(f"lambda_std\n{lambda_std};")

    elif local_test == LocalTests.TEST3:
        df = pd.DataFrame({'Date': ['2015-05-08', '2015-05-07', '2015-05-06', '2015-05-05', '2015-05-08', '2015-05-07',
                                    '2015-05-06', '2015-05-05'],
                           'Sym': ['aapl', 'aapl', 'aapl', 'aapl', 'aaww', 'aaww', 'aaww', 'aaww'],
                           'Data2': [11, 8, 10, 15, 110, 60, 100, 40],
                           'Data3': [5, 8, 6, 1, 50, 100, 60, 120]})
        print(df)
        df['Data4'] = df['Data3'].groupby(df['Date']).transform('sum')
        print(df)

    elif local_test == LocalTests.DESC_DF:
        df = pd.DataFrame({'Date': ['2015-05-08', '2015-05-07', '2015-05-06', '2015-05-05', '2015-05-08', '2015-05-07',
                                    '2015-05-06', '2015-05-05'],
                           'Data2': [11, 8, 10, 15, 110, 60, 100, 40],
                           'Data3': [5, 8, 6, 1, 50, 100, 60, 120]}).set_index('Date')
        print(df)
        df_desc_data = compute_df_desc_data(df=df, axis=0)
        print(df_desc_data)


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.DESC_DF)

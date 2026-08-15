import pandas as pd
import numpy as np
from enum import Enum
from qis.utils.df_ops import (align_df1_to_df2,
                              compute_last_score,
                              reindex_upto_last_nonnan,
                              get_nonnan_index,
                              merge_dfs_on_column)


class LocalTests(Enum):
    ALIGN = 1
    SCORES = 2
    NONNANINDEX = 3
    REINDEX_UPTO_LAST_NONAN = 4
    MERGE_DFS_ON_COLUMNS = 5
    NONNAN_INDEX = 6


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    if local_test == LocalTests.ALIGN:

        desc_dict1 = {'f0': (0.5, 0.5), 'f1': (0.5, 0.5), 'f2': (0.5, 0.5), 'f3': (0.5, 0.5), 'f4': (0.5, 0.5)}
        df1 = pd.DataFrame.from_dict(desc_dict1, orient='index', columns=['as1', 'as2'])

        desc_dict2 = {'f1': (1.0, 1.0), 'f2': (1.0, 1.0), 'f5': (1.0, 1.0), 'f6': (1.0, 1.0)}
        df2 = pd.DataFrame.from_dict(desc_dict2, orient='index', columns=['as1', 'as3'])

        print(f"df1=\n{df1}")
        print(f"df2=\n{df2}")

        df1_, df2_ = align_df1_to_df2(df1=df1, df2=df2, join='outer', axis=0)
        print(f"df1_=\n{df1_}")
        print(f"df2_=\n{df2_}")

        df1_, df2_ = align_df1_to_df2(df1=df1, df2=df2, join='outer', axis=None)
        print(f"df1_=\n{df1_}")
        print(f"df2_=\n{df2_}")

    elif local_test == LocalTests.SCORES:
        np.random.seed(1)
        nrows, ncols = 20, 5
        df = pd.DataFrame(data=np.random.normal(0.0, 1.0, size=(nrows, ncols)),
                          columns=[f"id{n + 1}" for n in range(ncols)])
        print(df)
        percentiles = compute_last_score(df=df)
        print(percentiles)

    elif local_test == LocalTests.REINDEX_UPTO_LAST_NONAN:

        values = [1.0, np.nan, 3.0, 4.0, np.nan, 6.0, np.nan, 1.0]
        dates = pd.date_range(start='1Jan2020', periods=len(values))
        ds = pd.Series({d: v for d, v in zip(dates, values)})
        print(ds)

        dates1 = pd.date_range(start='1Jan2020', periods=len(values)+2)
        post_filled = ds.reindex(index=dates1, method='ffill')
        print(post_filled)

        post_filled_up_nan = reindex_upto_last_nonnan(ds=ds, index=dates1, method='ffill')
        print(post_filled_up_nan)

    elif local_test == LocalTests.MERGE_DFS_ON_COLUMNS:
        data_entries = {'Bond1': pd.Series(['AAA', 100.00, 'A3'], index=['bbg_ticker', 'face', 'raiting']),
                        'Bond2': pd.Series(['AA', 100.00, 'A2'], index=['bbg_ticker', 'face', 'raiting']),
                        'Bond3': pd.Series(['A', 100.00, 'A1'], index=['bbg_ticker', 'face', 'raiting']),
                        'Bond4': pd.Series(['BBB', 100.00, 'B3'], index=['bbg_ticker', 'face', 'raiting'])}
        data_df = pd.DataFrame.from_dict(data_entries, orient='index')

        index_df = {'A': pd.Series([95.0], index=['price']),
                    'AA': pd.Series([99.0], index=['price']),
                    'AAA': pd.Series([101.0], index=['price']),
                    'B': pd.Series([90.0], index=['price'])}
        index_df = pd.DataFrame.from_dict(index_df, orient='index')
        print(data_df)
        print(index_df)
        df = merge_dfs_on_column(data_df=data_df, index_df=index_df)
        print(df)

    elif local_test == LocalTests.NONNAN_INDEX:
        # Create test data
        dates = pd.date_range('2024-01-01', periods=5, freq='D')

        # Test Series with NaNs at beginning
        series = pd.Series([np.nan, np.nan, 1.0, 2.0, np.nan], index=dates)
        result = get_nonnan_index(series, position='first')
        print(f"Expected {dates[2]}, got {result}")

        result = get_nonnan_index(series, position='last')
        print(f"Expected {dates[-2]}, got {result}")

        # Test Series with all NaNs (return last)
        series_all_nan = pd.Series([np.nan] * 5, index=dates)
        result = get_nonnan_index(series_all_nan, return_index_for_all_nans=-1)
        print(f"Expected {dates[-1]}, got {result}")

        # Test DataFrame
        df = pd.DataFrame({
            'A': [np.nan, 1.0, 2.0, 3.0, 4.0],
            'B': [np.nan, np.nan, np.nan, 5.0, 6.0],
            'C': [7.0, 8.0, 9.0, 10.0, 11.0]
        }, index=dates)

        result = get_nonnan_index(df)
        print(f"Expected [{dates[1]}, {dates[3]}, {dates[0]}], got {result}")


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.NONNAN_INDEX)

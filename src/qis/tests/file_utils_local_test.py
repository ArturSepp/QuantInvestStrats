"""
local tests for file_utils save/load round-trips and new validation behavior
"""

from enum import Enum
import warnings

import numpy as np
import pandas as pd

import qis.file_utils as fu
import qis.local_path as local_path


RESOURCE_PATH = local_path.get_paths()['RESOURCE_PATH']
TEST_FOLDER = 'test_file_utils'  # subfolder under RESOURCE_PATH to contain test artifacts


def _make_df(n: int = 10, ncols: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2024-01-01', periods=n, freq='D')
    return pd.DataFrame(rng.standard_normal((n, ncols)),
                        index=idx,
                        columns=[f'col_{i}' for i in range(ncols)])


def _make_tz_df(n: int = 10, seed: int = 0) -> pd.DataFrame:
    df = _make_df(n=n, seed=seed)
    df.index = df.index.tz_localize('UTC')
    return df


class LocalTests(Enum):
    EXCEL_SINGLE = 1
    EXCEL_LIST = 2
    EXCEL_DICT = 3
    EXCEL_TRANSPOSE_DICT = 4
    EXCEL_SERIES_COERCION = 5
    EXCEL_BAD_SHEET_NAMES = 6
    EXCEL_EMPTY_INPUT_RAISES = 7
    EXCEL_MISMATCHED_SHEET_NAMES_RAISES = 8
    EXCEL_APPEND_MODE = 9
    CSV_ROUNDTRIP = 10
    CSV_APPEND = 11
    CSV_DICT = 12
    CSV_NONE_RAISES = 13
    FEATHER_ROUNDTRIP = 14
    FEATHER_APPEND = 15
    FEATHER_DICT = 16
    PARQUET_ROUNDTRIP = 17
    PARQUET_DICT = 18
    PARQUET_SERIES_COERCION_IN_DICT = 19


def _save_kwargs() -> dict:
    return dict(local_path=RESOURCE_PATH, folder_name=TEST_FOLDER)


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    Writes test artifacts under RESOURCE_PATH/test_file_utils/ so they
    can be inspected after the run. Safe to re-run; files are overwritten.
    """

    if local_test == LocalTests.EXCEL_SINGLE:
        df = _make_tz_df()
        path = fu.save_df_to_excel(data=df, file_name='t_excel_single', **_save_kwargs())
        loaded = fu.load_df_from_excel(file_name='t_excel_single',
                                       sheet_name='Sheet1',
                                       local_path=RESOURCE_PATH,
                                       folder_name=TEST_FOLDER,
                                       delocalize=True)
        print(f"saved to {path}")
        print(loaded.head())

    elif local_test == LocalTests.EXCEL_LIST:
        dfs = [_make_df(seed=0), _make_df(seed=1), _make_df(seed=2)]
        path = fu.save_df_to_excel(data=dfs,
                                   sheet_names=['a', 'b', 'c'],
                                   file_name='t_excel_list',
                                   **_save_kwargs())
        print(f"saved to {path}")
        loaded = fu.load_df_dict_from_excel(file_name='t_excel_list',
                                            local_path=RESOURCE_PATH,
                                            folder_name=TEST_FOLDER)
        print('sheets:', list(loaded.keys()))

    elif local_test == LocalTests.EXCEL_DICT:
        dfs = {'alpha': _make_df(seed=0), 'beta': _make_df(seed=1)}
        path = fu.save_df_to_excel(data=dfs, file_name='t_excel_dict', **_save_kwargs())
        print(f"saved to {path}")
        loaded = fu.load_df_dict_from_excel(file_name='t_excel_dict',
                                            local_path=RESOURCE_PATH,
                                            folder_name=TEST_FOLDER)
        print('sheets:', list(loaded.keys()))

    elif local_test == LocalTests.EXCEL_TRANSPOSE_DICT:
        # verifies the transpose-for-dict fix (original code silently skipped it)
        df = _make_df(n=5, ncols=3)
        dfs = {'raw': df, 'transposed_too': df}
        path = fu.save_df_to_excel(data=dfs,
                                   file_name='t_excel_transpose_dict',
                                   transpose=True,
                                   **_save_kwargs())
        loaded = fu.load_df_dict_from_excel(file_name='t_excel_transpose_dict',
                                            local_path=RESOURCE_PATH,
                                            folder_name=TEST_FOLDER)
        print(f"saved to {path}")
        print('original shape:', df.shape)
        print('loaded raw shape:', loaded['raw'].shape, '(should be transposed)')

    elif local_test == LocalTests.EXCEL_SERIES_COERCION:
        s = _make_df().iloc[:, 0]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            path = fu.save_df_to_excel(data={'series_as_sheet': s},
                                       file_name='t_excel_series',
                                       **_save_kwargs())
            print(f"saved to {path}")
            print(f"warnings raised: {len(w)}")
            for item in w:
                print(f"  - {item.message}")

    elif local_test == LocalTests.EXCEL_BAD_SHEET_NAMES:
        # illegal chars, too long, duplicates
        dfs = {
            'a/b:c*name?[with]\\bad': _make_df(seed=0),
            'x' * 50: _make_df(seed=1),
            'dup': _make_df(seed=2),
            'dup ': _make_df(seed=3),  # trailing space -> not a duplicate by itself
        }
        # add an actual duplicate after sanitization
        dfs2 = {**dfs, 'dup2': _make_df(seed=4)}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            path = fu.save_df_to_excel(data=dfs2,
                                       file_name='t_excel_bad_names',
                                       **_save_kwargs())
            print(f"saved to {path}")
            for item in w:
                print(f"  warn: {item.message}")
        loaded = fu.load_df_dict_from_excel(file_name='t_excel_bad_names',
                                            local_path=RESOURCE_PATH,
                                            folder_name=TEST_FOLDER)
        print('final sheet names:', list(loaded.keys()))

    elif local_test == LocalTests.EXCEL_EMPTY_INPUT_RAISES:
        try:
            fu.save_df_to_excel(data=[None, None], file_name='t_excel_empty', **_save_kwargs())
        except ValueError as e:
            print(f"got expected ValueError: {e}")
        try:
            fu.save_df_to_excel(data=None, file_name='t_excel_none', **_save_kwargs())
        except (ValueError, AttributeError) as e:
            print(f"got expected error for None: {type(e).__name__}: {e}")

    elif local_test == LocalTests.EXCEL_MISMATCHED_SHEET_NAMES_RAISES:
        dfs = [_make_df(seed=0), _make_df(seed=1), _make_df(seed=2)]
        try:
            fu.save_df_to_excel(data=dfs,
                                sheet_names=['a', 'b'],  # too few
                                file_name='t_excel_mismatch',
                                **_save_kwargs())
        except ValueError as e:
            print(f"got expected ValueError (too few): {e}")
        try:
            fu.save_df_to_excel(data=dfs,
                                sheet_names='single_string',  # wrong type for list
                                file_name='t_excel_mismatch2',
                                **_save_kwargs())
        except TypeError as e:
            print(f"got expected TypeError (str vs list): {e}")

    elif local_test == LocalTests.EXCEL_APPEND_MODE:
        df1 = _make_df(seed=0)
        df2 = _make_df(seed=1)
        path = fu.save_df_to_excel(data={'first': df1},
                                   file_name='t_excel_append',
                                   mode='w',
                                   **_save_kwargs())
        print(f"wrote initial: {path}")
        fu.save_df_to_excel(data={'second': df2},
                            file_name='t_excel_append',
                            mode='a',
                            if_sheet_exists='replace',
                            **_save_kwargs())
        loaded = fu.load_df_dict_from_excel(file_name='t_excel_append',
                                            local_path=RESOURCE_PATH,
                                            folder_name=TEST_FOLDER)
        print('sheets after append:', list(loaded.keys()))

    elif local_test == LocalTests.CSV_ROUNDTRIP:
        df = _make_df()
        fu.save_df_to_csv(df=df, file_name='t_csv', **_save_kwargs())
        loaded = fu.load_df_from_csv(file_name='t_csv',
                                     local_path=RESOURCE_PATH,
                                     folder_name=TEST_FOLDER)
        print('round-trip equal:', np.allclose(df.values, loaded.values))

    elif local_test == LocalTests.CSV_APPEND:
        df1 = _make_df(n=5, seed=0)
        df2 = _make_df(n=5, seed=1)
        df2.index = df1.index + pd.Timedelta(days=5)
        fu.save_df_to_csv(df=df1, file_name='t_csv_append', **_save_kwargs())
        fu.update_df_in_csv(df=df2, file_name='t_csv_append', **_save_kwargs())
        loaded = fu.load_df_from_csv(file_name='t_csv_append',
                                     local_path=RESOURCE_PATH,
                                     folder_name=TEST_FOLDER)
        print(f"appended shape: {loaded.shape} (expected (10, 3))")

    elif local_test == LocalTests.CSV_DICT:
        dfs = {'one': _make_df(seed=0), 'two': _make_df(seed=1)}
        fu.save_df_dict_to_csv(datasets=dfs, **_save_kwargs())
        loaded = fu.load_df_dict_from_csv(dataset_keys=['one', 'two'],
                                          file_name=None,
                                          local_path=RESOURCE_PATH,
                                          folder_name=TEST_FOLDER)
        print('loaded keys:', list(loaded.keys()))

    elif local_test == LocalTests.CSV_NONE_RAISES:
        try:
            fu.save_df_to_csv(df=None, file_name='t_csv_none', **_save_kwargs())
        except ValueError as e:
            print(f"got expected ValueError: {e}")

    elif local_test == LocalTests.FEATHER_ROUNDTRIP:
        df = _make_df()
        fu.save_df_to_feather(df=df, file_name='t_feather', **_save_kwargs())
        loaded = fu.load_df_from_feather(file_name='t_feather',
                                         local_path=RESOURCE_PATH,
                                         folder_name=TEST_FOLDER)
        print('round-trip equal:', np.allclose(df.values, loaded.values))
        print('index preserved:', isinstance(loaded.index, pd.DatetimeIndex))

    elif local_test == LocalTests.FEATHER_APPEND:
        df1 = _make_df(n=5, seed=0)
        df2 = _make_df(n=5, seed=1)
        df2.index = df1.index + pd.Timedelta(days=5)
        fu.save_df_to_feather(df=df1, file_name='t_feather_append', **_save_kwargs())
        merged = fu.append_df_to_feather(df=df2, file_name='t_feather_append', **_save_kwargs())
        print(f"appended shape: {merged.shape} (expected (10, 3))")

    elif local_test == LocalTests.FEATHER_DICT:
        dfs = {'one': _make_df(seed=0), 'two': _make_df(seed=1)}
        fu.save_df_dict_to_feather(dfs=dfs, **_save_kwargs())
        loaded = fu.load_df_dict_from_feather(dataset_keys=['one', 'two'],
                                              file_name=None,
                                              local_path=RESOURCE_PATH,
                                              folder_name=TEST_FOLDER)
        print('loaded keys:', list(loaded.keys()))

    elif local_test == LocalTests.PARQUET_ROUNDTRIP:
        df = _make_df()
        fu.save_df_to_parquet(df=df, file_name='t_parquet', **_save_kwargs())
        loaded = fu.load_df_from_parquet(file_name='t_parquet',
                                         local_path=RESOURCE_PATH,
                                         folder_name=TEST_FOLDER)
        print('round-trip equal:', np.allclose(df.values, loaded.values))

    elif local_test == LocalTests.PARQUET_DICT:
        dfs = {'one': _make_df(seed=0), 'two': _make_df(seed=1)}
        fu.save_df_dict_to_parquet(datasets=dfs, **_save_kwargs())
        loaded = fu.load_df_dict_from_parquet(dataset_keys=['one', 'two'],
                                              file_name=None,
                                              local_path=RESOURCE_PATH,
                                              folder_name=TEST_FOLDER)
        print('loaded keys:', list(loaded.keys()))

    elif local_test == LocalTests.PARQUET_SERIES_COERCION_IN_DICT:
        # the original save_df_dict_to_parquet would crash on a Series; the refactor warns + coerces
        dfs = {'df_ok': _make_df(seed=0), 'series_gets_coerced': _make_df(seed=1).iloc[:, 0]}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            fu.save_df_dict_to_parquet(datasets=dfs, **_save_kwargs())
            for item in w:
                print(f"  warn: {item.message}")
        loaded = fu.load_df_dict_from_parquet(dataset_keys=['df_ok', 'series_gets_coerced'],
                                              file_name=None,
                                              local_path=RESOURCE_PATH,
                                              folder_name=TEST_FOLDER)
        print('loaded keys:', list(loaded.keys()))


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.EXCEL_SINGLE)

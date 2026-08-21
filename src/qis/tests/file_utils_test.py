"""
pytest unit tests for qis.file_utils save functions.

Uses tmp_path so tests never touch the configured RESOURCE_PATH / OUTPUT_PATH.
"""

import importlib.util

import numpy as np
import pandas as pd
import pytest

import qis.file_utils as fu

requires_pyarrow = pytest.mark.skipif(importlib.util.find_spec('pyarrow') is None,
                                      reason='parquet and feather io need pip install qis[io]')


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df() -> pd.DataFrame:
    idx = pd.date_range('2024-01-01', periods=10, freq='D')
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.standard_normal((10, 3)),
                        index=idx,
                        columns=['a', 'b', 'c'])


@pytest.fixture
def df_tz(df) -> pd.DataFrame:
    out = df.copy()
    out.index = out.index.tz_localize('UTC')
    return out


@pytest.fixture
def dfs_dict(df) -> dict:
    return {'one': df, 'two': df * 2}


# ---------------------------------------------------------------------------
# private helpers
# ---------------------------------------------------------------------------


class TestCoerceToDf:

    def test_dataframe_passthrough(self, df):
        assert fu._coerce_to_df(df, 'x') is df

    def test_series_to_frame(self):
        s = pd.Series([1, 2, 3], name='s')
        with pytest.warns(UserWarning, match='Series converted'):
            out = fu._coerce_to_df(s, 'x')
        assert isinstance(out, pd.DataFrame)
        assert out.shape == (3, 1)

    def test_none_returns_none(self):
        assert fu._coerce_to_df(None, 'x') is None

    def test_unsupported_warns_and_returns_none(self):
        with pytest.warns(UserWarning, match='unsupported type'):
            assert fu._coerce_to_df([1, 2, 3], 'x') is None
        with pytest.warns(UserWarning, match='unsupported type'):
            assert fu._coerce_to_df('hello', 'x') is None


class TestSanitizeSheetName:

    def test_truncates_long_name(self):
        taken: set = set()
        with pytest.warns(UserWarning, match='sanitized'):
            out = fu._sanitize_sheet_name('x' * 50, taken)
        assert len(out) == 31

    def test_replaces_illegal_chars(self):
        taken: set = set()
        with pytest.warns(UserWarning, match='sanitized'):
            out = fu._sanitize_sheet_name('a/b:c*d?e[f]g\\h', taken)
        assert all(ch not in out for ch in '[]:*?/\\')

    def test_deduplicates(self):
        taken = {'sheet'}
        with pytest.warns(UserWarning, match='Duplicate'):
            out = fu._sanitize_sheet_name('sheet', taken)
        assert out == 'sheet_2'
        assert 'sheet_2' in taken

    def test_clean_name_unchanged(self):
        taken: set = set()
        assert fu._sanitize_sheet_name('ok', taken) == 'ok'
        assert 'ok' in taken


# ---------------------------------------------------------------------------
# excel
# ---------------------------------------------------------------------------


class TestSaveDfToExcel:

    def test_single_roundtrip(self, tmp_path, df):
        fu.save_df_to_excel(data=df, file_name='t', local_path=str(tmp_path))
        loaded = fu.load_df_from_excel(file_name='t', sheet_name='Sheet1', local_path=str(tmp_path))
        pd.testing.assert_frame_equal(loaded, df, check_freq=False)

    def test_single_delocalizes_tz_index(self, tmp_path, df_tz):
        fu.save_df_to_excel(data=df_tz, file_name='t', local_path=str(tmp_path))
        loaded = fu.load_df_from_excel(file_name='t', sheet_name='Sheet1', local_path=str(tmp_path))
        assert loaded.index.tz is None

    def test_list_default_sheet_names(self, tmp_path, df):
        fu.save_df_to_excel(data=[df, df * 2], file_name='t', local_path=str(tmp_path))
        loaded = fu.load_df_dict_from_excel(file_name='t', local_path=str(tmp_path))
        assert list(loaded.keys()) == ['Sheet 1', 'Sheet 2']

    def test_list_custom_sheet_names(self, tmp_path, df):
        fu.save_df_to_excel(data=[df, df * 2],
                            sheet_names=['alpha', 'beta'],
                            file_name='t',
                            local_path=str(tmp_path))
        loaded = fu.load_df_dict_from_excel(file_name='t', local_path=str(tmp_path))
        assert list(loaded.keys()) == ['alpha', 'beta']

    def test_list_sheet_names_too_few_raises(self, tmp_path, df):
        with pytest.raises(ValueError, match='sheet_names'):
            fu.save_df_to_excel(data=[df, df, df],
                                sheet_names=['a', 'b'],
                                file_name='t',
                                local_path=str(tmp_path))

    def test_list_sheet_names_as_string_raises(self, tmp_path, df):
        with pytest.raises(TypeError, match='must be a list'):
            fu.save_df_to_excel(data=[df, df],
                                sheet_names='nope',
                                file_name='t',
                                local_path=str(tmp_path))

    def test_dict_uses_keys(self, tmp_path, dfs_dict):
        fu.save_df_to_excel(data=dfs_dict, file_name='t', local_path=str(tmp_path))
        loaded = fu.load_df_dict_from_excel(file_name='t', local_path=str(tmp_path))
        assert set(loaded.keys()) == {'one', 'two'}

    def test_dict_ignores_sheet_names_with_warning(self, tmp_path, dfs_dict):
        with pytest.warns(UserWarning, match='ignored for dict'):
            fu.save_df_to_excel(data=dfs_dict,
                                sheet_names=['x', 'y'],
                                file_name='t',
                                local_path=str(tmp_path))

    def test_transpose_applied_to_single(self, tmp_path, df):
        fu.save_df_to_excel(data=df, file_name='t', local_path=str(tmp_path), transpose=True)
        loaded = fu.load_df_from_excel(file_name='t', sheet_name='Sheet1', local_path=str(tmp_path))
        assert loaded.shape == (df.shape[1], df.shape[0])

    def test_transpose_applied_to_dict(self, tmp_path, df):
        """Regression: original code silently ignored transpose for dict inputs."""
        fu.save_df_to_excel(data={'x': df}, file_name='t', local_path=str(tmp_path), transpose=True)
        loaded = fu.load_df_from_excel(file_name='t', sheet_name='x', local_path=str(tmp_path))
        assert loaded.shape == (df.shape[1], df.shape[0])

    def test_series_coerced_in_list(self, tmp_path, df):
        s = df['a']
        with pytest.warns(UserWarning, match='Series converted'):
            fu.save_df_to_excel(data=[df, s],
                                sheet_names=['frame', 'series'],
                                file_name='t',
                                local_path=str(tmp_path))
        loaded = fu.load_df_dict_from_excel(file_name='t', local_path=str(tmp_path))
        assert 'series' in loaded

    def test_none_items_skipped_in_list(self, tmp_path, df):
        fu.save_df_to_excel(data=[df, None, df * 2],
                            sheet_names=['a', 'skip', 'c'],
                            file_name='t',
                            local_path=str(tmp_path))
        loaded = fu.load_df_dict_from_excel(file_name='t', local_path=str(tmp_path))
        assert set(loaded.keys()) == {'a', 'c'}

    def test_unsupported_type_warns_and_skipped(self, tmp_path, df):
        with pytest.warns(UserWarning, match='unsupported'):
            fu.save_df_to_excel(data=[df, 'not a df'],
                                sheet_names=['ok', 'bad'],
                                file_name='t',
                                local_path=str(tmp_path))
        loaded = fu.load_df_dict_from_excel(file_name='t', local_path=str(tmp_path))
        assert list(loaded.keys()) == ['ok']

    def test_all_none_raises(self, tmp_path):
        with pytest.raises(ValueError, match='No DataFrames'):
            fu.save_df_to_excel(data=[None, None],
                                file_name='t',
                                local_path=str(tmp_path))

    def test_sheet_name_sanitization_illegal_chars(self, tmp_path, df):
        with pytest.warns(UserWarning, match='sanitized'):
            fu.save_df_to_excel(data={'a/b:c*': df}, file_name='t', local_path=str(tmp_path))
        loaded = fu.load_df_dict_from_excel(file_name='t', local_path=str(tmp_path))
        sheet_name = next(iter(loaded.keys()))
        assert all(ch not in sheet_name for ch in '[]:*?/\\')

    def test_sheet_name_sanitization_truncation(self, tmp_path, df):
        long_name = 'x' * 50
        with pytest.warns(UserWarning, match='sanitized'):
            fu.save_df_to_excel(data={long_name: df}, file_name='t', local_path=str(tmp_path))
        loaded = fu.load_df_dict_from_excel(file_name='t', local_path=str(tmp_path))
        assert all(len(k) <= 31 for k in loaded.keys())

    def test_append_mode_adds_sheet(self, tmp_path, df):
        fu.save_df_to_excel(data={'first': df}, file_name='t', local_path=str(tmp_path), mode='w')
        fu.save_df_to_excel(data={'second': df * 2},
                            file_name='t',
                            local_path=str(tmp_path),
                            mode='a',
                            if_sheet_exists='replace')
        loaded = fu.load_df_dict_from_excel(file_name='t', local_path=str(tmp_path))
        assert set(loaded.keys()) == {'first', 'second'}


class TestSaveDfDictToExcel:

    def test_roundtrip(self, tmp_path, dfs_dict):
        fu.save_df_dict_to_excel(datasets=dfs_dict, file_name='t', local_path=str(tmp_path))
        loaded = fu.load_df_dict_from_excel(file_name='t', local_path=str(tmp_path))
        assert set(loaded.keys()) == {'one', 'two'}

    def test_delocalize(self, tmp_path, df_tz):
        fu.save_df_dict_to_excel(datasets={'tz': df_tz},
                                 file_name='t',
                                 local_path=str(tmp_path),
                                 delocalize=True)
        loaded = fu.load_df_from_excel(file_name='t', sheet_name='tz', local_path=str(tmp_path))
        assert loaded.index.tz is None

    def test_empty_dict_raises(self, tmp_path):
        with pytest.raises(ValueError, match='No DataFrames'):
            fu.save_df_dict_to_excel(datasets={}, file_name='t', local_path=str(tmp_path))

    def test_all_none_values_raises(self, tmp_path):
        with pytest.raises(ValueError, match='No DataFrames'):
            fu.save_df_dict_to_excel(datasets={'a': None, 'b': None},
                                     file_name='t',
                                     local_path=str(tmp_path))


# ---------------------------------------------------------------------------
# csv
# ---------------------------------------------------------------------------


class TestCsv:

    def test_roundtrip(self, tmp_path, df):
        fu.save_df_to_csv(df=df, file_name='t', local_path=str(tmp_path))
        loaded = fu.load_df_from_csv(file_name='t', local_path=str(tmp_path))
        np.testing.assert_allclose(loaded.values, df.values)

    def test_save_none_raises(self, tmp_path):
        with pytest.raises(ValueError, match='No DataFrame'):
            fu.save_df_to_csv(df=None, file_name='t', local_path=str(tmp_path))

    def test_save_series_coerced(self, tmp_path, df):
        with pytest.warns(UserWarning, match='Series converted'):
            fu.save_df_to_csv(df=df['a'], file_name='t', local_path=str(tmp_path))
        loaded = fu.load_df_from_csv(file_name='t', local_path=str(tmp_path))
        assert loaded.shape == (len(df), 1)

    def test_append_concatenates(self, tmp_path, df):
        df1 = df.iloc[:5]
        df2 = df.iloc[5:]
        fu.save_df_to_csv(df=df1, file_name='t', local_path=str(tmp_path))
        fu.update_df_in_csv(df=df2, file_name='t', local_path=str(tmp_path))
        loaded = fu.load_df_from_csv(file_name='t', local_path=str(tmp_path))
        assert len(loaded) == len(df)

    def test_append_none_raises(self, tmp_path):
        with pytest.raises(ValueError, match='No DataFrame'):
            fu.update_df_in_csv(df=None, file_name='t', local_path=str(tmp_path))

    def test_append_with_keep_drops_duplicates(self, tmp_path, df):
        fu.save_df_to_csv(df=df, file_name='t', local_path=str(tmp_path))
        fu.update_df_in_csv(df=df, file_name='t', local_path=str(tmp_path), keep='first')
        loaded = fu.load_df_from_csv(file_name='t', local_path=str(tmp_path))
        assert len(loaded) == len(df)

    def test_dict_roundtrip(self, tmp_path, dfs_dict):
        fu.save_df_dict_to_csv(datasets=dfs_dict, local_path=str(tmp_path))
        loaded = fu.load_df_dict_from_csv(dataset_keys=['one', 'two'],
                                          file_name=None,
                                          local_path=str(tmp_path))
        assert set(loaded.keys()) == {'one', 'two'}

    def test_dict_skips_unsupported(self, tmp_path, df):
        with pytest.warns(UserWarning, match='unsupported'):
            fu.save_df_dict_to_csv(datasets={'good': df, 'bad': 'not a df'},
                                   local_path=str(tmp_path))


# ---------------------------------------------------------------------------
# feather
# ---------------------------------------------------------------------------


@requires_pyarrow
class TestFeather:

    def test_roundtrip_preserves_index(self, tmp_path, df):
        fu.save_df_to_feather(df=df, file_name='t', local_path=str(tmp_path))
        loaded = fu.load_df_from_feather(file_name='t', local_path=str(tmp_path))
        pd.testing.assert_index_equal(loaded.index, df.index, check_names=False)

    def test_roundtrip_no_index_col(self, tmp_path, df):
        fu.save_df_to_feather(df=df, file_name='t', local_path=str(tmp_path), index_col=None)
        loaded = fu.load_df_from_feather(file_name='t', local_path=str(tmp_path), index_col=None)
        np.testing.assert_allclose(loaded.values, df.values)

    def test_save_none_raises(self, tmp_path):
        with pytest.raises(ValueError, match='No DataFrame'):
            fu.save_df_to_feather(df=None, file_name='t', local_path=str(tmp_path))

    def test_save_series_coerced(self, tmp_path, df):
        with pytest.warns(UserWarning, match='Series converted'):
            fu.save_df_to_feather(df=df['a'], file_name='t', local_path=str(tmp_path))

    def test_append_concatenates(self, tmp_path, df):
        df1 = df.iloc[:5]
        df2 = df.iloc[5:]
        fu.save_df_to_feather(df=df1, file_name='t', local_path=str(tmp_path))
        merged = fu.append_df_to_feather(df=df2, file_name='t', local_path=str(tmp_path))
        assert len(merged) == len(df)

    def test_append_none_raises(self, tmp_path):
        with pytest.raises(ValueError, match='No DataFrame'):
            fu.append_df_to_feather(df=None, file_name='t', local_path=str(tmp_path))

    def test_dict_roundtrip(self, tmp_path, dfs_dict):
        fu.save_df_dict_to_feather(dfs=dfs_dict, local_path=str(tmp_path))
        loaded = fu.load_df_dict_from_feather(dataset_keys=['one', 'two'],
                                              file_name=None,
                                              local_path=str(tmp_path))
        assert set(loaded.keys()) == {'one', 'two'}

    def test_dict_skips_unsupported(self, tmp_path, df):
        with pytest.warns(UserWarning, match='unsupported'):
            fu.save_df_dict_to_feather(dfs={'good': df, 'bad': 12345},
                                       local_path=str(tmp_path))


# ---------------------------------------------------------------------------
# parquet
# ---------------------------------------------------------------------------


@requires_pyarrow
class TestParquet:

    def test_roundtrip(self, tmp_path, df):
        fu.save_df_to_parquet(df=df, file_name='t', local_path=str(tmp_path))
        loaded = fu.load_df_from_parquet(file_name='t', local_path=str(tmp_path))
        pd.testing.assert_frame_equal(loaded, df, check_freq=False)

    def test_delocalize(self, tmp_path, df_tz):
        fu.save_df_to_parquet(df=df_tz, file_name='t', local_path=str(tmp_path), delocalize=True)
        loaded = fu.load_df_from_parquet(file_name='t', local_path=str(tmp_path))
        assert loaded.index.tz is None

    def test_save_none_raises(self, tmp_path):
        with pytest.raises(ValueError, match='No DataFrame'):
            fu.save_df_to_parquet(df=None, file_name='t', local_path=str(tmp_path))

    def test_save_series_coerced(self, tmp_path, df):
        with pytest.warns(UserWarning, match='Series converted'):
            fu.save_df_to_parquet(df=df['a'], file_name='t', local_path=str(tmp_path))

    def test_dict_roundtrip(self, tmp_path, dfs_dict):
        fu.save_df_dict_to_parquet(datasets=dfs_dict, local_path=str(tmp_path))
        loaded = fu.load_df_dict_from_parquet(dataset_keys=['one', 'two'],
                                              file_name=None,
                                              local_path=str(tmp_path))
        assert set(loaded.keys()) == {'one', 'two'}

    def test_dict_coerces_series(self, tmp_path, df):
        """Regression: original save_df_dict_to_parquet crashed on non-DataFrame values."""
        with pytest.warns(UserWarning, match='Series converted'):
            fu.save_df_dict_to_parquet(datasets={'df': df, 's': df['a']},
                                       local_path=str(tmp_path))
        loaded = fu.load_df_dict_from_parquet(dataset_keys=['df', 's'],
                                              file_name=None,
                                              local_path=str(tmp_path))
        assert set(loaded.keys()) == {'df', 's'}

"""Exact equivalence for the one-instrument group aggregation used by factsheets."""
import numpy as np
import pandas as pd
import pytest

from qis.utils.df_groups import agg_df_by_groups_ax1


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'int64', 'bool'])
def test_singleton_nansum_matches_row_reference_without_apply(dtype, monkeypatch):
    values = [[0, 1, -2], [3, 0, 4], [-5, 6, 0]]
    frame = pd.DataFrame(values, columns=['a', 'b', 'c'], dtype=dtype)
    if dtype.startswith('float'):
        frame.iloc[0, 0] = np.nan
        frame.iloc[1, 1] = -0.0
        frame.iloc[2, 2] = np.inf
    groups = pd.Series({'a': 'A', 'b': 'B', 'c': 'C'})
    expected = pd.DataFrame({group: frame[[asset]].apply(np.nansum, axis=1)
                             for asset, group in groups.items()})
    original_apply = pd.DataFrame.apply

    def reject_singleton_apply(self, func, *args, **kwargs):
        assert not (self.shape[1] == 1 and func is np.nansum), 'Slow singleton apply'
        return original_apply(self, func, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, 'apply', reject_singleton_apply)
    actual = agg_df_by_groups_ax1(frame, groups)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))


def test_mixed_groups_totals_and_custom_aggregator_keep_contract():
    frame = pd.DataFrame({'a': [1.0, np.nan], 'b': [3.0, 4.0], 'c': [5.0, np.nan]})
    groups = pd.Series({'a': 'AB', 'b': 'AB', 'c': 'C'})
    for aggregator in [np.nansum, np.nanmean]:
        expected = pd.DataFrame({
            'Total': frame.apply(aggregator, axis=1),
            'C': frame[['c']].apply(aggregator, axis=1),
            'AB': frame[['a', 'b']].apply(aggregator, axis=1),
        })
        actual = agg_df_by_groups_ax1(frame, groups, agg_func=aggregator,
                                      total_column='Total', group_order=['C', 'AB'])
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)

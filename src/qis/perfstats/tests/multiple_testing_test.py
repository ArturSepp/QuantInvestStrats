"""Independent family-level references and input contracts for Holm/BHY corrections."""

# packages
import numpy as np
import pandas as pd
import pytest
from fractions import Fraction

# qis
from qis.perfstats.multiple_testing import MultipleTestingMethod, adjust_pvalues


METHODS = [MultipleTestingMethod.HOLM, MultipleTestingMethod.BHY]


def _reference(values: np.ndarray, method: MultipleTestingMethod) -> np.ndarray:
    """Use the defining prefix maximum/suffix minimum, independently of statsmodels."""
    order = np.argsort(values, kind='stable')
    sorted_values = values[order]
    count = len(values)
    harmonic = sum(1.0 / rank for rank in range(1, count + 1))
    adjusted = np.empty(count, dtype=float)
    for position in range(count):
        if method is MultipleTestingMethod.HOLM:
            correction = max((count - j) * sorted_values[j] for j in range(position + 1))
        else:
            correction = min(count * harmonic * sorted_values[j] / (j + 1)
                             for j in range(position, count))
        adjusted[order[position]] = min(1.0, correction)
    return adjusted


def _hand_family() -> pd.Series:
    """Unsorted tied values requiring both a prefix maximum and a suffix minimum."""
    return pd.Series([0.07, 0.01, 0.03, 0.01, 0.8],
                     index=pd.Index(['late', 'first', 'middle', 'tie', 'poor'], name='hypothesis'),
                     name='raw evidence')


def test_holm_hand_calculation_and_default_rejection_boundary() -> None:
    """Sorted multipliers 0.05, 0.04, 0.09, 0.14, 0.8 need a forward maximum."""
    pvalues = _hand_family()
    result = adjust_pvalues(pvalues, family_id='five hypotheses')
    np.testing.assert_allclose(result['pvalue_adjusted'], [0.14, 0.05, 0.09, 0.05, 0.8],
                               rtol=1e-14, atol=0)
    assert result['reject'].tolist() == [False, True, False, True, False]
    assert result.attrs['method'] is MultipleTestingMethod.HOLM
    assert result.attrs['significance_level'] == 0.05
    assert result.attrs['error_control'] == 'FWER'


def test_bhy_hand_calculation_includes_harmonic_dependence_factor() -> None:
    """H_5=137/60 gives BHY values distinct from ordinary Benjamini-Hochberg."""
    result = adjust_pvalues(_hand_family(), family_id='five hypotheses',
                            method=MultipleTestingMethod.BHY)
    expected = [959 / 4800, 137 / 2400, 137 / 1200, 137 / 2400, 1.0]
    np.testing.assert_allclose(result['pvalue_adjusted'], expected, rtol=1e-14, atol=0)
    assert not result['reject'].any()
    # BH would reject the two tied low p-values at 5%; BHY must not silently use BH.
    assert result.loc['first', 'pvalue_adjusted'] > 0.05
    assert result.attrs['method'] is MultipleTestingMethod.BHY
    assert result.attrs['error_control'] == 'FDR'
    assert MultipleTestingMethod.HOLM.value == 'holm'
    assert MultipleTestingMethod.BHY.value == 'fdr_by'


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('values', [
    [0.0, 1.0],
    [0.0, 0.0, 1.0, 1.0],
    [0.01] * 5,
    [0.0001, 0.003, 0.02, 0.04, 0.049, 0.4, 0.02, 1.0],
    [0.6, 0.8, 0.9],
])
def test_independent_cumulative_reference_and_family_order(
        method: MultipleTestingMethod, values: list[float]) -> None:
    pvalues = pd.Series(values, index=[f'hypothesis-{i}' for i in range(len(values))])
    original = pvalues.copy(deep=True)
    for permutation in [np.arange(len(values)), np.arange(len(values))[::-1],
                        np.roll(np.arange(len(values)), 2)]:
        permuted = pvalues.iloc[permutation]
        result = adjust_pvalues(permuted, family_id='prespecified', method=method,
                                significance_level=0.1)
        expected = _reference(permuted.to_numpy(), method)
        np.testing.assert_allclose(result['pvalue_adjusted'], expected, rtol=1e-14, atol=0)
        np.testing.assert_array_equal(result['pvalue_raw'], permuted)
        np.testing.assert_array_equal(result['reject'], expected <= 0.1)
        pd.testing.assert_index_equal(result.index, permuted.index)
        assert np.all(result['pvalue_adjusted'] >= result['pvalue_raw'] - 1e-15)
        assert result['pvalue_adjusted'].between(0, 1).all()
        sorted_adjusted = result.loc[permuted.sort_values(kind='stable').index, 'pvalue_adjusted']
        assert np.all(np.diff(sorted_adjusted) >= -1e-15)
    pd.testing.assert_series_equal(pvalues, original)


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('value,reject', [(0.0, True), (0.05, True), (0.051, False), (1.0, False)])
def test_singleton_and_inclusive_threshold(method: MultipleTestingMethod,
                                          value: float, reject: bool) -> None:
    result = adjust_pvalues(pd.Series([value], index=['only']), family_id='singleton',
                            method=method)
    assert result.loc['only', 'pvalue_raw'] == result.loc['only', 'pvalue_adjusted'] == value
    assert bool(result.loc['only', 'reject']) is reject


@pytest.mark.parametrize('method', METHODS)
def test_metadata_labels_and_input_are_preserved(method: MultipleTestingMethod) -> None:
    pvalues = _hand_family()
    pvalues.attrs['upstream'] = 'do not overwrite'
    original = pvalues.copy(deep=True)
    result = adjust_pvalues(pvalues, family_id='research-grid-v1', method=method,
                            significance_level=0.125)
    assert result.columns.tolist() == ['pvalue_raw', 'pvalue_adjusted', 'reject']
    pd.testing.assert_index_equal(result.index, pvalues.index)
    assert pd.api.types.is_bool_dtype(result['reject'].dtype)
    assert result.attrs['family_id'] == 'research-grid-v1'
    assert result.attrs['family_members'] == tuple(pvalues.index)
    assert result.attrs['n_hypotheses'] == len(pvalues)
    assert result.attrs['method'] is method
    assert result.attrs['significance_level'] == 0.125
    pd.testing.assert_series_equal(pvalues, original)
    assert pvalues.attrs == {'upstream': 'do not overwrite'}


@pytest.mark.parametrize('method', METHODS)
def test_poor_results_remain_part_of_the_family(method: MultipleTestingMethod) -> None:
    singleton = adjust_pvalues(pd.Series([0.01], index=['candidate']), family_id='single',
                               method=method)
    complete = adjust_pvalues(pd.Series([0.01, 1.0], index=['candidate', 'poor']),
                              family_id='complete', method=method)
    expected = 0.02 if method is MultipleTestingMethod.HOLM else 0.03
    assert singleton.loc['candidate', 'pvalue_adjusted'] == 0.01
    assert complete.loc['candidate', 'pvalue_adjusted'] == pytest.approx(expected)
    assert complete.attrs['n_hypotheses'] == 2
    assert complete.attrs['family_members'] == ('candidate', 'poor')


@pytest.mark.parametrize('index', [
    pd.Index([7, 3, 2], name='hypothesis_id'),
    pd.MultiIndex.from_tuples([('mean', 'B'), ('alpha', 'A'), ('mean', 'A')],
                             names=['question', 'factor']),
])
def test_numeric_and_multiindex_labels_are_not_coerced(index: pd.Index) -> None:
    pvalues = pd.Series([0.1, 0.01, 0.5], index=index, dtype='Float64')
    result = adjust_pvalues(pvalues, family_id='labelled')
    pd.testing.assert_index_equal(result.index, pvalues.index)
    assert result.attrs['family_members'] == tuple(index)
    np.testing.assert_array_equal(result['pvalue_raw'], [0.1, 0.01, 0.5])


@pytest.mark.parametrize('bad_value', [np.nan, np.inf, -np.inf, -0.01, 1.01, pd.NA])
def test_invalid_pvalues_reject_the_whole_family_with_offending_label(bad_value: object) -> None:
    pvalues = pd.Series([0.01, bad_value, 0.9], index=['valid', 'invalid-hypothesis', 'poor'],
                        dtype='Float64')
    with pytest.raises(ValueError, match='invalid-hypothesis'):
        adjust_pvalues(pvalues, family_id='incomplete')


@pytest.mark.parametrize('pvalues', [
    pd.Series([], dtype=float),
    pd.Series([0.1, 0.2], index=['duplicate', 'duplicate']),
    pd.Series([0.1, 0.2], index=['valid', '']),
    pd.Series([0.1, 0.2], index=['valid', '  ']),
    pd.Series([0.1, 0.2], index=['valid', None]),
    pd.Series([0.1, 0.2], index=['valid', np.nan]),
    pd.Series([0.1, 0.2], index=['valid', pd.NA]),
    pd.Series([0.1, 0.2], index=pd.MultiIndex.from_tuples([('q', 'a'), ('q', '')])),
    pd.Series([0.1, 0.2], index=pd.MultiIndex.from_tuples([('q', 'a'), ('q', None)])),
    pd.Series([True, False], index=['a', 'b']),
    pd.Series([0.1 + 0j, 0.2 + 0j], index=['a', 'b']),
    pd.Series(['0.1', '0.2'], index=['a', 'b']),
])
def test_invalid_families_are_rejected(pvalues: pd.Series) -> None:
    with pytest.raises((TypeError, ValueError)):
        adjust_pvalues(pvalues, family_id='invalid')


@pytest.mark.parametrize('pvalues', [[0.1, 0.2], np.array([0.1, 0.2]),
                                   pd.DataFrame({'a': [0.1, 0.2]}), None])
def test_unlabelled_or_nonseries_inputs_are_rejected(pvalues: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        adjust_pvalues(pvalues, family_id='invalid')


@pytest.mark.parametrize('family_id', ['', ' \t', None, 7, True])
def test_family_id_is_required_and_nonblank(family_id: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        adjust_pvalues(_hand_family(), family_id=family_id)


@pytest.mark.parametrize('method', ['holm', 'fdr_by', 'fdr_bh', 'BHY', None, True, 7])
def test_unknown_methods_and_implicit_string_conversion_are_rejected(method: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        adjust_pvalues(_hand_family(), family_id='explicit-method', method=method)


@pytest.mark.parametrize('level', [0, 1, -0.1, 1.1, np.nan, np.inf, -np.inf, True, '0.05', None])
def test_invalid_targets_are_rejected(level: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        adjust_pvalues(_hand_family(), family_id='invalid-target', significance_level=level)


def test_real_target_normalization_and_overflow() -> None:
    """Real scalar targets normalize before NumPy finite checks; overflow is invalid."""
    rational = adjust_pvalues(_hand_family(), family_id='rational',
                             significance_level=Fraction(1, 20))
    expected = adjust_pvalues(_hand_family(), family_id='rational', significance_level=0.05)
    pd.testing.assert_frame_equal(rational, expected)
    assert rational.attrs == expected.attrs
    with pytest.raises(ValueError, match='significance_level'):
        adjust_pvalues(_hand_family(), family_id='overflow', significance_level=10 ** 400)


@pytest.mark.parametrize('label', [frozenset(), frozenset({'a'}), b'', b'  ', ()])
def test_nonscalar_or_empty_labels_are_rejected(label: object) -> None:
    pvalues = pd.Series([0.1], index=pd.Index([label], tupleize_cols=False))
    with pytest.raises(ValueError, match='labels'):
        adjust_pvalues(pvalues, family_id='invalid-label')

"""The one quantile-bucket rule of qis: each edge case against its rule, then pd.qcut parity."""

import numpy as np
import pandas as pd
import pytest

from qis.utils.df_cut import add_quantile_classification, x_bins_cut
from qis.utils.quantile_buckets import (
    EmptyQuantileBucketError,
    assign_bucket_codes,
    classify_quantile_buckets,
    compute_bucket_codes,
    compute_quantile_edges,
    compute_sample_quantiles,
    get_quantile_probabilities,
)

Q_VECTORS = ([0.0, 0.16, 0.84, 1.0], [0.0, 0.1, 0.9, 1.0], [0.0, 0.05, 0.95, 1.0],
             [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 1 / 3, 2 / 3, 1.0])
INT_Q = (2, 3, 4, 5, 7, 10)


def test_integer_partitions_never_fall_below_the_exact_fractions():
    """i / k is rounded up to the next double whenever k times it would fall short of i."""
    for k in range(2, 21):
        probs = get_quantile_probabilities(k)
        assert probs[0] == 0.0 and probs[-1] == 1.0
        assert np.all(k * probs >= np.arange(k + 1))
        np.testing.assert_allclose(probs, np.arange(k + 1) / k, rtol=0.0, atol=1e-15)
    np.testing.assert_array_equal(get_quantile_probabilities(4), [0.0, 0.25, 0.5, 0.75, 1.0])


@pytest.mark.parametrize('q', [0, -2, 2.5, True, [0.0, 0.5], [0.1, 0.5, 1.0], [0.0, 0.6, 0.4, 1.0]])
def test_invalid_partitions_raise(q):
    """A partition is a positive integer or probabilities increasing strictly from 0 to 1."""
    with pytest.raises(ValueError):
        get_quantile_probabilities(q)


def test_edges_are_numpy_linear_quantiles_of_the_finite_values():
    """Away from whole-number positions the edges are np.quantile's, bit for bit."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal(137)
    for q in Q_VECTORS:
        probs = np.asarray(q)[1:-1]
        np.testing.assert_array_equal(compute_quantile_edges(x, q), np.quantile(x, probs))
    with_gaps = np.concatenate([x, [np.nan, np.inf, -np.inf]])
    np.testing.assert_array_equal(compute_quantile_edges(with_gaps, 5),
                                  compute_quantile_edges(x, 5))


def test_a_position_within_tolerance_of_a_whole_number_is_snapped_to_it():
    """Float noise below an exact position does not move the edge off the order statistic."""
    x = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    noisy = 0.25 - 1e-15  # h = 4 p lands a hair below 1
    assert compute_sample_quantiles(x, [noisy])[0] == 20.0
    assert assign_bucket_codes(x, compute_sample_quantiles(x, [noisy]))[1] == 0


def test_a_tie_at_an_interior_edge_falls_in_the_lower_bucket():
    """Three copies of one value sit on the 16% edge of a resample and are all Bear."""
    x = np.concatenate([np.linspace(-0.10, -0.04, 16), [-0.035] * 3, np.linspace(0.0, 0.05, 88)])
    edges = compute_quantile_edges(x, [0.0, 0.16, 0.84, 1.0])
    assert edges[0] == -0.035
    codes = compute_bucket_codes(x, [0.0, 0.16, 0.84, 1.0])
    assert np.all(codes[x == -0.035] == 0)
    assert np.sum(codes == 0) == 19


def test_extremes_and_values_beyond_supplied_edges_fall_in_the_outer_buckets():
    """Open outer ends: the minimum is first, the maximum last, nothing is dropped."""
    x = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
    codes = compute_bucket_codes(x, 4)
    assert codes[0] == 0 and codes[-1] == 3
    other_sample_edges = np.array([-0.5, 0.0, 0.5])
    np.testing.assert_array_equal(assign_bucket_codes(np.array([-100.0, -0.5, 0.0, 0.25, 100.0]),
                                                      other_sample_edges), [0, 0, 1, 2, 3])


def test_missing_and_infinite_values_get_no_bucket():
    """NaN and +-inf are excluded from the edges and coded -1, a missing label."""
    x = pd.Series([np.nan, -np.inf, 1.0, 2.0, 3.0, 4.0, np.inf], index=list('abcdefg'))
    codes = compute_bucket_codes(x, 2)
    np.testing.assert_array_equal(codes, [-1, -1, 0, 0, 1, 1, -1])
    buckets = classify_quantile_buckets(x, 2, labels=['low', 'high'])
    assert buckets.isna().tolist() == [True, True, False, False, False, False, True]


def test_nullable_floats_are_classified_with_missing_values():
    """pd.NA in a nullable float series is a missing observation."""
    x = pd.Series([1.0, pd.NA, 2.0, 3.0, 4.0], dtype='Float64')
    np.testing.assert_array_equal(compute_bucket_codes(x, 2), [0, -1, 0, 1, 1])


def test_an_empty_bucket_raises_with_its_occupancy():
    """Estimated edges must leave every bucket occupied, unless the caller waives the check."""
    with pytest.raises(EmptyQuantileBucketError) as constant:
        compute_bucket_codes(np.full(12, 0.02), 4)
    assert constant.value.num_occupied == 1 and constant.value.num_buckets == 4
    assert constant.value.edges == [0.02]
    with pytest.raises(EmptyQuantileBucketError) as short:
        compute_bucket_codes(np.array([0.01, 0.03]), 4)
    assert short.value.num_occupied == 2
    np.testing.assert_array_equal(
        compute_bucket_codes(np.array([0.01, 0.03]), 4, is_require_occupied=False), [0, 3])


def test_categorical_output_keeps_the_index_and_every_category():
    """An ordered categorical in bucket order, unobserved labels kept, index and name preserved."""
    index = pd.date_range('2020-03-31', periods=6, freq='QE')
    x = pd.Series([3.0, 1.0, 2.0, 4.0, 5.0, 6.0], index=index, name='bench')
    buckets = classify_quantile_buckets(x, 3, labels=['Bear', 'Normal', 'Bull'])
    assert buckets.cat.ordered and list(buckets.cat.categories) == ['Bear', 'Normal', 'Bull']
    assert buckets.index.equals(x.index) and buckets.name == 'bench'
    assert buckets.tolist() == ['Normal', 'Bear', 'Bear', 'Normal', 'Bull', 'Bull']
    with pytest.raises(ValueError, match='need 3 labels'):
        classify_quantile_buckets(x, 3, labels=['a', 'b'])


@pytest.mark.parametrize('kind', ['continuous', 'bootstrap', 'exact_positions', 'gaps'])
def test_codes_equal_pd_qcut(kind):
    """Within the rules the classification is pd.qcut's, for explicit and integer partitions."""
    seeds = {'continuous': 1, 'bootstrap': 2, 'exact_positions': 3, 'gaps': 4}
    rng = np.random.default_rng(seeds[kind])
    compared = 0
    for n in range(20, 320, 7):
        base = rng.standard_normal(101 if kind == 'exact_positions' else n)
        if kind == 'bootstrap':
            base = base[rng.integers(0, base.size, base.size)]
        elif kind == 'gaps':
            base[rng.random(base.size) < 0.1] = np.nan
        for q in Q_VECTORS + INT_Q:
            try:
                expected = pd.qcut(base, q=q if np.isscalar(q) else list(q), labels=False)
            except ValueError:  # pandas refuses equal edges; the occupancy rule is tested above
                continue
            expected = np.where(np.isnan(expected), -1, expected).astype(int)
            np.testing.assert_array_equal(
                compute_bucket_codes(base, q, is_require_occupied=False), expected)
            compared += 1
    assert compared > 400


def test_df_cut_assigns_by_the_same_rule():
    """The open-ended default of x_bins_cut and the quantile hue buckets follow the one rule."""
    x = pd.Series([-2.0, -1.5, -0.5, 0.0, 0.0, 0.7, 1.5, 2.5], name='x')
    bins = np.array([-1.5, 0.0, 1.5])
    out, labels = x_bins_cut(a=x, bins=bins)
    np.testing.assert_array_equal(out.cat.codes.to_numpy(), assign_bucket_codes(x, bins))
    assert out.iloc[1] == labels[0] and out.iloc[3] == labels[1]  # values on edges go lower
    df, _ = add_quantile_classification(df=x.to_frame(), x_column='x', num_buckets=4,
                                        is_value_labels=False, hue_name='hue')
    expected = assign_bucket_codes(df['x'], compute_quantile_edges(x, 4))
    assert df['hue'].map({f'regime-{i + 1}': i for i in range(4)}).tolist() == expected.tolist()

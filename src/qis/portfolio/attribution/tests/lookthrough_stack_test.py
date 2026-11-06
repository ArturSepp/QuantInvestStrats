"""Static look-through contracts shared by portfolio analytics and reports."""
import numpy as np
import pandas as pd

from qis.portfolio.tests.brinson_wrapper_test import make_portfolios


def test_partial_matrix_overrides_and_falls_back_without_mutation():
    """Only explicitly split assets override their ordinary classifications."""
    from qis.portfolio.attribution.group_exposures import resolve_group_partition
    labels = pd.Series({"a": "Alts", "b": "Bonds"})
    matrix = pd.DataFrame({"PE": [.6], "Credit": [.4]}, index=["a"])
    original = matrix.copy()
    actual = resolve_group_partition(labels, labels.index, matrix)
    expected = pd.DataFrame({"PE": [.6, 0.], "Credit": [.4, 0.],
                             "Bonds": [0., 1.]}, index=labels.index)
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(matrix, original)


def test_group_weights_use_fractional_positions():
    """Reported groups preserve totals and split a mixed instrument's weight."""
    portfolio, _ = make_portfolios()
    matrix = pd.DataFrame({"PE": [.6], "Credit": [.4]}, index=["SEQ_US"])
    result = portfolio.get_weights(is_grouped=True, freq=None,
                                   lookthrough_matrix=matrix)
    np.testing.assert_allclose(result.PE, portfolio.weights.SEQ_US * .6)
    np.testing.assert_allclose(result.sum(axis=1), portfolio.weights.sum(axis=1))


def test_none_matrix_preserves_grouped_weights_exactly():
    """Explicit None follows the historical categorical path."""
    portfolio, _ = make_portfolios()
    expected = portfolio.get_weights(is_grouped=True, freq=None)
    actual = portfolio.get_weights(is_grouped=True, freq=None, lookthrough_matrix=None)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)

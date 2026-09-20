"""Regression tests for signed sum-to-one normalization at a zero net exposure.

A gross-positive signed vector can cancel to an exact or floating-point zero denominator, so no
finite proportional rescaling can make it sum to one. Zero-gross rows remain the established
zero allocation, while finite nonzero-net rows retain their labels, order, and proportions.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import qis


@pytest.mark.parametrize("dtype", ["float64", "Float64"])
@pytest.mark.parametrize(
    "values",
    ([1.0, -1.0], [0.1, 0.2, -0.3]),
    ids=["exact-cancellation", "floating-cancellation"],
)
def test_df_to_weight_allocation_sum1_rejects_zero_net_series(
    dtype: str,
    values: list[float],
) -> None:
    """Reject ordinary and nullable cancellation before producing non-finite weights."""
    scores = pd.Series(values, index=[f"asset-{n}" for n in range(len(values))], dtype=dtype)
    original = scores.copy(deep=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="positive gross exposure.*net sum.*zero"):
            qis.df_to_weight_allocation_sum1(scores)

    pd.testing.assert_series_equal(scores, original, check_exact=True)


def test_df_to_weight_allocation_sum1_preserves_defined_signed_series() -> None:
    """Retain the public Series path when its signed net exposure is nonzero."""
    scores = pd.Series([1.0, -0.5], index=["long", "short"], dtype="Float64", name="scores")
    original = scores.copy(deep=True)
    expected = pd.Series([2.0, -1.0], index=scores.index, dtype="Float64", name="scores")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = qis.df_to_weight_allocation_sum1(scores)

    pd.testing.assert_series_equal(actual, expected, check_exact=True)
    pd.testing.assert_series_equal(scores, original, check_exact=True)


def test_df_to_weight_allocation_sum1_preserves_nullable_zero_gross_series() -> None:
    """Return zeros without relying on nullable 0/0 fill behavior."""
    scores = pd.Series([0.0, 0.0], index=["first", "second"], dtype="Float64", name="scores")
    original = scores.copy(deep=True)
    expected = scores.copy(deep=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = qis.df_to_weight_allocation_sum1(scores)

    pd.testing.assert_series_equal(actual, expected, check_exact=True)
    pd.testing.assert_series_equal(scores, original, check_exact=True)


def test_df_to_weight_allocation_sum1_rejects_cancelling_dataframe_row() -> None:
    """Validate every row before division and identify the cancelling labeled row."""
    scores = pd.DataFrame(
        {
            "long": pd.array([2.0, 1.0, 0.0, pd.NA], dtype="Float64"),
            "short": pd.array([1.0, -1.0, 0.0, pd.NA], dtype="Float64"),
        },
        index=pd.Index(["positive", "cancelling", "zero", "missing"], name="case"),
    )
    original = scores.copy(deep=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="row 'cancelling'.*positive gross exposure"):
            qis.df_to_weight_allocation_sum1(scores)

    pd.testing.assert_frame_equal(scores, original, check_exact=True)


def test_df_to_weight_allocation_sum1_preserves_defined_rows() -> None:
    """Keep valid signed proportions and the established zero-gross result."""
    index = pd.Index(
        ["positive", "signed", "negative-net", "small-positive", "zero", "missing"],
        name="case",
    )
    scores = pd.DataFrame(
        {
            "first": pd.Series([2.0, 1.0, 1.0, 1.0e-20, 0.0, pd.NA], index=index, dtype="Float64"),
            "second": pd.Series(
                [1.0, -0.5, -2.0, 2.0e-20, 0.0, pd.NA], index=index, dtype="Float64"
            ),
        },
        index=index,
    )
    original = scores.copy(deep=True)
    expected = pd.DataFrame(
        {
            "first": pd.Series(
                [2.0 / 3.0, 2.0, -1.0, 1.0 / 3.0, 0.0, 0.0], index=index, dtype="Float64"
            ),
            "second": pd.Series(
                [1.0 / 3.0, -1.0, 2.0, 2.0 / 3.0, 0.0, 0.0], index=index, dtype="Float64"
            ),
        },
        index=index,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = qis.df_to_weight_allocation_sum1(scores)

    pd.testing.assert_frame_equal(actual, expected, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(
        actual.loc[["positive", "signed", "negative-net", "small-positive"]].sum(axis=1),
        np.ones(4),
        rtol=0.0,
        atol=1.0e-15,
    )
    pd.testing.assert_frame_equal(scores, original, check_exact=True)

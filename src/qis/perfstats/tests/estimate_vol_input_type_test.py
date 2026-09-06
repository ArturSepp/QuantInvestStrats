"""Regression coverage for ``estimate_vol`` container and numeric-type boundaries.

Return volatility is defined for real numeric observations held in the three public container
families: pandas Series, pandas DataFrames, and one- or two-dimensional NumPy arrays. Permissive
float coercion must not turn undeclared array-like containers, categorical booleans, text,
datetimes, or complex values into apparently valid return samples.

The accepted fixtures use the Pythagorean pairs ``[3, 4]`` and ``[5, 12]``. Their independently
calculated small-sample RMS values are ``5 / sqrt(2)`` and ``13 / sqrt(2)``. Tests cover integer,
floating, and nullable pandas storage; scalar-versus-vector return types; a mixed valid panel; a
mixed invalid panel; exact domain errors; warning behavior; and caller ownership.
"""

import warnings
from typing import cast

import numpy as np
import pandas as pd
import pytest

# qis
from qis.perfstats.returns import estimate_vol


# =============================================================================
# Shared deterministic fixtures and independent expectations
# =============================================================================

_CONTAINER_ERROR = "sampled_returns must be a pandas Series, pandas DataFrame, or NumPy array"
_FIRST_COLUMN = "Three Four"
_SECOND_COLUMN = "Five Twelve"
_TYPE_ERROR = "sampled_returns must contain only real numeric values or missing values"
_EXPECTED_FIRST_RMS = 5.0 / np.sqrt(2.0)
_EXPECTED_SECOND_RMS = 13.0 / np.sqrt(2.0)


def _call_with_untyped_input(sampled_returns: object) -> object:
    """Call the public function with an intentionally unsupported container.

    Args:
        sampled_returns: Candidate object outside the statically declared container union.

    Returns:
        Result produced if runtime validation fails to reject the object.
    """
    supported_input = cast(pd.Series | pd.DataFrame | np.ndarray, sampled_returns)
    return estimate_vol(supported_input)


def _mixed_numeric_frame(*, nullable: bool) -> pd.DataFrame:
    """Create equivalent ordinary or nullable real-numeric columns.

    Args:
        nullable: Whether to use pandas nullable integer and floating storage.

    Returns:
        Two-column panel containing the independent RMS fixtures and a missing row.
    """
    if nullable:
        return pd.DataFrame(
            {
                _FIRST_COLUMN: pd.Series((3, pd.NA, 4), dtype=pd.Int64Dtype()),
                _SECOND_COLUMN: pd.Series((5.0, pd.NA, 12.0), dtype=pd.Float64Dtype()),
            }
        )
    return pd.DataFrame(
        {
            _FIRST_COLUMN: (3.0, np.nan, 4.0),
            _SECOND_COLUMN: (5.0, np.nan, 12.0),
        }
    )


def _assert_input_unchanged(
    actual: pd.Series | pd.DataFrame | np.ndarray,
    expected: pd.Series | pd.DataFrame | np.ndarray,
) -> None:
    """Assert caller ownership with the comparison for the concrete container.

    Args:
        actual: Input after calling ``estimate_vol``.
        expected: Independent snapshot taken before the call.
    """
    if isinstance(actual, np.ndarray):
        assert isinstance(expected, np.ndarray)
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(actual, pd.Series):
        assert isinstance(expected, pd.Series)
        pd.testing.assert_series_equal(actual, expected)
    else:
        assert isinstance(actual, pd.DataFrame)
        assert isinstance(expected, pd.DataFrame)
        pd.testing.assert_frame_equal(actual, expected)


# =============================================================================
# Accepted real-numeric inputs
# =============================================================================


@pytest.mark.parametrize(
    "sampled_returns",
    (
        np.array((3, 4), dtype=np.int64),
        np.array((3.0, np.nan, 4.0), dtype=np.float32),
        np.array((3.0, np.nan, 4.0), dtype=np.float64),
    ),
    ids=("integer", "float32", "float64"),
)
def test_estimate_vol_accepts_real_numeric_numpy_dtypes(
    sampled_returns: np.ndarray,
) -> None:
    """Preserve the one-dimensional RMS across real NumPy storage types.

    Args:
        sampled_returns: Integer or floating NumPy representation of ``[3, 4]``.
    """
    original_returns = sampled_returns.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = estimate_vol(sampled_returns)

    assert isinstance(actual, np.float64)
    np.testing.assert_allclose(actual, _EXPECTED_FIRST_RMS)
    _assert_input_unchanged(sampled_returns, original_returns)


@pytest.mark.parametrize(
    "sampled_returns",
    (
        pd.Series((3, 4), dtype=np.int64, name=_FIRST_COLUMN),
        pd.Series((3.0, np.nan, 4.0), dtype=np.float64, name=_FIRST_COLUMN),
        pd.Series((3, pd.NA, 4), dtype=pd.Int64Dtype(), name=_FIRST_COLUMN),
        pd.Series((3.0, pd.NA, 4.0), dtype=pd.Float64Dtype(), name=_FIRST_COLUMN),
    ),
    ids=("integer", "float64", "nullable-integer", "nullable-float64"),
)
def test_estimate_vol_accepts_real_numeric_pandas_series(
    sampled_returns: pd.Series,
) -> None:
    """Preserve scalar RMS results across ordinary and nullable pandas storage.

    Args:
        sampled_returns: Real numeric Series representing the ``[3, 4]`` sample.
    """
    original_returns = sampled_returns.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = estimate_vol(sampled_returns)

    assert isinstance(actual, np.float64)
    np.testing.assert_allclose(actual, _EXPECTED_FIRST_RMS)
    _assert_input_unchanged(sampled_returns, original_returns)


@pytest.mark.parametrize("nullable", (False, True), ids=("ordinary", "nullable"))
def test_estimate_vol_accepts_mixed_real_numeric_dataframe(nullable: bool) -> None:
    """Calculate independent column RMS values in one mixed numeric panel.

    Args:
        nullable: Whether the columns use pandas nullable numeric storage.
    """
    sampled_returns = _mixed_numeric_frame(nullable=nullable)
    original_returns = sampled_returns.copy()
    expected = np.array((_EXPECTED_FIRST_RMS, _EXPECTED_SECOND_RMS))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = estimate_vol(sampled_returns)

    assert isinstance(actual, np.ndarray)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected)
    _assert_input_unchanged(sampled_returns, original_returns)


# =============================================================================
# Rejected containers and value types
# =============================================================================


@pytest.mark.parametrize(
    "sampled_returns",
    (
        [3.0, 4.0],
        (3.0, 4.0),
        pd.Index((3.0, 4.0)),
        pd.array((3.0, pd.NA, 4.0), dtype=pd.Float64Dtype()),
    ),
    ids=("list", "tuple", "index", "extension-array"),
)
def test_estimate_vol_rejects_undeclared_array_like_containers(
    sampled_returns: object,
) -> None:
    """Reject coercible objects outside the three declared container families.

    Args:
        sampled_returns: Undeclared array-like container accepted by float coercion.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(TypeError, match=f"^{_CONTAINER_ERROR}$"):
            _call_with_untyped_input(sampled_returns)


@pytest.mark.parametrize(
    "sampled_returns",
    (
        np.array((True, False)),
        np.array((1 + 2j, 3 + 4j)),
        np.array(("3", "4")),
        np.array((3.0, 4.0), dtype=object),
        np.array(("2024-01-01", "2024-01-02"), dtype="datetime64[D]"),
        np.array((1, 2), dtype="timedelta64[D]"),
        pd.Series((True, pd.NA, False), dtype=pd.BooleanDtype()),
        pd.Series((1 + 2j, 3 + 4j)),
        pd.Series(("3", "4"), dtype=pd.StringDtype()),
        pd.Series((3.0, 4.0), dtype=object),
        pd.Series(("3", "4"), dtype="category"),
        pd.Series(pd.to_datetime(("2024-01-01", "2024-01-02"))),
        pd.Series(pd.to_timedelta((1, 2), unit="D")),
    ),
    ids=(
        "numpy-boolean",
        "numpy-complex",
        "numpy-string",
        "numpy-object",
        "numpy-datetime",
        "numpy-timedelta",
        "pandas-nullable-boolean",
        "pandas-complex",
        "pandas-string",
        "pandas-object",
        "pandas-categorical",
        "pandas-datetime",
        "pandas-timedelta",
    ),
)
def test_estimate_vol_rejects_non_real_numeric_dtypes(
    sampled_returns: pd.Series | np.ndarray,
) -> None:
    """Reject categorical or lossy coercions before warnings or calculations.

    Args:
        sampled_returns: NumPy or pandas sample with an unsupported value type.
    """
    original_returns = sampled_returns.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(TypeError, match=f"^{_TYPE_ERROR}$"):
            estimate_vol(sampled_returns)

    _assert_input_unchanged(sampled_returns, original_returns)


def test_estimate_vol_rejects_mixed_dataframe_with_non_numeric_column() -> None:
    """Reject an entire mixed panel before reducing its valid numeric neighbor."""
    sampled_returns = pd.DataFrame(
        {
            _FIRST_COLUMN: pd.Series((3.0, pd.NA, 4.0), dtype=pd.Float64Dtype()),
            "Text": pd.Series(("5", pd.NA, "12"), dtype=pd.StringDtype()),
        }
    )
    original_returns = sampled_returns.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(TypeError, match=f"^{_TYPE_ERROR}$"):
            estimate_vol(sampled_returns)

    _assert_input_unchanged(sampled_returns, original_returns)

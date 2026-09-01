"""Regression coverage for ``estimate_vol`` input dimensionality.

A return history has one observation axis and, optionally, one column axis. NumPy arrays are
therefore meaningful inputs only when they are one-dimensional ``(observations,)`` or
two-dimensional ``(observations, columns)``. Scalar arrays have no time axis, while arrays with
three or more dimensions do not identify which axes should be flattened or treated as columns.

The invalid-shape matrix below covers zero-, three-, and four-dimensional arrays with an exact
domain error and caller-ownership assertion. Valid controls preserve independently calculated RMS
values, scalar-versus-column-vector output shapes, empty histories, and ordinary/nullable pandas
Series and DataFrame parity without warnings.
"""

import warnings
from typing import cast

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

# qis
from qis.perfstats.returns import estimate_vol


# =============================================================================
# Shared deterministic fixtures and independent expectations
# =============================================================================

_DIMENSION_ERROR = "sampled_returns must be a 1- or 2-dimensional NumPy array"
_FIRST_COLUMN = "First"
_SECOND_COLUMN = "Second"
_FIRST_VALUES = np.array((3.0, 4.0, np.nan, np.inf), dtype=float)
_SECOND_VALUES = np.array((5.0, 12.0, np.nan, -np.inf), dtype=float)
_EXPECTED_FIRST_RMS = np.sqrt((3.0**2 + 4.0**2) / 2.0)
_EXPECTED_SECOND_RMS = np.sqrt((5.0**2 + 12.0**2) / 2.0)


def _estimate_without_warnings(
    sampled_returns: pd.DataFrame | pd.Series | np.ndarray,
) -> object:
    """Call the public estimator while treating every warning as a failure.

    Args:
        sampled_returns: Supported public pandas object or NumPy array.

    Returns:
        Scalar or one-dimensional array produced by ``estimate_vol``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return estimate_vol(sampled_returns)


def _as_float_array(result: object) -> NDArray[np.float64]:
    """Narrow a public estimator result to its asserted array representation.

    Args:
        result: Value returned by ``estimate_vol``.

    Returns:
        Result narrowed to a floating-point NumPy array.
    """
    assert isinstance(result, np.ndarray)
    return cast(NDArray[np.float64], result)


def _pandas_inputs(*, nullable: bool) -> tuple[pd.Series, pd.DataFrame]:
    """Create equivalent Series and mixed-column DataFrame controls.

    Args:
        nullable: Whether to use pandas nullable ``Float64`` storage.

    Returns:
        First-column Series and two-column DataFrame with the same observations.
    """
    frame = pd.DataFrame(
        {
            _FIRST_COLUMN: _FIRST_VALUES,
            _SECOND_COLUMN: _SECOND_VALUES,
        },
        index=pd.date_range("2024-01-01", periods=len(_FIRST_VALUES), freq="D"),
    )
    if nullable:
        frame = frame.astype(pd.Float64Dtype())
    series = frame[_FIRST_COLUMN]
    assert isinstance(series, pd.Series)
    return series, frame


# =============================================================================
# Invalid NumPy dimensionality
# =============================================================================


@pytest.mark.parametrize(
    "sampled_returns",
    (
        np.array(0.01),
        np.arange(24.0).reshape(2, 3, 4),
        np.arange(24.0).reshape(1, 2, 3, 4),
    ),
    ids=("zero-dimensional", "three-dimensional", "four-dimensional"),
)
def test_estimate_vol_rejects_unsupported_numpy_dimensions(
    sampled_returns: np.ndarray,
) -> None:
    """Reject arrays without exactly one observation axis and optional column axis.

    Args:
        sampled_returns: Real-valued NumPy array with unsupported dimensionality.
    """
    original = sampled_returns.copy()

    with pytest.raises(ValueError, match=f"^{_DIMENSION_ERROR}$"):
        _estimate_without_warnings(sampled_returns)

    np.testing.assert_array_equal(sampled_returns, original)


# =============================================================================
# Valid NumPy shape and empty-history controls
# =============================================================================


def test_estimate_vol_preserves_supported_numpy_shapes_and_values() -> None:
    """Preserve independent RMS results and scalar-versus-vector output shapes.

    The first finite sample is ``[3, 4]``, whose RMS is ``sqrt((9 + 16) / 2)``. The second is
    ``[5, 12]``, whose RMS is ``sqrt((25 + 144) / 2)``. Missing and infinite padding does not
    contribute to either estimate.
    """
    sampled_series = _FIRST_VALUES.copy()
    sampled_column = sampled_series[:, np.newaxis]
    sampled_frame = np.column_stack((_FIRST_VALUES, _SECOND_VALUES))
    original_series = sampled_series.copy()
    original_column = sampled_column.copy()
    original_frame = sampled_frame.copy()

    series_result = _estimate_without_warnings(sampled_series)
    column_result = _as_float_array(_estimate_without_warnings(sampled_column))
    frame_result = _as_float_array(_estimate_without_warnings(sampled_frame))

    assert isinstance(series_result, np.float64)
    assert column_result.shape == (1,)
    assert frame_result.shape == (2,)
    np.testing.assert_allclose(series_result, _EXPECTED_FIRST_RMS)
    np.testing.assert_allclose(column_result, np.array((_EXPECTED_FIRST_RMS,)))
    np.testing.assert_allclose(
        frame_result,
        np.array((_EXPECTED_FIRST_RMS, _EXPECTED_SECOND_RMS)),
    )
    np.testing.assert_array_equal(sampled_series, original_series)
    np.testing.assert_array_equal(sampled_column, original_column)
    np.testing.assert_array_equal(sampled_frame, original_frame)


def test_estimate_vol_preserves_empty_numpy_shape_contracts() -> None:
    """Keep established scalar, missing-column, and empty-column results for valid shapes."""
    empty_series = np.array([], dtype=float)
    empty_rows = np.empty((0, 2), dtype=float)
    empty_columns = np.empty((3, 0), dtype=float)

    series_result = _estimate_without_warnings(empty_series)
    empty_rows_result = _as_float_array(_estimate_without_warnings(empty_rows))
    empty_columns_result = _as_float_array(_estimate_without_warnings(empty_columns))

    assert isinstance(series_result, np.float64)
    assert np.isnan(series_result)
    assert empty_rows_result.shape == (2,)
    assert np.isnan(empty_rows_result).all()
    assert empty_columns_result.shape == (0,)


# =============================================================================
# Pandas representation and ownership controls
# =============================================================================


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
def test_estimate_vol_preserves_pandas_shape_parity(nullable: bool) -> None:
    """Retain Series and DataFrame results across ordinary and nullable storage.

    Args:
        nullable: Whether the inputs use nullable ``Float64``/``pd.NA`` storage.
    """
    sampled_series, sampled_frame = _pandas_inputs(nullable=nullable)
    original_series = sampled_series.copy()
    original_frame = sampled_frame.copy()

    series_result = _estimate_without_warnings(sampled_series)
    frame_result = _as_float_array(_estimate_without_warnings(sampled_frame))

    assert isinstance(series_result, np.float64)
    assert frame_result.shape == (2,)
    np.testing.assert_allclose(series_result, _EXPECTED_FIRST_RMS)
    np.testing.assert_allclose(
        frame_result,
        np.array((_EXPECTED_FIRST_RMS, _EXPECTED_SECOND_RMS)),
    )
    pd.testing.assert_series_equal(sampled_series, original_series)
    pd.testing.assert_frame_equal(sampled_frame, original_frame)


@pytest.mark.parametrize("nullable", (False, True), ids=("float64", "nullable-float64"))
def test_estimate_vol_preserves_empty_pandas_shape_contracts(nullable: bool) -> None:
    """Retain valid empty Series, zero-row frame, and zero-column frame behavior.

    Args:
        nullable: Whether the declared values use nullable ``Float64`` storage.
    """
    dtype = pd.Float64Dtype() if nullable else float
    empty_series = pd.Series([], dtype=dtype, name=_FIRST_COLUMN)
    empty_rows = pd.DataFrame(columns=(_FIRST_COLUMN, _SECOND_COLUMN), dtype=dtype)
    empty_columns = pd.DataFrame(index=pd.RangeIndex(3), dtype=dtype)
    original_series = empty_series.copy()
    original_rows = empty_rows.copy()
    original_columns = empty_columns.copy()

    series_result = _estimate_without_warnings(empty_series)
    empty_rows_result = _as_float_array(_estimate_without_warnings(empty_rows))
    empty_columns_result = _as_float_array(_estimate_without_warnings(empty_columns))

    assert isinstance(series_result, np.float64)
    assert np.isnan(series_result)
    assert empty_rows_result.shape == (2,)
    assert np.isnan(empty_rows_result).all()
    assert empty_columns_result.shape == (0,)
    pd.testing.assert_series_equal(empty_series, original_series)
    pd.testing.assert_frame_equal(empty_rows, original_rows)
    pd.testing.assert_frame_equal(empty_columns, original_columns)

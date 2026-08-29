"""Regression coverage for empty providers in time-series backfills.

``bfill_timeseries`` defines its public shape and labels from the newer provider while using the
older provider only to extend available history. A provider with no observations should therefore
behave like an unavailable history rather than exposing incidental pandas indexing errors. A
zero-column newer DataFrame is different: it cannot define the documented output schema and must
be rejected explicitly.

The deterministic cases below exercise returns and prices through Series and DataFrame inputs,
NumPy-backed and nullable floating dtypes, every return fill policy, and one mixed panel containing
finite and all-missing states simultaneously. Literal expectations preserve the available provider
without calling another QIS backfill, return, or NAV path. The tests also pin frequency metadata,
labels, column order, warning behavior, and caller ownership.
"""

import warnings
from typing import Literal, cast

import numpy as np
import pandas as pd
import pytest

from qis.perfstats.timeseries_bfill import bfill_timeseries


# =============================================================================
# Shared deterministic fixtures and comparison helpers
# =============================================================================

_OLDER_DATES = pd.DatetimeIndex(("2024-01-01", "2024-01-02"), freq="B")
_NEWER_DATES = pd.DatetimeIndex(("2024-01-03", "2024-01-05"))
_EXPECTED_NEWER_DATES = pd.DatetimeIndex(
    ("2024-01-03", "2024-01-04", "2024-01-05"),
    freq="B",
)

_ABSENT = "Absent"
_ASSET = "Asset"
_FINITE = "Finite"
_RAGGED = "Ragged"

_TOLERANCE = 1.0e-12

_FloatDtype = Literal["float64", "Float64"]


def _astype_frame(frame: pd.DataFrame, dtype: _FloatDtype) -> pd.DataFrame:
    """Apply one explicitly supported floating representation to a frame.

    Args:
        frame: NumPy-backed deterministic fixture.
        dtype: NumPy-backed or pandas nullable floating dtype.

    Returns:
        Converted DataFrame with the requested physical column dtypes.
    """
    if dtype == "Float64":
        return frame.astype(pd.Float64Dtype())
    return frame.astype(np.float64)


def _astype_series(series: pd.Series, dtype: _FloatDtype) -> pd.Series:
    """Apply one explicitly supported floating representation to a Series.

    Args:
        series: NumPy-backed deterministic fixture.
        dtype: NumPy-backed or pandas nullable floating dtype.

    Returns:
        Converted Series with the requested dtype.
    """
    if dtype == "Float64":
        return series.astype(pd.Float64Dtype())
    return series.astype(np.float64)


def _call_warning_free(
    newer: pd.Series | pd.DataFrame,
    older: pd.Series | pd.DataFrame,
    *,
    is_prices: bool,
    fill_method: str | None = None,
) -> pd.Series | pd.DataFrame:
    """Call the public join while asserting warning and ownership contracts.

    Args:
        newer: Newer caller-owned provider.
        older: Older caller-owned provider.
        is_prices: Whether the providers contain prices rather than returns.
        fill_method: Missing-return policy supplied to the public function.

    Returns:
        Public backfill result.
    """
    original_newer = newer.copy()
    original_older = older.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = bfill_timeseries(
            df_newer=newer,
            df_older=older,
            freq="B",
            fill_method=fill_method,
            is_prices=is_prices,
        )

    if isinstance(newer, pd.Series):
        assert isinstance(older, pd.Series)
        pd.testing.assert_series_equal(newer, original_newer, check_exact=True)
        pd.testing.assert_series_equal(older, original_older, check_exact=True)
    else:
        assert isinstance(older, pd.DataFrame)
        pd.testing.assert_frame_equal(newer, original_newer, check_exact=True)
        pd.testing.assert_frame_equal(older, original_older, check_exact=True)
    return actual


def _assert_frame_close(actual: pd.Series | pd.DataFrame, expected: pd.DataFrame) -> None:
    """Assert complete DataFrame values, schema, dtypes, and frequency.

    Args:
        actual: Result carrying the public Series-or-DataFrame annotation.
        expected: Independently constructed DataFrame reference.
    """
    assert isinstance(actual, pd.DataFrame)
    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_exact=False,
        rtol=0.0,
        atol=_TOLERANCE,
    )
    actual_index = cast(pd.DatetimeIndex, actual.index)
    assert actual_index.freqstr == "B"


def _assert_series_close(actual: pd.Series | pd.DataFrame, expected: pd.Series) -> None:
    """Assert complete Series values, name, dtype, and frequency.

    Args:
        actual: Result carrying the public Series-or-DataFrame annotation.
        expected: Independently constructed Series reference.
    """
    assert isinstance(actual, pd.Series)
    pd.testing.assert_series_equal(
        actual,
        expected,
        check_exact=False,
        rtol=0.0,
        atol=_TOLERANCE,
    )
    actual_index = cast(pd.DatetimeIndex, actual.index)
    assert actual_index.freqstr == "B"


# =============================================================================
# Empty older-provider pass-through
# =============================================================================


@pytest.mark.parametrize("is_prices", (False, True), ids=("returns", "prices"))
@pytest.mark.parametrize("dtype", ("float64", "Float64"))
def test_bfill_timeseries_preserves_mixed_newer_panel_with_empty_older_provider(
    dtype: _FloatDtype,
    is_prices: bool,
) -> None:
    """Preserve finite and all-missing newer columns when older has no rows.

    For returns, Thursday has no supplied observation and remains missing under the default
    policy. For prices, the independently expected Thursday level carries Wednesday's 100 forward.
    The all-missing neighbor remains missing in both modes.

    Args:
        dtype: Floating representation applied to providers and expectation.
        is_prices: Whether the finite values are levels or returns.
    """
    finite = (100.0, 200.0) if is_prices else (0.10, 0.20)
    expected_finite = (100.0, 100.0, 200.0) if is_prices else (0.10, np.nan, 0.20)
    newer = _astype_frame(
        pd.DataFrame({_FINITE: finite, _ABSENT: (np.nan, np.nan)}, index=_NEWER_DATES),
        dtype,
    )
    older = _astype_frame(
        pd.DataFrame(columns=newer.columns, index=pd.DatetimeIndex([])),
        dtype,
    )
    expected = _astype_frame(
        pd.DataFrame(
            {_FINITE: expected_finite, _ABSENT: (np.nan, np.nan, np.nan)},
            index=_EXPECTED_NEWER_DATES,
        ),
        dtype,
    )

    actual = _call_warning_free(newer, older, is_prices=is_prices)

    _assert_frame_close(actual, expected)


@pytest.mark.parametrize(
    ("fill_method", "expected_values"),
    (
        pytest.param(None, (0.10, np.nan, 0.20), id="preserve-missing"),
        pytest.param("to_zero", (0.10, 0.00, 0.20), id="fill-zero"),
        pytest.param("ffill", (0.10, 0.10, 0.20), id="forward-fill"),
    ),
)
@pytest.mark.parametrize("dtype", ("float64", "Float64"))
def test_bfill_timeseries_applies_return_fill_policy_with_empty_older_series(
    dtype: _FloatDtype,
    fill_method: str | None,
    expected_values: tuple[float, float, float],
) -> None:
    """Apply each documented return-gap policy when only newer data are available.

    Args:
        dtype: Floating representation applied to providers and expectation.
        fill_method: Public missing-return policy under test.
        expected_values: Literal Wednesday-through-Friday returns.
    """
    newer = _astype_series(pd.Series((0.10, 0.20), index=_NEWER_DATES, name=_ASSET), dtype)
    older = _astype_series(
        pd.Series([], index=pd.DatetimeIndex([]), dtype=np.float64, name="Older name"),
        dtype,
    )
    expected = _astype_series(
        pd.Series(expected_values, index=_EXPECTED_NEWER_DATES, name=_ASSET),
        dtype,
    )

    actual = _call_warning_free(
        newer,
        older,
        is_prices=False,
        fill_method=fill_method,
    )

    _assert_series_close(actual, expected)


# =============================================================================
# Empty newer-provider fallback
# =============================================================================


@pytest.mark.parametrize("is_prices", (False, True), ids=("returns", "prices"))
@pytest.mark.parametrize("dtype", ("float64", "Float64"))
def test_bfill_timeseries_uses_older_mixed_panel_when_newer_has_no_rows(
    dtype: _FloatDtype,
    is_prices: bool,
) -> None:
    """Use compatible older histories under the empty newer provider's schema.

    The newer schema deliberately reverses the older columns and adds an absent column. Finite and
    ragged older values remain literal, while the newer-only column is missing on both older dates.

    Args:
        dtype: Floating representation applied to providers and expectation.
        is_prices: Whether the finite values are levels or returns.
    """
    finite = (100.0, 110.0) if is_prices else (0.01, 0.02)
    ragged = (np.nan, 50.0) if is_prices else (np.nan, 0.03)
    older = _astype_frame(
        pd.DataFrame({_FINITE: finite, _RAGGED: ragged}, index=_OLDER_DATES),
        dtype,
    )
    newer = _astype_frame(
        pd.DataFrame(columns=(_RAGGED, _FINITE, _ABSENT), index=pd.DatetimeIndex([])),
        dtype,
    )
    expected = _astype_frame(
        pd.DataFrame(
            {
                _RAGGED: ragged,
                _FINITE: finite,
                _ABSENT: (np.nan, np.nan),
            },
            index=_OLDER_DATES,
        ),
        dtype,
    )

    actual = _call_warning_free(newer, older, is_prices=is_prices)

    _assert_frame_close(actual, expected)


@pytest.mark.parametrize("is_prices", (False, True), ids=("returns", "prices"))
@pytest.mark.parametrize("dtype", ("float64", "Float64"))
def test_bfill_timeseries_uses_newer_series_name_with_empty_newer_provider(
    dtype: _FloatDtype,
    is_prices: bool,
) -> None:
    """Preserve older values while taking the public Series label from newer.

    Args:
        dtype: Floating representation applied to providers and expectation.
        is_prices: Whether the finite values are levels or returns.
    """
    values = (100.0, 110.0) if is_prices else (0.01, 0.02)
    older = _astype_series(pd.Series(values, index=_OLDER_DATES, name="Older name"), dtype)
    newer = _astype_series(
        pd.Series([], index=pd.DatetimeIndex([]), dtype=np.float64, name=_ASSET),
        dtype,
    )
    expected = _astype_series(pd.Series(values, index=_OLDER_DATES, name=_ASSET), dtype)

    actual = _call_warning_free(newer, older, is_prices=is_prices)

    _assert_series_close(actual, expected)


# =============================================================================
# Both-empty and schema-less DataFrame boundaries
# =============================================================================


@pytest.mark.parametrize("is_prices", (False, True), ids=("returns", "prices"))
@pytest.mark.parametrize("dtype", ("float64", "Float64"))
@pytest.mark.parametrize("input_kind", ("series", "frame"))
def test_bfill_timeseries_returns_newer_empty_schema_when_both_providers_have_no_rows(
    input_kind: Literal["series", "frame"],
    dtype: _FloatDtype,
    is_prices: bool,
) -> None:
    """Return an owned empty object carrying the newer provider's declared schema.

    Args:
        input_kind: Public pandas shape under test.
        dtype: Floating representation applied to providers and expectation.
        is_prices: Whether the empty providers nominally contain levels or returns.
    """
    empty_index = pd.DatetimeIndex([])
    expected_index = pd.DatetimeIndex([], dtype=empty_index.dtype, freq="B")
    if input_kind == "series":
        newer = _astype_series(
            pd.Series([], index=empty_index, dtype=np.float64, name=_ASSET), dtype
        )
        older = _astype_series(
            pd.Series([], index=empty_index, dtype=np.float64, name="Older name"),
            dtype,
        )
        expected = _astype_series(
            pd.Series([], index=expected_index, dtype=np.float64, name=_ASSET),
            dtype,
        )
        actual = _call_warning_free(newer, older, is_prices=is_prices)
        _assert_series_close(actual, expected)
    else:
        newer = _astype_frame(
            pd.DataFrame(columns=(_RAGGED, _FINITE), index=empty_index),
            dtype,
        )
        older = _astype_frame(
            pd.DataFrame(columns=(_FINITE, _RAGGED), index=empty_index),
            dtype,
        )
        expected = _astype_frame(
            pd.DataFrame(columns=(_RAGGED, _FINITE), index=expected_index),
            dtype,
        )
        actual = _call_warning_free(newer, older, is_prices=is_prices)
        _assert_frame_close(actual, expected)


@pytest.mark.parametrize("is_prices", (False, True), ids=("returns", "prices"))
@pytest.mark.parametrize("dtype", ("float64", "Float64"))
def test_bfill_timeseries_ignores_zero_column_older_dataframe(
    dtype: _FloatDtype,
    is_prices: bool,
) -> None:
    """Exclude dates from an older DataFrame that supplies no output column.

    Args:
        dtype: Floating representation applied to newer and expected panels.
        is_prices: Whether the finite values are levels or returns.
    """
    finite = (100.0, 200.0) if is_prices else (0.10, 0.20)
    expected_finite = (100.0, 100.0, 200.0) if is_prices else (0.10, np.nan, 0.20)
    newer = _astype_frame(
        pd.DataFrame({_FINITE: finite, _ABSENT: (np.nan, np.nan)}, index=_NEWER_DATES),
        dtype,
    )
    older = pd.DataFrame(index=_OLDER_DATES)
    expected = _astype_frame(
        pd.DataFrame(
            {_FINITE: expected_finite, _ABSENT: (np.nan, np.nan, np.nan)},
            index=_EXPECTED_NEWER_DATES,
        ),
        dtype,
    )

    actual = _call_warning_free(newer, older, is_prices=is_prices)

    _assert_frame_close(actual, expected)


@pytest.mark.parametrize("is_prices", (False, True), ids=("returns", "prices"))
def test_bfill_timeseries_rejects_zero_column_newer_dataframe(is_prices: bool) -> None:
    """Reject a newer DataFrame that cannot define the documented output schema.

    Args:
        is_prices: Whether the older provider contains levels or returns.
    """
    newer = pd.DataFrame(index=_NEWER_DATES)
    older = pd.DataFrame({_FINITE: (100.0, 110.0)}, index=_OLDER_DATES)
    original_newer = newer.copy()
    original_older = older.copy()

    with pytest.raises(ValueError, match="df_newer must contain at least one column"):
        bfill_timeseries(
            df_newer=newer,
            df_older=older,
            freq="B",
            is_prices=is_prices,
        )

    pd.testing.assert_frame_equal(newer, original_newer, check_exact=True)
    pd.testing.assert_frame_equal(older, original_older, check_exact=True)

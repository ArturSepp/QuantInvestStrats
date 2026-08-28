"""Regression coverage for price anchors in time-series backfills.

``bfill_timeseries(is_prices=True)`` joins providers in return space and then reconstructs price
levels. A first observed price has no preceding observation from which to calculate a return, but
it is still a supplied level and must not disappear merely because its provider has no usable
earlier return. These tests calculate the required levels directly rather than through QIS return
or NAV helpers.

The mixed panel exercises every material column state simultaneously under NumPy-backed and
pandas nullable floating dtypes. Separate Series/DataFrame controls cover physical one-row
providers, because another column's dates can otherwise hide that boundary. The regressions also
assert warning behavior, frequency metadata, labels, caller ownership, and chronological
equivalence where each property is material.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from qis.perfstats.timeseries_bfill import bfill_timeseries


# =============================================================================
# Shared deterministic fixtures and comparison helpers
# =============================================================================

_ABSENT = "Absent"
_FALLBACK = "Fallback"
_FINITE_ONLY = "Finite only"
_RAGGED = "Ragged"
_SHARED = "Shared"
_SINGLE_NEWER = "Single newer"
_SINGLE_OLDER = "Single older"

_MIXED_EXPECTED_DATES = pd.bdate_range("2024-01-08", "2024-01-10")
_MIXED_NEWER_DATES = pd.DatetimeIndex(("2024-01-09", "2024-01-10"))
_MIXED_OLDER_DATES = pd.DatetimeIndex(("2024-01-06", "2024-01-09"))

_TOLERANCE = 1.0e-12


def _assert_frame_close(actual: pd.Series | pd.DataFrame, expected: pd.DataFrame) -> None:
    """Assert DataFrame values, missingness, labels, dtypes, and frequency.

    Args:
        actual: Public result carrying a Series-or-DataFrame return annotation.
        expected: Independently constructed result in the requested dtype.
    """
    assert isinstance(actual, pd.DataFrame)
    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_exact=False,
        rtol=0.0,
        atol=_TOLERANCE,
    )
    assert isinstance(actual.index, pd.DatetimeIndex)
    assert actual.index.freqstr == "B"


def _assert_series_close(actual: pd.Series | pd.DataFrame, expected: pd.Series) -> None:
    """Assert Series values, missingness, labels, dtype, and frequency.

    Args:
        actual: Public result carrying a Series-or-DataFrame return annotation.
        expected: Independently constructed result in the requested dtype.
    """
    assert isinstance(actual, pd.Series)
    pd.testing.assert_series_equal(
        actual,
        expected,
        check_exact=False,
        rtol=0.0,
        atol=_TOLERANCE,
    )
    assert isinstance(actual.index, pd.DatetimeIndex)
    assert actual.index.freqstr == "B"


def _convert_dtype(
    frame: pd.DataFrame,
    dtype: np.dtype[np.float64] | pd.Float64Dtype,
) -> pd.DataFrame:
    """Convert every physical column to one supported floating representation.

    Args:
        frame: NumPy-backed deterministic fixture.
        dtype: NumPy or pandas nullable floating dtype.

    Returns:
        New DataFrame whose physical columns share the requested dtype.
    """
    return frame.astype(dtype)


def _make_expected_mixed_panel() -> pd.DataFrame:
    """Create literal expected prices for every simultaneous column state.

    Returns:
        Business-day panel preserving each supplied provider anchor.
    """
    return pd.DataFrame(
        {
            _ABSENT: (np.nan, np.nan, np.nan),
            _FINITE_ONLY: (np.nan, 200.0, 220.0),
            _RAGGED: (np.nan, 210.0, 231.0),
            _SINGLE_NEWER: (np.nan, np.nan, 300.0),
            _SINGLE_OLDER: (np.nan, 75.0, 75.0),
            _FALLBACK: (50.0, 55.0, 55.0),
            _SHARED: (100.0, 110.0, 121.0),
        },
        index=_MIXED_EXPECTED_DATES,
    )


def _make_mixed_providers() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create providers containing every material price-anchor boundary.

    Returns:
        Older and newer NumPy-backed price panels.
    """
    older = pd.DataFrame(
        {
            _RAGGED: (np.nan, 210.0),
            _SINGLE_OLDER: (np.nan, 75.0),
            _FALLBACK: (50.0, 55.0),
            _SHARED: (100.0, 110.0),
        },
        index=_MIXED_OLDER_DATES,
    )
    newer = pd.DataFrame(
        {
            _ABSENT: (np.nan, np.nan),
            _FINITE_ONLY: (200.0, 220.0),
            _RAGGED: (210.0, 231.0),
            _SINGLE_NEWER: (np.nan, 300.0),
            _SINGLE_OLDER: (np.nan, np.nan),
            _FALLBACK: (np.nan, np.nan),
            _SHARED: (110.0, 121.0),
        },
        index=_MIXED_NEWER_DATES,
    )
    return older, newer


# =============================================================================
# Simultaneous DataFrame price-anchor boundaries
# =============================================================================


@pytest.mark.parametrize(
    "dtype",
    (
        pytest.param(np.dtype(np.float64), id="float64"),
        pytest.param(pd.Float64Dtype(), id="nullable-float64"),
    ),
)
@pytest.mark.parametrize("reverse_rows", (False, True), ids=("sorted", "reversed"))
def test_bfill_timeseries_preserves_every_mixed_price_anchor(
    dtype: np.dtype[np.float64] | pd.Float64Dtype,
    reverse_rows: bool,
) -> None:
    """Preserve all materially different anchors in one physical panel.

    ``Finite only`` and ``Single newer`` have no older counterpart. ``Ragged`` begins on the
    same Tuesday in both providers but has no calculable older return. ``Single older`` has one
    observed older level and an all-missing newer history. ``Fallback`` and ``Shared`` are healthy
    controls, while ``Absent`` must remain entirely missing.

    Args:
        dtype: Floating representation applied to both providers and the literal expectation.
        reverse_rows: Whether to reverse physical provider row order before the public call.
    """
    older, newer = _make_mixed_providers()
    older = _convert_dtype(older, dtype)
    newer = _convert_dtype(newer, dtype)
    if reverse_rows:
        older = older.iloc[::-1]
        newer = newer.iloc[::-1]
    original_older = older.copy(deep=True)
    original_newer = newer.copy(deep=True)
    expected = _convert_dtype(_make_expected_mixed_panel(), dtype)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = bfill_timeseries(
            df_newer=newer,
            df_older=older,
            freq="B",
            is_prices=True,
        )

    _assert_frame_close(actual, expected)
    pd.testing.assert_frame_equal(older, original_older, check_exact=True)
    pd.testing.assert_frame_equal(newer, original_newer, check_exact=True)


# =============================================================================
# Series/DataFrame shape and physical-row boundaries
# =============================================================================


@pytest.mark.parametrize(
    "dtype",
    (
        pytest.param(np.dtype(np.float64), id="float64"),
        pytest.param(pd.Float64Dtype(), id="nullable-float64"),
    ),
)
def test_bfill_timeseries_preserves_finite_newer_series_start(
    dtype: np.dtype[np.float64] | pd.Float64Dtype,
) -> None:
    """Retain the first finite newer price through equivalent Series and DataFrame calls.

    Args:
        dtype: Floating representation applied to providers and literal expectations.
    """
    older = pd.Series(
        (np.nan, np.nan),
        index=pd.bdate_range("2024-01-08", periods=2),
        name="Asset",
    ).astype(dtype)
    newer = pd.Series(
        (200.0, 210.0, 220.5),
        index=pd.bdate_range("2024-01-10", periods=3),
        name="Asset",
    ).astype(dtype)
    expected = pd.Series(
        (np.nan, np.nan, 200.0, 210.0, 220.5),
        index=pd.bdate_range("2024-01-08", periods=5),
        name="Asset",
    ).astype(dtype)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        series_actual = bfill_timeseries(
            df_newer=newer,
            df_older=older,
            freq="B",
            is_prices=True,
        )
        frame_actual = bfill_timeseries(
            df_newer=newer.to_frame(),
            df_older=older.to_frame(),
            freq="B",
            is_prices=True,
        )

    _assert_series_close(series_actual, expected)
    _assert_frame_close(frame_actual, expected.to_frame())


@pytest.mark.parametrize(
    "dtype",
    (
        pytest.param(np.dtype(np.float64), id="float64"),
        pytest.param(pd.Float64Dtype(), id="nullable-float64"),
    ),
)
def test_bfill_timeseries_preserves_one_row_older_provider(
    dtype: np.dtype[np.float64] | pd.Float64Dtype,
) -> None:
    """Use a physical one-row older provider as a valid price anchor.

    The fallback has no newer price, so its supplied 50 must carry across the requested grid. The
    shared path scales the lone older 100 to the first newer 110 and then compounds to 121. Running
    the fallback alone through Series also proves that another column's dates are unnecessary.

    Args:
        dtype: Floating representation applied to providers and literal expectations.
    """
    older = _convert_dtype(
        pd.DataFrame({_FALLBACK: (50.0,), _SHARED: (100.0,)}, index=[pd.Timestamp("2024-01-01")]),
        dtype,
    )
    newer = _convert_dtype(
        pd.DataFrame(
            {_FALLBACK: (np.nan, np.nan), _SHARED: (110.0, 121.0)},
            index=pd.bdate_range("2024-01-02", periods=2),
        ),
        dtype,
    )
    expected = _convert_dtype(
        pd.DataFrame(
            {_FALLBACK: (50.0, 50.0, 50.0), _SHARED: (110.0, 110.0, 121.0)},
            index=pd.bdate_range("2024-01-01", periods=3),
        ),
        dtype,
    )
    expected_fallback = pd.Series(
        (50.0, 50.0, 50.0),
        index=expected.index,
        name=_FALLBACK,
    ).astype(dtype)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        frame_actual = bfill_timeseries(
            df_newer=newer,
            df_older=older,
            freq="B",
            is_prices=True,
        )
        series_actual = bfill_timeseries(
            df_newer=newer[_FALLBACK],
            df_older=older[_FALLBACK],
            freq="B",
            is_prices=True,
        )

    _assert_frame_close(frame_actual, expected)
    _assert_series_close(series_actual, expected_fallback)


@pytest.mark.parametrize(
    "dtype",
    (
        pytest.param(np.dtype(np.float64), id="float64"),
        pytest.param(pd.Float64Dtype(), id="nullable-float64"),
    ),
)
def test_bfill_timeseries_carries_one_off_grid_older_price(
    dtype: np.dtype[np.float64] | pd.Float64Dtype,
) -> None:
    """Carry one Saturday older price onto the complete business-day grid.

    Args:
        dtype: Floating representation applied to providers and literal expectations.
    """
    older = pd.Series(
        (75.0,),
        index=pd.DatetimeIndex(("2024-01-06",)),
        name="Asset",
    ).astype(dtype)
    newer = pd.Series(
        (np.nan, np.nan),
        index=pd.DatetimeIndex(("2024-01-09", "2024-01-10")),
        name="Asset",
    ).astype(dtype)
    expected = pd.Series(
        (75.0, 75.0, 75.0),
        index=pd.bdate_range("2024-01-08", "2024-01-10"),
        name="Asset",
    ).astype(dtype)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = bfill_timeseries(
            df_newer=newer,
            df_older=older,
            freq="B",
            is_prices=True,
        )

    _assert_series_close(actual, expected)

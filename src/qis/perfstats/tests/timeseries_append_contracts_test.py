"""Regression tests for plain time-series append input contracts.

``append_time_series`` combines two providers whose roles are asymmetric: the newer provider
declares the public result schema, while the older provider may supply a compatible historical
prefix. These deterministic tests cover Series names, DataFrame labels and axis metadata,
overlap diagnostics, nullable missing data, empty and zero-column providers, validation,
warnings, and caller/result ownership. Literal expected objects keep the contract independent of
the production splice implementation.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from qis.perfstats.timeseries_bfill import append_time_series


# =============================================================================
# Shared deterministic timeline and conversion helpers
# =============================================================================

_DAILY_DATES = pd.date_range("2024-01-01", periods=6, freq="D")
_NEWER_COLUMNS = ("newer_only", "ragged", "shared", "older_only")
_OLDER_COLUMNS = ("older_only", "shared", "ragged")
_NEWER_INDEX_NAME = "Newer date"
_NEWER_COLUMNS_NAME = "Newer asset"


def _as_float_dtype(frame: pd.DataFrame, use_nullable: bool) -> pd.DataFrame:
    """Convert a fresh frame to the requested floating-point representation.

    Args:
        frame: Frame to convert without modifying its caller-owned source.
        use_nullable: Use pandas nullable ``Float64`` when true and NumPy ``float64`` otherwise.

    Returns:
        A converted DataFrame with identical labels and values.
    """
    dtype = pd.Float64Dtype() if use_nullable else np.dtype("float64")
    return frame.astype(dtype)


def _as_float_series(series: pd.Series, use_nullable: bool) -> pd.Series:
    """Convert a fresh Series to the requested floating-point representation.

    Args:
        series: Series to convert without modifying its caller-owned source.
        use_nullable: Use pandas nullable ``Float64`` when true and NumPy ``float64`` otherwise.

    Returns:
        A converted Series with identical labels and values.
    """
    dtype = pd.Float64Dtype() if use_nullable else np.dtype("float64")
    return series.astype(dtype)


def _append_frames(
    newer: pd.DataFrame,
    older: pd.DataFrame,
    numerical_check_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Append frames while enforcing warning, caller-ownership, and result-type invariants.

    Args:
        newer: Newer provider passed to the public function.
        older: Older provider passed to the public function.
        numerical_check_columns: Optional ordered overlap-diagnostic selection.

    Returns:
        The appended DataFrame and optional overlap-difference Series.
    """
    original_newer = newer.copy(deep=True)
    original_older = older.copy(deep=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        if numerical_check_columns is None:
            actual, actual_diff = append_time_series(df_newer=newer, df_older=older)
        else:
            actual, actual_diff = append_time_series(
                df_newer=newer,
                df_older=older,
                numerical_check_columns=numerical_check_columns,
            )

    assert isinstance(actual, pd.DataFrame)
    assert actual is not newer
    assert actual is not older
    pd.testing.assert_frame_equal(newer, original_newer, check_exact=True)
    pd.testing.assert_frame_equal(older, original_older, check_exact=True)
    return actual, actual_diff


def _append_series(
    newer: pd.Series,
    older: pd.Series,
    numerical_check_columns: list[str] | None = None,
) -> tuple[pd.Series, pd.Series | None]:
    """Append Series while enforcing warning, caller-ownership, and result-type invariants.

    Args:
        newer: Newer provider passed to the public function.
        older: Older provider passed to the public function.
        numerical_check_columns: Optional ordered overlap-diagnostic selection.

    Returns:
        The appended Series and optional overlap-difference Series.
    """
    original_newer = newer.copy(deep=True)
    original_older = older.copy(deep=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        if numerical_check_columns is None:
            actual, actual_diff = append_time_series(df_newer=newer, df_older=older)
        else:
            actual, actual_diff = append_time_series(
                df_newer=newer,
                df_older=older,
                numerical_check_columns=numerical_check_columns,
            )

    assert isinstance(actual, pd.Series)
    assert actual is not newer
    assert actual is not older
    pd.testing.assert_series_equal(newer, original_newer, check_exact=True)
    pd.testing.assert_series_equal(older, original_older, check_exact=True)
    return actual, actual_diff


# =============================================================================
# Mixed-panel schema, metadata, and diagnostic contract
# =============================================================================


def _make_mixed_inputs(use_nullable: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct deliberately reordered providers containing every material column state.

    Args:
        use_nullable: Use pandas nullable ``Float64`` when true and NumPy ``float64`` otherwise.

    Returns:
        Newer and older frames with different axis names and with the older columns a reordered
        subset of the newer declaration.
    """
    newer = pd.DataFrame(
        {
            "newer_only": (40.0, 50.0, 60.0),
            "ragged": (np.nan, 50.0, 60.0),
            "shared": (400.0, 500.0, 600.0),
            "older_only": (np.nan, np.nan, np.nan),
        },
        index=_DAILY_DATES[3:].rename(_NEWER_INDEX_NAME),
    ).rename_axis(_NEWER_COLUMNS_NAME, axis="columns")
    older = pd.DataFrame(
        {
            "older_only": (1.0, 2.0, 3.0, 4.0, 5.0),
            "shared": (10.0, 20.0, 30.0, 40.0, 50.0),
            "ragged": (np.nan, 2.0, 3.0, 4.0, 5.0),
        },
        index=_DAILY_DATES[:5].rename("Older date"),
    ).rename_axis("Older asset", axis="columns")
    return _as_float_dtype(newer, use_nullable), _as_float_dtype(older, use_nullable)


def _make_expected_mixed_result(use_nullable: bool) -> pd.DataFrame:
    """Return the literal newer-schema splice for the mixed providers.

    Args:
        use_nullable: Use pandas nullable ``Float64`` when true and NumPy ``float64`` otherwise.

    Returns:
        The older January 1-3 prefix and newer January 4-6 values in exact newer column order.
    """
    expected = pd.DataFrame(
        {
            "newer_only": (np.nan, np.nan, np.nan, 40.0, 50.0, 60.0),
            "ragged": (np.nan, 2.0, 3.0, np.nan, 50.0, 60.0),
            "shared": (10.0, 20.0, 30.0, 400.0, 500.0, 600.0),
            "older_only": (1.0, 2.0, 3.0, np.nan, np.nan, np.nan),
        },
        index=_DAILY_DATES.rename(_NEWER_INDEX_NAME),
    ).rename_axis(_NEWER_COLUMNS_NAME, axis="columns")
    return _as_float_dtype(expected, use_nullable)


@pytest.mark.parametrize("use_nullable", (False, True), ids=("float64", "Float64"))
def test_append_time_series_uses_newer_mixed_panel_schema(use_nullable: bool) -> None:
    """Use the newer schema while preserving all four material column states.

    The overlap is January 4-5. ``ragged`` has one jointly observed difference of ``45`` and
    ``shared`` has ``(abs(40 - 400) + abs(50 - 500)) / 2 = 405``. Both one-sided columns have an
    undefined diagnostic. This single mixed panel prevents one column state from hiding another.

    Args:
        use_nullable: Exercise NumPy-backed and pandas nullable missing-value representations.
    """
    newer, older = _make_mixed_inputs(use_nullable)
    actual, actual_diff = _append_frames(newer, older, list(_NEWER_COLUMNS))
    expected_diff = pd.Series(
        (np.nan, 45.0, 405.0, np.nan),
        index=pd.Index(_NEWER_COLUMNS, name=_NEWER_COLUMNS_NAME),
    )
    expected_diff = _as_float_series(expected_diff, use_nullable)

    pd.testing.assert_frame_equal(
        actual,
        _make_expected_mixed_result(use_nullable),
        check_exact=True,
        check_freq=False,
    )
    assert isinstance(actual_diff, pd.Series)
    pd.testing.assert_series_equal(actual_diff, expected_diff, check_exact=True)


# =============================================================================
# Series naming and empty-provider contract
# =============================================================================


@pytest.mark.parametrize(
    ("newer_name", "older_name"),
    (
        pytest.param("New asset", "New asset", id="same"),
        pytest.param("New asset", "Old asset", id="different"),
        pytest.param("New asset", None, id="older-unnamed"),
        pytest.param(None, "Old asset", id="newer-unnamed"),
        pytest.param(None, None, id="both-unnamed"),
    ),
)
@pytest.mark.parametrize("use_nullable", (False, True), ids=("float64", "Float64"))
def test_append_time_series_uses_exact_newer_series_name(
    newer_name: str | None,
    older_name: str | None,
    use_nullable: bool,
) -> None:
    """Interpret compatible older values under the exact newer name, including ``None``.

    Args:
        newer_name: Public name declared by the newer provider.
        older_name: Independently varied older provider name.
        use_nullable: Exercise NumPy-backed and pandas nullable floating-point dtypes.
    """
    older = _as_float_series(
        pd.Series((1.0, 2.0, 3.0), index=_DAILY_DATES[:3], name=older_name),
        use_nullable,
    )
    newer = _as_float_series(
        pd.Series((30.0, 40.0), index=_DAILY_DATES[2:4], name=newer_name),
        use_nullable,
    )
    expected = _as_float_series(
        pd.Series((1.0, 2.0, 30.0, 40.0), index=_DAILY_DATES[:4], name=newer_name),
        use_nullable,
    )
    diagnostic_columns = [newer_name] if newer_name is not None else None

    actual, actual_diff = _append_series(newer, older, diagnostic_columns)

    pd.testing.assert_series_equal(actual, expected, check_exact=True, check_freq=False)
    if newer_name is None:
        assert actual_diff is None
    else:
        expected_diff = _as_float_series(
            pd.Series((27.0,), index=pd.Index([newer_name])),
            use_nullable,
        )
        assert isinstance(actual_diff, pd.Series)
        pd.testing.assert_series_equal(actual_diff, expected_diff, check_exact=True)


@pytest.mark.parametrize("empty_side", ("older", "newer", "both"))
@pytest.mark.parametrize("use_nullable", (False, True), ids=("float64", "Float64"))
def test_append_time_series_handles_empty_series_providers(
    empty_side: str,
    use_nullable: bool,
) -> None:
    """Return available history under the newer Series declaration when a provider is empty.

    Args:
        empty_side: Whether the older, newer, or both providers contain no observations.
        use_nullable: Exercise NumPy-backed and pandas nullable floating-point dtypes.
    """
    newer_values = () if empty_side in {"newer", "both"} else (6.0, 4.0, 5.0)
    newer_index = _DAILY_DATES[:0] if not newer_values else _DAILY_DATES[[5, 3, 4]]
    older_values = () if empty_side in {"older", "both"} else (3.0, 1.0, 2.0)
    older_index = _DAILY_DATES[:0] if not older_values else _DAILY_DATES[[2, 0, 1]]
    newer = _as_float_series(
        pd.Series(newer_values, index=newer_index, name="New asset"), use_nullable
    )
    older = _as_float_series(
        pd.Series(older_values, index=older_index, name="Old asset"), use_nullable
    )
    expected_values = (
        (4.0, 5.0, 6.0)
        if older_values == () and newer_values
        else ((1.0, 2.0, 3.0) if older_values else ())
    )
    expected_index = (
        _DAILY_DATES[:0]
        if expected_values == ()
        else (_DAILY_DATES[3:] if older_values == () else _DAILY_DATES[:3])
    )
    expected = _as_float_series(
        pd.Series(expected_values, index=expected_index, name="New asset"),
        use_nullable,
    )

    actual, actual_diff = _append_series(newer, older, ["New asset"])

    pd.testing.assert_series_equal(actual, expected, check_exact=True, check_freq=False)
    assert actual_diff is None


# =============================================================================
# DataFrame empty-provider, zero-column, and validation contract
# =============================================================================


@pytest.mark.parametrize("empty_side", ("older", "newer", "both"))
@pytest.mark.parametrize("use_nullable", (False, True), ids=("float64", "Float64"))
def test_append_time_series_handles_empty_dataframe_providers(
    empty_side: str,
    use_nullable: bool,
) -> None:
    """Preserve the newer declaration across all zero-row DataFrame combinations.

    The newer schema deliberately contains a finite, ragged, and all-missing column together.
    When newer has no rows, the compatible older subset fills its declared columns and the
    newer-only column remains missing.

    Args:
        empty_side: Whether the older, newer, or both providers contain no observations.
        use_nullable: Exercise NumPy-backed and pandas nullable floating-point dtypes.
    """
    columns = ("newer_only", "ragged", "finite", "all_missing")
    newer_values = {
        "newer_only": (40.0, 50.0, 60.0),
        "ragged": (np.nan, 5.0, 6.0),
        "finite": (4.0, 5.0, 6.0),
        "all_missing": (np.nan, np.nan, np.nan),
    }
    newer = pd.DataFrame(newer_values, index=_DAILY_DATES[[5, 3, 4]])
    if empty_side in {"newer", "both"}:
        newer = newer.head(0)
    newer.index.name = _NEWER_INDEX_NAME
    newer.columns.name = _NEWER_COLUMNS_NAME
    older = pd.DataFrame(
        {
            "finite": (3.0, 1.0, 2.0),
            "ragged": (3.0, np.nan, 2.0),
            "all_missing": (np.nan, np.nan, np.nan),
        },
        index=_DAILY_DATES[[2, 0, 1]],
    )
    if empty_side in {"older", "both"}:
        older = older.head(0)
    newer = _as_float_dtype(newer, use_nullable)
    older = _as_float_dtype(older, use_nullable)

    if empty_side == "older":
        expected = newer.sort_index(kind="stable")
    elif empty_side == "newer":
        expected = pd.DataFrame(
            {
                "newer_only": (np.nan, np.nan, np.nan),
                "ragged": (np.nan, 2.0, 3.0),
                "finite": (1.0, 2.0, 3.0),
                "all_missing": (np.nan, np.nan, np.nan),
            },
            index=_DAILY_DATES[:3],
        )
        expected.index.name = _NEWER_INDEX_NAME
        expected.columns.name = _NEWER_COLUMNS_NAME
    else:
        expected = newer.copy(deep=True)
    expected = _as_float_dtype(expected.reindex(columns=list(columns)), use_nullable)

    actual, actual_diff = _append_frames(newer, older, list(columns))

    pd.testing.assert_frame_equal(actual, expected, check_exact=True, check_freq=False)
    assert actual_diff is None


def test_append_time_series_ignores_zero_column_older_provider_dates() -> None:
    """Ignore dates from an older provider that contains no values to append."""
    newer = pd.DataFrame({"asset": (2.0, 1.0)}, index=_DAILY_DATES[[2, 1]])
    older = pd.DataFrame(index=_DAILY_DATES[:2])
    expected = pd.DataFrame({"asset": (1.0, 2.0)}, index=_DAILY_DATES[1:3])

    actual, actual_diff = _append_frames(newer, older)

    pd.testing.assert_frame_equal(actual, expected, check_exact=True, check_freq=False)
    assert actual_diff is None


def test_append_time_series_rejects_zero_column_newer_provider() -> None:
    """Reject a newer DataFrame that cannot declare any public result columns."""
    newer = pd.DataFrame(index=_DAILY_DATES[3:])
    older = pd.DataFrame({"asset": (1.0, 2.0, 3.0)}, index=_DAILY_DATES[:3])

    with pytest.raises(ValueError, match="df_newer must contain at least one column"):
        append_time_series(df_newer=newer, df_older=older)


@pytest.mark.parametrize("provider", ("newer", "older"))
def test_append_time_series_rejects_duplicate_columns(provider: str) -> None:
    """Reject ambiguous duplicate labels before pandas raises an incidental alignment error.

    Args:
        provider: Provider whose DataFrame contains the duplicate column label.
    """
    newer = pd.DataFrame(((4.0, 40.0), (5.0, 50.0)), index=_DAILY_DATES[2:4], columns=("a", "b"))
    older = pd.DataFrame(
        ((1.0, 10.0), (2.0, 20.0), (3.0, 30.0)), index=_DAILY_DATES[:3], columns=("a", "b")
    )
    duplicate = pd.DataFrame(
        ((1.0, 10.0), (2.0, 20.0)),
        index=_DAILY_DATES[:2],
        columns=("a", "a"),
    )
    if provider == "newer":
        newer = duplicate
    else:
        older = duplicate

    with pytest.raises(ValueError, match=rf"df_{provider} columns must be unique"):
        append_time_series(df_newer=newer, df_older=older)


def test_append_time_series_rejects_unknown_diagnostic_column() -> None:
    """Reject diagnostics outside the newer schema before attempting overlap arithmetic."""
    newer = pd.DataFrame({"asset": (30.0, 40.0)}, index=_DAILY_DATES[2:4])
    older = pd.DataFrame({"asset": (1.0, 2.0, 3.0)}, index=_DAILY_DATES[:3])

    with pytest.raises(
        ValueError, match=r"numerical_check_columns not found in df_newer: \['missing'\]"
    ):
        append_time_series(
            df_newer=newer,
            df_older=older,
            numerical_check_columns=["missing"],
        )


def test_append_time_series_preserves_empty_diagnostic_selection() -> None:
    """Return an empty ordered diagnostic when an overlapping call selects no columns."""
    newer = pd.DataFrame({"asset": (30.0, 40.0)}, index=_DAILY_DATES[2:4])
    older = pd.DataFrame({"asset": (1.0, 2.0, 3.0)}, index=_DAILY_DATES[:3])

    _, actual_diff = _append_frames(newer, older, [])

    assert isinstance(actual_diff, pd.Series)
    expected_diff = pd.Series(index=pd.Index([], dtype="str"), dtype="float64")
    pd.testing.assert_series_equal(actual_diff, expected_diff, check_exact=True)


def test_append_time_series_owns_nonextending_result() -> None:
    """Return an independent chronological result when older history cannot extend newer data."""
    newer = pd.DataFrame({"asset": (3.0, 1.0, 2.0)}, index=_DAILY_DATES[[2, 0, 1]])
    older = pd.DataFrame({"asset": (20.0, 30.0)}, index=_DAILY_DATES[1:3])
    expected = pd.DataFrame({"asset": (1.0, 2.0, 3.0)}, index=_DAILY_DATES[:3])

    actual, actual_diff = _append_frames(newer, older)

    pd.testing.assert_frame_equal(actual, expected, check_exact=True, check_freq=False)
    assert actual_diff is None

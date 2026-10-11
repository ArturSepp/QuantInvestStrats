"""Grouped row sums preserve numerical results and labeled callable inputs."""

import numpy as np
import pandas as pd
import pytest

from qis.utils.df_groups import agg_df_by_groups_ax1


@pytest.mark.parametrize("dtype", ["float32", "float64", "int8", "complex128", "bool"])
def test_agg_df_by_groups_ax1_avoids_series_rows_for_numeric_sums(dtype, monkeypatch):
    """A raw-row sum avoids Series construction without changing values or metadata."""
    dates = pd.date_range("2026-01-01", periods=2, tz="UTC", name="date")
    frame = pd.DataFrame([[1, 0, 1], [0, 1, 1]], index=dates, columns=["A", "B", "C"])
    frame = frame.astype(dtype)
    groups = pd.Series({"C": "Other", "B": "Pair", "A": "Pair"})
    before = frame.copy(deep=True)
    calls = []
    original_apply = pd.DataFrame.apply

    def observed_apply(self, func, *args, **kwargs):
        calls.append(kwargs.get("raw", False))
        return original_apply(self, func, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "apply", observed_apply)
    actual = agg_df_by_groups_ax1(frame, groups, total_column="Total", group_order=["Other"])
    # Literal sums, not a second call to the production grouping/reduction rule.
    result_dtype = np.dtype(np.int_) if dtype in ("bool", "int8") else dtype
    expected = pd.DataFrame(
        {"Total": [2, 2], "Other": [1, 1], "Pair": [1, 1]}, index=dates, dtype=result_dtype
    )
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    pd.testing.assert_frame_equal(frame, before, check_exact=True)
    assert calls == [True, True, True]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_agg_df_by_groups_ax1_preserves_row_reduction_order_and_missingness(dtype):
    """Cancellation and mixed missing states forbid replacing row-wise NumPy sums."""
    frame = pd.DataFrame(
        [[1e16, 1, -1e16], [np.nan, 2, 3], [np.nan] * 3, [1, np.nan, 3]],
        columns=["A", "B", "C"],
        dtype=dtype,
    )
    groups = pd.Series("Group", index=frame.columns)
    expected = pd.DataFrame({"Group": [0, 5, 0, 4]}, dtype=dtype)
    with np.errstate(all="raise"):
        actual = agg_df_by_groups_ax1(frame, groups)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


@pytest.mark.parametrize("dtype", ["Float64", "Int64", "object"])
def test_agg_df_by_groups_ax1_keeps_series_path_for_non_numpy_storage(dtype, monkeypatch):
    """Nullable missing scalars and object result inference must retain pandas semantics."""
    frame = pd.DataFrame([[1, None], [None, None]], columns=["A", "B"], dtype=dtype)
    if dtype == "object":
        # Raw application infers its output buffer from the first row: a later fraction
        # could otherwise be truncated when that first row produces an integer scalar.
        frame = pd.DataFrame([[1, 2], [0.25, 0.5]], columns=["A", "B"], dtype=object)
    groups = pd.Series("Group", index=frame.columns)
    original_apply = pd.DataFrame.apply
    calls = []

    def observed_apply(self, func, *args, **kwargs):
        calls.append(kwargs.get("raw", False))
        return original_apply(self, func, *args, **kwargs)

    expected = frame.apply(np.nansum, axis=1).to_frame("Group")
    monkeypatch.setattr(pd.DataFrame, "apply", observed_apply)
    actual = agg_df_by_groups_ax1(frame, groups)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert calls == [False]


def test_agg_df_by_groups_ax1_dispatches_each_group_in_a_mixed_nullable_panel(monkeypatch):
    """A nullable neighbor changes only its own group and the all-instrument total path."""
    frame = pd.DataFrame(
        {
            "A": [1.0, np.nan],
            "B": [2.0, np.nan],
            "C": pd.Series([3.0, None], dtype="Float64"),
            "D": pd.Series([4, None], dtype="Int64"),
        }
    )
    groups = pd.Series({"A": "Ordinary", "B": "Mixed", "C": "Mixed", "D": "Nullable"})
    before = frame.copy(deep=True)
    original_apply = pd.DataFrame.apply
    calls = []

    def observed_apply(self, func, *args, **kwargs):
        calls.append(kwargs.get("raw", False))
        return original_apply(self, func, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "apply", observed_apply)
    actual = agg_df_by_groups_ax1(
        frame, groups, total_column="Total", group_order=["Ordinary", "Mixed", "Nullable"]
    )
    # The all-missing second observation keeps the established nansum identity, zero.
    # Its nullable-integer sum is floating zero, so pandas infers float64 for that
    # output column even though the input storage is Int64.
    expected = pd.DataFrame(
        {"Total": [10.0, 0.0], "Ordinary": [1.0, 0.0], "Mixed": [5.0, 0.0], "Nullable": [4.0, 0.0]}
    )
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    pd.testing.assert_frame_equal(frame, before, check_exact=True)
    assert calls == [False, True, False, False]


def test_agg_df_by_groups_ax1_custom_callable_retains_labels_and_input_isolation():
    """User callables can inspect and modify labeled rows without mutating the caller."""
    dates = pd.date_range("2026-01-01", periods=2, tz="UTC", name="date")
    frame = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]], index=dates, columns=pd.Index(["A", "B"], name="asset")
    )
    before = frame.copy(deep=True)
    seen = []

    def labeled_sum(row):
        assert isinstance(row, pd.Series)
        pd.testing.assert_index_equal(row.index, frame.columns)
        seen.append(row.name)
        result = row["A"] + row["B"]
        row.iloc[0] = 999.0
        return result

    actual = agg_df_by_groups_ax1(frame, pd.Series("Group", index=frame.columns), labeled_sum)
    pd.testing.assert_frame_equal(actual, pd.DataFrame({"Group": [3.0, 7.0]}, index=dates))
    pd.testing.assert_frame_equal(frame, before, check_exact=True)
    assert seen == list(dates)


def test_agg_df_by_groups_ax1_preserves_empty_and_duplicate_axes():
    """Raw application must not create rows, drop repeated dates, or deduplicate assets."""
    groups = pd.Series({"A": "Group"})
    frame = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=["same", "same"], columns=["A", "A"])
    # Accepted selection repeats both A columns for each matching A ticker: four values.
    expected = pd.DataFrame({"Group": [6.0, 14.0]}, index=frame.index)
    pd.testing.assert_frame_equal(agg_df_by_groups_ax1(frame, groups), expected, check_exact=True)
    empty = pd.DataFrame(columns=["A"], dtype=float)
    pd.testing.assert_frame_equal(
        agg_df_by_groups_ax1(empty, groups), pd.DataFrame({"Group": pd.Series([], dtype=float)})
    )
    absent = pd.Series({"missing": "Absent"})
    pd.testing.assert_frame_equal(agg_df_by_groups_ax1(frame, absent), pd.DataFrame())

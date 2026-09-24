"""Regression tests for Series truncation with a retained prior anchor."""

import pandas as pd
import pytest

import qis


@pytest.mark.parametrize(
    ("dtype", "timezone", "values"),
    (
        ("float64", None, (1.0, 2.0, 3.0, 4.0)),
        ("Float64", "UTC", (1.0, 2.0, pd.NA, 4.0)),
        ("string", None, ("first", "anchor", pd.NA, "fourth")),
    ),
    ids=("ordinary", "nullable-timezone", "text"),
)
def test_truncate_prior_to_start_series_matches_dataframe_anchor(
    dtype: str,
    timezone: str | None,
    values: tuple[object, ...],
) -> None:
    """Preserve Series metadata while matching the DataFrame anchor result.

    Args:
        dtype: Ordinary, nullable floating, or text dtype for the public input.
        timezone: Optional timezone carried by the dated index.
        values: Representative numeric or text values, including nullable missing data.
    """
    index = pd.date_range(
        "2024-01-01",
        periods=4,
        freq="D",
        tz=timezone,
        name="observation_date",
    )
    series = pd.Series(values, index=index, dtype=dtype, name="price")
    original = series.copy(deep=True)
    start = pd.Timestamp("2024-01-02 12:00", tz=timezone)
    expected = pd.Series(values[1:], index=index[1:], dtype=dtype, name="price")

    actual = qis.truncate_prior_to_start(series, start=start)
    frame_actual = qis.truncate_prior_to_start(series.to_frame(), start=start)

    pd.testing.assert_series_equal(actual, expected)
    pd.testing.assert_frame_equal(
        actual.to_frame(),
        frame_actual,
        check_names=False,
        check_freq=False,
    )
    pd.testing.assert_series_equal(series, original)


def test_truncate_prior_to_start_exact_single_observation_is_unchanged() -> None:
    """Keep the exact-cutoff control on the existing no-anchor path."""
    index = pd.DatetimeIndex([pd.Timestamp("2024-01-01", tz="UTC")], name="observation_date")
    series = pd.Series([1.0], index=index, dtype="Float64", name="price")
    original = series.copy(deep=True)

    actual = qis.truncate_prior_to_start(series, start=index[0])

    pd.testing.assert_series_equal(actual, series)
    pd.testing.assert_series_equal(series, original)

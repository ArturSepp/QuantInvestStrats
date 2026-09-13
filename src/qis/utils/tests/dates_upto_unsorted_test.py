"""Regression tests for chronological latest-at-or-before date lookup.

The public lookup accepts a DatetimeIndex or timestamp list without requiring sorted input. Its
result must equal the greatest finite supplied timestamp not later than the target, independent of
input order, while retaining the established warning and timestamp metadata behavior.
"""

from typing import cast

import pandas as pd
import pytest

import qis


_UNSORTED_DATES = (
    pd.Timestamp("2024-01-01"),
    pd.Timestamp("2024-01-03"),
    pd.Timestamp("2024-01-04"),
    pd.Timestamp("2024-01-02"),
)


@pytest.mark.parametrize("as_list", (False, True), ids=("datetime-index", "list"))
@pytest.mark.parametrize(
    ("target", "expected"),
    (
        (pd.Timestamp("2024-01-02 12:00"), pd.Timestamp("2024-01-02")),
        (pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-04")),
        (pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-02")),
    ),
    ids=("interior", "after-final", "exact"),
)
def test_find_upto_date_from_datetime_index_is_order_invariant(
    as_list: bool,
    target: pd.Timestamp,
    expected: pd.Timestamp,
) -> None:
    """Select the literal greatest eligible timestamp for both public input containers.

    Args:
        as_list: Supply a timestamp list instead of a DatetimeIndex.
        target: Exact or off-grid lookup target.
        expected: Independently selected greatest timestamp not later than ``target``.
    """
    datetime_index = pd.DatetimeIndex(_UNSORTED_DATES, name="observation_date")
    index: pd.DatetimeIndex | list[pd.Timestamp]
    index = datetime_index.to_list() if as_list else datetime_index
    original_order = list(index)

    actual = qis.find_upto_date_from_datetime_index(index=index, date=target)

    assert actual == expected
    assert list(index) == original_order


@pytest.mark.parametrize(
    ("target", "expected"),
    (
        (pd.Timestamp("2024-01-02", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC")),
        (pd.Timestamp("2024-01-04", tz="UTC"), pd.Timestamp("2024-01-03", tz="UTC")),
    ),
    ids=("interior", "after-final"),
)
def test_find_upto_date_from_datetime_index_ignores_nat_with_compatible_timezone(
    target: pd.Timestamp,
    expected: pd.Timestamp,
) -> None:
    """Exclude ``NaT`` while retaining the timezone of the greatest finite match.

    Args:
        target: Compatible timezone-aware lookup target.
        expected: Independently selected finite UTC timestamp.
    """
    index = pd.DatetimeIndex(
        (pd.Timestamp("2024-01-03", tz="UTC"), pd.NaT, pd.Timestamp("2024-01-01", tz="UTC")),
        name="estimate_date",
    )
    original = index.copy()

    actual = qis.find_upto_date_from_datetime_index(index=index, date=target)

    assert actual == expected
    pd.testing.assert_index_equal(index, original, exact=True)


def test_find_upto_date_from_datetime_index_warns_against_chronological_first_date() -> None:
    """Return None before the global minimum and name that minimum in the warning."""
    index = pd.DatetimeIndex(("2024-01-03", "2024-01-01", "2024-01-02"))

    with pytest.warns(UserWarning, match=r"index=2024-01-01 00:00:00"):
        actual = qis.find_upto_date_from_datetime_index(
            index=index,
            date=cast(pd.Timestamp, pd.Timestamp("2023-12-31")),
        )

    assert actual is None


@pytest.mark.parametrize("index", (pd.DatetimeIndex([pd.NaT]), [pd.NaT]))
def test_find_upto_date_from_datetime_index_returns_none_without_finite_dates(
    index: pd.DatetimeIndex | list[pd.Timestamp],
) -> None:
    """Return None with a useful warning when no finite timestamp can match.

    Args:
        index: Public DatetimeIndex or timestamp-list input containing only ``NaT``.
    """
    with pytest.warns(UserWarning, match="contains no finite timestamps"):
        actual = qis.find_upto_date_from_datetime_index(
            index=index,
            date=cast(pd.Timestamp, pd.Timestamp("2024-01-02")),
        )

    assert actual is None


def test_find_upto_date_from_datetime_index_preserves_sorted_duplicate_control() -> None:
    """Keep exact and off-grid lookup unchanged when equal labels share one timestamp."""
    index = pd.DatetimeIndex(("2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"))

    assert qis.find_upto_date_from_datetime_index(
        index=index,
        date=cast(pd.Timestamp, pd.Timestamp("2024-01-02")),
    ) == pd.Timestamp("2024-01-02")
    assert qis.find_upto_date_from_datetime_index(
        index=index,
        date=cast(pd.Timestamp, pd.Timestamp("2024-01-02 12:00")),
    ) == pd.Timestamp("2024-01-02")

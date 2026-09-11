"""Regression tests for ragged support in multi-asset periodic-return tables.

Periodic returns need two observed price boundaries. These tests distinguish missing
pre-inception support from a genuine zero return while preserving the established interior- and
trailing-fill behavior used by periodic tables.
"""

import warnings
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import qis


def _make_mixed_prices(storage: str = "float64") -> pd.DataFrame:
    """Create price columns spanning the periodic-table support boundaries.

    Args:
        storage: Pandas floating dtype used for the panel.

    Returns:
        A complete, ragged, interior-missing, trailing-missing, and all-missing panel.
    """
    index = pd.date_range("2024-01-31", periods=4, freq="ME")
    return pd.DataFrame(
        {
            "Complete": [100.0, 110.0, 110.0, 121.0],
            "Late": [np.nan, 100.0, 110.0, 121.0],
            "One": [np.nan, np.nan, 50.0, np.nan],
            "Interior": [100.0, np.nan, 110.0, 121.0],
            "Trailing": [100.0, 110.0, np.nan, np.nan],
            "Missing": [np.nan, np.nan, np.nan, np.nan],
        },
        index=index,
    ).astype(storage)


@pytest.mark.parametrize("storage", ["float64", "Float64"])
def test_compute_periodic_returns_requires_two_observed_boundaries(storage: str) -> None:
    """Report returns only after a column supplies two observed price boundaries."""
    prices = _make_mixed_prices(storage=storage)
    before = prices.copy(deep=True)
    dates = prices.index

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        actual = qis.compute_periodic_returns(prices=prices, freq="ME", add_total=True)

    # Expected values are direct endpoint ratios; no QIS return helper constructs this oracle.
    expected = pd.DataFrame(
        [
            [0.10, np.nan, np.nan, 0.00, 0.10, np.nan],
            [0.00, 0.10, np.nan, 0.10, 0.00, np.nan],
            [0.10, 0.10, np.nan, 0.10, 0.00, np.nan],
            [0.21, 0.21, np.nan, 0.21, 0.10, np.nan],
        ],
        index=pd.Index([dates[1], dates[2], dates[3], "YTD"]),
        columns=prices.columns,
    ).astype(storage)
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(prices, before)


def test_compute_periodic_returns_counts_support_inside_time_period() -> None:
    """Require two boundaries inside the requested window, not merely in the source."""
    prices = cast(pd.DataFrame, _make_mixed_prices().loc[:, ["Late"]])
    time_period = qis.TimePeriod(start="31Jan2024", end="29Feb2024")

    actual = qis.compute_periodic_returns(
        prices=prices, freq="ME", time_period=time_period, add_total=True
    )

    expected = pd.DataFrame(
        {"Late": [np.nan, np.nan]},
        index=pd.Index([pd.Timestamp("2024-02-29"), "YTD"]),
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_compute_periodic_returns_preserves_off_grid_ragged_start() -> None:
    """Keep pre-inception periods missing when observations fall between month ends."""
    dates = pd.to_datetime(["2024-01-15", "2024-02-15", "2024-03-15", "2024-04-15"])
    prices = pd.DataFrame(
        {
            "Complete": [100.0, 110.0, 121.0, 133.1],
            "Late": [np.nan, 100.0, 110.0, 121.0],
        },
        index=dates,
    )

    actual = qis.compute_periodic_returns(prices=prices, freq="ME", add_total=True)

    expected = pd.DataFrame(
        [
            [0.000, np.nan],
            [0.100, np.nan],
            [0.100, 0.10],
            [0.100, 0.10],
            [0.331, 0.21],
        ],
        index=pd.Index(
            [
                pd.Timestamp("2024-01-31"),
                pd.Timestamp("2024-02-29"),
                pd.Timestamp("2024-03-31"),
                pd.Timestamp("2024-04-15"),
                "YTD",
            ]
        ),
        columns=prices.columns,
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_compute_periodic_returns_preserves_yearly_pre_inception_missing() -> None:
    """Apply the same two-boundary rule when the public frequency selects years."""
    dates = pd.date_range("2022-12-31", periods=3, freq="YE")
    prices = pd.DataFrame({"Late": [np.nan, 100.0, 110.0]}, index=dates)

    actual = qis.compute_periodic_returns(prices=prices, freq="YE", add_total=False)

    expected = pd.DataFrame({"Late": [np.nan, 0.10]}, index=dates[1:])
    pd.testing.assert_frame_equal(actual, expected)


def test_compute_periodic_returns_handles_duplicate_labels_by_position() -> None:
    """Keep column-local support independent when two assets share a label."""
    dates = pd.date_range("2024-01-31", periods=3, freq="ME")
    prices = pd.DataFrame(
        np.column_stack(([np.nan, 100.0, 110.0], [100.0, 100.0, 110.0])),
        index=dates,
        columns=["Asset", "Asset"],
    )

    actual = qis.compute_periodic_returns(prices=prices, freq="ME", add_total=True)

    expected = pd.DataFrame(
        [[np.nan, 0.00], [0.10, 0.10], [0.10, 0.10]],
        index=pd.Index([dates[1], dates[2], "YTD"]),
        columns=prices.columns,
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_plot_periodic_returns_table_preserves_blank_zero_display() -> None:
    """Render ragged missing and genuine zero cells using the established blank display."""
    prices = cast(pd.DataFrame, _make_mixed_prices().loc[:, ["Complete", "Late"]])
    fig, ax = plt.subplots()
    try:
        qis.plot_periodic_returns_table(prices=prices, freq="ME", ax=ax)
        fig.canvas.draw()

        annotations = {
            tuple(float(value) for value in text.get_position()): text.get_text()
            for text in ax.texts
        }
        assert annotations == {
            (0.5, 0.5): "10%",
            (2.5, 0.5): "10%",
            (3.5, 0.5): "21%",
            (1.5, 1.5): "10%",
            (2.5, 1.5): "10%",
            (3.5, 1.5): "21%",
        }
    finally:
        plt.close(fig)

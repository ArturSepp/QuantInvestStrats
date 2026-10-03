"""Asset-label regressions for the public total-return Series wrapper.

The endpoint ratio is independent of its container. A Series must retain the asset identity
used by its one-column DataFrame, including pandas' default column for an unnamed Series.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from qis.perfstats.returns import to_total_returns


def test_to_total_returns_keeps_one_asset_in_one_concat_row() -> None:
    """Equivalent price containers must not split an asset into separate summary rows."""
    prices = pd.Series(
        [100.0, 105.0, 110.0],
        index=pd.date_range("2024-01-01", periods=3),
        name="Asset",
    )
    expected_return = 110.0 / 100.0 - 1.0
    actual = pd.concat(
        [to_total_returns(prices).rename("Series"), to_total_returns(prices.to_frame())],
        axis=1,
    )
    actual.columns = ["Series", "DataFrame"]
    expected = pd.DataFrame(
        {"Series": [expected_return], "DataFrame": [expected_return]}, index=["Asset"]
    )

    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


@pytest.mark.parametrize("storage", ["float64", "Float64"])
@pytest.mark.parametrize(
    "name",
    ["Asset", None, 0, False, ("Asset", "USD"), pd.Timestamp("2024-01-01", tz="UTC")],
    ids=["named", "unnamed", "zero", "false", "tuple", "timestamp"],
)
def test_to_total_returns_matches_native_one_column_labels(name, storage: str) -> None:
    """Preserve index class and dtype as well as the existing Series result name."""
    prices = pd.Series(
        [100.0, 105.0, 110.0],
        index=pd.date_range("2024-01-01", periods=3, tz="UTC", name="date"),
        name=name,
        dtype=storage,
    )
    frame = prices.to_frame()
    expected = pd.Series(110.0 / 100.0 - 1.0, index=frame.columns, name=name)

    actual = to_total_returns(prices)
    frame_actual = to_total_returns(frame)

    pd.testing.assert_series_equal(actual, expected, check_exact=True)
    # The wrapper already names Series results after the input, but frame results are unnamed.
    assert frame_actual.name is None
    pd.testing.assert_series_equal(actual, frame_actual.rename(name), check_exact=True)


@pytest.mark.parametrize("storage", ["float64", "Float64"])
@pytest.mark.parametrize(
    ("values", "expected_return", "warning_count"),
    [
        ([np.nan, 100.0, 110.0], 110.0 / 100.0 - 1.0, 1),
        ([100.0, 110.0, np.nan], 110.0 / 100.0 - 1.0, 1),
        ([100.0, np.nan, 110.0], 110.0 / 100.0 - 1.0, 0),
        ([np.nan, np.nan, np.nan], np.nan, 3),
        ([100.0], np.nan, 0),
    ],
    ids=["leading", "trailing", "interior", "all-missing", "single-observation"],
)
def test_to_total_returns_labels_ragged_and_undefined_results(
    storage: str, values: list[float], expected_return: float, warning_count: int
) -> None:
    """Label finite and undefined returns without changing the reduction's warnings."""
    prices = pd.Series(
        values, index=pd.date_range("2024-01-01", periods=len(values)), name="Asset", dtype=storage
    )
    original = prices.copy(deep=True)
    expected = pd.Series([expected_return], index=["Asset"], name="Asset")

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        actual = to_total_returns(prices)

    pd.testing.assert_series_equal(actual, expected, check_exact=True)
    assert len(recorded) == warning_count
    assert all(item.category is UserWarning for item in recorded)
    pd.testing.assert_series_equal(prices, original, check_exact=True)


def test_to_total_returns_result_mutation_does_not_change_prices() -> None:
    """Replacing result labels and values must not change the price history."""
    prices = pd.Series(
        [100.0, 105.0, 110.0],
        index=pd.date_range("2024-01-01", periods=3, name="date"),
        name="Asset",
    )
    frame = prices.to_frame()
    frame.columns.name = "instrument"
    original = prices.copy(deep=True)
    original_frame = frame.copy(deep=True)

    expected_return = 110.0 / 100.0 - 1.0
    for source, expected in (
        (prices, pd.Series([expected_return], index=["Asset"], name="Asset")),
        (frame, pd.Series([expected_return], index=frame.columns)),
    ):
        result = to_total_returns(source)
        pd.testing.assert_series_equal(result, expected, check_exact=True)
        result.iloc[0] = -1.0
        result.index = result.index.rename("summary")
        result.name = "changed"

    pd.testing.assert_series_equal(prices, original, check_exact=True)
    pd.testing.assert_frame_equal(frame, original_frame, check_exact=True)

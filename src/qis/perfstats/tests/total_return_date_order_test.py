"""Chronological endpoint regressions for the public total-return family.

Dated price histories are mappings from observation time to value, so permuting valid unique rows
must not change total return, elapsed years, annualized return, or the endpoint metadata exposed by
performance tables. The fixtures below derive every expected endpoint and ratio directly rather
than through another QIS return calculation.
"""

from numbers import Real
from typing import cast
import warnings

import numpy as np
import pandas as pd
import pytest

from qis.perfstats.config import PerfParams, PerfStat
from qis.perfstats.perf_stats import compute_ra_perf_table
from qis.perfstats.returns import compute_num_years, compute_pa_return, compute_total_return


_DAYS_PER_YEAR = 365.25
_TOLERANCE = 1.0e-12


def _real_table_value(table: pd.DataFrame, statistic: PerfStat) -> float:
    """Extract one real-valued statistic from the single-asset table."""
    label = statistic.value.name
    if not isinstance(label, str):
        raise TypeError("expected a string performance-statistic label")
    value = cast(object, table.loc["Asset", label])
    if not isinstance(value, Real):
        raise TypeError("expected a real performance-table value")
    return float(value)


@pytest.mark.parametrize(
    ("dates", "values"),
    [
        (
            pd.to_datetime(["2024-01-02", "2024-01-04", "2024-01-03"]),
            [100.0, 120.0, 110.0],
        ),
        (
            pd.to_datetime(["2024-01-04", "2024-01-02", "2024-01-03"], utc=True),
            [120.0, 100.0, 110.0],
        ),
    ],
    ids=["silent-physical-endpoints", "reverse-physical-endpoints-utc"],
)
def test_compute_total_return_uses_chronological_series_endpoints(
    dates: pd.DatetimeIndex,
    values: list[float],
) -> None:
    """Return 20% for the same uniquely dated path under either physical permutation."""
    prices = pd.Series(values, index=dates, name="Asset")
    original = prices.copy(deep=True)

    actual = compute_total_return(prices)

    assert isinstance(actual, Real)
    assert np.isclose(float(actual), 0.20, rtol=0.0, atol=_TOLERANCE)
    pd.testing.assert_series_equal(prices, original, check_exact=True)


def test_compute_total_return_uses_chronological_finite_endpoints_by_column() -> None:
    """Select each ragged column's earliest and latest finite price after ordering dates."""
    prices = pd.DataFrame(
        {
            "Complete": [100.0, 110.0, 105.0, 120.0],
            "Leading": [np.nan, 110.0, 100.0, 120.0],
            "Trailing": [100.0, 110.0, 105.0, np.nan],
            "All missing": [np.nan, np.nan, np.nan, np.nan],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-02", "2024-01-04"]),
    )
    original = prices.copy(deep=True)
    expected = np.array([0.20, 0.20, 0.10, np.nan])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        actual = compute_total_return(prices)

    assert isinstance(actual, np.ndarray)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=_TOLERANCE, equal_nan=True)
    pd.testing.assert_frame_equal(prices, original, check_exact=True)


def test_compute_pa_return_uses_chronological_elapsed_years_and_endpoints() -> None:
    """Annualize a two-year doubling from literal dates and prices, independent of row order."""
    prices = pd.Series(
        [100.0, 200.0, 120.0],
        index=pd.to_datetime(["2020-01-01", "2022-01-01", "2021-01-01"]),
        name="Asset",
    )
    expected_years = 731.0 / _DAYS_PER_YEAR
    expected_pa_return = 2.0 ** (1.0 / expected_years) - 1.0

    actual_years = compute_num_years(prices)
    actual_pa_return = compute_pa_return(prices)

    assert np.isclose(actual_years, expected_years, rtol=0.0, atol=_TOLERANCE)
    assert isinstance(actual_pa_return, Real)
    assert np.isclose(float(actual_pa_return), expected_pa_return, rtol=0.0, atol=_TOLERANCE)


def test_compute_ra_perf_table_reports_chronological_visible_endpoints() -> None:
    """Expose chronological total-return dates and prices through the public performance table."""
    dates = pd.bdate_range("2024-01-02", periods=40)
    steps = np.arange(len(dates), dtype=float)
    sorted_prices = pd.DataFrame(
        {"Asset": 100.0 + steps + 2.0 * np.sin(steps / 3.0)},
        index=dates,
    )
    # Keep the true first row but move the second row to the physical end, making every visible
    # endpoint field wrong without changing the dated observations themselves.
    permutation = [0, *range(2, len(dates)), 1]
    prices = sorted_prices.take(permutation)
    original = prices.copy(deep=True)
    expected_start_price = 100.0
    expected_end_price = 139.0 + 2.0 * float(np.sin(13.0))
    expected_total = expected_end_price / expected_start_price - 1.0
    expected_years = 55.0 / _DAYS_PER_YEAR
    expected_start = pd.Timestamp("2024-01-02")
    expected_end = pd.Timestamp("2024-02-26")

    table = compute_ra_perf_table(
        prices,
        PerfParams(freq="B", freq_skewness="B", freq_drawdown="B"),
    )

    assert np.isclose(
        _real_table_value(table, PerfStat.TOTAL_RETURN),
        expected_total,
        rtol=0.0,
        atol=_TOLERANCE,
    )
    assert np.isclose(
        _real_table_value(table, PerfStat.NUM_YEARS),
        expected_years,
        rtol=0.0,
        atol=_TOLERANCE,
    )
    assert _real_table_value(table, PerfStat.START_PRICE) == expected_start_price
    assert _real_table_value(table, PerfStat.END_PRICE) == expected_end_price
    assert table.loc["Asset", PerfStat.START_DATE.value.name] == expected_start
    assert table.loc["Asset", PerfStat.END_DATE.value.name] == expected_end
    pd.testing.assert_frame_equal(prices, original, check_exact=True)

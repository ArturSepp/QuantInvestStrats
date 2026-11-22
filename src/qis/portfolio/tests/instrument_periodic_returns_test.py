"""Opening price marks are baselines, not the first investable return."""

import numpy as np
import pandas as pd
import pytest

import qis


def _portfolio(prices: pd.DataFrame) -> qis.PortfolioData:
    """A fully held single asset; panel cases exercise only instrument return reconstruction."""
    return qis.PortfolioData(nav=prices.iloc[:, 0].rename("Portfolio"), prices=prices)


@pytest.mark.parametrize("dtype", ["float64", "Float64"])
@pytest.mark.parametrize("freq", [None, "ME"])
@pytest.mark.parametrize("length", [2, 4])
def test_get_instruments_periodic_returns_and_realised_attribution(dtype, freq, length) -> None:
    dates = pd.date_range("2025-01-31", periods=length, freq="ME", name="date")
    prices = pd.DataFrame(
        {"Asset": [100.0, 120.0, 120.0, 132.0][:length]}, index=dates, dtype=dtype
    )
    portfolio = _portfolio(prices)
    before = prices.copy(deep=True)
    expected = pd.DataFrame({"Asset": [np.nan, 0.2, 0.0, 0.1][:length]}, index=dates, dtype=dtype)

    pd.testing.assert_frame_equal(portfolio.get_instruments_periodic_returns(freq=freq), expected)
    pd.testing.assert_frame_equal(
        portfolio.get_attribution_table_by_instrument(freq=freq), expected.iloc[1:]
    )
    pd.testing.assert_frame_equal(prices, before)
    # A copied Index may share its mutable metadata; compare frequency with an immutable value.
    assert prices.index.freqstr == "ME"


@pytest.mark.parametrize("dtype", ["float64", "Float64"])
def test_get_instruments_periodic_returns_ragged_gaps_and_selected_baseline(dtype) -> None:
    dates = pd.date_range("2025-01-31", periods=5, freq="ME", name="date")
    prices = pd.DataFrame(
        {
            "Complete": [100.0, 120.0, 120.0, 132.0, 145.2],
            "Ragged": [np.nan, 100.0, 120.0, 120.0, 132.0],
            "Gap": [100.0, 120.0, np.nan, 132.0, 145.2],
            "Missing": [np.nan] * 5,
        },
        index=dates,
        dtype=dtype,
    )
    expected = pd.DataFrame(
        {
            "Complete": [np.nan, 0.2, 0.0, 0.1, 0.1],
            "Ragged": [np.nan, np.nan, 0.2, 0.0, 0.1],
            # Retain native NAV reconstruction: unobservable gap returns do not become price jumps.
            "Gap": [np.nan, 0.2, 0.0, 0.0, 0.1],
            "Missing": [np.nan] * 5,
        },
        index=dates,
        dtype=dtype,
    )
    portfolio = _portfolio(prices)
    pd.testing.assert_frame_equal(portfolio.get_instruments_periodic_returns(), expected)

    selected = expected.iloc[1:].copy()
    selected.iloc[0] = np.nan
    pd.testing.assert_frame_equal(
        portfolio.get_instruments_periodic_returns(
            time_period=qis.TimePeriod("2025-02-28", "2025-05-31")
        ),
        selected,
    )


def test_compute_tracking_error_table_retains_first_arithmetic_contribution() -> None:
    dates = pd.date_range("2025-01-31", periods=4, freq="ME", name="date")
    prices = pd.DataFrame({"Asset": [100.0, 120.0, 120.0, 132.0]}, index=dates)
    strategy = _portfolio(prices)
    zero = pd.DataFrame(0.0, index=dates, columns=prices.columns)
    benchmark = qis.PortfolioData(
        nav=pd.Series(100.0, index=dates, name="Benchmark"),
        prices=prices,
        weights=zero.copy(),
        units=zero.copy(),
    )
    actual = qis.MultiPortfolioData([strategy, benchmark]).compute_tracking_error_table(freq="ME")
    # This report sums weighted period returns: 20% + 0% + 10%, not compounded 32%.
    expected = pd.Series(
        [0.3, 0.3, 0.0],
        name="Asset",
        index=["Portfolio-Benchmark total perf", "Portfolio total perf", "Benchmark total perf"],
    )
    pd.testing.assert_series_equal(actual.loc["Asset", expected.index], expected)


@pytest.mark.parametrize("structured", [False, True])
def test_get_instruments_periodic_returns_preserves_distinct_column_keys(structured) -> None:
    dates = pd.date_range("2025-01-31", periods=3, freq="ME", name="date")
    columns = (
        pd.MultiIndex.from_tuples([("x", 1), ("x", np.nan)]) if structured else pd.Index([2, "2"])
    )
    prices = pd.DataFrame(
        [[100.0, 100.0], [120.0, 110.0], [132.0, 110.0]], index=dates, columns=columns
    )
    portfolio = qis.PortfolioData(
        nav=prices.iloc[:, 0].rename("Portfolio"),
        prices=prices,
        group_data=pd.Series("Group", index=columns),
    )
    expected = pd.DataFrame(
        [[np.nan, np.nan], [0.2, 0.1], [0.1, 0.0]], index=dates, columns=columns
    )
    pd.testing.assert_frame_equal(portfolio.get_instruments_periodic_returns(), expected)


@pytest.mark.parametrize(
    "kind,freq",
    [
        ("descending", None),
        ("duplicate", None),
        ("duplicate", "ME"),
        ("missing", "ME"),
    ],
)
def test_get_instruments_periodic_returns_irregular_dates_keep_native_behavior(kind, freq) -> None:
    dates = pd.date_range("2025-01-31", periods=4, freq="ME", name="date")
    if kind == "descending":
        index = dates[::-1]
    elif kind == "duplicate":
        index = pd.DatetimeIndex([dates[0], dates[0], dates[2], dates[3]], name="date")
    else:
        index = pd.DatetimeIndex([dates[0], pd.NaT, dates[2], dates[3]], name="date")
    portfolio = _portfolio(pd.DataFrame({"Asset": [100.0, 120.0, 120.0, 132.0]}, index=index))
    if freq is not None:
        match = "duplicate labels" if kind == "duplicate" else "must not contain NaT"
        with pytest.raises(ValueError, match=match):
            portfolio.get_instruments_periodic_returns(freq=freq)
    else:
        expected = (
            pd.DataFrame({"Asset": [np.nan, -1.0 / 11.0, 0.0]}, index=dates[:3])
            if kind == "descending"
            else pd.DataFrame({"Asset": [np.nan, np.nan, 0.0, 0.1]}, index=index)
        )
        pd.testing.assert_frame_equal(
            portfolio.get_instruments_periodic_returns(freq=None), expected, check_freq=False
        )


def test_get_attribution_table_by_instrument_target_weights_retain_native_path() -> None:
    dates = pd.date_range("2025-01-31", periods=3, freq="ME", name="date")
    prices = pd.DataFrame({"Asset": [100.0, 120.0, 132.0]}, index=dates)
    portfolio = _portfolio(prices)
    portfolio.input_weights = pd.DataFrame({"Asset": [0.5, 0.75, 0.25]}, index=dates)
    expected = pd.DataFrame({"Asset": [0.1, 0.075]}, index=dates[1:])
    pd.testing.assert_frame_equal(
        portfolio.get_attribution_table_by_instrument(is_input_weights=True), expected
    )

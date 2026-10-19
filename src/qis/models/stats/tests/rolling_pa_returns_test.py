"""Protect batched rolling annualised returns and the native exceptional-input path.

Literal calendar/endpoint expectations establish the financial convention; the original scalar
composition separately protects exact pandas results, warnings and errors outside the fast path.
"""

import warnings

import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import numpy as np
import pandas as pd
import pytest

import qis
import qis.models.stats.rolling_stats as rolling


def _prices() -> pd.DataFrame:
    """Use simple positive levels so window endpoints can be checked independently."""
    return pd.DataFrame(
        {"rising": np.arange(100.0, 112.0), "constant": np.full(12, 80.0)},
        index=pd.date_range("2020-01-01", periods=12, freq="D", name="date"),
    )


def _native(prices, window):
    """Retain the accepted scalar composition as a compatibility, not financial, oracle."""
    return prices.rolling(window).apply(rolling.ret.compute_pa_return)


def _assert_equal(actual, expected) -> None:
    compare = (
        pd.testing.assert_series_equal
        if isinstance(expected, pd.Series)
        else (pd.testing.assert_frame_equal)
    )
    compare(actual, expected, check_exact=True)
    assert actual.attrs == expected.attrs
    assert actual.flags.allows_duplicate_labels == expected.flags.allows_duplicate_labels


def _observe(function, prices, window):
    """Compare native diagnostics without suppressing or changing their expectations."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = function(prices, window)
            error = None
        except Exception as exc:
            result = None
            error = (type(exc), str(exc))
    return result, error, [(item.category, str(item.message)) for item in caught]


def test_compute_rolling_pa_returns_batches_public_window_preparation(monkeypatch) -> None:
    """A complete public calculation must not reconstruct every scalar price window."""
    prices = _prices()
    expected = _native(prices, 3)

    def unexpected_scalar_window(*args, **kwargs):
        pytest.fail("ordinary rolling PA returns must batch endpoint/calendar preparation")

    monkeypatch.setattr(rolling.ret, "compute_pa_return", unexpected_scalar_window)
    actual, title = qis.compute_rolling_perf_stat(
        prices, qis.RollingPerfStat.PA_RETURNS, roll_freq="ME", roll_periods=3
    )

    _assert_equal(actual, expected)
    assert title == "P.a. returns with roll_period=3"


@pytest.mark.parametrize(
    ("dates", "expected"),
    [
        (["2020-01-01", "2020-06-01", "2020-12-31"], 1.0),
        (["2020-01-01", "2020-06-01", "2021-01-01"], 2.0 ** (365.25 / 366) - 1),
        (["2024-03-10 00:00", "2024-03-10 03:00", "2024-03-10 05:00"], 1.0),
    ],
    ids=["365-days-unannualised", "366-days-geometric", "subday-minimum-one-day"],
)
def test_compute_rolling_pa_returns_preserves_literal_calendar_convention(dates, expected) -> None:
    """Three rows are not three return periods; annualisation uses their elapsed calendar span."""
    prices = pd.Series([100.0, 120.0, 200.0], index=pd.to_datetime(dates), name="Asset")
    actual = rolling.compute_rolling_pa_returns(prices, 3)

    assert actual.iloc[:2].isna().all()
    assert actual.iloc[2] == pytest.approx(expected, rel=1e-14, abs=1e-14)
    _assert_equal(actual, _native(prices, 3))


@pytest.mark.parametrize("container", ["frame", "series", "multiindex", "duplicates"])
def test_compute_rolling_pa_returns_preserves_native_container_metadata(container) -> None:
    """Batch construction must retain native labels, metadata and independently owned values."""
    prices = _prices()
    if container == "series":
        prices = prices["rising"]
        prices.flags.allows_duplicate_labels = False
    elif container == "multiindex":
        prices.columns = pd.MultiIndex.from_tuples([(1, "A"), (2, "B")], names=["id", "name"])
        prices.flags.allows_duplicate_labels = False
    elif container == "duplicates":
        prices.columns = pd.Index([7, 7], name="asset")
    prices.attrs["source"] = {"kind": "literal"}
    before = prices.copy(deep=True)
    expected = _native(prices, 3)

    actual = rolling.compute_rolling_pa_returns(prices, 3)

    _assert_equal(actual, expected)
    # Metadata must be changed first: a value assignment can detach pandas axes and hide aliases.
    if isinstance(actual, pd.DataFrame):
        actual.columns.names = ["result"] * actual.columns.nlevels
        expected.columns.names = ["result"] * expected.columns.nlevels
    else:
        actual.name = expected.name = "result"
    _assert_equal(prices, before)
    _assert_equal(actual, expected)
    actual.iloc[-1] = 9.0
    _assert_equal(prices, before)


@pytest.mark.parametrize(
    "case",
    [
        "mixed-missing",
        "nullable",
        "float32",
        "integer",
        "nonpositive",
        "infinity",
        "overflow",
        "underflow",
        "reordered",
        "duplicate-date",
        "nat",
        "nondate",
        "one",
        "zero",
        "oversize",
        "offset",
        "unsigned-window",
    ],
)
def test_compute_rolling_pa_returns_keeps_native_fallback_observations(case) -> None:
    """Ineligible representations must not acquire new value, warning or rejection policies."""
    prices = _prices()
    window = 3
    if case == "mixed-missing":
        prices.iloc[5, 0] = np.nan
        prices["late"] = [np.nan] * 4 + [50.0] * 8
        prices["never"] = np.nan
    elif case in ("nullable", "float32", "integer"):
        prices = prices.astype(
            {"nullable": "Float64", "float32": "float32", "integer": "int64"}[case]
        )
    elif case == "nonpositive":
        prices.iloc[4, 0] = 0.0
        prices.iloc[5, 0] = -1.0
    elif case == "infinity":
        prices.iloc[4, 0] = np.inf
    elif case in ("overflow", "underflow"):
        prices.iloc[:3, 0] = [1e-308, 1.0, 1e308] if case == "overflow" else [1e308, 1.0, 1e-308]
    elif case == "reordered":
        prices = prices.iloc[::-1]
    elif case == "duplicate-date":
        prices.index = pd.DatetimeIndex([prices.index[0], *prices.index[:-1]])
    elif case == "nat":
        prices.index = pd.DatetimeIndex([pd.NaT, *prices.index[1:]])
    elif case == "nondate":
        prices.index = pd.RangeIndex(len(prices))
    else:
        window = {
            "one": 1,
            "zero": 0,
            "oversize": 20,
            "offset": "3D",
            "unsigned-window": np.uint64(3),
        }[case]
    before = prices.copy(deep=True)
    expected, error, diagnostics = _observe(_native, prices, window)

    actual, actual_error, actual_diagnostics = _observe(
        rolling.compute_rolling_pa_returns, prices, window
    )

    assert actual_error == error
    assert actual_diagnostics == diagnostics
    if error is None:
        _assert_equal(actual, expected)
    _assert_equal(prices, before)


def test_compute_rolling_pa_returns_rechecks_edited_and_replaced_inputs() -> None:
    """Eligibility belongs to the current call, not an earlier complete panel or date axis."""
    prices = _prices()
    for stage in ("complete", "missing", "restored", "new-dates", "copy"):
        if stage == "missing":
            prices.iloc[5, 0] = np.nan
        elif stage == "restored":
            prices.iloc[5, 0] = 105.0
        elif stage == "new-dates":
            prices.index = pd.date_range("2010-01-01", periods=12, freq="180D")
        elif stage == "copy":
            prices = prices.copy()
        _assert_equal(rolling.compute_rolling_pa_returns(prices, 3), _native(prices, 3))


def test_plot_rolling_perf_stat_preserves_pa_data_and_drawn_labels() -> None:
    """The public chart must draw the same native-row statistic and public title."""
    prices = _prices()
    expected = _native(prices, 3)
    fig, ax = plt.subplots()
    try:
        qis.plot_rolling_perf_stat(
            prices,
            rolling_perf_stat=qis.RollingPerfStat.PA_RETURNS,
            roll_freq="ME",
            roll_periods=3,
            ax=ax,
        )
        fig.canvas.draw()
        # Seaborn omits unavailable points and also creates empty lines for legend handles.
        # Compare the actual data/date pairs, not those handles or the leading rolling NaNs.
        data_lines = [line for line in ax.lines if len(line.get_ydata())]
        for line, (_, column) in zip(data_lines, expected.items(), strict=True):
            observed = column.dropna()
            np.testing.assert_array_equal(line.get_ydata(), observed.to_numpy())
            np.testing.assert_array_equal(line.get_xdata(), date2num(observed.index))
        assert ax.get_title() == "P.a. returns with roll_period=3"
    finally:
        plt.close(fig)

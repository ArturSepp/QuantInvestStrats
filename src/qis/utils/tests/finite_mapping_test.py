"""Python finite mapping avoids JIT startup without changing native numerical contracts."""

import math

import numpy as np
import pandas as pd
import pytest
from numba import njit

import qis
from qis.utils import np_ops as npo


def _unexpected_compilation(*args, **kwargs):
    raise AssertionError("Ordinary Python finite mapping must not invoke the compiled mapper")


def _input(kind):
    values = np.array([0.5, -0.0, np.nan, np.inf, -np.inf, np.nextafter(0.0, 1.0)])
    if kind == "array1":
        return values
    if kind == "array2":
        return values.reshape(3, 2)
    if kind == "empty":
        return np.empty((0, 2))
    if kind == "series":
        return pd.Series(values, index=pd.bdate_range("2025-01-06", periods=6), name="Asset")
    return pd.DataFrame(
        values.reshape(3, 2), index=pd.bdate_range("2025-01-06", periods=3), columns=[2, "2"]
    )


@pytest.mark.parametrize("kind", ["array1", "array2", "frame", "series", "empty"])
def test_to_finite_np_maps_bits_and_preserves_caller_without_compilation(monkeypatch, kind):
    data = _input(kind)
    pandas_input = isinstance(data, (pd.Series, pd.DataFrame))
    before = data.copy(deep=True) if pandas_input else data.copy()
    values = data.to_numpy() if pandas_input else data
    expected = np.array(
        [x if math.isfinite(x) else 0.0 for x in values.flat], dtype=np.float64
    ).reshape(values.shape)
    monkeypatch.setattr(npo, "to_finite", _unexpected_compilation)

    actual = npo.to_finite_np(data)

    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    # Exercise returned-value ownership, including pandas' read-only array conversion.
    if actual.size:
        actual.flat[0] = 99.0
    if isinstance(data, pd.DataFrame):
        pd.testing.assert_frame_equal(data, before)
    elif isinstance(data, pd.Series):
        pd.testing.assert_series_equal(data, before)
    else:
        np.testing.assert_array_equal(data, before)
    if pandas_input:
        assert data.index.freqstr == "B"


@pytest.mark.parametrize("clip", [True, False])
def test_to_finite_np_preserves_clipping_after_mapping(monkeypatch, clip):
    monkeypatch.setattr(npo, "to_finite", _unexpected_compilation)
    data = np.array([-2.0, 0.5, 2.0, np.nan, np.inf])
    expected = [-1.0, 0.5, 1.0, 0.0, 0.0] if clip else [np.nan, 0.5, np.nan, 0.0, 0.0]
    np.testing.assert_array_equal(
        npo.to_finite_np(data, a_min=-1.0, a_max=1.0, is_min_max_clip_fill=clip), expected
    )


def test_to_finite_np_preserves_native_float32_promotion():
    # A Python float fill promotes this native result to float64; NumPy 2 where keeps float32.
    data = np.array([0.5, np.nan, np.inf], dtype=np.float32)
    actual = npo.to_finite_np(data, fill_value=0.25)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, [0.5, 0.25, 0.25])


class _ArrayWithUfuncOverride(np.ndarray):
    def __array_ufunc__(self, *args, **kwargs):
        raise AssertionError("The native mapper does not dispatch Python ufunc overrides")


@pytest.mark.parametrize("kind", ["nullable", "endian", "rank3", "subclass"])
def test_to_finite_np_preserves_native_fallback_results_and_diagnostics(kind):
    if kind == "nullable":
        data = pd.DataFrame({"Asset": pd.Series([1.0, pd.NA], dtype="Float64")})
        converted = data.to_numpy()
    else:
        data = np.array([1.0, np.nan], dtype=">f8" if kind == "endian" else "float64")
        if kind == "rank3":
            data = data.reshape(1, 1, 2)
        elif kind == "subclass":
            data = data.view(_ArrayWithUfuncOverride)
        converted = data.copy()
    # Native admission can differ by pandas/Numba version; preserve either complete outcome.
    try:
        expected = npo.to_finite(converted, 0.0)
    except Exception as expected_error:
        with pytest.raises(type(expected_error)) as actual_error:
            npo.to_finite_np(data)
        assert str(actual_error.value) == str(expected_error)
    else:
        actual = npo.to_finite_np(data)
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)


def test_to_finite_remains_callable_from_compiled_code():
    @njit
    def compiled_caller(values):
        return npo.to_finite(values, 0.0)

    actual = compiled_caller(np.array([-0.0, np.nan, np.inf, 1.0]))
    expected = np.array([-0.0, 0.0, 0.0, 1.0])
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    assert compiled_caller.nopython_signatures and npo.to_finite.nopython_signatures


@pytest.mark.parametrize("freq", [None, "W-WED"])
def test_get_grouped_long_short_weights_preserves_complete_exposures_without_jit(monkeypatch, freq):
    dates = pd.bdate_range("2025-01-06", periods=6, name="date")
    weights = pd.DataFrame(
        [
            [0.5, -0.25, 0.125, -0.125],
            [0.25, -0.5, 0.25, 0.0],
            [0.5, -0.25, np.nan, 0.125],
            [0.75, -0.125, 0.25, -0.25],
            [0.125, 0.0, 0.5, -0.5],
            [0.25, -0.125, 0.5, -0.25],
        ],
        index=dates,
        columns=["A", "B", "C", "D"],
    )
    before = weights.copy(deep=True)
    prices = pd.DataFrame(100.0, index=dates, columns=weights.columns)
    portfolio = qis.PortfolioData(
        nav=pd.Series(100.0, index=dates, name="Portfolio"),
        prices=prices,
        weights=weights,
        units=weights.copy(),
        instrument_pnl=prices * 0.0,
        group_data=pd.Series(["Equity", "Equity", "Rates", "Rates"], index=weights.columns),
        group_order=["Equity", "Rates"],
    )
    expected = {
        "Equity": pd.DataFrame(
            {
                "Total": [0.25, -0.25, 0.25, 0.625, 0.125, 0.125],
                "Net Long": [0.5, 0.25, 0.5, 0.75, 0.125, 0.25],
                "Net Short": [-0.25, -0.5, -0.25, -0.125, 0.0, -0.125],
            },
            index=dates,
        ),
        "Rates": pd.DataFrame(
            {
                "Total": [0.0, 0.25, 0.125, 0.0, 0.0, 0.25],
                "Net Long": [0.125, 0.25, 0.125, 0.25, 0.5, 0.5],
                "Net Short": [-0.125, 0.0, 0.0, -0.25, -0.5, -0.25],
            },
            index=dates,
        ),
    }
    monkeypatch.setattr(npo, "to_finite", _unexpected_compilation)
    aggregate, instruments = portfolio.get_grouped_long_short_weights(weights_freq=freq)
    assert list(aggregate) == list(instruments) == ["Equity", "Rates"]
    for group, columns in [("Equity", ["A", "B"]), ("Rates", ["C", "D"])]:
        selected = before[columns]
        if freq is not None:
            # W-WED selects each column's last nonmissing value; C's first-bin mark is Jan 7.
            index = pd.date_range("2025-01-08", periods=2, freq="W-WED", name="date")
            selected = selected.iloc[[2, 5]].set_axis(index)
            expected[group] = expected[group].iloc[[2, 5]].set_axis(index)
            if group == "Rates":
                selected.iloc[0, 0] = 0.25
                expected[group].iloc[0] = [0.375, 0.375, 0.0]
        pd.testing.assert_frame_equal(aggregate[group], expected[group])
        pd.testing.assert_frame_equal(instruments[group], selected)
    instruments["Equity"].iloc[0, 0] = 99.0
    pd.testing.assert_frame_equal(weights, before)
    assert weights.index.freqstr == "B"

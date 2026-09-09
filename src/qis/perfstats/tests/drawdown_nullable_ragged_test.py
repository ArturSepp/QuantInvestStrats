"""Regression coverage for nullable maximum and current drawdown reductions.

Drawdown values depend on each column's running peak and observed price history, not on pandas'
missing-value representation or neighboring column states. The mixed panels below exercise every
material missingness shape together and independently state the expected reductions.
"""

import warnings
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# qis
from qis.perfstats.config import PerfParams, PerfStat
from qis.perfstats.perf_stats import compute_max_current_drawdown, compute_risk_table


_DATES = pd.date_range("2023-01-31", periods=8, freq="ME")
_COLUMNS = ("complete", "leading", "interior", "trailing", "missing")
_EXPECTED_WORST = np.asarray((-1.0 / 9.0, -0.20, -0.10, -1.0 / 9.0, np.nan), dtype=float)
_EXPECTED_BEST = np.asarray((0.25, 0.50, 0.50, 0.50, np.nan), dtype=float)
_EXPECTED_MAXIMUM = np.asarray((-0.25, -0.20, -0.20, -0.20, np.nan), dtype=float)
_EXPECTED_CURRENT = np.asarray((-0.25, -1.0 / 13.0, -1.0 / 6.0, 0.0, np.nan), dtype=float)


def _ordinary_mixed_prices() -> pd.DataFrame:
    """Create complete, ragged, and all-missing price histories in one panel.

    Returns:
        Ordinary floating price panel with independently calculable drawdowns.
    """
    return pd.DataFrame(
        {
            "complete": (100.0, 90.0, 80.0, 100.0, 120.0, 110.0, 100.0, 90.0),
            "leading": (np.nan, np.nan, 100.0, 80.0, 120.0, 110.0, 130.0, 120.0),
            "interior": (100.0, 90.0, np.nan, 80.0, 120.0, np.nan, 110.0, 100.0),
            "trailing": (100.0, 90.0, 80.0, 120.0, np.nan, np.nan, np.nan, np.nan),
            "missing": (np.nan,) * len(_DATES),
        },
        index=_DATES,
    )


def _column_name(perf_stat: PerfStat) -> str:
    """Return the typed full label carried by a performance statistic."""
    name = perf_stat.value.name
    if not isinstance(name, str):
        raise TypeError(f"expected a string column label, got {type(name)!r}")
    return name


def _assert_drawdown_arrays(
    maximum_drawdowns: object,
    current_drawdowns: object,
) -> None:
    """Compare public drawdown arrays with the independent positional reference.

    Args:
        maximum_drawdowns: Public maximum-drawdown result.
        current_drawdowns: Public current-drawdown result.
    """
    assert isinstance(maximum_drawdowns, np.ndarray)
    assert isinstance(current_drawdowns, np.ndarray)
    maximum_array = cast(NDArray[np.float64], maximum_drawdowns)
    current_array = cast(NDArray[np.float64], current_drawdowns)
    assert maximum_array.dtype == np.dtype(float)
    assert current_array.dtype == np.dtype(float)
    np.testing.assert_allclose(maximum_array, _EXPECTED_MAXIMUM, equal_nan=True)
    np.testing.assert_allclose(current_array, _EXPECTED_CURRENT, equal_nan=True)


def test_compute_max_current_drawdown_normalizes_nullable_mixed_panels() -> None:
    """Match ordinary, nullable, and mixed-storage drawdowns without warnings.

    Complete, leading, interior, trailing, and all-missing columns remain independent when pandas
    nullable columns introduce ``pd.NA`` at the NumPy reduction boundary. The result order follows
    the input columns, and normalization must not mutate caller-owned data.
    """
    ordinary_prices = _ordinary_mixed_prices()
    nullable_prices = ordinary_prices.astype("Float64")
    mixed_prices = ordinary_prices.astype(
        {"leading": "Float64", "trailing": "Float64", "missing": "Float64"}
    )

    for prices in (ordinary_prices, nullable_prices, mixed_prices):
        original_prices = prices.copy(deep=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            maximum_drawdowns, current_drawdowns = compute_max_current_drawdown(prices=prices)

        _assert_drawdown_arrays(maximum_drawdowns, current_drawdowns)
        pd.testing.assert_frame_equal(prices, original_prices)


def test_compute_max_current_drawdown_normalizes_nullable_all_missing_series() -> None:
    """Return scalar NaNs for an all-missing nullable Series without warnings."""
    prices = pd.Series(
        pd.array([pd.NA] * len(_DATES), dtype="Float64"), index=_DATES, name="missing"
    )
    original_prices = prices.copy(deep=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        maximum_drawdown, current_drawdown = compute_max_current_drawdown(prices=prices)

    assert isinstance(maximum_drawdown, float)
    assert isinstance(current_drawdown, float)
    assert np.isnan(maximum_drawdown)
    assert np.isnan(current_drawdown)
    pd.testing.assert_series_equal(prices, original_prices)


def test_compute_risk_table_reports_nullable_mixed_drawdowns() -> None:
    """Carry normalized drawdowns and two-endpoint returns through the risk table."""
    prices = _ordinary_mixed_prices().astype("Float64")
    original_prices = prices.copy(deep=True)

    table = compute_risk_table(
        prices=prices,
        perf_params=PerfParams(freq_vol="ME", freq_drawdown="ME", freq_skewness="ME"),
    )

    assert tuple(table.index) == _COLUMNS
    maximum_values = cast(
        NDArray[np.float64],
        table[_column_name(PerfStat.MAX_DD)].to_numpy(dtype=float, na_value=np.nan),
    )
    current_values = cast(
        NDArray[np.float64],
        table[_column_name(PerfStat.CURRENT_DD)].to_numpy(dtype=float, na_value=np.nan),
    )
    worst_values = cast(
        NDArray[np.float64],
        table[_column_name(PerfStat.WORST)].to_numpy(dtype=float, na_value=np.nan),
    )
    best_values = cast(
        NDArray[np.float64],
        table[_column_name(PerfStat.BEST)].to_numpy(dtype=float, na_value=np.nan),
    )
    np.testing.assert_allclose(worst_values, _EXPECTED_WORST, equal_nan=True)
    np.testing.assert_allclose(best_values, _EXPECTED_BEST, equal_nan=True)
    np.testing.assert_allclose(
        maximum_values,
        _EXPECTED_MAXIMUM,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        current_values,
        _EXPECTED_CURRENT,
        equal_nan=True,
    )
    pd.testing.assert_frame_equal(prices, original_prices)

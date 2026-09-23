"""Regression tests for allocation-free rolling volatility and Sharpe windows.

The rolling reducers must receive raw NumPy windows without changing the established log-return,
sample-standard-deviation, annualisation, missing-value, or labeled-container contracts.
"""

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pandas as pd
import pytest
from pandas.core.window.rolling import Rolling

from qis.models.stats.rolling_stats import (
    compute_rolling_sharpes,
    compute_rolling_vols,
    compute_sharpe,
)


_WINDOW = 3
_ANNUALISATION = 252.0


def _nullable_prices() -> pd.DataFrame:
    """Return a labeled panel that crosses finite, interior-missing, and constant paths."""
    index = pd.bdate_range("2024-01-02", periods=8, name="date")
    return pd.DataFrame(
        {
            "complete": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
            "interior_missing": [90.0, 91.0, pd.NA, 93.0, 94.0, 95.0, 96.0, 97.0],
            "constant": [100.0] * 8,
        },
        index=index,
        dtype="Float64",
    )


def _reference_rolling_stat(
    prices: pd.DataFrame,
    reducer: Callable[[np.ndarray], float],
) -> pd.DataFrame:
    """Calculate rolling results directly from the stated log-return window convention."""
    log_returns = prices.ffill().astype(float).map(np.log).diff()
    expected = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns, dtype=float)
    for column in prices.columns:
        values = log_returns[column].to_numpy(dtype=float, na_value=np.nan)
        for end in range(_WINDOW, len(values) + 1):
            window = values[end - _WINDOW : end]
            if np.isfinite(window).all():
                expected.loc[prices.index[end - 1], column] = reducer(window)
    return expected


def test_rolling_volatility_and_sharpe_request_raw_array_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid reconstructing a labeled Series for every rolling reduction."""
    observed_raw: list[bool] = []
    observed_window_types: list[type[Any]] = []
    original_apply = Rolling.apply

    def recording_apply(
        self: Rolling,
        func: Callable[..., Any],
        raw: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> pd.Series | pd.DataFrame:
        observed_raw.append(raw)

        def recording_reducer(values: Any, *call_args: Any, **call_kwargs: Any) -> Any:
            observed_window_types.append(type(values))
            return func(values, *call_args, **call_kwargs)

        return original_apply(
            self,
            recording_reducer,
            raw=raw,
            engine=engine,
            engine_kwargs=engine_kwargs,
            args=args,
            kwargs=kwargs,
        )

    monkeypatch.setattr(Rolling, "apply", recording_apply)
    prices = _nullable_prices()["complete"].astype(float)

    compute_rolling_vols(prices=prices, roll_periods=_WINDOW)
    compute_rolling_sharpes(prices=prices, roll_periods=_WINDOW)

    assert observed_raw == [True, True]
    assert observed_window_types
    assert set(observed_window_types) == {np.ndarray}


def test_rolling_raw_windows_preserve_values_labels_missingness_and_ownership() -> None:
    """Match independent formulas across nullable, missing, and zero-volatility columns."""
    prices = _nullable_prices()
    original = prices.copy(deep=True)
    scale = np.sqrt(_ANNUALISATION)

    expected_vols = _reference_rolling_stat(
        prices,
        lambda values: float(scale * np.std(values, ddof=1)),
    )
    expected_sharpes = _reference_rolling_stat(
        prices,
        lambda values: (
            float(scale * np.expm1(np.mean(values)) / np.std(values, ddof=1))
            if np.std(values, ddof=1) > 0.0
            else np.nan
        ),
    )

    actual_vols = compute_rolling_vols(prices=prices, roll_periods=_WINDOW)
    actual_sharpes = compute_rolling_sharpes(prices=prices, roll_periods=_WINDOW)
    complete_log_returns = prices["complete"].astype(float).map(np.log).diff().dropna()
    expected_direct_sharpe = (
        scale * np.expm1(complete_log_returns.mean()) / complete_log_returns.std()
    )

    pd.testing.assert_frame_equal(actual_vols, expected_vols, check_exact=True)
    pd.testing.assert_frame_equal(actual_sharpes, expected_sharpes, check_exact=True)
    assert compute_sharpe(complete_log_returns) == expected_direct_sharpe
    pd.testing.assert_frame_equal(prices, original)

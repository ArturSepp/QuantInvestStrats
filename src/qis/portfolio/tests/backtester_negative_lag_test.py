"""Regression tests for the implementation-lag domain in the portfolio backtester.

Dated targets may trade on their observation or a later price observation, but never before the
target exists. These tests cover that causal boundary, the complete scalar-validation domain, and
the unchanged rule that the lag is unused for static weights.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest

import qis


def _prices() -> pd.DataFrame:
    """Return a small price panel with unambiguous schedule positions."""
    dates = pd.bdate_range("2024-01-02", periods=5)
    return pd.DataFrame(
        {
            "A": [100.0, 101.0, 102.0, 103.0, 104.0],
            "B": [100.0, 100.0, 100.0, 100.0, 100.0],
        },
        index=dates,
    )


def _dated_weights(prices: pd.DataFrame, position: int = 1) -> pd.DataFrame:
    """Return one dated target whose mapped trade observation is easy to inspect."""
    return pd.DataFrame(
        [[1.0, 0.0]],
        index=prices.index[[position]],
        columns=prices.columns,
    )


@pytest.mark.parametrize("decision_position", [0, 2], ids=["first-date", "mid-history"])
@pytest.mark.parametrize("invalid_lag", [-1, np.int64(-1)], ids=["python-int", "numpy-int"])
def test_backtest_model_portfolio_rejects_negative_implementation_lag_before_mapping(
    decision_position: int,
    invalid_lag: Any,
) -> None:
    """A negative lag cannot wrap or expose a target before its decision date."""
    prices = _prices()
    weights = _dated_weights(prices, position=decision_position)
    caller_weights = weights.copy()

    with pytest.raises(
        ValueError,
        match="weight_implementation_lag must be a non-negative integer or None",
    ):
        qis.backtest_model_portfolio(
            prices=prices,
            weights=weights,
            weight_implementation_lag=invalid_lag,
        )

    pd.testing.assert_frame_equal(weights, caller_weights)


@pytest.mark.parametrize(
    "invalid_lag",
    [
        pytest.param(True, id="python-bool"),
        pytest.param(np.bool_(False), id="numpy-bool"),
        pytest.param(1.0, id="integral-float"),
        pytest.param(0.5, id="fractional-float"),
        pytest.param(np.nan, id="nan"),
        pytest.param(np.inf, id="positive-infinity"),
        pytest.param(-np.inf, id="negative-infinity"),
        pytest.param(pd.NA, id="pandas-missing"),
    ],
)
def test_backtest_model_portfolio_rejects_nonintegral_implementation_lag(
    invalid_lag: Any,
) -> None:
    """Unsupported scalar classes fail at the public boundary with one clear error."""
    prices = _prices()
    weights = _dated_weights(prices)
    caller_weights = weights.copy()

    with pytest.raises(
        ValueError,
        match="weight_implementation_lag must be a non-negative integer or None",
    ):
        qis.backtest_model_portfolio(
            prices=prices,
            weights=weights,
            weight_implementation_lag=invalid_lag,
        )

    pd.testing.assert_frame_equal(weights, caller_weights)


@pytest.mark.parametrize(
    ("lag", "expected_trade_position"),
    [
        pytest.param(None, 1, id="none"),
        pytest.param(0, 1, id="python-zero"),
        pytest.param(np.int64(0), 1, id="numpy-zero"),
        pytest.param(1, 2, id="python-positive"),
        pytest.param(np.int64(1), 2, id="numpy-positive"),
    ],
)
def test_backtest_model_portfolio_preserves_valid_implementation_lag(
    lag: Any,
    expected_trade_position: int,
) -> None:
    """None and non-negative integer scalars retain observation-count scheduling."""
    prices = _prices()
    weights = _dated_weights(prices)

    result = qis.backtest_model_portfolio(
        prices=prices,
        weights=weights,
        weight_implementation_lag=lag,
    )

    expected_rebalancing = pd.Series(False, index=prices.index)
    expected_rebalancing.iloc[expected_trade_position] = True
    pd.testing.assert_series_equal(result.is_rebalancing, expected_rebalancing)


def test_backtest_model_portfolio_ignores_implementation_lag_for_static_weights() -> None:
    """The dated-weight-only parameter does not change a static-weight schedule."""
    prices = _prices()
    weights = pd.Series({"A": 0.6, "B": 0.4})

    expected = qis.backtest_model_portfolio(
        prices=prices,
        weights=weights,
        rebalancing_freq="QE",
        is_rebalanced_at_first_date=True,
    )
    result = qis.backtest_model_portfolio(
        prices=prices,
        weights=weights,
        rebalancing_freq="QE",
        weight_implementation_lag=-1,
        is_rebalanced_at_first_date=True,
    )

    pd.testing.assert_series_equal(result.nav, expected.nav)
    pd.testing.assert_frame_equal(result.units, expected.units)
    pd.testing.assert_frame_equal(result.weights, expected.weights)

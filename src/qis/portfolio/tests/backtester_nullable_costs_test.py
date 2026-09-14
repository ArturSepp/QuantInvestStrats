"""Regression tests for nullable transaction-cost schedules.

The public backtester accepts dated pandas cost DataFrames and treats missing schedule cells as
zero. These tests require pandas nullable floating storage to preserve the same traded-notional
accounting, labels, and caller ownership as an ordinary floating schedule before the Numba kernel.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import qis


DATES = pd.bdate_range("2026-01-05", periods=4, name="date")


@pytest.mark.parametrize(
    ("opening_b_rate", "expected_nav", "expected_units", "expected_costs"),
    [
        pytest.param(
            0.02,
            [98.5, 98.5, 97.9015, 97.9015],
            [[0.5, 0.5], [0.5, 0.5], [0.6895, 0.2955], [0.6895, 0.2955]],
            [[0.5, 1.0], [0.0, 0.0], [0.1895, 0.409], [0.0, 0.0]],
            id="all-finite",
        ),
        pytest.param(
            None,
            [99.5, 99.5, 98.9005, 98.9005],
            [[0.5, 0.5], [0.5, 0.5], [0.6965, 0.2985], [0.6965, 0.2985]],
            [[0.5, 0.0], [0.0, 0.0], [0.1965, 0.403], [0.0, 0.0]],
            id="missing-cell",
        ),
    ],
)
def test_backtest_model_portfolio_normalizes_nullable_cost_dataframe(
    opening_b_rate: float | None,
    expected_nav: list[float],
    expected_units: list[list[float]],
    expected_costs: list[list[float]],
) -> None:
    """Nullable cost schedules preserve exact opening and later-trade accounting."""
    prices = pd.DataFrame(100.0, index=DATES, columns=["A", "B"])
    weights = pd.DataFrame(
        [[0.5, 0.5], [0.7, 0.3]],
        index=DATES[[0, 2]],
        columns=prices.columns,
    )
    opening_b_value = pd.NA if opening_b_rate is None else opening_b_rate
    costs = pd.DataFrame(
        [[0.02, 0.01], [opening_b_value, 0.01]],
        index=DATES[[1, 0]],
        columns=["B", "A"],
        dtype="Float64",
    )
    caller_costs = costs.copy(deep=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = qis.backtest_model_portfolio(
            prices=prices,
            weights=weights,
            initial_nav=100.0,
            rebalancing_costs=costs,
        )

    expected_nav_array = np.asarray(expected_nav)
    expected_units_array = np.asarray(expected_units)
    expected_costs_array = np.asarray(expected_costs)
    expected_weights = 100.0 * expected_units_array / expected_nav_array[:, np.newaxis]
    np.testing.assert_allclose(result.nav.to_numpy(), expected_nav_array)
    np.testing.assert_allclose(result.units.to_numpy(), expected_units_array)
    np.testing.assert_allclose(result.weights.to_numpy(), expected_weights)
    np.testing.assert_allclose(result.realized_costs.to_numpy(), expected_costs_array)
    pd.testing.assert_index_equal(pd.DatetimeIndex(result.nav.index), DATES)
    pd.testing.assert_index_equal(result.realized_costs.index, DATES)
    pd.testing.assert_index_equal(result.realized_costs.columns, prices.columns)
    pd.testing.assert_frame_equal(costs, caller_costs)

"""Regression tests for strategy isolation from retained discrete state."""

# packages
import pandas as pd

# qis
from qis.discrete_portfolio import DiscretePortfolioState, Order, backtest_discrete_portfolio


def test_strategy_mutation_cannot_rewrite_retained_state() -> None:
    """Strategy-owned pandas mutations stay isolated from history and reporting."""
    index = pd.date_range("2026-01-05", periods=2, freq="D")
    columns = pd.Index(["SPY", "TLT"], name="ticker")
    prices = pd.DataFrame([[100.0, 50.0], [101.0, 51.0]], index=index, columns=columns)
    original_prices = prices.copy(deep=True)

    class MutateCallbackObjects:
        def on_bar(
            self,
            timestamp: pd.Timestamp,
            current_prices: pd.Series,
            state: DiscretePortfolioState,
        ) -> list[Order]:
            # High-level pandas writes may replace a read-only buffer, and frozen dataclasses do
            # not prevent metadata changes on their Series fields.
            for values in (
                state.units,
                state.prices,
                state.position_values,
                state.weights,
            ):
                values.index = pd.Index(["BROKEN_A", "BROKEN_B"], name="broken_axis")
                values.name = "broken_series"
                values.attrs["owner"] = "strategy"
                values.iloc[0] = -999.0
            current_prices.index = pd.Index(["BROKEN_A", "BROKEN_B"], name="broken_axis")
            current_prices.name = "broken_prices"
            current_prices.attrs["owner"] = "strategy"
            current_prices.iloc[0] = -1.0
            if timestamp == index[0]:
                return [Order("entry", timestamp, "SPY", 1.0)]
            return []

    result = backtest_discrete_portfolio(
        prices=prices,
        strategy=MutateCallbackObjects(),
        initial_cash=1_000.0,
    )

    pd.testing.assert_frame_equal(prices, original_prices)
    expected_units = pd.DataFrame([[0.0, 0.0], [1.0, 0.0]], index=index, columns=columns)
    expected_values = pd.DataFrame([[0.0, 0.0], [101.0, 0.0]], index=index, columns=columns)
    expected_weights = pd.DataFrame([[0.0, 0.0], [0.101, 0.0]], index=index, columns=columns)
    expected_nav = pd.Series(1_000.0, index=index, name="DiscretePortfolio")
    expected_cash = [1_000.0, 899.0]

    portfolio_data = result.portfolio_data
    assert portfolio_data is not None
    pd.testing.assert_frame_equal(portfolio_data.units, expected_units)
    pd.testing.assert_frame_equal(portfolio_data.prices, original_prices)
    pd.testing.assert_frame_equal(portfolio_data.weights, expected_weights)
    pd.testing.assert_series_equal(portfolio_data.nav, expected_nav)

    assert result.order_ledger.loc[0, "order_id"] == "entry"
    assert result.trade_ledger.loc[0, "filled_quantity"] == 1.0
    assert result.trade_ledger.loc[0, "fill_time"] == index[1]

    for row, (state, expected_prices) in enumerate(
        zip(result.states, original_prices.itertuples(index=False))
    ):
        pd.testing.assert_series_equal(
            state.units,
            expected_units.iloc[row].rename("units"),
        )
        pd.testing.assert_series_equal(
            state.prices,
            pd.Series(expected_prices, index=columns, name="prices"),
        )
        pd.testing.assert_series_equal(
            state.position_values,
            expected_values.iloc[row].rename(None),
        )
        pd.testing.assert_series_equal(
            state.weights,
            expected_weights.iloc[row].rename(None),
        )
        assert state.units.attrs == {}
        assert state.prices.attrs == {}
        assert state.position_values.attrs == {}
        assert state.weights.attrs == {}
        assert state.cash == expected_cash[row]
        assert state.nav == 1_000.0
    pd.testing.assert_frame_equal(
        portfolio_data.instrument_pnl,
        pd.DataFrame(0.0, index=index, columns=columns),
    )

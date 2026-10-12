"""Named carry rates retain their asset identity before positional backtest execution."""

import warnings

import numpy as np
import pandas as pd
import pytest
from numba.core.errors import TypingError

import qis


@pytest.mark.parametrize("price_order", [["A", "B"], ["B", "A"]])
@pytest.mark.parametrize("carry_order", [["A", "B"], ["B", "A"]])
@pytest.mark.parametrize("dated_weights", [False, True])
def test_backtest_model_portfolio_pairs_named_carry_with_held_instrument(
    price_order, carry_order, dated_weights
):
    """A's carry belongs to held A, regardless of either panel's column order."""
    dates = pd.date_range("2026-01-01", periods=3, tz="UTC", name="date")
    prices = pd.DataFrame(100.0, index=dates, columns=pd.Index(price_order, name="asset"))
    carry = pd.DataFrame({"A": 0.365, "B": 0.0}, index=dates)[carry_order].iloc[[2, 0, 1]]
    weights = (
        pd.DataFrame({"A": [1.0], "B": [0.0]}, index=dates[:1])
        if dated_weights
        else {"A": 1.0, "B": 0.0}
    )
    before_prices, before_carry = prices.copy(deep=True), carry.copy(deep=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = qis.backtest_model_portfolio(
            prices,
            weights,
            instruments_carry=carry,
            rebalancing_freq="YE",
            is_rebalanced_at_first_date=True,
        )

    # One unit of A at 100 earns 100 * .365 / 365 = .1 per calendar day; B earns nothing.
    nav = pd.Series([100.0, 100.1, 100.2], index=dates, name="Portfolio")
    units = pd.DataFrame({"A": 1.0, "B": 0.0}, index=dates)[price_order]
    units.columns.name = "asset"
    pd.testing.assert_series_equal(result.nav, nav)
    pd.testing.assert_frame_equal(result.units, units)
    pd.testing.assert_frame_equal(result.weights, (100.0 * units).div(nav, axis=0))
    pd.testing.assert_frame_equal(result.realized_costs, units * 0.0)
    pd.testing.assert_frame_equal(prices, before_prices)
    pd.testing.assert_frame_equal(carry, before_carry)


@pytest.mark.parametrize("carry_order", [["A", "B"], ["B", "A"]])
def test_backtest_model_portfolio_preserves_dated_carry_cash_and_trade_ledger(carry_order):
    """Named selection leaves as-of quotes, ACT/365, held units and trade costs intact."""
    dates = pd.DatetimeIndex(["2026-01-02", "2026-01-05", "2026-01-06"], tz="UTC", name="date")
    chronological = pd.DataFrame(
        {"A": [100.0, 110.0, 121.0], "B": [100.0, 90.0, 81.0]}, index=dates
    )
    prices = chronological[["B", "A"]].iloc[[2, 0, 1]]
    quote_dates = pd.DatetimeIndex(["2026-01-01", "2026-01-04", "2026-01-06"], tz="UTC")
    carry = pd.DataFrame(
        {"A": [0.365, 0.73, 0.365], "B": [-0.1825, -0.365, np.nan]}, index=quote_dates
    )[carry_order].iloc[[2, 0, 1]]
    targets = pd.DataFrame({"A": [0.3, 0.8], "B": [0.7, 0.2]}, index=dates[[2, 0]])
    costs = pd.Series({"A": 0.01, "B": 0.02})
    before_prices, before_carry, before_targets = (
        prices.copy(deep=True),
        carry.copy(deep=True),
        targets.copy(deep=True),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = qis.backtest_model_portfolio(
            prices, targets, instruments_carry=carry, rebalancing_costs=costs
        )

    # Opening units .8/.2 cost .8 + .4 = 1.2. Monday receives the Sunday quote over three
    # days: 88*.73*3/365 - 18*.365*3/365 = .474. Tuesday adds 96.8*.365/365 = .0968;
    # B's missing quote contributes no carry under the existing kernel policy.
    pretrade_nav = 112.3708
    final_units = np.array([0.3 * pretrade_nav / 121.0, 0.7 * pretrade_nav / 81.0])
    final_costs = np.array([0.6308876, 1.2491912])
    nav = pd.Series([98.8, 105.274, 110.4907212], index=dates, name="Portfolio")
    units = pd.DataFrame([[0.8, 0.2], [0.8, 0.2], final_units], index=dates, columns=["A", "B"])
    realized_costs = pd.DataFrame(
        [[0.8, 0.4], [0.0, 0.0], final_costs], index=dates, columns=["A", "B"]
    )
    pd.testing.assert_series_equal(result.nav, nav)
    pd.testing.assert_frame_equal(result.units, units[["B", "A"]])
    pd.testing.assert_frame_equal(result.realized_costs, realized_costs[["B", "A"]])
    pd.testing.assert_frame_equal(
        result.weights, (units * chronological).div(nav, axis=0)[["B", "A"]]
    )
    pd.testing.assert_frame_equal(prices, before_prices)
    pd.testing.assert_frame_equal(carry, before_carry)
    pd.testing.assert_frame_equal(targets, before_targets)

    # A future price/quote must not rewrite the already-accounted prefix.
    future_date = dates[-1] + pd.Timedelta(days=1)
    future_index = pd.DatetimeIndex([future_date], name=dates.name)
    future_prices = pd.concat(
        [prices, pd.DataFrame({"B": [10000.0], "A": [1.0]}, index=future_index)]
    )
    future_carry = pd.concat(
        [carry, pd.DataFrame({"A": [-10.0], "B": [10.0]}, index=[future_date])]
    )
    future = qis.backtest_model_portfolio(
        future_prices, targets, instruments_carry=future_carry, rebalancing_costs=costs
    )
    pd.testing.assert_series_equal(future.nav.loc[dates], result.nav)
    pd.testing.assert_frame_equal(future.units.loc[dates], result.units)


@pytest.mark.parametrize("schema", ["single", "single-unknown", "unknown", "duplicate", "series"])
def test_backtest_model_portfolio_preserves_legacy_nonmatching_carry_schemas(schema):
    """A column-order fix must not impose new missing-label or broadcast semantics."""
    dates = pd.date_range("2026-01-01", periods=3)
    prices = pd.DataFrame(100.0, index=dates, columns=["A", "B"])
    carry = pd.DataFrame({"A": 0.365, "B": 0.0}, index=dates)
    if schema == "single":
        carry = carry[["A"]]
    elif schema == "single-unknown":
        carry = carry[["A"]].set_axis(["shared"], axis=1)
    elif schema == "unknown":
        carry = carry.set_axis(["A", "X"], axis=1)
    elif schema == "duplicate":
        carry = carry.set_axis(["A", "A"], axis=1)
    else:
        carry = carry["A"]
    before = carry.copy(deep=True)

    result = qis.backtest_model_portfolio(
        prices,
        {"A": 0.8, "B": 0.2},
        instruments_carry=carry,
        rebalancing_freq="YE",
        is_rebalanced_at_first_date=True,
    )

    # Singleton and Series rates broadcast to all 100 notional; two-column rates stay positional.
    daily_credit = 0.1 if schema in ("single", "single-unknown", "series") else 0.08
    np.testing.assert_allclose(result.nav, [100.0, 100.0 + daily_credit, 100.0 + 2 * daily_credit])
    if isinstance(carry, pd.DataFrame):
        pd.testing.assert_frame_equal(carry, before)
    else:
        pd.testing.assert_series_equal(carry, before)


def test_backtest_model_portfolio_preserves_ambiguous_price_column_carry_path():
    """Duplicate price labels cannot define a new name-based carry mapping."""
    dates = pd.date_range("2026-01-01", periods=3)
    prices = pd.DataFrame(100.0, index=dates, columns=["A", "A"])
    carry = pd.DataFrame({"B": 0.0, "A": 0.365}, index=dates)
    result = qis.backtest_model_portfolio(
        prices,
        [0.8, 0.2],
        instruments_carry=carry,
        rebalancing_freq="YE",
        is_rebalanced_at_first_date=True,
    )
    np.testing.assert_allclose(result.nav, [100.0, 100.02, 100.04])


@pytest.mark.parametrize("width", [0, 3])
def test_backtest_model_portfolio_preserves_carry_shape_rejection(width):
    """Empty and excess-column carry panels must not be silently filled or trimmed."""
    dates = pd.date_range("2026-01-01", periods=3)
    prices = pd.DataFrame(100.0, index=dates, columns=["A", "B"])
    carry = pd.DataFrame(0.365, index=dates, columns=["A", "B", "C"][:width])
    before = carry.copy(deep=True)
    with pytest.raises(ValueError, match="unable to broadcast"):
        qis.backtest_model_portfolio(
            prices,
            {"A": 1.0},
            instruments_carry=carry,
            rebalancing_freq="YE",
            is_rebalanced_at_first_date=True,
        )
    pd.testing.assert_frame_equal(carry, before)


@pytest.mark.parametrize("storage", ["nullable", "mixed"])
@pytest.mark.parametrize("missing", [False, True])
def test_backtest_model_portfolio_preserves_nullable_carry_conversion_rejection(storage, missing):
    """Aligning labels does not promise a new object-array conversion for the Numba kernel."""
    dates = pd.date_range("2026-01-01", periods=3)
    prices = pd.DataFrame(100.0, index=dates, columns=["A", "B"])
    carry = pd.DataFrame({"B": 0.0, "A": 0.365}, index=dates)
    if storage == "nullable":
        carry = carry.astype("Float64")
    else:
        carry["A"] = carry["A"].astype("Float64")
    if missing:
        carry.loc[dates[1], "A"] = pd.NA
    before = carry.copy(deep=True)
    with pytest.raises(TypingError, match="nopython"):
        qis.backtest_model_portfolio(
            prices,
            {"A": 1.0},
            instruments_carry=carry,
            rebalancing_freq="YE",
            is_rebalanced_at_first_date=True,
        )
    pd.testing.assert_frame_equal(carry, before)

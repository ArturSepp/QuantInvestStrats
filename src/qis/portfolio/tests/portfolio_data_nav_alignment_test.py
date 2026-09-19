"""Regression tests for date-aligned portfolio cost ratios.

NAV-normalized costs must divide each report row by the NAV carrying the same label. These tests
cover row permutations, pandas nullable values, malformed NAV/report axes, aggregate labels,
downstream cost modes, and the raw currency-cost bypass.
"""

import pandas as pd
import pytest

import qis


DATES = pd.DatetimeIndex(["2026-01-05", "2026-01-06"], tz="UTC", name="date")
TICKERS = pd.Index(["A", "B"], name="ticker")


def _portfolio(*, nullable: bool = False) -> qis.PortfolioData:
    """Return report panels in chronological order and NAV in reverse label order."""
    dtype = "Float64" if nullable else "float64"
    nav = pd.Series([200.0, 100.0], index=DATES[::-1], name="strategy", dtype=dtype)
    prices = pd.DataFrame([[10.0, 20.0], [20.0, 20.0]], index=DATES, columns=TICKERS, dtype=dtype)
    units = pd.DataFrame(1.0, index=DATES, columns=TICKERS, dtype=dtype)
    costs = pd.DataFrame([[1.0, 2.0], [2.0, 4.0]], index=DATES, columns=TICKERS, dtype=dtype)
    return qis.PortfolioData(
        nav=nav,
        prices=prices,
        units=units,
        realized_costs=costs,
        group_data=pd.Series(["risk", "defensive"], index=TICKERS),
        group_order=["risk", "defensive"],
    )


@pytest.mark.parametrize("nullable", [False, True], ids=["float64", "nullable-float64"])
def test_portfolio_data_get_costs_aligns_nav_by_date(nullable: bool) -> None:
    """A NAV row permutation preserves per-instrument values and aggregate date labels."""
    portfolio = _portfolio(nullable=nullable)
    original_nav = portfolio.nav.copy(deep=True)
    original_costs = portfolio.realized_costs.copy(deep=True)
    expected = pd.DataFrame([[0.01, 0.02], [0.01, 0.02]], index=DATES, columns=TICKERS)

    actual = portfolio.get_costs(add_total=False, roll_period=None)
    aggregate = portfolio.get_costs(is_agg=True, roll_period=None)

    pd.testing.assert_frame_equal(actual.astype(float), expected)
    pd.testing.assert_series_equal(
        aggregate.astype(float), pd.Series([0.03, 0.03], index=DATES, name="strategy")
    )
    pd.testing.assert_series_equal(portfolio.nav, original_nav)
    pd.testing.assert_frame_equal(portfolio.realized_costs, original_costs)


def test_portfolio_data_get_costs_aligns_nav_before_downstream_aggregation() -> None:
    """Total-column and grouped reports consume the same date-aligned cost frame."""
    portfolio = _portfolio()

    with_total = portfolio.get_costs(add_total=True, roll_period=None)
    grouped = portfolio.get_costs(is_grouped=True, add_total=True, roll_period=None)

    expected_total = pd.Series([0.03, 0.03], index=DATES, name="strategy")
    pd.testing.assert_series_equal(with_total["strategy"], expected_total)
    pd.testing.assert_series_equal(grouped["strategy"], expected_total)
    pd.testing.assert_series_equal(
        grouped["risk"], pd.Series([0.01, 0.01], index=DATES, name="risk")
    )
    pd.testing.assert_series_equal(
        grouped["defensive"], pd.Series([0.02, 0.02], index=DATES, name="defensive")
    )


def _replace_nav_index(portfolio: qis.PortfolioData, case: str) -> None:
    """Apply one malformed NAV-label case while retaining its literal values."""
    if case == "missing":
        portfolio.nav = portfolio.nav.drop(DATES[1])
    elif case == "extra":
        portfolio.nav.loc[pd.Timestamp("2026-01-07", tz="UTC")] = 300.0
    elif case == "different":
        portfolio.nav.index = pd.DatetimeIndex(
            [DATES[0], pd.Timestamp("2026-01-07", tz="UTC")], name=DATES.name
        )
    elif case == "duplicate":
        portfolio.nav.index = pd.DatetimeIndex([DATES[0], DATES[0]], name=DATES.name)
    else:  # pragma: no cover - parametrization owns this closed set
        raise AssertionError(f"unknown test case: {case}")


def _normalized_costs(portfolio: qis.PortfolioData) -> pd.DataFrame | pd.Series:
    """Return the normalized cost report used by malformed-axis cases."""
    return portfolio.get_costs(add_total=False, roll_period=None)


@pytest.mark.parametrize("case", ["missing", "extra", "different", "duplicate"])
def test_portfolio_data_nav_normalized_costs_reject_malformed_nav_labels(
    case: str,
) -> None:
    """Missing, extra, and duplicate NAV labels fail before financial division."""
    portfolio = _portfolio()
    _replace_nav_index(portfolio, case)
    original_nav = portfolio.nav.copy(deep=True)
    original_costs = portfolio.realized_costs.copy(deep=True)
    original_prices = portfolio.prices.copy(deep=True)
    original_units = portfolio.units.copy(deep=True)

    with pytest.raises(ValueError, match="NAV index"):
        _normalized_costs(portfolio)

    pd.testing.assert_series_equal(portfolio.nav, original_nav)
    pd.testing.assert_frame_equal(portfolio.realized_costs, original_costs)
    pd.testing.assert_frame_equal(portfolio.prices, original_prices)
    pd.testing.assert_frame_equal(portfolio.units, original_units)


def test_portfolio_data_nav_normalized_costs_reject_duplicate_report_labels() -> None:
    """A duplicated numerator label cannot be paired unambiguously with one NAV value."""
    portfolio = _portfolio()
    duplicate_dates = pd.DatetimeIndex([DATES[0], DATES[0]], name=DATES.name)

    portfolio.realized_costs.index = duplicate_dates
    original_nav = portfolio.nav.copy(deep=True)
    original_costs = portfolio.realized_costs.copy(deep=True)
    original_prices = portfolio.prices.copy(deep=True)
    original_units = portfolio.units.copy(deep=True)

    with pytest.raises(ValueError, match="report index"):
        portfolio.get_costs(add_total=False, roll_period=None)

    pd.testing.assert_series_equal(portfolio.nav, original_nav)
    pd.testing.assert_frame_equal(portfolio.realized_costs, original_costs)
    pd.testing.assert_frame_equal(portfolio.prices, original_prices)
    pd.testing.assert_frame_equal(portfolio.units, original_units)


def test_portfolio_data_get_costs_raw_currency_mode_ignores_nav_labels() -> None:
    """Raw realized costs remain available without imposing a NAV denominator contract."""
    portfolio = _portfolio()
    portfolio.nav = portfolio.nav.iloc[:1]
    expected = portfolio.realized_costs.copy(deep=True)

    actual = portfolio.get_costs(
        add_total=False,
        is_unit_based_traded_volume=False,
        roll_period=None,
    )

    pd.testing.assert_frame_equal(actual, expected)

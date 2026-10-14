"""Reconstructed instrument pnl values missing quotes at the last observed mark (#189).

``PortfolioData`` reconstructs ``instrument_pnl`` from ``prices`` and lagged ``weights``
when it is not supplied. Raw ``pct_change`` cannot cross a missing quote, so a resumed
quote used to contribute nothing and instrument contributions no longer summed to the
nav return, unlike the marks convention the backtester adopts in #189.
"""

import numpy as np
import pandas as pd

from qis.portfolio.portfolio_data import PortfolioData


def _held_portfolio(prices: pd.DataFrame, weight: float = 0.5):
    """Hold a fixed share of each instrument through the panel, valued on last marks."""
    units = pd.DataFrame(weight, index=prices.index, columns=prices.columns)
    marks = prices.ffill()
    nav = (units * marks).sum(axis=1).rename("nav")
    weights = units.multiply(marks).divide(nav, axis=0)
    return nav, units, weights


def _reconstruct(prices: pd.DataFrame, weight: float = 0.5) -> PortfolioData:
    nav, units, weights = _held_portfolio(prices, weight=weight)
    return PortfolioData(nav=nav, units=units, weights=weights, prices=prices)


def test_resumed_quote_books_full_move_on_resume_date():
    prices = pd.DataFrame(
        {"A": [100., 100., 102.], "B": [100., np.nan, 110.]},
        index=pd.bdate_range("2024-01-01", periods=3),
    )
    portfolio = _reconstruct(prices)
    # B is frozen at its 100 mark on the gap date and books 110/100 - 1 on the resume date.
    np.testing.assert_allclose(portfolio.instrument_pnl["B"], [0., 0., 0.05], atol=1e-15)
    nav_returns = portfolio.nav.pct_change(fill_method=None).fillna(0.0)
    np.testing.assert_allclose(portfolio.instrument_pnl.sum(axis=1), nav_returns, atol=1e-15)


def test_interior_gap_attributes_frozen_gap_date_and_resume_move():
    prices = pd.DataFrame(
        {"A": [100., 100., 100., 102.], "B": [100., np.nan, np.nan, 110.]},
        index=pd.bdate_range("2024-01-01", periods=4),
    )
    portfolio = _reconstruct(prices)
    np.testing.assert_allclose(portfolio.instrument_pnl["B"], [0., 0., 0., 0.05], atol=1e-15)
    nav_returns = portfolio.nav.pct_change(fill_method=None).fillna(0.0)
    np.testing.assert_allclose(portfolio.instrument_pnl.sum(axis=1), nav_returns, atol=1e-15)


def test_trailing_gap_never_writes_holding_down():
    prices = pd.DataFrame(
        {"A": [100., 100., 100.], "B": [100., 100., np.nan]},
        index=pd.bdate_range("2024-01-01", periods=3),
    )
    portfolio = _reconstruct(prices)
    np.testing.assert_allclose(portfolio.instrument_pnl["B"], 0.0, atol=1e-15)


def test_never_quoted_instrument_stays_flat():
    prices = pd.DataFrame(
        {"A": [100., 101., 102.], "B": [np.nan, np.nan, 110.]},
        index=pd.bdate_range("2024-01-01", periods=3),
    )
    portfolio = _reconstruct(prices)
    np.testing.assert_allclose(portfolio.instrument_pnl["B"], 0.0, atol=1e-15)


def test_complete_prices_reconstruction_is_unchanged():
    prices = pd.DataFrame(
        {"A": [100., 101., 99., 100.5], "B": [50., 52., 51., 53.]},
        index=pd.bdate_range("2024-01-01", periods=4),
    )
    portfolio = _reconstruct(prices)
    weights = _held_portfolio(prices)[2]
    legacy = prices.pct_change(fill_method=None).multiply(weights.shift(1)).fillna(0.0)
    pd.testing.assert_frame_equal(portfolio.instrument_pnl, legacy)

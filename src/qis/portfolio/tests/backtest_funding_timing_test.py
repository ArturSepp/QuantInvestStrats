"""The backtest cash leg accrues the funding rate known at the start of each period.

A cash balance held over ``(t-1, t]`` earns the annual rate known at ``t-1`` times the ACT/365
fraction of the period, the convention of ``qis.compute_excess_returns``. A portfolio with zero
weights is pure cash, so its NAV return must equal the cash return that the excess-return
helper subtracts. Expected values are hand arithmetic: 36.5% a year is 0.1% a calendar day.
"""

import numpy as np
import pandas as pd
import pytest

import qis


_DATES = pd.date_range('2024-01-01', periods=4, freq='D')


def _cash_only_nav(funding_rate: pd.Series) -> pd.Series:
    """NAV of a zero-weight portfolio that holds only its initial cash."""
    prices = pd.DataFrame({'asset': 100.0}, index=_DATES)
    portfolio = qis.backtest_model_portfolio(prices=prices, weights={'asset': 0.0},
                                             funding_rate=funding_rate, initial_nav=100.0,
                                             is_rebalanced_at_first_date=True)
    return portfolio.get_portfolio_nav()


def test_cash_accrues_the_rate_known_at_the_start_of_the_period() -> None:
    """A rising daily rate is applied one day later than it is quoted."""
    funding = pd.Series([0.365, 0.73, 1.095, 1.46], index=_DATES)

    nav = _cash_only_nav(funding)

    expected = 100.0 * np.cumprod([1.0, 1.001, 1.002, 1.003])
    np.testing.assert_allclose(nav.to_numpy(), expected, rtol=1e-14)


def test_cash_leg_matches_the_excess_return_convention() -> None:
    """The cash-only NAV return is the cash return subtracted by compute_excess_returns."""
    funding = pd.Series([0.02, 0.05, 0.03, 0.08], index=_DATES)

    nav = _cash_only_nav(funding)
    cash_returns = -qis.compute_excess_returns(returns=pd.Series(0.0, index=_DATES),
                                               rates_data=funding)

    np.testing.assert_allclose(nav.pct_change().to_numpy()[1:], cash_returns.to_numpy()[1:],
                               rtol=0.0, atol=1e-15)


def test_funding_that_starts_late_is_reported() -> None:
    """A period with no known funding rate leaves the NAV missing and warns."""
    funding = pd.Series(0.365, index=_DATES[1:])

    with pytest.warns(UserWarning, match='funding_rate'):
        nav = _cash_only_nav(funding)

    assert nav.iloc[0] == 100.0 and nav.iloc[1:].isna().all()

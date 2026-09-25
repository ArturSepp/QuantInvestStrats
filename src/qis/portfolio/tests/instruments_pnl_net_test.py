"""Net instrument P&L reconciles to the NAV return period by period.

With no management fee, funding or carry, a held-unit backtest changes NAV only through price
moves on the units held over the period and through the trading cost paid at its end. The
arithmetic instrument contributions net of each period's cost divided by the preceding NAV must
therefore sum to the simple NAV return of that period. The check is independent of the P&L
code: it compares against the NAV path the backtester records.
"""

# packages
import numpy as np
import pandas as pd

# qis
import qis


def _portfolio(periods: int = 300) -> qis.PortfolioData:
    """Monthly-rebalanced two-asset backtest with costs over more than one rolling year.

    Args:
        periods: number of business-day price observations

    Returns:
        the backtested portfolio
    """
    dates = pd.bdate_range('2020-01-01', periods=periods)
    steps = np.arange(periods)
    prices = pd.DataFrame({'Asset A': 100.0 * np.exp(0.0004 * steps + 0.02 * np.sin(steps / 7.0)),
                           'Asset B': 100.0 * np.exp(0.0001 * steps + 0.01 * np.cos(steps / 5.0))},
                          index=dates)
    return qis.backtest_model_portfolio(prices=prices, weights=np.array([0.6, 0.4]),
                                        rebalancing_freq='ME', rebalancing_costs=0.002,
                                        initial_nav=100.0, ticker='net pnl test')


def test_net_instrument_pnl_reconciles_to_nav_returns() -> None:
    """Every period's summed net contributions equal that period's NAV return."""
    portfolio = _portfolio()
    net = portfolio.get_instruments_pnl(is_net=True)
    nav_returns = portfolio.get_portfolio_nav().pct_change()
    assert portfolio.realized_costs.iloc[1:].to_numpy().sum() > 0.0, 'the test needs later trades'
    np.testing.assert_allclose(net.sum(axis=1).iloc[1:], nav_returns.iloc[1:], rtol=0.0,
                               atol=1e-12)


def test_net_minus_gross_is_the_period_cost_on_preceding_nav() -> None:
    """The net adjustment is each period's own cost, not a trailing sum of costs."""
    portfolio = _portfolio()
    gross = portfolio.get_instruments_pnl(is_net=False)
    net = portfolio.get_instruments_pnl(is_net=True)
    expected = portfolio.realized_costs.divide(portfolio.get_portfolio_nav().shift(1), axis=0)
    np.testing.assert_allclose((gross - net).iloc[1:], expected.iloc[1:], rtol=0.0, atol=1e-15)
    # the opening trade is paid out of the baseline NAV rather than attributed to a return
    np.testing.assert_allclose(net.iloc[0], gross.iloc[0], rtol=0.0, atol=0.0)

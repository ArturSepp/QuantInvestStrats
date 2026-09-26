"""``PortfolioData.compute_portfolio_vol`` compares like with like.

Both columns are EWM volatilities of simple returns on the same grid: the instrument-weighted
column contracts the lagged weights with a point-in-time EWM covariance of simple instrument
returns, and the strategy column is the EWM volatility of the simple NAV return, which is the
weighted sum of those instrument returns. Log NAV returns would mix return conventions.
"""

# packages
import numpy as np
import pandas as pd

# qis
import qis


def _portfolio() -> qis.PortfolioData:
    """A monthly-rebalanced three-asset backtest over two years of business days.

    Returns:
        The backtested portfolio.
    """
    dates = pd.bdate_range('2021-01-01', periods=520)
    rng = np.random.default_rng(3)
    returns = pd.DataFrame(rng.normal(0.0003, 0.012, size=(len(dates), 3)), index=dates,
                           columns=['Asset A', 'Asset B', 'Asset C'])
    returns.iloc[0, :] = 0.0
    prices = 100.0 * (1.0 + returns).cumprod()
    return qis.backtest_model_portfolio(prices=prices, weights=np.array([0.5, 0.3, 0.2]),
                                        rebalancing_freq='ME', ticker='vol basis')


def test_strategy_vol_uses_simple_nav_returns() -> None:
    """The strategy column is the EWM volatility of simple NAV returns on the grid."""
    portfolio = _portfolio()
    actual = portfolio.compute_portfolio_vol(freq='W-WED', span=13)

    nav_returns = qis.to_returns(portfolio.get_portfolio_nav(freq='W-WED'),
                                 is_log_returns=False)
    expected = qis.compute_ewm_vol(data=nav_returns, span=13, annualize=True)
    pd.testing.assert_series_equal(actual['strategy returns vol'],
                                   expected.reindex(index=actual.index),
                                   check_names=False, rtol=1e-12)


def test_instrument_vol_is_point_in_time_on_simple_returns() -> None:
    """The instrument column is sqrt(52 w_{t-1}' S_t w_{t-1}) from a zero-seeded recursion."""
    portfolio = _portfolio()
    actual = portfolio.compute_portfolio_vol(freq='W-WED', span=13)['instrument weighted vol']

    returns = portfolio.get_instruments_periodic_returns(freq='W-WED')
    weights = portfolio.weights.reindex(index=returns.index, method='ffill')
    r = returns.fillna(0.0).to_numpy()
    w = weights.shift(1).fillna(0.0).to_numpy()
    ewm_lambda = 1.0 - 2.0 / 14.0
    state, path = np.zeros((3, 3)), []
    for row, x in zip(r, w):
        state = (1.0 - ewm_lambda) * np.outer(row, row) + ewm_lambda * state
        path.append(np.sqrt(52.0 * x @ state @ x))
    np.testing.assert_allclose(actual.reindex(index=returns.index).to_numpy(), np.array(path),
                               rtol=1e-12, atol=0.0)

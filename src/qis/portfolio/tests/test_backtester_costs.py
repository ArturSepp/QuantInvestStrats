"""
rebalancing costs as a panel of dates x tickers.

The scalar and per-instrument forms are broadcast to the panel form, so the three must agree
exactly where they overlap; an era schedule must switch rates at its boundary and nowhere
else; and the two silent misreads — a date-indexed Series taken as per-instrument, and a
schedule row lost because its boundary falls on a non-trading day — must raise or be handled,
not swallowed. The motivating consumer is the ``trendfollowing`` volume-cost panel, a
per-era, per-asset-class schedule aligned with prices.
"""
# packages
import numpy as np
import pandas as pd
import pytest
# qis / project
import qis

WEIGHTS = {'A': 0.5, 'B': 0.3, 'C': 0.2}
COST = 0.0010  # fractional units: 10 bp


def _panel() -> pd.DataFrame:
    """seeded three-instrument business-day price panel with no nans."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range('2020-01-01', '2023-12-29')
    returns = 0.0002 + 0.01 * rng.standard_normal((len(dates), 3))
    prices = 100.0 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=['A', 'B', 'C'])


def _run(prices: pd.DataFrame, rebalancing_costs) -> qis.PortfolioData:
    """the backtest under test, quarterly rebalancing, everything else default."""
    return qis.backtest_model_portfolio(prices=prices,
                                        weights=WEIGHTS,
                                        rebalancing_freq='QE',
                                        rebalancing_costs=rebalancing_costs,
                                        ticker='S')


def test_scalar_series_and_constant_frame_agree() -> None:
    """the three input forms produce bit-identical navs when they state the same cost."""
    prices = _panel()
    nav_float = _run(prices, COST).get_portfolio_nav()
    nav_series = _run(prices, pd.Series(COST, index=prices.columns)).get_portfolio_nav()
    frame = pd.DataFrame(COST, index=prices.index, columns=prices.columns)
    nav_frame = _run(prices, frame).get_portfolio_nav()
    np.testing.assert_array_equal(nav_float.to_numpy(), nav_series.to_numpy())
    np.testing.assert_array_equal(nav_float.to_numpy(), nav_frame.to_numpy())


def test_era_schedule_applies_at_the_boundary() -> None:
    """
    before the boundary the era run equals the constant run; at the first rebalancing under
    the new rate the trade is identical, so realised costs scale by exactly the rate ratio.
    """
    prices = _panel()
    boundary = pd.Timestamp('2022-01-01')
    era_costs = pd.DataFrame(COST, index=prices.index, columns=prices.columns)
    era_costs.loc[boundary:, :] = 5.0 * COST
    p_frame = _run(prices, era_costs)
    p_const = _run(prices, COST)

    rebalancing_dates = p_frame.realized_costs.index[p_frame.realized_costs.abs().sum(axis=1) > 0]
    post = [d for d in rebalancing_dates if d >= boundary]
    assert len(post) > 2, 'the panel must span several rebalancings after the boundary'
    first_post = post[0]

    nav_frame = p_frame.get_portfolio_nav()
    nav_const = p_const.get_portfolio_nav()
    before = nav_frame.index < first_post
    np.testing.assert_array_equal(nav_frame[before].to_numpy(), nav_const[before].to_numpy())
    np.testing.assert_allclose(p_frame.realized_costs.loc[first_post].to_numpy(),
                               5.0 * p_const.realized_costs.loc[first_post].to_numpy(),
                               rtol=1.0e-12)
    assert float(nav_frame.iloc[-1]) < float(nav_const.iloc[-1])


def test_schedule_stated_on_era_boundaries_forward_fills() -> None:
    """
    a two-row schedule whose boundary falls on a non-trading day equals the dense panel:
    each price date takes the last schedule row at or before it.
    """
    prices = _panel()
    boundary = pd.Timestamp('2022-01-01')  # a Saturday, deliberately not in the price index
    sparse = pd.DataFrame([[COST] * 3, [5.0 * COST] * 3],
                          index=[prices.index[0], boundary],
                          columns=prices.columns)
    dense = pd.DataFrame(COST, index=prices.index, columns=prices.columns)
    dense.loc[boundary:, :] = 5.0 * COST
    np.testing.assert_array_equal(_run(prices, sparse).get_portfolio_nav().to_numpy(),
                                  _run(prices, dense).get_portfolio_nav().to_numpy())


def test_date_indexed_series_raises() -> None:
    """a Series indexed by dates is ambiguous and must raise, not be misread by ticker."""
    prices = _panel()
    with pytest.raises(ValueError, match='date-indexed'):
        _run(prices, pd.Series(COST, index=prices.index))


def test_frame_missing_a_price_column_raises() -> None:
    """a cost panel that does not cover every price column must raise with the culprits."""
    prices = _panel()
    with pytest.raises(ValueError, match='missing price columns'):
        _run(prices, pd.DataFrame(COST, index=prices.index, columns=['A', 'B']))

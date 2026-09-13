"""Brinson inputs retain the opening report month and realised trading costs."""
import numpy as np
import pandas as pd
import pytest
import qis


@pytest.mark.parametrize('cost', [0.0, .002])
def test_brinson_inputs_reconcile_to_monthly_nav_returns(cost):
    """Instrument contributions equal fee-free NAV returns, including rebalance costs."""
    dates = pd.date_range('2020-12-31', periods=8, freq='ME')
    prices = pd.DataFrame({
        'A': [100, 105, 103, 110, 114, 111, 120, 118],
        'B': [100, 101, 100, 102, 103, 104, 105, 106]}, index=dates, dtype=float)
    weights = pd.DataFrame({
        'A': [.6, .8, .3, .7, .5, .9, .4, .6],
        'B': [.4, .2, .7, .3, .5, .1, .6, .4]}, index=dates)
    portfolio = qis.backtest_model_portfolio(
        prices, weights, management_fee=0.0, rebalancing_costs=cost)
    period = qis.TimePeriod('2021-01-01', '2021-07-31')
    pnl, applied_weights = portfolio.get_brinson_inputs(period, freq='ME', is_net=True)
    expected = portfolio.nav.pct_change().loc['2021-01-31':]
    assert pnl.index[0] == pd.Timestamp('2021-01-31')
    assert np.isfinite(pnl.to_numpy()).all()
    np.testing.assert_allclose(pnl.sum(axis=1), expected, atol=1e-14)
    np.testing.assert_allclose(applied_weights.iloc[0], portfolio.weights.iloc[0], atol=1e-14)
    gross, _ = portfolio.get_brinson_inputs(period, freq='ME', is_net=False)
    costs = portfolio.realized_costs.div(portfolio.nav.shift(1), axis=0).loc[pnl.index]
    np.testing.assert_allclose(gross - pnl, costs, atol=1e-14)

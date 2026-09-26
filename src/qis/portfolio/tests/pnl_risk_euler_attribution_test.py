"""The P&L risk attribution is the ex-post Euler split of the portfolio P&L variance.

With x_{i,t} the arithmetic P&L contribution of instrument i and x_{p,t} = sum_i x_{i,t}, the
share of instrument i is Cov(x_i, x_p) / Var(x_p). The shares sum to one because the sample
covariance is bilinear, and a hedge that lowered the realised volatility has a negative share,
which the former standalone volatility shares could not show.
"""

# packages
import numpy as np
import pandas as pd
import pytest

# qis
import qis


def _hedged_portfolio() -> qis.PortfolioData:
    """A three-asset backtest in which one asset moves against the other two.

    Returns:
        The backtested portfolio.
    """
    dates = pd.bdate_range('2021-01-01', periods=300)
    rng = np.random.default_rng(11)
    market = rng.normal(0.0, 0.01, size=len(dates))
    returns = pd.DataFrame({'Equity': market + rng.normal(0.0, 0.004, size=len(dates)),
                            'Credit': 0.5 * market + rng.normal(0.0, 0.003, size=len(dates)),
                            'Hedge': -0.8 * market + rng.normal(0.0, 0.004, size=len(dates))},
                           index=dates)
    returns.iloc[0, :] = 0.0
    prices = 100.0 * (1.0 + returns).cumprod()
    return qis.backtest_model_portfolio(prices=prices, weights=np.array([0.5, 0.3, 0.2]),
                                        rebalancing_freq='ME', ticker='hedged')


def test_pnl_risk_shares_are_euler_contributions() -> None:
    """Shares equal the covariance of each P&L with the total over the total's variance."""
    portfolio = _hedged_portfolio()
    actual = portfolio.get_instruments_pnl_risk_attribution()

    pnl = portfolio.get_instruments_pnl().fillna(0.0)
    total = pnl.sum(axis=1).to_numpy()
    expected = np.array([np.cov(pnl[column].to_numpy(), total)[0, 1] for column in pnl.columns])
    expected = expected / np.var(total, ddof=1)
    np.testing.assert_allclose(actual.to_numpy(), expected, rtol=1e-12, atol=1e-15)
    assert actual.sum() == pytest.approx(1.0, abs=1e-12)
    assert actual['Hedge'] < 0.0 < actual['Equity']
    assert actual.name == portfolio.nav.name

    # the factsheet panel reads the same numbers
    panel = portfolio.get_performance_attribution_data(
        attribution_metric=qis.AttributionMetric.PNL_RISK)
    pd.testing.assert_series_equal(panel, actual)

    # the shares times the P&L volatility are contributions summing to that volatility
    contributions = actual * np.std(total, ddof=1)
    assert contributions.sum() == pytest.approx(np.std(total, ddof=1), rel=1e-12)


def test_standalone_shares_remain_available() -> None:
    """``is_standalone=True`` returns the former non-negative standalone volatility shares."""
    portfolio = _hedged_portfolio()
    standalone = portfolio.get_instruments_pnl_risk_attribution(is_standalone=True)

    nonzero = portfolio.get_instruments_pnl().replace({0.0: np.nan})
    risk = nonzero.std(ddof=0)
    np.testing.assert_allclose(standalone.to_numpy(), (risk / risk.sum()).to_numpy(),
                               rtol=1e-12, atol=0.0)
    assert (standalone > 0.0).all()


def test_zero_pnl_variance_gives_nan_shares(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no portfolio P&L variance there is no risk to attribute.

    Args:
        monkeypatch: Pytest fixture used to provide a flat P&L.
    """
    portfolio = _hedged_portfolio()
    flat = pd.DataFrame(0.0, index=pd.RangeIndex(5), columns=['Equity', 'Credit', 'Hedge'])
    monkeypatch.setattr(portfolio, 'get_instruments_pnl', lambda time_period=None: flat)
    actual = portfolio.get_instruments_pnl_risk_attribution()
    assert actual.isna().all()

"""Covariance-implied volatility and contributions select weights as of each covariance date.

With ``freq=None`` the two ``PortfolioData`` methods evaluate sqrt(w' S w) and the Euler
contributions on each covariance date with the latest input weights dated at or before it, the
policy of ``RiskModel``. Weights are rarely dated exactly on covariance dates, so an exact-date
match would silently report zero risk.
"""

# packages
import numpy as np
import pandas as pd
import pytest

# qis
import qis


ASSETS = ['Asset A', 'Asset B', 'Asset C']


def _portfolio_and_covars() -> tuple:
    """A backtest rebalanced on its own dates and covariances dated in between.

    Returns:
        The portfolio, its dated input weights and a covariance dictionary.
    """
    dates = pd.bdate_range('2021-01-01', periods=260)
    steps = np.arange(len(dates))
    log_prices = {'Asset A': 0.0004 * steps + 0.02 * np.sin(steps / 7.0),
                  'Asset B': 0.0001 * steps + 0.01 * np.cos(steps / 5.0),
                  'Asset C': -0.0002 * steps + 0.03 * np.sin(steps / 11.0)}
    prices = 100.0 * np.exp(pd.DataFrame(log_prices, index=dates))
    weight_dates = dates[[0, 60, 120, 180]]
    weights = pd.DataFrame([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.6, 0.1, 0.3], [0.3, 0.3, 0.4]],
                           index=weight_dates, columns=ASSETS)
    portfolio = qis.backtest_model_portfolio(prices=prices, weights=weights, ticker='as-of test')

    vols = np.array([0.20, 0.10, 0.15])
    corr = np.array([[1.0, 0.3, -0.2], [0.3, 1.0, 0.1], [-0.2, 0.1, 1.0]])
    covar_dates = dates[[30, 60, 95, 150, 250]]  # one of them coincides with a weight date
    covar_dict = {date: pd.DataFrame(np.outer(vols, vols) * corr * (1.0 + 0.1 * k),
                                     index=ASSETS, columns=ASSETS)
                  for k, date in enumerate(covar_dates)}
    return portfolio, weights, covar_dict


def _asof_weights(weights: pd.DataFrame, date: pd.Timestamp) -> np.ndarray:
    """Latest weights dated at or before ``date``.

    Args:
        weights: dated weights.
        date: evaluation date.

    Returns:
        The weight vector as an array.
    """
    return weights.loc[weights.index <= date].iloc[-1].to_numpy()


def test_ex_ante_vol_uses_asof_weights() -> None:
    """Each covariance date uses the last rebalancing weights before it, never zero."""
    portfolio, weights, covar_dict = _portfolio_and_covars()
    actual = portfolio.compute_ex_anti_portfolio_vol_implied_by_covar(covar_dict=covar_dict)

    expected = {}
    for date, covar in covar_dict.items():
        w = _asof_weights(weights, date)
        expected[date] = float(np.sqrt(w @ covar.to_numpy() @ w))
    expected = pd.Series(expected)
    assert actual.index.equals(expected.index)
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), rtol=1e-14, atol=0.0)
    assert (actual > 0.0).all()


@pytest.mark.parametrize('normalise', [False, True])
def test_risk_contributions_use_asof_weights(normalise: bool) -> None:
    """Contributions are the Euler split of the as-of weights and add up to the volatility.

    Args:
        normalise: whether the rows are rescaled to sum to one.
    """
    portfolio, weights, covar_dict = _portfolio_and_covars()
    actual = portfolio.compute_risk_contributions_implied_by_covar(covar_dict=covar_dict,
                                                                   normalise=normalise)

    for date, covar in covar_dict.items():
        w = _asof_weights(weights, date)
        sigma_w = covar.to_numpy() @ w
        vol = np.sqrt(w @ sigma_w)
        expected = w * sigma_w / vol
        if normalise:
            expected = expected / vol
        np.testing.assert_allclose(actual.loc[date, ASSETS].to_numpy(), expected,
                                   rtol=1e-13, atol=0.0)

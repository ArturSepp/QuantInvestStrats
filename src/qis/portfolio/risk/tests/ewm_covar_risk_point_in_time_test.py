"""Point-in-time EWM portfolio variance and consistent correlated and undiversified VaR.

``compute_portfolio_var_np`` runs S_t = (1 - lambda) r_t r_t' + lambda S_{t-1}. Its seed must
not use observations after t (a full-sample seed is look-ahead), and the requested decay must
apply to every step. The correlated and the undiversified VaR must pair the same weights with
the same covariance, so that sum_i |w_i| sigma_i >= sqrt(w' S w) holds on every date.
"""

import numpy as np
import pandas as pd
import pytest

import qis
from qis.datasets import generate_synthetic_prices
from qis.portfolio.risk.ewm_covar_risk import VAR99


def _returns_and_weights(n_rows: int = 120,
                         n_assets: int = 3,
                         seed: int = 7) -> tuple:
    """Fixed-seed returns and slowly varying weights.

    Args:
        n_rows: number of observations.
        n_assets: number of assets.
        seed: seed of the generator.

    Returns:
        Returns and weights as two (n_rows, n_assets) arrays.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.01, size=(n_rows, n_assets))
    weights = 0.2 + 0.1 * rng.random(size=(n_rows, n_assets))
    return returns, weights


def _reference_path(returns: np.ndarray,
                    weights: np.ndarray,
                    ewm_lambda: float,
                    seed_covar: np.ndarray = None) -> np.ndarray:
    """Independent loop: w_t' S_t w_t with S_t seeded at ``seed_covar`` (zeros by default).

    Args:
        returns: (T, N) returns.
        weights: (T, N) weights paired with the covariance on the same row.
        ewm_lambda: decay of the recursion.
        seed_covar: covariance before the first row.

    Returns:
        Portfolio variance path.
    """
    state = np.zeros((returns.shape[1], returns.shape[1])) if seed_covar is None else seed_covar
    path = []
    for row, w in zip(returns, weights):
        state = (1.0 - ewm_lambda) * np.outer(row, row) + ewm_lambda * state
        path.append(w @ state @ w)
    return np.array(path)


def test_portfolio_var_np_does_not_use_future_observations() -> None:
    """The estimate on each row is unchanged when later rows are removed."""
    returns, weights = _returns_and_weights()
    full = qis.compute_portfolio_var_np(returns=returns, weights=weights, span=33)
    for n_rows in (1, 10, 60):
        truncated = qis.compute_portfolio_var_np(returns=returns[:n_rows],
                                                 weights=weights[:n_rows], span=33)
        np.testing.assert_allclose(truncated, full[:n_rows], rtol=1e-13, atol=0.0)


@pytest.mark.parametrize('kwargs, ewm_lambda', [
    (dict(span=33), 1.0 - 2.0 / 34.0),
    (dict(ewm_lambda=0.9), 0.9),
    (dict(), 0.94),
])
def test_portfolio_var_np_uses_the_requested_decay_from_a_zero_seed(kwargs: dict,
                                                                    ewm_lambda: float) -> None:
    """The recursion starts from zero and every step uses the requested decay.

    Args:
        kwargs: decay arguments passed to the function.
        ewm_lambda: the decay they imply.
    """
    returns, weights = _returns_and_weights()
    actual = qis.compute_portfolio_var_np(returns=returns, weights=weights, **kwargs)
    np.testing.assert_allclose(actual, _reference_path(returns, weights, ewm_lambda),
                               rtol=1e-12, atol=0.0)


def test_portfolio_var_np_accepts_an_explicit_seed() -> None:
    """``covar0`` seeds the recursion; its weight on row t is lambda^(t+1)."""
    returns, weights = _returns_and_weights()
    seed_covar = np.diag([4e-4, 1e-4, 9e-4])
    actual = qis.compute_portfolio_var_np(returns=returns, weights=weights, ewm_lambda=0.9,
                                          covar0=seed_covar)
    np.testing.assert_allclose(actual, _reference_path(returns, weights, 0.9, seed_covar),
                               rtol=1e-12, atol=0.0)


def test_compute_portfolio_vol_lags_weights_and_honours_span() -> None:
    """The pandas wrapper pairs w_{t-1} with S_t by default and w_t with ``weight_lag=0``."""
    returns, weights = _returns_and_weights()
    index = pd.bdate_range('2024-01-01', periods=returns.shape[0])
    returns_df = pd.DataFrame(returns, index=index, columns=list('abc'))
    weights_df = pd.DataFrame(weights, index=index, columns=list('abc'))
    ewm_lambda = 1.0 - 2.0 / 34.0

    lagged = qis.compute_portfolio_vol(returns=returns_df, weights=weights_df, span=33,
                                       is_return_vol=False)
    lagged_weights = np.vstack([np.zeros((1, 3)), weights[:-1]])
    np.testing.assert_allclose(lagged.to_numpy(),
                               _reference_path(returns, lagged_weights, ewm_lambda),
                               rtol=1e-12, atol=0.0)

    same_date = qis.compute_portfolio_vol(returns=returns_df, weights=weights_df, span=33,
                                          is_return_vol=False, weight_lag=0)
    np.testing.assert_allclose(same_date.to_numpy(),
                               _reference_path(returns, weights, ewm_lambda),
                               rtol=1e-12, atol=0.0)

    with pytest.raises(ValueError, match='weight_lag'):
        qis.compute_portfolio_vol(returns=returns_df, weights=weights_df, weight_lag=-1)


def _synthetic_prices_and_switching_weights() -> tuple:
    """Three synthetic instruments with a downward weight switch in mid-sample.

    Returns:
        Prices and business-day weights on the price index.
    """
    tickers = ['SEQ_US', 'SBD_TSY', 'SCM_GLD']
    prices = generate_synthetic_prices(start='2022-01-03', end='2023-06-30',
                                       apply_quirks=False)[tickers]
    weights = pd.DataFrame(np.tile([0.5, 0.3, 0.2], (len(prices.index), 1)),
                           index=prices.index, columns=tickers)
    weights.loc['2022-09-01':, :] = [0.1, 0.1, 0.1]
    return prices, weights


def test_undiversified_var_bounds_correlated_var_on_every_date() -> None:
    """Warm-up days and the weight switch included, sum |w| sigma >= sqrt(w' S w)."""
    prices, weights = _synthetic_prices_and_switching_weights()
    groups = pd.Series(['Risky', 'Defensive', 'Defensive'], index=prices.columns)

    correlated = qis.compute_portfolio_correlated_var_by_groups(
        prices=prices, weights=weights, vol_span=33)['Total VAR']
    _, undiversified = qis.compute_portfolio_independent_var_by_ac(
        prices=prices, weights=weights, vol_span=33)
    assert correlated.index.equals(undiversified.index)
    assert (undiversified - correlated).min() >= -1e-15

    correlated_groups = qis.compute_portfolio_correlated_var_by_groups(
        prices=prices, weights=weights, group_data=groups, vol_span=33)
    _, undiversified_groups = qis.compute_portfolio_independent_var_by_ac(
        prices=prices, weights=weights, group_data=groups, vol_span=33)
    for column in ['Total', 'Risky', 'Defensive']:
        gap = undiversified_groups[column] - correlated_groups[column]
        assert gap.min() >= -1e-15, column


def test_var_functions_share_same_date_weights_and_covariance() -> None:
    """Both VaR figures on date t use w_t and the zero-seeded EWM covariance through t."""
    prices, weights = _synthetic_prices_and_switching_weights()
    returns = qis.to_returns(prices=prices, freq='B', is_log_returns=True)
    ewm_lambda = 1.0 - 2.0 / 34.0

    state, covars = np.zeros((3, 3)), []
    for row in np.nan_to_num(returns.to_numpy(), nan=0.0):
        state = (1.0 - ewm_lambda) * np.outer(row, row) + ewm_lambda * state
        covars.append(state)
    w = weights.reindex(index=returns.index).to_numpy()
    expected_correlated = np.array([VAR99 * np.sqrt(x @ s @ x) for x, s in zip(w, covars)])
    expected_undiversified = np.array([VAR99 * np.abs(x) @ np.sqrt(np.diag(s))
                                       for x, s in zip(w, covars)])

    correlated = qis.compute_portfolio_correlated_var_by_groups(
        prices=prices, weights=weights, vol_span=33)['Total VAR']
    instrument_var, undiversified = qis.compute_portfolio_independent_var_by_ac(
        prices=prices, weights=weights, vol_span=33)
    np.testing.assert_allclose(correlated.to_numpy(), expected_correlated, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(undiversified.to_numpy(), expected_undiversified,
                               rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(instrument_var.sum(axis=1).to_numpy(), expected_undiversified,
                               rtol=1e-12, atol=1e-18)

    switch = returns.index.get_loc(pd.Timestamp('2022-09-01'))
    np.testing.assert_allclose(w[switch], [0.1, 0.1, 0.1])
    assert correlated.iloc[switch] == pytest.approx(expected_correlated[switch], rel=1e-12)


def test_var_limit_annualises_like_business_day_returns() -> None:
    """The default converts annual volatility to one day with AN=252, qis's 'B' factor."""
    weights = np.array([0.5, 0.3, 0.2])
    vols = np.array([0.20, 0.10, 0.15])
    an = qis.get_annualization_factor('B')
    capped = qis.limit_weights_to_max_var_limit(weights=weights, vols=vols,
                                                max_var_limit_bp=100.0)
    np.testing.assert_allclose(capped, [100.0 * np.sqrt(an) / (VAR99 * 1e4 * 0.20), 0.3, 0.2],
                               rtol=1e-12)
    np.testing.assert_allclose(
        capped, qis.limit_weights_to_max_var_limit(weights=weights, vols=vols,
                                                   max_var_limit_bp=100.0,
                                                   annualization_factor=252.0),
        rtol=0.0, atol=0.0)

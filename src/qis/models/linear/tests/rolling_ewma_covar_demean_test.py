"""estimate_rolling_ewma_covar: unbiased demeaning and the time_period window."""

# packages
import numpy as np
import pandas as pd
import pytest

# qis
import qis.utils.dates as da
from qis.models.linear.corr_cov_matrix import estimate_rolling_ewma_covar


def _iid_weekly_prices(num_weeks: int, seed: int = 7) -> pd.DataFrame:
    """Two assets with iid normal weekly log returns and a non-zero drift."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range('1900-01-03', periods=num_weeks + 1, freq='W-WED')
    log_returns = np.vstack([np.zeros(2),
                             [0.002, 0.001] + rng.standard_normal((num_weeks, 2)) * [0.03, 0.01]])
    return pd.DataFrame(100.0 * np.exp(np.cumsum(log_returns, axis=0)), index=dates,
                        columns=['a', 'b'])


def test_demeaned_matrix_is_scaled_prior_mean_recursion() -> None:
    """The matrix equals N/(N+1) times the EWM of (x_t - m_{t-1})(x_t - m_{t-1})'."""
    prices = _iid_weekly_prices(num_weeks=300)
    span = 52
    covars = estimate_rolling_ewma_covar(prices=prices, returns_freq='W-WED',
                                         rebalancing_freq='QE', span=span,
                                         apply_an_factor=False)
    returns = np.diff(np.log(prices.to_numpy()), axis=0)
    lam = 1.0 - 2.0 / (span + 1.0)
    # the mean starts from a zero prior, so the first residual is the first return itself; with
    # the former seed at the first return it was exactly zero, which gave the vol-normalised
    # estimator a zero first volatility and a NaN matrix
    prior_mean = np.zeros(2)
    state = np.zeros((2, 2))
    path = []
    for x_t in returns:
        e_t = x_t - prior_mean
        state = lam * state + (1.0 - lam) * np.outer(e_t, e_t)
        path.append(0.5 * (1.0 + lam) * state)
        prior_mean = lam * prior_mean + (1.0 - lam) * x_t
    index = prices.index[1:]
    for date, covar in covars.items():
        np.testing.assert_allclose(covar.to_numpy(), path[index.get_loc(date)], rtol=1e-10)


def test_demeaned_estimator_is_unbiased_for_iid_returns() -> None:
    """Time-averaged over a long iid sample, the variance matches the sample variance."""
    prices = _iid_weekly_prices(num_weeks=5200)
    covars = estimate_rolling_ewma_covar(prices=prices, returns_freq='W-WED',
                                         rebalancing_freq='W-WED', span=52,
                                         apply_an_factor=False)
    returns = np.diff(np.log(prices.to_numpy()), axis=0)
    dates = list(covars)[260:]  # skip five spans of warm-up from the zero seed
    average = np.mean([covars[date].to_numpy() for date in dates], axis=0)
    ratio = np.diag(average) / np.var(returns[260:], axis=0, ddof=1)
    # the former demeaning with m_t scaled the covariance by 2 lambda^2 / (1 + lambda) = 0.944
    np.testing.assert_allclose(ratio, 1.0, atol=0.01)


def test_time_period_restricts_both_ends() -> None:
    """Only rebalancing dates inside [start, end] are returned."""
    prices = _iid_weekly_prices(num_weeks=300)
    full = estimate_rolling_ewma_covar(prices=prices, rebalancing_freq='QE')
    window = da.TimePeriod('1901-01-01', '1902-06-30')
    covars = estimate_rolling_ewma_covar(prices=prices, rebalancing_freq='QE',
                                         time_period=window)
    expected = [d for d in full if window.start <= d <= window.end]
    assert list(covars) == expected and len(expected) == 6
    for date in expected:
        pd.testing.assert_frame_equal(covars[date], full[date])


def test_demean_false_is_second_moment_about_zero() -> None:
    """demean=False keeps the uncentred recursion, annualised by 52 on a weekly grid."""
    prices = _iid_weekly_prices(num_weeks=120)
    covars = estimate_rolling_ewma_covar(prices=prices, rebalancing_freq='QE', demean=False)
    returns = np.diff(np.log(prices.to_numpy()), axis=0)
    lam = 1.0 - 2.0 / 53.0
    state = np.zeros((2, 2))
    path = []
    for x_t in returns:
        state = lam * state + (1.0 - lam) * np.outer(x_t, x_t)
        path.append(state)
    index = prices.index[1:]
    assert len(covars) > 0
    for date, covar in covars.items():
        assert covar.to_numpy() == pytest.approx(52.0 * path[index.get_loc(date)], rel=1e-10)

"""ewm_xy_convolution runs at every horizon and is point in time."""

# packages
import numpy as np
import pandas as pd
import pytest

# qis
from qis.models.linear.ewm_convolution import ConvolutionType, ewm_xy_convolution


def _daily_returns(num_rows: int = 700, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2012-01-02', periods=num_rows)
    return pd.DataFrame(0.01 * rng.standard_normal((num_rows, 2)), index=dates,
                        columns=['a', 'b'])


def _zero_seeded_ewm(values: np.ndarray, ewm_lambda: float) -> np.ndarray:
    """EWM from a zero seed, started at the first finite row, NaN before it."""
    out = np.full_like(values, np.nan)
    state = None
    for t, value in enumerate(values):
        if state is None:
            if np.isfinite(value):
                state = (1.0 - ewm_lambda) * value
                out[t] = state
            continue
        if np.isfinite(value):
            state = ewm_lambda * state + (1.0 - ewm_lambda) * value
        out[t] = state
    return out


@pytest.mark.parametrize('freq', ['ME', 'QE', 'W-WED', 'YE', 'B'])
def test_horizon_from_float_factor_runs(freq: str) -> None:
    """A frequency whose annualisation factor is a float gives an integer horizon."""
    corr = ewm_xy_convolution(returns=_daily_returns(), freq=freq)
    finite = corr.to_numpy()[np.isfinite(corr.to_numpy())]
    assert finite.size > 0 and np.all(np.abs(finite) <= 1.0 + 1e-12)


def test_monthly_horizon_matches_direct_zero_seeded_recursion() -> None:
    """freq='ME': h=12 row sums, lagged by 12 rows, EWM correlation with lambda = 11/13."""
    returns = _daily_returns()
    corr = ewm_xy_convolution(returns=returns, freq='ME',
                              convolution_type=ConvolutionType.AUTO_CORR)
    h, lam = 12, 1.0 - 2.0 / 13.0
    for column in returns.columns:
        y = returns[column].rolling(h).sum().to_numpy()
        x = np.concatenate([np.full(h, np.nan), y[:-h]])
        cross = _zero_seeded_ewm(x * y, lam)
        x_var = _zero_seeded_ewm(x * x, lam)
        y_var = _zero_seeded_ewm(y * y, lam)
        expected = cross / np.sqrt(x_var * y_var)
        np.testing.assert_allclose(corr[column].to_numpy(), expected, atol=1e-12)
        assert np.isnan(corr[column].iloc[2 * h - 2]) and np.isfinite(corr[column].iloc[2 * h - 1])


def test_annual_horizon_falls_back_to_unsummed_lag_one() -> None:
    """freq='YE' gives h=1: returns are not summed, x is lagged one row and lambda is 0.2."""
    returns = _daily_returns(num_rows=200)
    corr = ewm_xy_convolution(returns=returns, freq='YE')
    y = returns['a'].to_numpy()
    x = np.concatenate([[np.nan], y[:-1]])
    lam = 0.2
    cross = _zero_seeded_ewm(x * y, lam)
    x_var = _zero_seeded_ewm(x * x, lam)
    y_var = _zero_seeded_ewm(y * y, lam)
    # compare after the start-up rows, where lambda^t is negligible whatever the seed-row rule
    np.testing.assert_allclose(corr['a'].to_numpy()[30:], (cross / np.sqrt(x_var * y_var))[30:],
                               atol=1e-12)


@pytest.mark.parametrize('freq', ['ME', 'B'])
def test_estimates_are_point_in_time(freq: str) -> None:
    """A run on a prefix reproduces the prefix of the full run: no full-sample seed."""
    returns = _daily_returns(num_rows=900)
    full = ewm_xy_convolution(returns=returns, freq=freq)
    prefix = ewm_xy_convolution(returns=returns.iloc[:600], freq=freq)
    pd.testing.assert_frame_equal(prefix, full.iloc[:600], atol=1e-12, rtol=0.0)

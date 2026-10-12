"""Scheduled covariance masking preserves availability without a full-history copy."""

import numpy as np
import pandas as pd
import pytest

import qis
import qis.models.linear.corr_cov_matrix as ccm


@pytest.mark.parametrize("normalised", [False, True])
@pytest.mark.parametrize("warmup", [None, 0, 1])
@pytest.mark.parametrize("annualised", [False, True])
def test_estimate_rolling_ewma_covar_masks_only_selected_matrices(
    monkeypatch, normalised, warmup, annualised
):
    """Mask only returned dates, without modifying the engine's tensor or sibling outputs."""
    dates = pd.date_range("2024-01-29", periods=9, freq="B")
    labels = [7, 7, "never", "zero"]  # duplicate labels must remain positional
    returns = pd.DataFrame(
        [
            [0.01, np.nan, np.nan, 0.0],
            [0.02, 0.03, np.nan, 0.0],
            [0.01, np.nan, np.nan, 0.0],
            [0.02, np.nan, np.nan, 0.0],
            [0.01, 0.04, np.nan, 0.0],
            [0.02, np.nan, np.nan, 0.0],
            [0.01, np.nan, np.nan, 0.0],
            [0.02, np.nan, np.nan, 0.0],
            [0.01, 0.05, np.nan, 0.0],
        ],
        index=dates,
        columns=labels,
    )
    prices = pd.DataFrame(100.0, index=dates, columns=labels)
    before = prices.copy(deep=True)
    tensor = np.arange(9 * 16, dtype=float).reshape(9, 4, 4)
    original = tensor.copy()
    tensor.setflags(write=False)
    schedule = pd.Series(False, index=dates)
    schedule.iloc[[0, 3, 8]] = True
    monkeypatch.setattr(ccm.ret, "to_returns", lambda **kwargs: returns)
    monkeypatch.setattr(ccm.da, "generate_rebalancing_indicators", lambda **kwargs: schedule)
    if normalised:
        monkeypatch.setattr(
            ccm.ewm,
            "compute_ewm_covar_tensor_vol_norm_returns",
            lambda **kwargs: (tensor, None, None),
        )
    else:
        monkeypatch.setattr(ccm.ewm, "compute_ewm_covar_tensor", lambda **kwargs: tensor)

    native_where = np.where
    matrix_masks = []

    def selected_matrix_where(condition, *args, **kwargs):
        # A full-history np.where recreates the allocation this optimization removes.
        assert condition.ndim != 3, "availability must not copy the covariance history"
        if condition.ndim == 2:
            matrix_masks.append(condition.shape)
        return native_where(condition, *args, **kwargs)

    monkeypatch.setattr(ccm.np, "where", selected_matrix_where)
    actual = qis.estimate_rolling_ewma_covar(
        prices,
        time_period=qis.TimePeriod(dates[3], dates[-1]),
        returns_freq="B",
        rebalancing_freq="ME",
        span=12,
        demean=False,
        is_apply_vol_normalised_returns=normalised,
        warmup_period=warmup,
        apply_an_factor=annualised,
    )
    assert list(actual) == [dates[3], dates[8]]
    assert matrix_masks == [(4, 4), (4, 4)]
    factor = 252.0 if annualised else 1.0
    for position in (3, 8):
        expected = original[position].copy()
        # The sparse asset has one observed return at date 3, and three at date 8.
        unavailable = [2] + ([1] if warmup == 1 and position == 3 else [])
        expected[unavailable, :] = np.nan
        expected[:, unavailable] = np.nan
        pd.testing.assert_frame_equal(
            actual[dates[position]], pd.DataFrame(factor * expected, index=labels, columns=labels)
        )
    np.testing.assert_array_equal(tensor, original)
    pd.testing.assert_frame_equal(prices, before)
    sibling = actual[dates[8]].copy(deep=True)
    actual[dates[3]].iloc[0, 0] = -1.0
    actual[dates[3]].index = pd.Index(["changed", 7, "never", "zero"])
    pd.testing.assert_frame_equal(actual[dates[8]], sibling)
    np.testing.assert_array_equal(tensor, original)
    pd.testing.assert_frame_equal(prices, before)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("demean", [False, True])
@pytest.mark.parametrize("warmup", [None, 0, 3])
def test_estimate_rolling_ewma_covar_matches_independent_recursion(dtype, demean, warmup):
    """A mixed panel keeps own-inception masks and the point-in-time covariance recursion."""
    dates = pd.date_range("2023-01-02", periods=45, freq="B")
    steps = np.arange(45)
    values = 100.0 * np.exp(
        np.column_stack(
            [
                0.01 * np.sin(steps),
                0.02 * np.cos(steps),
                0.005 * steps,
                np.zeros(45),
            ]
        )
    )
    values[:8, 1] = np.nan
    values[18:20, 1] = np.nan
    values[:, 2] = np.nan
    prices = pd.DataFrame(
        values.astype(dtype), index=dates, columns=["healthy", "late", "never", "zero"]
    )
    before = prices.copy(deep=True)
    actual = qis.estimate_rolling_ewma_covar(
        prices,
        returns_freq="B",
        rebalancing_freq="B",
        span=7,
        demean=demean,
        apply_an_factor=False,
        warmup_period=warmup,
    )
    # Derive returns independently, respecting the accepted no-fill rule on the native B grid.
    log_prices = np.log(prices.to_numpy())
    returns = np.diff(log_prices, axis=0)
    mean = np.zeros(4)
    state = np.zeros((4, 4))
    observed = np.zeros(4, dtype=int)
    decay = 0.75  # span 7 -> 1 - 2 / (7 + 1)
    for date, row in zip(dates[1:], returns):
        finite = np.isfinite(row)
        observed[finite] += 1
        residual = row - mean if demean else row
        update = np.outer(residual, residual)
        state = decay * state + (1.0 - decay) * update
        state[~np.isfinite(update)] = 0.0  # ZERO_FILL resets missing pair updates
        if demean:
            mean[finite] = decay * mean[finite] + (1.0 - decay) * row[finite]
        if date not in actual:
            continue
        expected = state * (0.875 if demean else 1.0)
        unavailable = observed <= (0 if warmup is None else warmup)
        expected[unavailable, :] = np.nan
        expected[:, unavailable] = np.nan
        np.testing.assert_allclose(actual[date].to_numpy(), expected, rtol=3e-6, atol=1e-10)
        assert actual[date].index.equals(prices.columns)
        assert actual[date].columns.equals(prices.columns)
        assert all(actual[date].dtypes == np.float64)
    assert actual
    pd.testing.assert_frame_equal(prices, before)


@pytest.mark.parametrize("normalised", [False, True])
def test_estimate_rolling_ewma_covar_window_and_future_prices_preserve_selected_history(normalised):
    """Output selection and future observations cannot alter an earlier eligible matrix."""
    dates = pd.date_range("2020-01-01", periods=100, freq="B", tz="America/New_York")
    rng = np.random.default_rng(9)
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (100, 2)), axis=0)),
        index=dates,
        columns=["healthy", "late"],
    )
    prices.iloc[:20, 1] = np.nan
    kwargs = dict(
        returns_freq="B",
        rebalancing_freq="ME",
        span=12,
        warmup_period=3,
        is_apply_vol_normalised_returns=normalised,
    )
    full = qis.estimate_rolling_ewma_covar(prices, **kwargs)
    window = qis.TimePeriod("2020-02-01", "2020-03-31")
    selected = qis.estimate_rolling_ewma_covar(prices, time_period=window, **kwargs)
    expected_dates = [d for d in full if d.month in (2, 3)]
    assert list(selected) == expected_dates and len(expected_dates) == 2
    cutoff = expected_dates[-1]
    prefix = qis.estimate_rolling_ewma_covar(prices.loc[:cutoff], **kwargs)
    perturbed = prices.copy()
    perturbed.loc[perturbed.index > cutoff] *= 1.5
    future = qis.estimate_rolling_ewma_covar(perturbed, **kwargs)
    for date in expected_dates:
        pd.testing.assert_frame_equal(selected[date], full[date])
        pd.testing.assert_frame_equal(prefix[date], full[date])
        pd.testing.assert_frame_equal(future[date], full[date])

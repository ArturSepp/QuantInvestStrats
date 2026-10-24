"""Zero-seeded unsmoothing and warm-up state regression contracts."""
import importlib
import numpy as np
import pandas as pd
import pytest
import qis


def data():
    """Build a deterministic quarterly panel with a large first return."""
    dates = pd.date_range('2000-03-31', periods=72, freq='QE')
    rng = np.random.default_rng(438)
    values = rng.normal(.01, .025, (72, 2))
    values[0] = [-.12, .09]
    return pd.DataFrame(values, index=dates, columns=['a', 'b'])


@pytest.mark.parametrize('kind', ['ar', 'joint'])
def test_zero_seeds_and_warmup_precede_smoother(monkeypatch, kind):
    """Excluded coefficients cannot initialise or update the secondary filter."""
    module = importlib.import_module(
        'qis.models.unsmoothing.' + ('ar_lag' if kind == 'ar' else 'joint_lag'))
    seen, means = [], []
    original, original_mean = module.compute_ewm, module.compute_rolling_mean_adj
    def capture(*args, **kwargs):
        """Observe the smoother's explicit seed and eligible observations."""
        seen.append((kwargs['data'].copy(), kwargs.get('init_type')))
        return original(*args, **kwargs)
    def capture_mean(*args, **kwargs):
        """Observe the mean seed without altering the numerical engine."""
        means.append(kwargs.get('init_type'))
        return original_mean(*args, **kwargs)
    monkeypatch.setattr(module, 'compute_ewm', capture)
    monkeypatch.setattr(module, 'compute_rolling_mean_adj', capture_mean)
    returns = data()
    if kind == 'ar':
        module.adjust_returns_with_ar(returns, ar_order=1, span=40, warmup_period=4)
    else:
        module.adjust_returns_with_joint_unsmoothing(returns, returns.a * .3 + returns.b,
                                                    span=40, warmup_period=4)
    assert means and all(seed == qis.InitType.ZERO for seed in means)
    assert seen and all(seed == qis.InitType.ZERO for _, seed in seen)
    for coefficients, _ in seen:
        assert coefficients.iloc[:5].isna().all().all()


def test_ar1_reference_zero_seed_and_single_warmup():
    """Compare the recursion to an independently weighted matrix calculation."""
    returns = data()[['a']]
    span, warmup = 12, 4
    corrected, theta, _ = qis.adjust_returns_with_ar(
        returns, ar_order=1, span=span, warmup_period=warmup, return_diagnostics=True)
    x = returns.a.to_numpy()
    n = len(x)
    lag = np.r_[np.nan, x[:-1]]
    steps = np.arange(n)[:, None] - np.arange(n)[None, :]
    weights = np.where(steps >= 0, (2/(span+1)) * (1-2/(span+1))**np.maximum(steps, 0), 0.)
    y_adj = x - weights @ x
    x_adj = lag - weights @ np.nan_to_num(lag)
    xx = weights @ np.nan_to_num(x_adj*x_adj)
    xy = weights @ np.nan_to_num(x_adj*y_adj)
    raw = np.divide(xy, xx, out=np.full(n, np.nan), where=xx > 0).clip(-.25, .75)
    raw[:warmup+1] = np.nan
    expected_theta = weights @ np.nan_to_num(raw)
    expected_theta[:warmup+1] = np.nan
    np.testing.assert_allclose(theta.a, expected_theta, atol=1e-12, equal_nan=True)
    applied = np.r_[np.nan, expected_theta[:-1]]
    expected = (x-applied*lag)/(1-applied)
    np.testing.assert_allclose(corrected.a, expected, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize('kind', ['ar', 'joint'])
def test_prefix_and_leading_empty_calendar_invariance(kind):
    """Future observations and another asset's older calendar cannot change the fit."""
    returns = data()
    factor = returns.a * .3 + returns.b
    def run(panel):
        """Use the same economic observations with different frame boundaries."""
        if kind == 'ar':
            return qis.adjust_returns_with_ar(panel, ar_order=1, span=40, warmup_period=4)
        return qis.adjust_returns_with_joint_unsmoothing(panel, factor.reindex(panel.index),
                                                        span=40, warmup_period=4)
    full = run(returns)
    pd.testing.assert_frame_equal(run(returns.iloc[:40]), full.iloc[:40])
    older = pd.date_range('1990-03-31', periods=40, freq='QE')
    padded = returns.reindex(older.union(returns.index))
    pd.testing.assert_frame_equal(run(padded).loc[returns.index], full)

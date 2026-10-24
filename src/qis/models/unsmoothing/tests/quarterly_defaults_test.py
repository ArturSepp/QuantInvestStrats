"""Quarterly unsmoothing defaults and their numerical forwarding contracts."""
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import lsq_linear
from scipy.signal import lfilter

import qis


AR_SETTINGS = dict(
    ar_order=1, span=40, warmup_period=4, mean_adj_type=qis.MeanAdjType.NONE,
    min_value_for_beta=0.0, max_value_for_beta=0.75, apply_ewma_mean_smoother=False,
)
JOINT_SETTINGS = dict(
    span=40, warmup_period=4, mean_adj_type=qis.MeanAdjType.NONE,
    min_ar_coeff=0.0, max_ar_coeff=0.75, min_factor_beta=0.0, max_factor_beta=1.0,
    apply_ewma_mean_smoother=False,
)


def _sample():
    """Generate quarterly returns whose factor coefficient exceeds the default cap."""
    rng = np.random.default_rng(61024)
    index = pd.date_range('1990-03-31', periods=140, freq='QE')
    factor = pd.Series(rng.normal(0.01, 0.03, len(index)), index=index)
    innovation = 1.8 * factor.shift(fill_value=0) + rng.normal(0, 0.003, len(index))
    returns = pd.DataFrame({'asset': lfilter([1.0], [1.0, -0.45], innovation)}, index=index)
    return returns, factor


@pytest.mark.parametrize('entry', ['ar', 'joint', 'ar1', 'prices', 'prices_ar1'])
def test_default_entry_points_equal_explicit_quarterly_policy(entry):
    """Omitting settings gives the approved quarterly policy through every wrapper."""
    returns, factor = _sample()
    if entry == 'joint':
        actual = qis.adjust_returns_with_joint_unsmoothing(
            returns, factor, return_diagnostics=True)
        expected = qis.adjust_returns_with_joint_unsmoothing(
            returns, factor, return_diagnostics=True, **JOINT_SETTINGS)
    elif entry in ('prices', 'prices_ar1'):
        prices = 100.0 * np.exp(returns.cumsum())
        func = (qis.compute_ar_unsmoothed_prices if entry == 'prices'
                else qis.compute_ar1_unsmoothed_prices)
        actual = func(prices)
        expected = qis.compute_ar_unsmoothed_prices(
            prices, freq='QE', is_log_returns=True, **AR_SETTINGS)
    else:
        actual = (qis.adjust_returns_with_ar(returns, return_diagnostics=True)
                  if entry == 'ar' else qis.unsmooth_returns_ar1_ewma(returns))
        expected = qis.adjust_returns_with_ar(
            returns, return_diagnostics=True, **AR_SETTINGS)
    for observed, reference in zip(actual, expected, strict=True):
        pd.testing.assert_frame_equal(observed, reference, check_exact=True)


@pytest.mark.parametrize('kind', ['ar', 'joint'])
def test_default_fit_matches_independently_weighted_observations(kind):
    """Default estimates use raw returns and bounded weighted least squares."""
    returns, factor = _sample()
    if kind == 'ar':
        corrected, phi, _ = qis.adjust_returns_with_ar(returns, return_diagnostics=True)
        beta = None
    else:
        corrected, phi, beta = qis.adjust_returns_with_joint_unsmoothing(
            returns, factor, return_diagnostics=True)
    assert phi.asset.first_valid_index() == returns.index[5]
    assert corrected.asset.first_valid_index() == returns.index[6]
    for pos in (20, 70, 138):
        y = returns.asset.iloc[1:pos + 1].to_numpy()
        ownlag = returns.asset.iloc[:pos].to_numpy()
        weights = (39 / 41) ** np.arange(pos - 1, -1, -1)
        if kind == 'ar':
            expected_phi = np.clip(np.sum(weights * ownlag * y) /
                                   np.sum(weights * ownlag ** 2), 0.0, 0.75)
            expected_beta = 0.0
        else:
            design = np.column_stack([ownlag, factor.iloc[:pos]])
            fit = lsq_linear(design * np.sqrt(weights[:, None]), y * np.sqrt(weights),
                             bounds=([0., 0.], [.75, 1.]), method='bvls', tol=1e-14)
            assert fit.success
            expected_phi, expected_beta = fit.x
            assert expected_beta == pytest.approx(1.0)
            assert beta.iloc[pos, 0] == pytest.approx(expected_beta, abs=1e-10)
        assert phi.iloc[pos, 0] == pytest.approx(expected_phi, abs=1e-10)
        expected_return = (returns.iloc[pos + 1, 0] - expected_phi * returns.iloc[pos, 0]
                           + expected_beta * (factor.iloc[pos + 1] - factor.iloc[pos]))
        expected_return /= 1.0 - expected_phi
        assert corrected.iloc[pos + 1, 0] == pytest.approx(expected_return, abs=1e-10)


@pytest.mark.parametrize('kind', ['ar', 'joint'])
def test_quarterly_defaults_preserve_prefix_causality(kind):
    """Later observations cannot change released default estimates or returns."""
    returns, factor = _sample()
    def run(panel):
        """Fit the same default policy on each prefix."""
        if kind == 'joint':
            return qis.adjust_returns_with_joint_unsmoothing(
                panel, factor.reindex(panel.index), return_diagnostics=True)
        return qis.adjust_returns_with_ar(panel, return_diagnostics=True)
    for short, full in zip(run(returns.iloc[:80]), run(returns), strict=True):
        pd.testing.assert_frame_equal(short, full.iloc[:80], check_exact=True)

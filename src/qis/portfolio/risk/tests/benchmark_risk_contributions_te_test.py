"""Active-risk contributions of ``compute_benchmark_portfolio_risk_contributions``.

The function decomposes tracking error, TE = sqrt(d' S d) with d = w_p - w_b, into Euler
contributions d_i (S d)_i / TE that add up to TE, as ``RiskModel.compute_marginal_tre_at_date``
does. Both weight vectors are aligned to the covariance labels, and a zero benchmark or a zero
tracking error does not produce infinities.
"""

import numpy as np
import pandas as pd
import pytest

import qis


ASSETS = ['Equity', 'Credit', 'Hedge']
VOLS = np.array([0.20, 0.10, 0.15])
CORR = np.array([[1.0, 0.5, -0.6],
                 [0.5, 1.0, -0.2],
                 [-0.6, -0.2, 1.0]])
COVAR = pd.DataFrame(np.outer(VOLS, VOLS) * CORR, index=ASSETS, columns=ASSETS)
W_PORTFOLIO = pd.Series([0.5, 0.3, 0.2], index=ASSETS)
W_BENCHMARK = pd.Series([0.6, 0.4, 0.0], index=ASSETS)


def _euler_te_contributions(w_portfolio: pd.Series, w_benchmark: pd.Series) -> pd.Series:
    """Independent reference: d_i (S d)_i / sqrt(d' S d) by hand arithmetic on arrays.

    Args:
        w_portfolio: portfolio weights in covariance order.
        w_benchmark: benchmark weights in covariance order.

    Returns:
        Euler contributions to tracking error indexed by asset.
    """
    d = w_portfolio.to_numpy() - w_benchmark.to_numpy()
    sigma_d = COVAR.to_numpy() @ d
    return pd.Series(d * sigma_d / np.sqrt(d @ sigma_d), index=ASSETS)


def test_contributions_sum_to_tracking_error_and_match_risk_model() -> None:
    """The contributions are the Euler split of TE, not TE^2 scaled by benchmark volatility."""
    actual = qis.compute_benchmark_portfolio_risk_contributions(
        w_portfolio=W_PORTFOLIO, w_benchmark=W_BENCHMARK, covar=COVAR)

    # hand arithmetic: S d = (-0.0086, -0.0026, 0.0066) and d' S d = 0.00244
    te = np.sqrt(0.00244)
    np.testing.assert_allclose(actual.to_numpy(),
                               np.array([-0.1, -0.1, 0.2]) * np.array([-0.0086, -0.0026, 0.0066])
                               / te, atol=1e-15)
    assert actual.sum() == pytest.approx(te, abs=1e-15)
    pd.testing.assert_series_equal(actual, _euler_te_contributions(W_PORTFOLIO, W_BENCHMARK),
                                   check_names=False, atol=1e-15)

    date = pd.Timestamp('2024-12-31')
    mcte = qis.RiskModel(covar={date: COVAR}).compute_marginal_tre_at_date(
        benchmark_weights=W_BENCHMARK, portfolio_weights=W_PORTFOLIO, date=date)['mcte']
    np.testing.assert_allclose(actual.to_numpy(), mcte.to_numpy(), atol=1e-15)


def test_benchmark_is_aligned_to_the_covariance_labels() -> None:
    """A benchmark with a missing, an extra or a reordered label is aligned by label."""
    expected = _euler_te_contributions(W_PORTFOLIO, W_BENCHMARK)

    missing_hedge = W_BENCHMARK.drop('Hedge')  # a zero weight left out
    extra_label = pd.concat([W_BENCHMARK, pd.Series({'Outside': 0.0})])
    reordered = W_BENCHMARK[['Hedge', 'Credit', 'Equity']]
    for benchmark in (missing_hedge, extra_label, reordered):
        actual = qis.compute_benchmark_portfolio_risk_contributions(
            w_portfolio=W_PORTFOLIO, w_benchmark=benchmark, covar=COVAR)
        assert actual.index.tolist() == ASSETS
        np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), atol=1e-15)

    reordered_portfolio = W_PORTFOLIO[['Credit', 'Hedge', 'Equity']]
    actual = qis.compute_benchmark_portfolio_risk_contributions(
        w_portfolio=reordered_portfolio, w_benchmark=reordered, covar=COVAR)
    assert actual.index.tolist() == ASSETS
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), atol=1e-15)


def test_zero_benchmark_and_zero_tracking_error_are_finite() -> None:
    """A zero benchmark gives the portfolio's own Euler split; zero TE gives zeros."""
    zero_benchmark = pd.Series(0.0, index=ASSETS)
    actual = qis.compute_benchmark_portfolio_risk_contributions(
        w_portfolio=W_PORTFOLIO, w_benchmark=zero_benchmark, covar=COVAR)
    assert np.all(np.isfinite(actual.to_numpy()))
    np.testing.assert_allclose(
        actual.to_numpy(),
        qis.compute_portfolio_risk_contributions(w=W_PORTFOLIO, covar=COVAR).to_numpy(),
        atol=1e-15)

    identical = qis.compute_benchmark_portfolio_risk_contributions(
        w_portfolio=W_PORTFOLIO, w_benchmark=W_PORTFOLIO, covar=COVAR)
    pd.testing.assert_series_equal(identical, pd.Series(0.0, index=ASSETS))


def test_numpy_inputs_give_the_same_contributions() -> None:
    """Arrays are taken in covariance order and return an array."""
    actual = qis.compute_benchmark_portfolio_risk_contributions(
        w_portfolio=W_PORTFOLIO.to_numpy(), w_benchmark=W_BENCHMARK.to_numpy(),
        covar=COVAR.to_numpy())
    assert isinstance(actual, np.ndarray)
    np.testing.assert_allclose(actual, _euler_te_contributions(W_PORTFOLIO, W_BENCHMARK)
                               .to_numpy(), atol=1e-15)


def test_standalone_option_is_the_undiversified_bound() -> None:
    """``is_independent_risk=True`` returns |d_i| sigma_i, whose sum bounds TE from above."""
    standalone = qis.compute_benchmark_portfolio_risk_contributions(
        w_portfolio=W_PORTFOLIO, w_benchmark=W_BENCHMARK[['Hedge', 'Credit', 'Equity']],
        covar=COVAR, is_independent_risk=True)
    assert standalone.index.tolist() == ASSETS
    np.testing.assert_allclose(standalone.to_numpy(), [0.02, 0.01, 0.03], atol=1e-15)
    assert standalone.sum() >= np.sqrt(0.00244)

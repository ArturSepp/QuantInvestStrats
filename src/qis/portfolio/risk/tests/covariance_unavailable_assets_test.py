"""Risk computations with assets that have no covariance yet.

``estimate_rolling_ewma_covar`` leaves the row and column of an asset NaN before its first return.
A zero weight on such an asset contributes nothing, so portfolio risk, tracking error, betas of
held assets and Euler contributions must equal those computed on the available block. A nonzero
weight on it makes the portfolio's risk unknown, so the result is NaN.
"""

# packages
import numpy as np
import pandas as pd
import pytest

# qis
import qis

_ASSETS = ['a', 'b', 'late']
_DATE = pd.Timestamp('2024-03-29')


def _covar() -> pd.DataFrame:
    """A valid 2x2 block for 'a' and 'b'; 'late' has no covariance yet."""
    block = np.array([[0.04, 0.006], [0.006, 0.01]])
    values = np.full((3, 3), np.nan)
    values[:2, :2] = block
    return pd.DataFrame(values, index=_ASSETS, columns=_ASSETS)


def _block() -> pd.DataFrame:
    return _covar().loc[['a', 'b'], ['a', 'b']]


def test_risk_contributions_ignore_an_unheld_unavailable_asset() -> None:
    """Contributions of held assets equal the available block's; the unheld asset gets zero."""
    weights = pd.Series([0.6, 0.4, 0.0], index=_ASSETS)
    actual = qis.compute_portfolio_risk_contributions(w=weights, covar=_covar())
    expected = qis.compute_portfolio_risk_contributions(w=weights.iloc[:2], covar=_block())
    np.testing.assert_allclose(actual.iloc[:2].to_numpy(), expected.to_numpy(), rtol=1e-14)
    assert actual['late'] == 0.0


def test_risk_contributions_are_missing_when_the_unavailable_asset_is_held() -> None:
    weights = pd.Series([0.5, 0.3, 0.2], index=_ASSETS)
    actual = qis.compute_portfolio_risk_contributions(w=weights, covar=_covar())
    assert actual.isna().all()


def test_risk_model_accepts_unavailable_assets_and_ignores_them_when_unheld() -> None:
    """Tracking error and betas use the available block when the late asset is not held."""
    model = qis.RiskModel(covar={_DATE: _covar()})
    portfolio = pd.Series([0.6, 0.4, 0.0], index=_ASSETS)
    benchmark = pd.Series([0.5, 0.5, 0.0], index=_ASSETS)
    block_model = qis.RiskModel(covar={_DATE: _block()})
    te = model.compute_tre_at_date(benchmark_weights=benchmark, portfolio_weights=portfolio,
                                   date=_DATE)
    te_block = block_model.compute_tre_at_date(benchmark_weights=benchmark.iloc[:2],
                                               portfolio_weights=portfolio.iloc[:2], date=_DATE)
    assert abs(te - te_block) < 1e-15
    beta = model.compute_benchmark_beta_at_date(benchmark_weights=benchmark,
                                                portfolio_weights=portfolio, date=_DATE)
    beta_block = block_model.compute_benchmark_beta_at_date(
        benchmark_weights=benchmark.iloc[:2], portfolio_weights=portfolio.iloc[:2], date=_DATE)
    assert abs(beta - beta_block) < 1e-15
    loadings = model.compute_benchmark_beta_loadings_at_date(benchmark_weights=benchmark,
                                                             date=_DATE)
    assert np.isnan(loadings['late'])
    assert np.all(np.isfinite(loadings[['a', 'b']]))


def test_risk_model_tracking_error_is_missing_when_the_unavailable_asset_is_held() -> None:
    model = qis.RiskModel(covar={_DATE: _covar()})
    portfolio = pd.Series([0.5, 0.3, 0.2], index=_ASSETS)
    benchmark = pd.Series([0.5, 0.5, 0.0], index=_ASSETS)
    te = model.compute_tre_at_date(benchmark_weights=benchmark, portfolio_weights=portfolio,
                                   date=_DATE)
    assert np.isnan(te)


def test_risk_model_still_rejects_scattered_missing_values() -> None:
    """A NaN outside the full row and column of an unavailable asset is invalid input."""
    covar = _block().copy()
    covar.loc['a', 'b'] = np.nan
    with pytest.raises(ValueError, match='non-finite'):
        qis.RiskModel(covar={_DATE: covar})


def test_marginal_tracking_error_and_beta_history_ignore_an_unheld_unavailable_asset() -> None:
    model = qis.RiskModel(covar={_DATE: _covar()})
    block_model = qis.RiskModel(covar={_DATE: _block()})
    portfolio = pd.Series([0.6, 0.4, 0.0], index=_ASSETS)
    benchmark = pd.Series([0.5, 0.5, 0.0], index=_ASSETS)
    mcte = model.compute_marginal_tre_at_date(benchmark_weights=benchmark,
                                              portfolio_weights=portfolio, date=_DATE)['mcte']
    mcte_block = block_model.compute_marginal_tre_at_date(
        benchmark_weights=benchmark.iloc[:2], portfolio_weights=portfolio.iloc[:2],
        date=_DATE)['mcte']
    np.testing.assert_allclose(mcte.iloc[:2].to_numpy(), mcte_block.to_numpy(), rtol=1e-14)
    assert mcte['late'] == 0.0
    beta = model.compute_benchmark_beta_history(benchmark_weights=benchmark,
                                                portfolio_weights=portfolio)
    beta_block = block_model.compute_benchmark_beta_history(
        benchmark_weights=benchmark.iloc[:2], portfolio_weights=portfolio.iloc[:2])
    np.testing.assert_allclose(beta.to_numpy(), beta_block.to_numpy(), rtol=1e-14)


def test_active_contributions_ignore_an_unavailable_asset_with_zero_active_weight() -> None:
    """Equal portfolio and benchmark weights on the late asset leave tracking error defined."""
    portfolio = pd.Series([0.5, 0.3, 0.2], index=_ASSETS)
    benchmark = pd.Series([0.4, 0.4, 0.2], index=_ASSETS)
    for is_independent_risk in (False, True):
        actual = qis.compute_benchmark_portfolio_risk_contributions(
            w_portfolio=portfolio, w_benchmark=benchmark, covar=_covar(),
            is_independent_risk=is_independent_risk)
        expected = qis.compute_benchmark_portfolio_risk_contributions(
            w_portfolio=portfolio.iloc[:2], w_benchmark=benchmark.iloc[:2], covar=_block(),
            is_independent_risk=is_independent_risk)
        np.testing.assert_allclose(actual.iloc[:2].to_numpy(), expected.to_numpy(), rtol=1e-14)
        assert actual['late'] == 0.0

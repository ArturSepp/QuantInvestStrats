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


def test_compute_marginal_tre_at_date_keeps_empty_universe_total_zero() -> None:
    """No assets is zero risk, not a nonempty portfolio with unknown covariance."""
    model = qis.RiskModel(covar={_DATE: pd.DataFrame(dtype=float)})
    weights = pd.Series(dtype=float)
    actual = model.compute_marginal_tre_at_date(
        weights, weights, _DATE, group_data=pd.Series(dtype=object),
    )
    pd.testing.assert_frame_equal(actual, pd.DataFrame({'mcte': [0.]}, index=['Total']))


@pytest.mark.parametrize('held', [False, True])
def test_compute_group_portfolio_risk_contribution_ratios_keeps_unused_categories_zero(
        monkeypatch: pytest.MonkeyPatch, held: bool) -> None:
    """Exercise the unobserved-category mode used by older supported pandas defaults."""
    groupby = pd.Series.groupby

    def include_unobserved(series, *args, **kwargs):
        kwargs.setdefault('observed', False)
        return groupby(series, *args, **kwargs)

    monkeypatch.setattr(pd.Series, 'groupby', include_unobserved)
    categories = ['risk', 'new', 'empty']
    groups = pd.Series(pd.Categorical(['risk', 'risk', 'new'], categories=categories),
                       index=_ASSETS)
    weights = pd.Series([.5, .3, .2] if held else [.6, .4, 0.], index=_ASSETS)
    actual = qis.compute_group_portfolio_risk_contribution_ratios(weights, _covar(), groups)
    expected = pd.Series([np.nan, np.nan, 0.] if held else [1., 0., 0.],
                         index=pd.CategoricalIndex(categories, categories=categories),
                         name='risk_contribution')
    pd.testing.assert_series_equal(actual, expected, atol=1e-14, rtol=1e-14)


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


@pytest.mark.parametrize('dtype', ['float64', 'Float64'])
@pytest.mark.parametrize('state', ['held', 'unheld', 'cash'])
def test_compute_group_portfolio_risk_contribution_ratios_preserves_unknown_risk(
        dtype: str, state: str) -> None:
    """Unknown variance cannot become zero when missing asset shares are grouped."""
    values = {'held': [.5, .3, .2], 'unheld': [.6, .4, 0.], 'cash': [0., 0., 0.]}
    weights = pd.Series(values[state], index=_ASSETS, dtype=dtype)
    covar = _covar().astype(dtype)
    groups = pd.Series(['risk', 'risk', 'new'], index=_ASSETS)
    before = covar.copy(deep=True)

    actual = qis.compute_group_portfolio_risk_contribution_ratios(weights, covar, groups)

    # Both known assets belong to risk: their signed Euler shares sum to one. The
    # unavailable asset contributes zero only when it is unheld, not when risk is unknown.
    expected_values = {'held': [np.nan, np.nan], 'unheld': [1., 0.], 'cash': [0., 0.]}
    expected_dtype = dtype if state != 'cash' else 'float64'
    expected = pd.Series(expected_values[state], index=['risk', 'new'],
                         name='risk_contribution', dtype=expected_dtype)
    pd.testing.assert_series_equal(actual, expected, atol=1e-14, rtol=1e-14)
    pd.testing.assert_frame_equal(covar, before)
    pd.testing.assert_series_equal(weights, pd.Series(values[state], index=_ASSETS, dtype=dtype))


@pytest.mark.parametrize('factor_block', [False, True])
@pytest.mark.parametrize('state', ['held', 'unheld', 'cash'])
def test_compute_marginal_tre_at_date_preserves_unknown_group_and_total_risk(
        factor_block: bool, state: str) -> None:
    """Every Euler column retains unknown risk; finite factor splits still reconcile."""
    fields = {}
    if factor_block:
        # B B' + D reproduces the available covariance block independently: .2*.03
        # gives .006 cross-covariance, and .03**2 + .0091 gives .01 variance.
        fields = dict(
            factor_loadings={_DATE: pd.DataFrame({'factor': [.2, .03, 0.]}, index=_ASSETS)},
            factor_covar={_DATE: pd.DataFrame([[1.]], index=['factor'], columns=['factor'])},
            residual_vars={_DATE: pd.Series([0., .0091, 0.], index=_ASSETS)},
        )
    model = qis.RiskModel(covar={_DATE: _covar()}, **fields)
    values = {'held': [.5, .3, .2], 'unheld': [.6, .4, 0.], 'cash': [0., 0., 0.]}
    weights = pd.Series(values[state], index=_ASSETS)
    groups = pd.Series(['risk', 'risk', 'new'], index=_ASSETS)
    before = model.covar[_DATE].copy(deep=True)

    actual = model.compute_marginal_tre_at_date(
        benchmark_weights=weights * 0., portfolio_weights=weights,
        date=_DATE, group_data=groups,
    )

    vol = np.sqrt(.6**2 * .04 + 2 * .6 * .4 * .006 + .4**2 * .01)
    expected = pd.DataFrame({'mcte': [vol, vol, 0.]}, index=['Total', 'risk', 'new'])
    if factor_block:
        systematic = (.6 * .2 + .4 * .03)**2 / vol
        residual = .4**2 * .0091 / vol
        expected['mcte_systematic'] = [systematic, systematic, 0.]
        expected['mcte_residual'] = [residual, residual, 0.]
    if state != 'unheld':
        expected.loc[:, :] = np.nan if state == 'held' else 0.
    pd.testing.assert_frame_equal(actual, expected, atol=1e-14, rtol=1e-14)
    pd.testing.assert_frame_equal(model.covar[_DATE], before)


@pytest.mark.parametrize('dtype', ['float64', 'Float64'])
@pytest.mark.parametrize('freq', [None, 'ME'])
@pytest.mark.parametrize('normalise', [False, True])
@pytest.mark.parametrize('grouped', [False, True])
def test_compute_risk_contributions_implied_by_covar_preserves_unknown_rows(
        dtype: str, freq: str, normalise: bool, grouped: bool) -> None:
    """Grouping and normalization cannot confuse unknown risk with an all-cash row."""
    dates = pd.date_range('2024-01-31', periods=4, freq='ME')
    weights = pd.DataFrame([[.6, .4, 0.], [.5, .3, .2], [0., 0., 0.], [.6, .4, 0.]],
                           index=dates, columns=_ASSETS, dtype=dtype)
    groups = pd.Series(['risk', 'risk', 'new'], index=_ASSETS)
    portfolio = qis.PortfolioData(
        nav=pd.Series(100., index=dates, name='portfolio'),
        prices=pd.DataFrame(100., index=dates, columns=_ASSETS),
        weights=weights, input_weights=weights, group_data=groups,
        covar_dict={date: _covar().astype(dtype) for date in dates},
    )
    before = portfolio.weights.copy(deep=True)

    actual = portfolio.compute_risk_contributions_implied_by_covar(
        freq=freq, normalise=normalise,
        group_data=groups if grouped else None, group_order=['risk', 'new'],
    )

    # The literal quadratic form supplies the denominator, not a production risk
    # helper. Keeping healthy rows around the gap catches an over-broad missing mask.
    vol = np.sqrt(.6**2 * .04 + 2 * .6 * .4 * .006 + .4**2 * .01)
    known = np.array([.6 * (.04 * .6 + .006 * .4),
                      .4 * (.006 * .6 + .01 * .4), 0.]) / vol
    if grouped:
        known = np.array([known[0] + known[1], 0.])
    if normalise:
        known = known / vol
    expected = pd.DataFrame([known, np.full(len(known), np.nan),
                             np.zeros(len(known)), known], index=dates,
                            columns=['risk', 'new'] if grouped else _ASSETS)
    if freq is not None:
        # The resampled public path starts strictly after the first covariance date.
        expected = expected.iloc[1:]
    if dtype == 'Float64' and freq is None and not grouped:
        # Native from-dict construction retains object storage when an asset row has pd.NA.
        expected = expected.astype(object)
        if not normalise:
            expected.loc[dates[1], :] = pd.NA
    expected.index = expected.index.copy(deep=True)
    expected.index.freq = None
    pd.testing.assert_frame_equal(actual, expected, atol=1e-14, rtol=1e-14)
    pd.testing.assert_frame_equal(portfolio.weights, before)

"""LinearModel diagnostics, exposures and attributions report undefined values as NaN.

Each test pins a defect recorded by the factor-risk-model handbook chapter against an independent
calculation: the average residual correlation, the start-up of the EWM R², the as-of alignment
of weights in aggregated exposures and risk contributions, and the warm-up rows of the asset and
benchmark attributions.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import qis
from qis.portfolio.risk.factor_model import (
    LinearModel,
    compute_benchmarks_beta_attribution_from_prices,
    compute_benchmarks_beta_attribution_from_returns,
)

ASSETS = ['A1', 'A2', 'A3']
FACTORS = ['Equity', 'Rates']
LOADINGS = pd.DataFrame([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], index=ASSETS, columns=FACTORS)


def _fitted_noisy_model(span: int = 26) -> tuple[qis.EwmLinearModel, pd.DataFrame, pd.DataFrame]:
    """Fit an EWM model on 156 weeks of factor returns plus independent noise."""
    rng = np.random.default_rng(20260725)
    weeks = pd.date_range('2022-01-05', periods=156, freq='W-WED')
    factor_returns = pd.DataFrame(rng.normal(0.0, [0.03, 0.015], size=(156, 2)),
                                  index=weeks, columns=FACTORS)
    noise = pd.DataFrame(rng.normal(0.0, [0.02, 0.01, 0.01], size=(156, 3)),
                         index=weeks, columns=ASSETS)
    asset_returns = factor_returns @ LOADINGS.T + noise
    model = qis.EwmLinearModel(x=factor_returns, y=asset_returns)
    model.fit(span=span)
    return model, factor_returns, asset_returns


def test_average_residual_correlation_is_the_mean_off_diagonal_correlation() -> None:
    """The average is the mean of the n - 1 off-diagonal correlations, not (n-1)/(2n) of it."""
    model, _, _ = _fitted_noisy_model()

    corr, avg_corr = model.get_model_residuals_corrs(span=52)

    c = corr.to_numpy()
    expected = (c.sum(axis=1) - np.diag(c)) / (c.shape[0] - 1)
    np.testing.assert_allclose(avg_corr.to_numpy(), expected, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(avg_corr['A1'], 0.5 * (c[0, 1] + c[0, 2]), atol=1.0e-12)


@pytest.mark.parametrize('lag', [0, 1])
def test_ewm_r2_aligns_residual_and_return_moments(lag: int) -> None:
    """Both EWM second moments run over the same observations with the same weights."""
    model, _, asset_returns = _fitted_noisy_model()
    span = 52
    lam = 1.0 - 2.0 / (span + 1.0)

    r2 = model.get_model_ewm_r2(span=span, lag=lag)

    residuals, _ = model.get_factor_alpha(lag=lag)
    e2 = np.square(residuals.to_numpy())
    y2 = np.square(asset_returns.to_numpy())
    first = int(np.flatnonzero(np.isfinite(e2).all(axis=1))[0])
    expected = np.full(e2.shape, np.nan)
    for t in range(first, e2.shape[0]):
        weights = lam ** np.arange(t - first, -1, -1)[:, None]
        expected[t] = 1.0 - (weights * e2[first:t + 1]).sum(axis=0) / (
            weights * y2[first:t + 1]).sum(axis=0)
    expected = np.clip(expected, 0.0, 1.0)

    # The misaligned seeding reported 0.97 to 0.999 on the first rows after the warm-up.
    assert r2.iloc[:first].isna().all().all()
    np.testing.assert_allclose(r2.to_numpy()[first:], expected[first:], rtol=0.0, atol=1.0e-12)


def test_agg_factor_exposures_select_weights_as_of_each_loading_date() -> None:
    """Month-end weights reach weekly loading dates as of, and warm-up rows are NaN."""
    weeks = pd.date_range('2024-01-03', periods=10, freq='W-WED')
    loadings = {
        'Equity': pd.DataFrame(1.0, index=weeks, columns=ASSETS),
        'Rates': pd.DataFrame(0.5, index=weeks, columns=ASSETS),
    }
    for frame in loadings.values():
        frame.iloc[:2] = np.nan  # warm-up rows
    model = LinearModel(x=pd.DataFrame(0.0, index=weeks, columns=FACTORS),
                        y=pd.DataFrame(0.0, index=weeks, columns=ASSETS),
                        loadings=loadings)
    month_ends = pd.DatetimeIndex(['2023-12-31', '2024-01-31', '2024-02-29'])
    weights = pd.DataFrame([[0.2, 0.0, 0.0], [0.5, 0.3, 0.0], [1.0, 0.0, 0.0]],
                           index=month_ends, columns=ASSETS)

    exposures = model.compute_agg_factor_exposures(weights=weights)

    assert exposures.iloc[:2].isna().all().all()
    as_of = weights.reindex(weights.index.union(weeks)).ffill().reindex(weeks)
    expected_equity = as_of.sum(axis=1).to_numpy()
    np.testing.assert_allclose(exposures['Equity'].to_numpy()[2:], expected_equity[2:],
                               atol=1.0e-15)
    np.testing.assert_allclose(exposures['Rates'].to_numpy()[2:], 0.5 * expected_equity[2:],
                               atol=1.0e-15)
    assert exposures.loc['2024-02-07', 'Equity'] == pytest.approx(0.8)


def test_agg_factor_exposures_ignore_missing_loadings_of_unheld_assets() -> None:
    """A missing loading matters only for an asset the portfolio holds."""
    date = pd.Timestamp('2024-01-31')
    loadings = {'Equity': pd.DataFrame([[1.0, np.nan, 2.0]], index=[date], columns=ASSETS)}
    model = LinearModel(x=pd.DataFrame(0.0, index=[date], columns=['Equity']),
                        y=pd.DataFrame(0.0, index=[date], columns=ASSETS),
                        loadings=loadings)

    unheld = model.compute_agg_factor_exposures(
        weights=pd.DataFrame([[0.5, 0.0, 0.5]], index=[date], columns=ASSETS))
    held = model.compute_agg_factor_exposures(
        weights=pd.DataFrame([[0.5, 0.2, 0.3]], index=[date], columns=ASSETS))

    assert unheld.loc[date, 'Equity'] == pytest.approx(1.5)
    assert np.isnan(held.loc[date, 'Equity'])


def test_asset_factor_attribution_total_is_nan_while_betas_are_missing() -> None:
    """The total is undefined until every lagged beta of the asset exists."""
    model, factor_returns, _ = _fitted_noisy_model()

    attribution = model.get_asset_factor_attribution(asset='A1')

    lagged = pd.DataFrame({q: model.loadings[q]['A1'].shift(1) for q in FACTORS})
    missing = lagged.isna().any(axis=1)
    assert missing.sum() == 22
    assert attribution.loc[missing, 'Total'].isna().all()
    expected_total = (lagged * factor_returns).sum(axis=1)
    np.testing.assert_allclose(attribution.loc[~missing, 'Total'], expected_total[~missing],
                               atol=1.0e-15)


def _risk_contribution_model(residual_dates: pd.DatetimeIndex) -> LinearModel:
    covar_dates = pd.DatetimeIndex(['2024-03-31', '2024-06-30'])
    loadings = {q: LOADINGS[[q]].T.set_axis([covar_dates[0]]) for q in FACTORS}
    factor_covar = pd.DataFrame([[0.04, 0.004], [0.004, 0.01]], index=FACTORS, columns=FACTORS)
    residual_vars = pd.DataFrame([[0.01, 0.0004, 0.0004]] * len(residual_dates),
                                 index=residual_dates, columns=ASSETS)
    return LinearModel(x=pd.DataFrame(0.0, index=covar_dates, columns=FACTORS),
                       y=pd.DataFrame(0.0, index=covar_dates, columns=ASSETS),
                       loadings=loadings,
                       x_covars={date: factor_covar for date in covar_dates},
                       residual_vars=residual_vars)


def test_factor_risk_contribution_reads_residual_variances_as_of() -> None:
    """Residual variances follow the same as-of rule as weights and loadings."""
    model = _risk_contribution_model(residual_dates=pd.DatetimeIndex(['2024-01-31']))
    weights = pd.DataFrame([[0.4, 0.4, 0.2]], index=[pd.Timestamp('2024-01-31')],
                           columns=ASSETS)

    _, total_shares, systematic_shares, variances = model.compute_factor_risk_contribution(
        weights=weights)

    for date in total_shares.index:
        np.testing.assert_allclose(total_shares.loc[date], [192 / 245, 32 / 245, 3 / 35],
                                   atol=1.0e-12)
        np.testing.assert_allclose(systematic_shares.loc[date], [6 / 7, 1 / 7], atol=1.0e-12)
    np.testing.assert_allclose(variances.to_numpy(), [[0.01792, 0.00168]] * 2, atol=1.0e-15)


def test_factor_risk_contribution_is_nan_when_a_held_loading_is_missing() -> None:
    """A missing loading of a held asset makes the date undefined instead of a zero exposure."""
    model = _risk_contribution_model(residual_dates=pd.DatetimeIndex(['2024-03-31']))
    for q in FACTORS:
        model.loadings[q].loc[:, 'A3'] = np.nan
    held = pd.DataFrame([[0.4, 0.4, 0.2]], index=[pd.Timestamp('2024-03-31')], columns=ASSETS)
    unheld = pd.DataFrame([[0.5, 0.5, 0.0]], index=[pd.Timestamp('2024-03-31')], columns=ASSETS)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        ratios, total_shares, _, variances = model.compute_factor_risk_contribution(weights=held)
    _, unheld_shares, _, unheld_variances = model.compute_factor_risk_contribution(
        weights=unheld)

    assert total_shares.isna().all().all()
    assert ratios.isna().all().all()
    assert variances['Systematic'].isna().all()
    exposures = np.array([0.75, 0.25])
    factor_covar = np.array([[0.04, 0.004], [0.004, 0.01]])
    np.testing.assert_allclose(unheld_variances['Systematic'],
                               exposures @ factor_covar @ exposures, atol=1.0e-15)
    assert np.isfinite(unheld_shares.to_numpy()).all()


def test_price_attribution_alpha_is_nan_while_the_lagged_beta_is_missing() -> None:
    """During the warm-up the residual is undefined, not the whole portfolio return."""
    index = pd.date_range('2024-01-31', periods=6, freq='ME')
    benchmark_prices = pd.DataFrame({'Bench': [100.0, 101.0, 99.0, 102.0, 103.0, 104.0]},
                                    index=index)
    portfolio_nav = pd.Series([1.0, 1.02, 1.01, 1.03, 1.05, 1.04], index=index, name='nav')
    betas = pd.DataFrame({'Bench': [np.nan, np.nan, 0.5, 0.6, 0.7, 0.8]}, index=index)

    attribution = compute_benchmarks_beta_attribution_from_prices(
        portfolio_nav=portfolio_nav, benchmark_prices=benchmark_prices,
        portfolio_benchmark_betas=betas)

    assert attribution.iloc[:3].isna().all().all()
    bench_returns = benchmark_prices['Bench'].pct_change()
    expected_alpha = portfolio_nav.pct_change() - betas['Bench'].shift(1) * bench_returns
    np.testing.assert_allclose(attribution['Alpha'].iloc[3:], expected_alpha.iloc[3:],
                               atol=1.0e-15)


def test_return_attribution_keeps_warm_up_rows_missing_instead_of_zero() -> None:
    """The returns variant no longer overwrites its first row with zeros."""
    index = pd.date_range('2024-01-31', periods=5, freq='ME')
    portfolio_returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01], index=index, name='p')
    benchmark_returns = pd.DataFrame({'Bench': [0.02, 0.01, -0.02, 0.02, 0.0]}, index=index)
    betas = pd.DataFrame({'Bench': [np.nan, 0.5, 0.6, 0.7, 0.8]}, index=index)

    attribution = compute_benchmarks_beta_attribution_from_returns(
        portfolio_returns=portfolio_returns, benchmark_returns=benchmark_returns,
        portfolio_benchmark_betas=betas, total_name='Total')

    assert attribution.iloc[:2][['Bench', 'Alpha']].isna().all().all()
    pd.testing.assert_series_equal(attribution['Total'], portfolio_returns.rename('Total'))
    expected_alpha = portfolio_returns - betas['Bench'].shift(1) * benchmark_returns['Bench']
    np.testing.assert_allclose(attribution['Alpha'].iloc[2:], expected_alpha.iloc[2:],
                               atol=1.0e-15)

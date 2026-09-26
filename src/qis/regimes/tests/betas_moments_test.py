"""Regime betas, EWMA moments and the mixture covariance against closed forms and identities."""

import ast
import pathlib

import numpy as np
import pandas as pd
import pytest
from scipy import integrate
from scipy.stats import norm

from qis.regimes import (
    compute_gaussian_regime_moments,
    compute_regime_betas,
    compute_regime_betas_bootstrap,
    compute_regime_ewm_avg,
    compute_regime_ewm_betas,
    compute_regime_mixture_covar,
    create_sampled_returns_with_regime_id,
    get_regime_probabilities,
)


@pytest.fixture(scope='module')
def monthly():
    """Twelve years of monthly returns: a benchmark and three assets with regime-dependent betas."""
    rng = np.random.default_rng(20260926)
    b = 0.005 + 0.04 * rng.standard_normal(144)
    tail = b < np.quantile(b, 0.16)
    assets = {'A': 0.3 * b + 0.02 * rng.standard_normal(144),
              'B': np.where(tail, -1.5, 0.2) * b + 0.03 * rng.standard_normal(144),
              'C': -0.2 * b + 0.025 * rng.standard_normal(144)}
    index = pd.date_range('2012-01-31', periods=144, freq='ME')
    return pd.DataFrame({'BM': b, **assets}, index=index)


def test_regime_betas_equal_the_covariance_ratio_per_regime(monthly):
    """The OLS slope within each regime is cov(asset, benchmark) / var(benchmark) there."""
    sampled = create_sampled_returns_with_regime_id(monthly, benchmark='BM')
    betas = compute_regime_betas(sampled, benchmark='BM', af=12.0)
    for regime in ('Bear', 'Normal', 'Bull'):
        block = sampled[sampled['regime'] == regime]
        for asset in ('A', 'B', 'C'):
            expected = np.cov(block[asset], block['BM'])[0, 1] / np.var(block['BM'], ddof=1)
            assert np.isclose(betas.loc[asset, f"beta_{regime.lower()}"], expected, rtol=1e-10)
    assert betas.loc['B', 'beta_bear'] < -1.0 < 0.0 < betas.loc['B', 'beta_normal']
    assert list(betas.columns) == ['beta_bear', 'n_bear', 'beta_normal', 'n_normal', 'beta_bull',
                                   'n_bull', 'beta_total', 'idio_vol']


def test_bootstrap_se_is_seeded_and_the_point_is_the_full_sample_estimate(monthly):
    """Point estimates equal compute_regime_betas; standard errors are positive and reproducible."""
    first = compute_regime_betas_bootstrap(monthly, benchmark='BM', af=12.0, n_boot=200)
    second = compute_regime_betas_bootstrap(monthly, benchmark='BM', af=12.0, n_boot=200)
    pd.testing.assert_frame_equal(first, second)
    point = compute_regime_betas(create_sampled_returns_with_regime_id(monthly, benchmark='BM'),
                                 benchmark='BM', af=12.0)
    np.testing.assert_allclose(first['beta_bear'], point['beta_bear'].astype(float))
    assert (first[['beta_bear_se', 'beta_normal_se', 'beta_bull_se']] > 0.0).to_numpy().all()


def test_regime_ewm_avg_tends_to_the_equal_weighted_regime_means(monthly):
    """With the stream-mean seed and a span far beyond the stream, the EWMA is the plain mean."""
    sampled = create_sampled_returns_with_regime_id(monthly, benchmark='BM')
    ewm = compute_regime_ewm_avg(sampled, span=1e9)
    plain = sampled.groupby('regime', observed=True).mean().reindex(['Bear', 'Normal', 'Bull'])
    np.testing.assert_allclose(ewm.to_numpy(), plain[ewm.columns].to_numpy(), rtol=1e-6)
    assert list(ewm.index) == ['Bear', 'Normal', 'Bull']


def test_regime_ewm_betas_tend_to_the_ols_betas(monthly):
    """At a very long span the EWMA betas are the per-regime OLS betas."""
    sampled = create_sampled_returns_with_regime_id(monthly, benchmark='BM')
    ewm_betas, idio_vars = compute_regime_ewm_betas(sampled, benchmark='BM', span=1e9)
    ols = compute_regime_betas(sampled, benchmark='BM', af=12.0)
    for regime in ('Bear', 'Normal', 'Bull'):
        np.testing.assert_allclose(ewm_betas[regime], ols[f"beta_{regime.lower()}"].astype(float),
                                   rtol=1e-6)
    assert (idio_vars > 0.0).all()


def test_gaussian_moments_obey_total_expectation_and_the_published_constants():
    """sum p m = mu and sum p S = sigma^2 + mu^2; Bear mean -1.521 sigma on the one-sigma cut."""
    for q in (None, [0.0, 0.1, 0.5, 0.9, 1.0], [0.0, 0.3, 1.0]):
        means, second = compute_gaussian_regime_moments(benchmark_vol=0.04, benchmark_mean=0.006,
                                                        q=q)
        probs = get_regime_probabilities(q).to_numpy()
        assert np.isclose(np.sum(probs * means.to_numpy()), 0.006, rtol=0.0, atol=1e-14)
        assert np.isclose(np.sum(probs * second.to_numpy()), 0.04 ** 2 + 0.006 ** 2, rtol=1e-12)
    means, second = compute_gaussian_regime_moments(benchmark_vol=1.0)
    assert np.isclose(means['Bear'], -1.521, atol=1e-3)
    assert np.isclose(second['Bear'] - means['Bear'] ** 2, 0.200, atol=1e-3)
    assert np.isclose(second['Normal'], 0.288, atol=1e-3)
    z = norm.ppf(0.84)
    tail_second, _ = integrate.quad(lambda x: x * x * norm.pdf(x), -np.inf, -z)
    assert np.isclose(second['Bear'], tail_second / 0.16, rtol=1e-9)


def test_mixture_covariance_with_equal_betas_is_the_single_factor_covariance():
    """Equal regime betas reduce the mixture to beta beta' var_B plus the residual diagonal."""
    means, second = compute_gaussian_regime_moments(benchmark_vol=0.04, benchmark_mean=0.006)
    betas = pd.DataFrame({g: [1.0, 0.5, -0.3] for g in ('Bear', 'Normal', 'Bull')},
                         index=['BM', 'A', 'C'])
    idio = pd.Series({'BM': 0.0, 'A': 0.0004, 'C': 0.0009})
    covar = compute_regime_mixture_covar(betas, idio, means, second, af=12.0)
    beta = betas['Bear'].to_numpy()
    expected = 12.0 * (np.outer(beta, beta) * 0.04 ** 2 + np.diag(idio.to_numpy()))
    np.testing.assert_allclose(covar.to_numpy(), expected, rtol=1e-10)


def test_regimes_depend_only_on_utils_perfstats_and_models():
    """The subpackage never imports plotting, the portfolio layer or the package root."""
    package = pathlib.Path(__file__).resolve().parents[1]
    forbidden = ('qis.plots', 'qis.portfolio', 'qis.market_data', 'matplotlib', 'seaborn')
    failures = []
    for path in sorted(package.glob('*.py')):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name == 'qis' or name.startswith(forbidden):
                    failures.append(f"{path.name}: {name}")
    assert not failures, failures

"""The Gaussian and Student-t nulls against numerical integration and their published constants."""

import numpy as np
import pytest
from scipy import integrate
from scipy.stats import norm
from scipy.stats import t as student_t

from qis.regimes import (
    calibrate_student_t_nu,
    compute_convexity_premium,
    compute_null_regime_contributions,
    compute_overlay_blend_frontier,
    compute_portfolio_bear_sharpe,
    compute_regime_kappa,
    compute_regime_null_loadings,
)


def _truncated_first_moment(a: float, b: float, nu: float = None) -> float:
    """E[Z 1{a < Z < b}] for a unit-variance normal or Student-t margin, by quadrature."""
    if nu is None:
        value, _ = integrate.quad(lambda z: z * norm.pdf(z), a, b)
        return value
    scale = np.sqrt(nu / (nu - 2.0))
    value, _ = integrate.quad(lambda z: z * student_t.pdf(z * scale, df=nu) * scale, a, b)
    return value


def _unit_variance_quantile(p: float, nu: float = None) -> float:
    """Quantile of the unit-variance margin."""
    if p in (0.0, 1.0):
        return -np.inf if p == 0.0 else np.inf
    return norm.ppf(p) if nu is None else student_t.ppf(p, df=nu) / np.sqrt(nu / (nu - 2.0))


def test_kappa_matches_the_published_constants():
    """Quarterly and monthly kappa of the one-sigma cut."""
    assert np.isclose(compute_regime_kappa(af=4.0), 0.4866, atol=1e-4)
    assert np.isclose(compute_regime_kappa(af=12.0), 0.8429, atol=1e-4)


@pytest.mark.parametrize('nu', [None, 5.0, 12.0])
@pytest.mark.parametrize('q', [[0.0, 0.16, 0.84, 1.0], [0.0, 0.1, 0.3, 0.7, 0.9, 1.0],
                               [0.0, 0.05, 0.5, 1.0]])
def test_loadings_equal_the_integrated_truncated_moments(q, nu):
    """k_s = sqrt(af) E[Z 1{bucket}], for symmetric and asymmetric partitions and both nulls."""
    af = 4.0
    loadings = compute_regime_null_loadings(af=af, q=q, nu=nu).to_numpy()
    edges = [_unit_variance_quantile(p, nu) for p in q]
    expected = [np.sqrt(af) * _truncated_first_moment(a, b, nu)
                for a, b in zip(edges[:-1], edges[1:])]
    np.testing.assert_allclose(loadings, expected, rtol=1e-7, atol=1e-9)
    assert abs(loadings.sum()) < 1e-12


def test_symmetric_partition_has_exactly_antisymmetric_loadings():
    """Mirrored buckets carry opposite loadings to the last bit, and the middle one zero."""
    loadings = compute_regime_null_loadings(af=12.0, q=[0.0, 0.16, 0.84, 1.0])
    assert loadings['Bear'] == -loadings['Bull']
    assert loadings['Normal'] == 0.0
    assert loadings['Bull'] == compute_regime_kappa(af=12.0)


def test_student_t_kappa_sits_below_at_one_sigma_above_in_far_tails_and_tends_to_the_gaussian():
    """The t margin lowers kappa at the one-sigma cut and raises it at 5%; large nu is Gaussian."""
    for tail_prob, below in ((0.16, True), (0.05, False)):
        gaussian = compute_regime_kappa(af=4.0, tail_prob=tail_prob)
        student = compute_regime_kappa(af=4.0, tail_prob=tail_prob, nu=5.0)
        assert (student < gaussian) == below
        assert np.isclose(compute_regime_kappa(af=4.0, tail_prob=tail_prob, nu=1e6), gaussian,
                          rtol=1e-5)


def test_null_contributions_sum_to_the_sharpe_ratio():
    """The null decomposes the Sharpe ratio exactly, for any partition."""
    for q in ([0.0, 0.16, 0.84, 1.0], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]):
        null = compute_null_regime_contributions(sr=0.63, rho=-0.4, af=4.0, q=q)
        assert np.isclose(null.sum(), 0.63, rtol=0.0, atol=1e-12)
    null = compute_null_regime_contributions(sr=0.5, rho=0.9, af=4.0)
    assert np.isclose(null['Bear'], 0.16 * 0.5 - 0.9 * compute_regime_kappa(af=4.0))


def test_convexity_premium_is_zero_at_the_null():
    """An asset whose Bear contribution equals its null has no premium."""
    bear = compute_null_regime_contributions(sr=0.5, rho=0.9, af=4.0)['Bear']
    assert np.isclose(compute_convexity_premium(sr_bear=bear, sr=0.5, rho=0.9, af=4.0), 0.0)


def test_calibrated_nu_matches_the_kurtosis_and_respects_the_floor():
    """nu = 4 + 6 / kurt, floored, and no t null for thin tails."""
    assert np.isclose(calibrate_student_t_nu(excess_kurtosis=6.0), 5.0)
    assert calibrate_student_t_nu(excess_kurtosis=100.0) == 4.5
    assert calibrate_student_t_nu(excess_kurtosis=-0.2) is None


def test_portfolio_bear_sharpe_follows_the_aggregation_identity():
    """Risk-weighted null values plus risk-weighted premia."""
    weights, vols = np.array([0.4, 0.6]), np.array([0.10, 0.15])
    srs, rhos, premia = np.array([0.50, 0.63]), np.array([1.0, 0.0]), np.array([0.0, 0.30])
    portfolio_vol = float(np.sqrt(np.sum(np.square(weights * vols))))
    risk_weights = weights * vols / portfolio_vol
    kappa = compute_regime_kappa(af=4.0)
    expected = (0.16 * np.sum(risk_weights * srs) - kappa * np.sum(risk_weights * rhos)
                + np.sum(risk_weights * premia))
    actual = compute_portfolio_bear_sharpe(weights=weights, vols=vols, srs=srs, rhos=rhos,
                                           premia=premia, portfolio_vol=portfolio_vol, af=4.0)
    assert np.isclose(actual, expected)


def test_blend_frontier_endpoints_and_the_uncorrelated_maximum():
    """The frontier runs from the benchmark to the overlay and peaks at hypot(SR_b, SR_a)."""
    frontier = compute_overlay_blend_frontier(sr_b=0.5, vol_b=0.10, sr_a=0.63, vol_a=0.15, rho=0.0,
                                              af=4.0)
    assert np.isclose(frontier['sharpe'].iloc[0], 0.5)
    assert np.isclose(frontier['sharpe'].iloc[-1], 0.63)
    assert np.isclose(frontier['sharpe'].max(), np.hypot(0.5, 0.63), atol=5e-3)


def test_invalid_inputs_raise():
    """Out-of-range inputs fail loudly."""
    with pytest.raises(ValueError):
        compute_regime_kappa(af=4.0, tail_prob=0.6)
    with pytest.raises(ValueError):
        compute_regime_null_loadings(af=4.0, nu=2.0)
    with pytest.raises(ValueError):
        compute_regime_null_loadings(af=4.0, q=[0.0, 0.5, 0.4, 1.0])
    with pytest.raises(ValueError):
        compute_null_regime_contributions(sr=0.5, rho=1.5, af=4.0)

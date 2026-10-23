"""Independent four-moment and input-contract checks for paired Sharpe HAC inference."""

from dataclasses import astuple, replace

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import toeplitz
from scipy.stats import norm

import qis
from qis.datasets.synthetic import generate_synthetic_prices
from qis.utils import regression


@pytest.fixture(scope='module')
def returns():
    prices = generate_synthetic_prices(end='2006-06-30', apply_quirks=False)
    return prices[['SEQ_US', 'SBD_TSY']].pct_change(fill_method=None).dropna()


def _moment_reference(values, lags, af):
    """Paper's four raw moments and dense Bartlett kernel, not the scalar influence code."""
    n = len(values)
    mu = np.mean(values, axis=0)
    gamma = np.mean(values ** 2, axis=0)
    sigma = np.sqrt(gamma - mu ** 2)
    gradient = np.array([gamma[0] / sigma[0] ** 3, -gamma[1] / sigma[1] ** 3,
                         -mu[0] / (2 * sigma[0] ** 3), mu[1] / (2 * sigma[1] ** 3)])
    moments = np.column_stack([values - mu, values ** 2 - gamma])
    kernel = toeplitz(np.maximum(1 - np.arange(n) / (lags + 1), 0))
    covariance = moments.T @ kernel @ moments / n
    se = np.sqrt(gradient @ covariance @ gradient / n)
    sharpes = mu / sigma
    diff = sharpes[0] - sharpes[1]
    return np.array([*(sharpes * np.sqrt(af)), diff * np.sqrt(af), se * np.sqrt(af),
                     diff / se, 2 * norm.sf(abs(diff / se))])


@pytest.mark.parametrize('lags', [0, 1, 3, 6, 50])
def test_matches_independent_moment_reference(returns, lags):
    out = qis.sharpe_difference_test(returns.iloc[:, 0], returns.iloc[:, 1], lags, af=260)
    np.testing.assert_allclose(astuple(out)[:6], _moment_reference(returns.values, lags, 260),
                               rtol=2e-12, atol=1e-13)
    assert out.n_obs == len(returns)
    assert out.n_lags == lags


def test_sharpe_inference_consumes_qis_hac_estimator(returns, monkeypatch):
    """Shared QIS HAC changes must flow through to the Sharpe standard error."""
    r1, r2 = returns.iloc[:, 0], returns.iloc[:, 1]
    baseline = qis.sharpe_difference_test(r1, r2, n_lags=3)
    estimate_hac_mean = regression.estimate_hac_mean
    calls = []

    def doubled_standard_error(y, hac_lags=3, confidence_level=0.95):
        calls.append((len(y), hac_lags))
        result = estimate_hac_mean(y, hac_lags=hac_lags, confidence_level=confidence_level)
        return replace(result, hac_se=2 * result.hac_se)

    monkeypatch.setattr(regression, 'estimate_hac_mean', doubled_standard_error)
    actual = qis.sharpe_difference_test(r1, r2, n_lags=3)
    assert calls == [(len(returns), 3)]
    assert actual.diff_an == baseline.diff_an
    assert actual.se_an == pytest.approx(2 * baseline.se_an)
    assert actual.t_stat == pytest.approx(baseline.t_stat / 2)
    assert actual.p_value == pytest.approx(2 * norm.sf(abs(baseline.t_stat / 2)))


def test_hac_accounts_for_serial_dependence(returns):
    # Deliberately smooth the frozen synthetic observations to expose a missing-HAC regression.
    paired = returns.ewm(alpha=0.10, adjust=False).mean()
    independent = qis.sharpe_difference_test(paired.iloc[:, 0], paired.iloc[:, 1], 0)
    dependent = qis.sharpe_difference_test(paired.iloc[:, 0], paired.iloc[:, 1], 20)
    assert dependent.se_an > 2 * independent.se_an
    np.testing.assert_allclose(astuple(dependent)[:6],
                               _moment_reference(paired.values, 20, 12), rtol=2e-12)


@pytest.mark.parametrize('scale1,scale2', [(2.0, 0.1), (1e-180, 1e180)])
def test_positive_leverage_and_annualisation_invariance(returns, scale1, scale2):
    r1, r2 = returns.iloc[:, 0], returns.iloc[:, 1]
    base = qis.sharpe_difference_test(r1, r2, af=1)
    out = qis.sharpe_difference_test(scale1 * r1, scale2 * r2, af=260)
    np.testing.assert_allclose(np.array(astuple(out)[:4]) / np.sqrt(260),
                               astuple(base)[:4], rtol=1e-12)
    np.testing.assert_allclose(astuple(out)[4:], astuple(base)[4:], rtol=1e-12)


def test_swapping_series_changes_only_sign_and_order(returns):
    r1, r2 = returns.iloc[:, 0], returns.iloc[:, 1]
    a = qis.sharpe_difference_test(r1, r2)
    b = qis.sharpe_difference_test(r2, r1)
    np.testing.assert_allclose(astuple(b)[:6],
                               [a.sr2_an, a.sr1_an, -a.diff_an, a.se_an, -a.t_stat, a.p_value])


def test_default_three_lags_and_explicit_automatic_option(returns):
    """Omitting the lag argument uses three; explicit None still selects the sample rule."""
    r1, r2 = returns.iloc[:, 0], returns.iloc[:, 1]
    default = qis.sharpe_difference_test(r1, r2)
    assert default.n_lags == 3
    assert default == qis.sharpe_difference_test(r1, r2, n_lags=3)
    automatic = qis.sharpe_difference_test(r1, r2, n_lags=None)
    expected_lags = int(np.floor(4 * (len(returns) / 100) ** (2 / 9)))
    assert expected_lags != 3
    assert automatic.n_lags == expected_lags
    np.testing.assert_allclose(astuple(automatic)[:6],
                               _moment_reference(returns.values, expected_lags, 12), rtol=2e-12)


def test_alignment_nan_handling_and_automatic_lags(returns):
    r1 = returns.iloc[:-10, 0].astype('Float64')
    r2 = returns.iloc[7:, 1].copy()
    r1.iloc[20] = pd.NA
    r2.iloc[30] = np.nan
    complete = pd.concat([r1, r2], axis=1).dropna()
    out = qis.sharpe_difference_test(r1.iloc[::-1], r2.iloc[::-1], n_lags=None)
    expected = qis.sharpe_difference_test(complete.iloc[:, 0], complete.iloc[:, 1], n_lags=None)
    assert out == expected
    assert out.n_obs == len(complete)
    assert out.n_lags == int(np.floor(4 * (len(complete) / 100) ** (2 / 9)))


@pytest.mark.parametrize('scale', [1.0, 3.0])
def test_identical_and_proportional_returns_are_neutral(returns, scale):
    out = qis.sharpe_difference_test(returns.iloc[:, 0], scale * returns.iloc[:, 0])
    assert (out.diff_an, out.se_an, out.t_stat, out.p_value) == (0.0, 0.0, 0.0, 1.0)


def test_normal_tail_does_not_cancel_to_zero(returns):
    r1 = returns.iloc[:, 0]
    out = qis.sharpe_difference_test(r1 + 0.01, r1)
    assert out.t_stat > 8
    assert 0 < out.p_value < 1e-15


def test_nav_conversion_uses_simple_returns_and_no_initial_zero(returns):
    navs = pd.concat([pd.DataFrame([[1.0, 1.0]], columns=returns.columns,
                                  index=[returns.index[0] - pd.offsets.BDay()]),
                      (1 + returns).cumprod()])
    recovered = qis.to_returns(navs, is_log_returns=False, drop_first=True)
    out = qis.sharpe_difference_test(recovered.iloc[:, 0], recovered.iloc[:, 1], af=260)
    expected = qis.sharpe_difference_test(returns.iloc[:, 0], returns.iloc[:, 1], af=260)
    np.testing.assert_allclose(astuple(out), astuple(expected), rtol=1e-11)
    assert out.n_obs == len(navs) - 1


@pytest.mark.parametrize('af', [0, -1, np.nan, np.inf, True, '12'])
def test_invalid_annualisation(returns, af):
    with pytest.raises(ValueError, match='af'):
        qis.sharpe_difference_test(returns.iloc[:, 0], returns.iloc[:, 1], af=af)


@pytest.mark.parametrize('lags', [-1, 0.5, True, np.nan, 10000])
def test_invalid_lags(returns, lags):
    with pytest.raises(ValueError, match='n_lags'):
        qis.sharpe_difference_test(returns.iloc[:, 0], returns.iloc[:, 1], n_lags=lags)


@pytest.mark.parametrize('bad', ['short', 'constant', 'zero', 'infinity', 'duplicate',
                                'missing_index', 'string', 'complex', 'array', 'disjoint'])
def test_invalid_inputs(returns, bad):
    r1, r2 = returns.iloc[:, 0].copy(), returns.iloc[:, 1].copy()
    if bad == 'short':
        r1 = r1.iloc[:23]
    elif bad == 'constant':
        r1[:] = 0.01
    elif bad == 'zero':
        r1[:] = 0.0
    elif bad == 'infinity':
        r1.iloc[5] = np.inf
    elif bad == 'duplicate':
        r1 = pd.concat([r1, r1.iloc[:1]])
    elif bad == 'missing_index':
        r1.index = [pd.NaT] + list(r1.index[1:])
    elif bad == 'string':
        r1 = r1.astype(str)
    elif bad == 'complex':
        r1 = r1.astype(complex)
    elif bad == 'array':
        r1 = r1.values
    elif bad == 'disjoint':
        r1.index = r1.index + pd.DateOffset(years=20)
    with pytest.raises(ValueError):
        qis.sharpe_difference_test(r1, r2)

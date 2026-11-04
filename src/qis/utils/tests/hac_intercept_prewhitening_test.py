"""Deterministic references for the AR(1)-prewhitened intercept standard errors.

The dense SVD and D'KD references are built differently from the QR/lag-sum production
code, so a formula defect in either fails here. No RNG or market data is used.
"""

# packages
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

# qis
from qis.utils import hac_intercept_prewhitening as prewhitening
from qis.utils import regression
from qis.utils.hac_intercept_prewhitening import estimate_prewhitened_intercepts


def _panel(nobs=80, factor_count=3):
    t = np.arange(nobs, dtype=float)
    index = pd.date_range('2000-01-31', periods=nobs, freq='ME')
    factors = np.column_stack([
        .013*np.cos((j+1)*.137*t) + .007*np.sin((j+2)*.211*t) + .004*(j+1)
        for j in range(factor_count)
    ]) if factor_count else np.empty((nobs, 0))
    errors = np.column_stack([
        .007*np.cos((.31+.8*j)*t+.1) + .004*np.sin((1.3+.4*j)*t)
        + .002*np.cos(.13*t*t+j) for j in range(3)
    ])
    loadings = .2+.07*np.arange(factor_count*3).reshape(factor_count, 3)
    y = pd.DataFrame(errors + factors @ loadings + [.001, .002, .003],
                     index=index, columns=['A', 'B', 'C'])
    x = None if not factor_count else pd.DataFrame(
        factors, index=index, columns=[f'F{j}' for j in range(factor_count)],
    )
    return y, x


def _dense(y, x, lags):
    """Full-design SVD, dense filter operator and dense kernel matrix reference."""
    values = y.to_numpy()
    design = (np.ones((len(y), 1)) if x is None else
              np.column_stack((np.ones(len(y)), x.to_numpy())))
    left, singular, right = np.linalg.svd(design, full_matrices=False)
    coefficient_map = (right.T / singular) @ left.T
    coefficients = coefficient_map @ values
    influence = len(y)*coefficient_map[0, :, None]*(values-design @ coefficients)
    difference = np.eye(len(y)-1, len(y), k=1)
    lagged = np.eye(len(y)-1, len(y))
    distance = np.abs(np.subtract.outer(np.arange(len(y)-1), np.arange(len(y)-1)))
    kernel = np.maximum(1.0-distance/(lags+1.0), 0.0)
    variances, rhos, innovations = [], [], []
    for column in influence.T:
        # Independent scalar least-squares fit instead of the production dot-ratio expression.
        rho = np.linalg.lstsq(column[:-1, None], column[1:], rcond=None)[0][0]
        transformed = difference-rho*lagged
        variance = (len(y)/(len(y)-design.shape[1]) / len(y)**2 / (1-rho)**2
                    * column @ transformed.T @ kernel @ transformed @ column)
        variances.append(variance)
        rhos.append(rho)
        innovations.append(transformed @ column)
    return coefficients[0], influence, np.array(variances), np.array(rhos), np.array(innovations).T


@pytest.mark.parametrize('factor_count', [0, 1, 3, 6])
@pytest.mark.parametrize('lags', [0, 3, 27])
def test_dense_svd_reference(factor_count, lags):
    y, x = _panel(factor_count=factor_count)
    estimate, influence, variance, rho, innovations = _dense(y, x, lags)
    result = estimate_prewhitened_intercepts(y, x, hac_lags=lags)
    expected_se = np.sqrt(variance)
    expected_z = estimate/expected_se
    np.testing.assert_allclose(result.table.estimate, estimate, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(result.table.standard_error, expected_se, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(result.table.statistic, expected_z, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(result.table.pvalue_normal, 2*norm.sf(np.abs(expected_z)),
                               rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(result.table.ci_lower_pointwise,
                               estimate-norm.ppf(.975)*expected_se, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(result.table.ci_upper_pointwise,
                               estimate+norm.ppf(.975)*expected_se, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(result.table.rho, rho, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(result.influence, influence, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(result.innovations, innovations, rtol=1e-9, atol=1e-12)
    assert result.influence.index.equals(y.index)
    assert result.innovations.index.equals(y.index[1:])
    assert (result.nobs, result.n_parameters, result.hac_lags) == (len(y), factor_count+1, lags)


@pytest.mark.parametrize('factor_count', [0, 1, 3, 6])
@pytest.mark.parametrize('lags', [0, 3, 27])
def test_unfiltered_influence_equals_canonical_sandwich(factor_count, lags):
    y, x = _panel(factor_count=factor_count)
    result = estimate_prewhitened_intercepts(y, x, hac_lags=lags)
    canonical = regression.estimate_hac_regression(y, x, span=None, hac_lags=lags)
    nobs = len(y)
    distance = np.abs(np.subtract.outer(np.arange(nobs), np.arange(nobs)))
    kernel = np.maximum(1.0-distance/(lags+1.0), 0.0)
    psi = result.influence.to_numpy()
    unfiltered = nobs/(nobs-(1 if x is None else 1+x.shape[1])) / nobs**2 * np.diag(
        psi.T @ kernel @ psi,
    )
    np.testing.assert_allclose(unfiltered, canonical.hac_se.loc['Intercept']**2,
                               rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(result.table.estimate, canonical.coefficients.loc['Intercept'],
                               rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(psi.sum(axis=0), 0.0, atol=1e-9)


def test_canonical_fit_is_used_once_on_unit_scaled_inputs(monkeypatch):
    y, x = _panel()
    canonical = regression.estimate_hac_regression
    calls = []

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return canonical(*args, **kwargs)

    monkeypatch.setattr(regression, 'estimate_hac_regression', spy)
    estimate_prewhitened_intercepts(y, x, hac_lags=3)
    assert len(calls) == 1 and calls[0]['span'] is None and calls[0]['hac_lags'] == 3
    np.testing.assert_array_equal(calls[0]['y'].abs().max(), np.ones(y.shape[1]))
    np.testing.assert_array_equal(calls[0]['x'].abs().max(), np.ones(x.shape[1]))


def test_confidence_level_sets_pointwise_intervals():
    y, x = _panel()
    result = estimate_prewhitened_intercepts(y, x, confidence_level=.9)
    se = result.table.standard_error
    z = result.table.estimate / se
    np.testing.assert_allclose(result.table.statistic, z, rtol=1e-12)
    np.testing.assert_allclose(result.table.pvalue_normal, 2*norm.sf(np.abs(z)), rtol=1e-12)
    np.testing.assert_allclose(result.table.ci_lower_pointwise,
                               result.table.estimate-norm.ppf(.95)*se, rtol=1e-12)
    np.testing.assert_allclose(result.table.ci_upper_pointwise,
                               result.table.estimate+norm.ppf(.95)*se, rtol=1e-12)
    assert result.confidence_level == .9


@pytest.mark.parametrize('factor_count', [0, 3])
def test_family_order_duplicates_subsets_and_shifts(factor_count):
    y, x = _panel(factor_count=factor_count)
    base = estimate_prewhitened_intercepts(y, x)
    expanded = y.assign(copy=y['A'])
    result = estimate_prewhitened_intercepts(expanded, x)
    np.testing.assert_allclose(result.table.loc['A'], result.table.loc['copy'], rtol=1e-12)
    np.testing.assert_allclose(result.table.loc[y.columns], base.table, rtol=1e-9, atol=1e-12)
    subset = estimate_prewhitened_intercepts(y[['C', 'A']], x)
    np.testing.assert_allclose(subset.table, base.table.loc[['C', 'A']], rtol=1e-9, atol=1e-12)
    shifted = estimate_prewhitened_intercepts(y + [.07, -.02, .05], x)
    np.testing.assert_allclose(shifted.table.estimate, base.table.estimate+[.07, -.02, .05],
                               rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(shifted.table.standard_error, base.table.standard_error,
                               rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(shifted.table.rho, base.table.rho, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize('scale', [1e-150, -3., 12., 1e150])
def test_response_scale_and_sign_invariance(scale):
    y, x = _panel()
    base = estimate_prewhitened_intercepts(y, x)
    changed = estimate_prewhitened_intercepts(y*scale, x)
    np.testing.assert_allclose(changed.table.estimate/scale, base.table.estimate,
                               rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(changed.table.standard_error/abs(scale), base.table.standard_error,
                               rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(changed.table.rho, base.table.rho, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(changed.table.pvalue_normal, base.table.pvalue_normal,
                               rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize('factor_count', [0, 3])
def test_maximum_valid_lag_and_numpy_integer_setting(factor_count):
    y, x = _panel(factor_count=factor_count)
    lags = np.int64(len(y)-2)
    result = estimate_prewhitened_intercepts(y, x, hac_lags=lags)
    expected = _dense(y, x, lags)
    np.testing.assert_allclose(result.table.standard_error, np.sqrt(expected[2]),
                               rtol=1e-9, atol=1e-12)
    assert result.hac_lags == len(y)-2


def test_zero_fitted_rho_still_drops_first_observation():
    # psi = (1, 0, -1): rho = 0, innovations (0, -1), variance = 3/2 * 1 / 9 = 1/6.
    y = pd.Series([1., 0., -1.], name='zero_rho')
    result = estimate_prewhitened_intercepts(y, hac_lags=0)
    assert result.table.loc['zero_rho', 'rho'] == 0.
    np.testing.assert_allclose(result.table.standard_error, np.sqrt(1/6), rtol=1e-12)
    np.testing.assert_allclose(result.innovations.to_numpy().ravel(), [0., -1.],
                               rtol=1e-12, atol=1e-12)
    unfiltered = regression.estimate_hac_regression(y, span=None, hac_lags=0)
    np.testing.assert_allclose(unfiltered.hac_se.loc['Intercept'], np.sqrt(1/3), rtol=1e-12)


def test_factor_basis_preserves_intercept_functional_not_translations():
    y, x = _panel()
    base = estimate_prewhitened_intercepts(y, x)
    transform = np.array([[100., .3, -.1], [0., -.02, .5], [0., 0., 2.]])
    rotated = pd.DataFrame(x.to_numpy() @ transform, index=x.index, columns=x.columns[::-1])
    result = estimate_prewhitened_intercepts(y, rotated)
    np.testing.assert_allclose(result.table, base.table, rtol=1e-9, atol=1e-12)
    shift = np.array([.01, -.02, .015])
    translated = estimate_prewhitened_intercepts(y, x+shift)
    fit = regression.estimate_hac_regression(y, x, span=None)
    expected = base.table.estimate.to_numpy()-shift @ fit.coefficients.iloc[1:].to_numpy()
    np.testing.assert_allclose(translated.table.estimate, expected, rtol=1e-9, atol=1e-12)
    assert not np.allclose(translated.table.estimate, base.table.estimate)


def test_series_and_nullable_inputs_without_mutation():
    y, x = _panel(factor_count=1)
    y0, x0 = y.copy(deep=True), x.copy(deep=True)
    result = estimate_prewhitened_intercepts(y.astype('Float64'), x.astype('Float64'))
    single = estimate_prewhitened_intercepts(y['A'], x.iloc[:, 0])
    np.testing.assert_allclose(result.table.loc['A'], single.table.loc['A'],
                               rtol=1e-9, atol=1e-12)
    pd.testing.assert_frame_equal(y, y0)
    pd.testing.assert_frame_equal(x, x0)


@pytest.mark.parametrize('lags', [-1, True, 1.5, None, 79])
def test_invalid_lags(lags):
    y, x = _panel()
    with pytest.raises(ValueError, match='hac_lags'):
        estimate_prewhitened_intercepts(y, x, hac_lags=lags)


@pytest.mark.parametrize('confidence', [0, 1, True, np.nan, np.inf, '0.95'])
def test_invalid_confidence(confidence):
    y, x = _panel()
    with pytest.raises(ValueError, match='confidence_level'):
        estimate_prewhitened_intercepts(y, x, confidence_level=confidence)


@pytest.mark.parametrize('defect', ['nan', 'inf', 'complex', 'duplicate', 'order', 'dates',
                                    'rank', 'zero_factor', 'constant', 'zero', 'empty', 'text'])
def test_invalid_complete_sample_and_identification(defect):
    y, x = _panel()
    if defect in ('nan', 'inf'):
        y.iloc[3, 0] = np.nan if defect == 'nan' else np.inf
    elif defect == 'complex':
        y = y.astype(complex)+1j
    elif defect == 'duplicate':
        y.columns = ['A', 'A', 'C']
    elif defect == 'order':
        y, x = y.iloc[::-1], x.iloc[::-1]
    elif defect == 'dates':
        x = x.iloc[::-1]
    elif defect == 'rank':
        x['copy'] = x['F0']
    elif defect == 'zero_factor':
        x['F0'] = 0.
    elif defect in ('constant', 'zero'):
        y['A'] = 1. if defect == 'constant' else 0.
    elif defect == 'empty':
        y = y.iloc[:, :0]
    else:
        y['A'] = 'not a return'
    with pytest.raises(ValueError):
        estimate_prewhitened_intercepts(y, x)


@pytest.mark.parametrize('tail', [[1., 2.], [1., -3.]])
def test_unstable_filter_aborts_without_fallback(tail):
    values = np.r_[np.full(48, -sum(tail)/48), tail]
    y = pd.DataFrame({'unstable': values})
    regression.estimate_hac_regression(y, span=None, hac_lags=0)
    with pytest.raises(ValueError, match="unstable.*responses=.*unstable"):
        estimate_prewhitened_intercepts(y, hac_lags=0)


def test_low_level_degeneracy_and_nonpositive_variance():
    columns = pd.Index(['edge'])
    for values, message in [(np.zeros(6), 'zero influence'),
                            (np.r_[np.zeros(5), 1.], 'AR denominator'),
                            (np.array([1., 0., 0., 0., 0., 0.]), 'nonpositive')]:
        with pytest.raises(ValueError, match=message):
            prewhitening._ar1_bartlett(values[:, None], 1, 0, columns)


def test_unrepresentable_output_names_the_response():
    t = np.arange(40, dtype=float)
    factor = np.linspace(1., 2., 40)
    y = pd.DataFrame({'too_large': (
        2.5-factor+.02*np.sin(1.3*t)+.005*np.cos(.2*t*t))*1e308})
    x = pd.DataFrame({'F': factor})
    with pytest.raises(ValueError, match='unrepresentable.*too_large'):
        estimate_prewhitened_intercepts(y, x)

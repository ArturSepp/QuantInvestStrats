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
    compute_regime_mixture_covar_from_sample,
    compute_sample_regime_moments,
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


@pytest.mark.parametrize('dtype', ['float64', 'Float64', 'float32', 'Float32'])
@pytest.mark.parametrize('levels', [[-.03125, 0., .03125], [-.1, .1, .3]],
                         ids=['binary-constants', 'decimal-mean-roundoff'])
@pytest.mark.filterwarnings('error')
def test_compute_regime_ewm_betas_returns_nan_for_constant_groups(dtype, levels):
    """Occupied quantile buckets need not identify a slope within any bucket."""
    benchmark = np.repeat(levels, [5, 20, 5])
    panel = pd.DataFrame({'BM': benchmark, 'A': .5 * benchmark + np.arange(30) / 1000},
                         index=pd.date_range('2020-01-31', periods=30, freq='ME')).astype(dtype)
    sampled = create_sampled_returns_with_regime_id(panel, benchmark='BM')
    before = sampled.copy(deep=True)
    assert sampled.groupby('regime', observed=True)['BM'].nunique().tolist() == [1, 1, 1]
    betas, idio_vars = compute_regime_ewm_betas(sampled, benchmark='BM')
    pd.testing.assert_frame_equal(
        betas, pd.DataFrame(np.nan, index=['A'], columns=['Bear', 'Normal', 'Bull']))
    pd.testing.assert_series_equal(idio_vars, pd.Series(np.nan, index=['A']))
    pd.testing.assert_frame_equal(sampled, before)


def _regime_stream_weights(size, span):
    """Closed-form weights include the stream-mean seed and every subsequent update."""
    decay = 1. - 2. / (span + 1.)
    return decay ** size / size + (1. - decay) * decay ** np.arange(size - 1, -1, -1)


@pytest.mark.parametrize('dtype', ['float64', 'Float64', 'float32', 'Float32'])
@pytest.mark.parametrize('regime_column', ['regime', 'state'])
@pytest.mark.parametrize('single_observation', [False, True])
@pytest.mark.filterwarnings('error')
def test_compute_regime_ewm_betas_preserves_identified_neighbors(
        dtype, regime_column, single_observation):
    """One unidentified group must not erase healthy slopes or their pooled residual variance."""
    sampled = pd.DataFrame({'BM': [.1, 0., .03125, .0625, .1, .125, .1875],
                            'A': [.2, .003, .061, .128, .201, .254, .373],
                            'Cash': [0.] * 7,
                            'Ragged': [np.nan, 0., .015625, .03125, np.nan, .0625, .09375]},
                           index=pd.date_range('2020-01-31', periods=7, freq='ME')).astype(dtype)
    sampled[regime_column] = pd.Categorical(
        ['Bear', 'Normal', 'Normal', 'Normal', 'Bear', 'Bull', 'Bull'],
        categories=['Bull', 'Bear', 'Normal'], ordered=True)
    if single_observation:
        sampled = sampled.drop(sampled.index[4])
    before = sampled.copy(deep=True)
    assets = ['A', 'Cash', 'Ragged']
    expected = pd.DataFrame(np.nan, index=assets, columns=['Bull', 'Bear', 'Normal'])
    residuals = pd.DataFrame(np.nan, index=sampled.index, columns=assets)
    for regime in ['Normal', 'Bull']:
        block = sampled.loc[sampled[regime_column] == regime, ['BM', *assets]].astype(float)
        weights = _regime_stream_weights(len(block), 40.)
        x = block['BM'].to_numpy()
        xm = weights @ x
        for asset in assets:
            y = block[asset].to_numpy()
            ym = weights @ y
            beta = (weights @ ((x - xm) * (y - ym))) / (weights @ ((x - xm) ** 2))
            expected.loc[asset, regime] = beta
            residuals.loc[block.index, asset] = y - (ym - beta * xm) - beta * x
    expected_vars = residuals.apply(
        lambda r: _regime_stream_weights(r.notna().sum(), 40.) @ r.dropna().to_numpy() ** 2)
    actual, idio_vars = compute_regime_ewm_betas(sampled, 'BM', regime_column=regime_column)
    tolerance = 1e-5 if '32' in dtype else 1e-11
    pd.testing.assert_frame_equal(actual, expected, rtol=tolerance, atol=1e-12)
    pd.testing.assert_series_equal(idio_vars, expected_vars, rtol=tolerance, atol=1e-12)
    pd.testing.assert_frame_equal(sampled, before)


@pytest.mark.parametrize('dtype', ['float64', 'Float64', 'float32', 'Float32'])
@pytest.mark.parametrize('span', [2., 40., 1e9])
@pytest.mark.filterwarnings('error')
def test_compute_regime_ewm_betas_preserves_small_positive_variance(dtype, span):
    """Two distinct nearby returns identify beta 2; a numerical tolerance must not reject them."""
    spacing = 1e-5 if '32' in dtype else 1e-8
    sampled = pd.DataFrame({'BM': [.1, .1 + spacing] * 6}).astype(dtype)
    sampled['A'] = 2. * sampled['BM']
    sampled['regime'] = pd.Categorical(['Low'] * 6 + ['High'] * 6,
                                        categories=['Low', 'High'], ordered=True)
    betas, idio_vars = compute_regime_ewm_betas(sampled, 'BM', span=span)
    pd.testing.assert_frame_equal(betas, pd.DataFrame([[2., 2.]],
                                                     index=['A'], columns=['Low', 'High']))
    np.testing.assert_allclose(idio_vars, 0., atol=1e-28)


@pytest.mark.filterwarnings('error')
def test_compute_regime_ewm_betas_handles_zero_variance_at_span_one(monthly):
    """Span one centres on the final observation, whose instantaneous covariance is zero."""
    sampled = create_sampled_returns_with_regime_id(monthly, benchmark='BM')
    betas, idio_vars = compute_regime_ewm_betas(sampled, 'BM', span=1.)
    pd.testing.assert_frame_equal(
        betas, pd.DataFrame(np.nan, index=['A', 'B', 'C'], columns=['Bear', 'Normal', 'Bull']))
    pd.testing.assert_series_equal(idio_vars, pd.Series(np.nan, index=['A', 'B', 'C']))


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


@pytest.fixture
def mixture_inputs():
    """Two assets with a known single-factor covariance and one zero residual variance."""
    regimes = pd.Index(['Bear', 'Normal', 'Bull'], name='regime')
    assets = pd.Index(['BM', 'A'], name='asset')
    return dict(
        betas=pd.DataFrame([[1., 1., 1.], [.5, .5, .5]], index=assets, columns=regimes),
        idio_vars=pd.Series([0., .0004], index=assets),
        benchmark_regime_means=pd.Series([0., 0., 0.], index=regimes),
        benchmark_regime_second_moments=pd.Series([.0016, .0016, .0016], index=regimes),
        regime_probs=get_regime_probabilities(), af=12.,
    )


@pytest.mark.parametrize('change', ['missing', 'negative', 'nan', 'infinite'])
def test_compute_regime_mixture_covar_rejects_invalid_residual_variances(mixture_inputs, change):
    """An absent variance is not zero, and an impossible variance cannot enter the diagonal."""
    if change == 'missing':
        mixture_inputs['idio_vars'] = mixture_inputs['idio_vars'].drop('A')
    else:
        invalid = dict(negative=-1., nan=np.nan, infinite=np.inf)
        mixture_inputs['idio_vars'].loc['A'] = invalid[change]
    with pytest.raises(ValueError, match='idio_vars'):
        compute_regime_mixture_covar(**mixture_inputs)


@pytest.mark.parametrize('field', ['idio_vars', 'benchmark_regime_means',
                                 'benchmark_regime_second_moments', 'regime_probs'])
@pytest.mark.parametrize('change', ['missing', 'extra', 'duplicate'])
def test_compute_regime_mixture_covar_requires_exact_unique_labels(mixture_inputs, field, change):
    """Label-based arithmetic must not drop a supplied component or choose duplicate entries."""
    values = mixture_inputs[field]
    if change == 'missing':
        values = values.iloc[:-1]
    elif change == 'extra':
        values = pd.concat([values, pd.Series([0.], index=['extra'])])
    else:
        values = pd.concat([values, values.iloc[:1]])
    mixture_inputs[field] = values
    with pytest.raises(ValueError, match=field):
        compute_regime_mixture_covar(**mixture_inputs)


@pytest.mark.parametrize('field', ['betas', 'idio_vars', 'benchmark_regime_means',
                                 'benchmark_regime_second_moments', 'regime_probs'])
@pytest.mark.parametrize('dtype', ['float64', 'Float64'])
@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_compute_regime_mixture_covar_rejects_nonfinite_before_arithmetic(
        mixture_inputs, monkeypatch, field, dtype, value):
    """Nullable missing scalars must not bypass the same preflight as ordinary NaN."""
    mixture_inputs[field] = mixture_inputs[field].astype(dtype)
    values = mixture_inputs[field]
    if field == 'betas':
        values.iloc[-1, -1] = value
    else:
        values.iloc[-1] = value
    before = values.copy(deep=True)

    def unexpected_outer(*args, **kwargs):
        pytest.fail('invalid components must fail before factor arithmetic')

    monkeypatch.setattr(np, 'outer', unexpected_outer)
    with pytest.raises(ValueError, match=field):
        compute_regime_mixture_covar(**mixture_inputs)
    if field == 'betas':
        pd.testing.assert_frame_equal(values, before)
    else:
        pd.testing.assert_series_equal(values, before)


@pytest.mark.parametrize('axis', ['index', 'columns'])
def test_compute_regime_mixture_covar_rejects_duplicate_beta_labels(mixture_inputs, axis):
    """A covariance needs one loading per asset and regime."""
    betas = mixture_inputs['betas']
    if axis == 'index':
        betas = pd.concat([betas, betas.iloc[:1]])
    else:
        betas = pd.concat([betas, betas.iloc[:, :1]], axis=1)
    mixture_inputs['betas'] = betas
    with pytest.raises(ValueError, match='betas'):
        compute_regime_mixture_covar(**mixture_inputs)


@pytest.mark.parametrize('af', [0., -1., np.nan, np.inf, None, pd.NA, 1j, [12.]])
def test_compute_regime_mixture_covar_rejects_invalid_annualisation(mixture_inputs, af):
    """Annualisation must preserve the variance domain."""
    mixture_inputs['af'] = af
    with pytest.raises(ValueError, match='af'):
        compute_regime_mixture_covar(**mixture_inputs)


@pytest.mark.parametrize('probabilities', [[-.1, .7, .4], [1.1, 0., -.1],
                                          [0., 0., 0.], [.1, .6, .2]])
def test_compute_regime_mixture_covar_rejects_invalid_probabilities(mixture_inputs, probabilities):
    """Weights in total covariance are probabilities, not arbitrary factor weights."""
    mixture_inputs['regime_probs'] = pd.Series(probabilities, index=mixture_inputs['betas'].columns)
    with pytest.raises(ValueError, match='regime_probs'):
        compute_regime_mixture_covar(**mixture_inputs)


@pytest.mark.parametrize('mean,second', [(0., -.001), (.1, .001), (1e200, 1e300)])
def test_compute_regime_mixture_covar_rejects_inconsistent_moments(mixture_inputs, mean, second):
    """A conditional second moment cannot be materially smaller than the squared mean."""
    mixture_inputs['benchmark_regime_means'].iloc[0] = mean
    mixture_inputs['benchmark_regime_second_moments'].iloc[0] = second
    with pytest.raises(ValueError, match='benchmark_regime_second_moments'):
        compute_regime_mixture_covar(**mixture_inputs)


@pytest.mark.parametrize('dtype', ['float64', 'Float64', 'float32', 'Float32'])
@pytest.mark.filterwarnings('error')
def test_compute_regime_mixture_covar_preserves_reordered_zero_boundaries(mixture_inputs, dtype):
    """Zero residuals and zero-probability regimes remain valid across labeled representations."""
    mixture_inputs['idio_vars'].iloc[-1] = 0.
    mixture_inputs['regime_probs'] = pd.Series([0., 0., 1.],
                                              index=mixture_inputs['betas'].columns)
    for field in ('betas', 'idio_vars', 'benchmark_regime_means',
                  'benchmark_regime_second_moments', 'regime_probs'):
        mixture_inputs[field] = mixture_inputs[field].astype(dtype)
    originals = {field: value.copy(deep=True) for field, value in mixture_inputs.items()
                 if isinstance(value, (pd.Series, pd.DataFrame))}
    actual = compute_regime_mixture_covar(**mixture_inputs)
    expected = pd.DataFrame([[.0192, .0096], [.0096, .0048]],
                            index=mixture_inputs['betas'].index,
                            columns=mixture_inputs['betas'].index,
                            dtype='float32' if '32' in dtype else 'float64')
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-6 if '32' in dtype else 1e-12)
    reordered = {field: value.iloc[::-1] if isinstance(value, pd.Series) else value
                 for field, value in mixture_inputs.items()}
    reordered['betas'] = reordered['betas'].iloc[:, ::-1]
    pd.testing.assert_frame_equal(compute_regime_mixture_covar(**reordered), actual)
    np.testing.assert_allclose(actual, actual.T, rtol=0., atol=0.)
    assert np.linalg.eigvalsh(actual).min() >= -1e-9
    actual.iloc[0, 0] = 99.
    for field, before in originals.items():
        if field == 'betas':
            pd.testing.assert_frame_equal(mixture_inputs[field], before)
        else:
            pd.testing.assert_series_equal(mixture_inputs[field], before)


def test_compute_regime_mixture_covar_matches_centered_observation_covariance():
    """Explicit observations provide an oracle independent of raw-moment subtraction."""
    betas = pd.DataFrame([[1., 1.], [-.5, 2.]], index=['BM', 'A'], columns=['Low', 'High'])
    observations = np.array([[-.04, .02], [-.02, .01], [.01, .02], [.03, .06]])
    weights = np.array([.125, .125, .375, .375])
    centered = observations - weights @ observations
    expected = centered.T @ (weights[:, None] * centered) * 12.
    expected[1, 1] += .0012
    actual = compute_regime_mixture_covar(
        betas, pd.Series([0., .0001], index=betas.index),
        pd.Series([-.03, .02], index=betas.columns),
        pd.Series([.001, .0005], index=betas.columns), 12.,
        pd.Series([.25, .75], index=betas.columns))
    pd.testing.assert_frame_equal(actual, pd.DataFrame(expected, index=betas.index,
                                                     columns=betas.index), rtol=1e-12)


@pytest.mark.parametrize('dtype', ['float64', 'Float64', 'float32', 'Float32'])
@pytest.mark.parametrize('frozen', [False, True])
@pytest.mark.filterwarnings('error')
def test_compute_regime_mixture_covar_from_sample_preserves_constant_roundoff(dtype, frozen):
    """Frozen and benchmark-only dispatch do not require positive conditional variance."""
    sample = pd.DataFrame({'BM': pd.Series([.1] * 6, dtype=dtype),
                           'regime': ['Low'] * 3 + ['High'] * 3})
    sheet = None
    if frozen:
        sample['A'] = .5 * sample['BM']
        sheet = pd.DataFrame({'beta_low': [.5], 'beta_high': [.5], 'idio_vol': [0.]}, index=['A'])
    before = sample.copy(deep=True)
    moments = compute_sample_regime_moments(sample, 'BM')
    actual = compute_regime_mixture_covar_from_sample(sample, 'BM', 12., betas=sheet)
    # Zero is the independent oracle; exact native roundoff is retained, not projected away.
    expected_variance = 12. * (float(moments['second_moment'].mean())
                               - float(moments['mean'].mean()) ** 2)
    loading = np.array([1., .5] if frozen else [1.])
    expected = pd.DataFrame(expected_variance * np.outer(loading, loading),
                            index=sample.columns[:-1] if not frozen else ['BM', 'A'],
                            columns=sample.columns[:-1] if not frozen else ['BM', 'A'])
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-6, atol=1e-15)
    np.testing.assert_allclose(actual, 0., atol=1e-8 if '32' in dtype else 1e-15)
    pd.testing.assert_frame_equal(sample, before)


@pytest.mark.parametrize('dtype', ['float64', 'Float64', 'float32', 'Float32'])
@pytest.mark.filterwarnings('error')
def test_compute_regime_mixture_covar_from_sample_preserves_nearly_constant_estimation(dtype):
    """A minimal identified fit needs 24 periods and two distinct returns per regime."""
    # Float32 needs wider spacing to stay above the native least-squares rank cutoff.
    spacing = 1e-5 if '32' in dtype else 1e-8
    sample = pd.DataFrame({'BM': pd.Series([.1, .1 + spacing] * 12, dtype=dtype),
                           'regime': ['Low'] * 12 + ['High'] * 12})
    sample['A'] = .5 * sample['BM']
    before = sample.copy(deep=True)
    benchmark = sample['BM'].to_numpy(dtype=float)
    centered = benchmark - benchmark.mean()
    annual_variance = 12. * (centered @ centered) / len(benchmark)
    expected = annual_variance * np.array([[1., .5], [.5, .25]])
    actual = compute_regime_mixture_covar_from_sample(sample, 'BM', 12.)
    precision = np.finfo('float32' if '32' in dtype else 'float64').eps
    # Raw-moment subtraction has cancellation error on this vanishing-variance panel.
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=.2 * precision)
    assert actual.index.tolist() == actual.columns.tolist() == ['BM', 'A']
    assert (actual.dtypes == np.dtype(float)).all()
    pd.testing.assert_frame_equal(sample, before)


@pytest.mark.parametrize('field', ['betas', 'idio_vars', 'benchmark_regime_means',
                                 'benchmark_regime_second_moments', 'regime_probs'])
@pytest.mark.parametrize('dtype,value', [('object', 'not numeric'), ('object', 1j),
                                        ('complex128', 1j)])
@pytest.mark.filterwarnings('error')
def test_compute_regime_mixture_covar_rejects_nonreal_components(
        mixture_inputs, field, dtype, value):
    """Conversion must not reinterpret text or discard an imaginary component."""
    mixture_inputs[field] = mixture_inputs[field].astype(dtype)
    if field == 'betas':
        mixture_inputs[field].iloc[-1, -1] = value
    else:
        mixture_inputs[field].iloc[-1] = value
    with pytest.raises(ValueError, match=field):
        compute_regime_mixture_covar(**mixture_inputs)


@pytest.mark.parametrize('field', ['betas', 'idio_vars', 'benchmark_regime_second_moments', 'af'])
@pytest.mark.filterwarnings('error')
def test_compute_regime_mixture_covar_rejects_nonrepresentable_arithmetic(mixture_inputs, field):
    """Finite components do not guarantee that an outer product or annualisation stays finite."""
    if field == 'betas':
        mixture_inputs[field].iloc[-1, -1] = 1e308
    elif field == 'af':
        mixture_inputs[field] = 1e308
        mixture_inputs['idio_vars'].iloc[-1] = 100.
    else:
        mixture_inputs[field].iloc[-1] = 1e308
    with pytest.raises(ValueError, match='covariance arithmetic'):
        compute_regime_mixture_covar(**mixture_inputs)


def test_compute_regime_mixture_covar_preserves_default_and_empty_assets(mixture_inputs):
    """The probability default and the native empty asset universe need no invented components."""
    actual = compute_regime_mixture_covar(**mixture_inputs)
    mixture_inputs['regime_probs'] = None
    pd.testing.assert_frame_equal(compute_regime_mixture_covar(**mixture_inputs), actual)
    mixture_inputs['betas'] = mixture_inputs['betas'].iloc[:0]
    mixture_inputs['idio_vars'] = mixture_inputs['idio_vars'].iloc[:0]
    empty = mixture_inputs['betas'].index
    pd.testing.assert_frame_equal(compute_regime_mixture_covar(**mixture_inputs),
                                  pd.DataFrame(index=empty, columns=empty, dtype=float))
    mixture_inputs['betas'] = mixture_inputs['betas'].iloc[:, :0]
    with pytest.raises(ValueError, match='betas'):
        compute_regime_mixture_covar(**mixture_inputs)


@pytest.mark.parametrize('dtype', ['float64', 'Float64', 'float32', 'Float32'])
@pytest.mark.filterwarnings('error')
def test_compute_regime_mixture_covar_tolerates_precision_without_normalizing(
        mixture_inputs, dtype):
    """Representational noise is not a negative residual or permission to renormalize weights."""
    mixture_inputs['regime_probs'] = pd.Series([.1, .2, .7],
                                              index=mixture_inputs['betas'].columns, dtype=dtype)
    mixture_inputs['idio_vars'] *= 0.
    total = mixture_inputs['regime_probs'].to_numpy(dtype=float).sum()
    actual = compute_regime_mixture_covar(**mixture_inputs)
    expected = pd.DataFrame([[.0192, .0096], [.0096, .0048]],
                            index=actual.index, columns=actual.columns) * total
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-12)
    mixture_inputs['benchmark_regime_means'] = pd.Series(
        [.1] * 3, index=mixture_inputs['betas'].columns, dtype=dtype)
    mixture_inputs['benchmark_regime_second_moments'] = pd.Series(
        [.01] * 3, index=mixture_inputs['betas'].columns, dtype=dtype)
    compute_regime_mixture_covar(**mixture_inputs)
    mixture_inputs['benchmark_regime_second_moments'] *= .99
    with pytest.raises(ValueError, match='benchmark_regime_second_moments'):
        compute_regime_mixture_covar(**mixture_inputs)
    mixture_inputs['benchmark_regime_second_moments'] *= 2.
    mixture_inputs['idio_vars'].iloc[-1] = -np.finfo(float).tiny
    with pytest.raises(ValueError, match='idio_vars'):
        compute_regime_mixture_covar(**mixture_inputs)


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

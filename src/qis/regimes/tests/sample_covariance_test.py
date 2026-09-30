"""Sample regime moments and covariance against independent population identities."""
import numpy as np
import pandas as pd
import pytest

import qis.regimes as rg


@pytest.fixture
def sample():
    """Build a reproducible monthly benchmark and two signed-beta assets."""
    rng = np.random.default_rng(921)
    benchmark = rng.normal(0.004, 0.03, 120)
    panel = pd.DataFrame({'A': 0.4 * benchmark + rng.normal(0, 0.02, 120),
                          'BM': benchmark,
                          'C': -0.7 * benchmark + rng.normal(0, 0.01, 120)},
                         index=pd.date_range('2000-01-31', periods=120, freq='ME'))
    return rg.create_sampled_returns_with_regime_id(panel, 'BM')


def test_sample_moments_and_benchmark_diagonal(sample):
    """Total expectation and variance reconstruct the observed benchmark."""
    moments = rg.compute_sample_regime_moments(sample, 'BM')
    assert moments.index.tolist() == ['Bear', 'Normal', 'Bull']
    assert moments.columns.tolist() == ['probability', 'mean', 'second_moment']
    assert moments.probability.sum() == pytest.approx(1.0)
    assert moments.probability @ moments['mean'] == pytest.approx(sample.BM.mean())
    assert moments.probability @ moments.second_moment == pytest.approx(sample.BM.pow(2).mean())
    covariance = rg.compute_regime_mixture_covar_from_sample(sample, 'BM', af=12.)
    assert covariance.index.tolist() == covariance.columns.tolist() == ['BM', 'A', 'C']
    assert covariance.loc['BM', 'BM'] == pytest.approx(12. * sample.BM.var(ddof=0))
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-15)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-12


def test_frozen_betas_match_estimation_and_direct_time_series_covariance(sample):
    """The factor covariance also equals covariance of each period's beta times benchmark."""
    betas = rg.compute_regime_betas(sample, 'BM', af=12.)
    actual = rg.compute_regime_mixture_covar_from_sample(sample, 'BM', 12., betas=betas)
    pd.testing.assert_frame_equal(
        actual, rg.compute_regime_mixture_covar_from_sample(sample, 'BM', 12.))
    factors = pd.DataFrame({'BM': sample.BM})
    for asset in ('A', 'C'):
        slopes = sample.regime.astype(str).map(
            {name: betas.loc[asset, f'beta_{name.lower()}']
             for name in ('Bear', 'Normal', 'Bull')})
        factors[asset] = slopes * sample.BM
    expected = factors.cov(ddof=0) * 12.
    expected += np.diag([0., betas.loc['A', 'idio_vol'] ** 2, betas.loc['C', 'idio_vol'] ** 2])
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)
    pd.testing.assert_frame_equal(
        actual, rg.compute_regime_mixture_covar_from_sample(
            sample, 'BM', 12., betas=betas.iloc[::-1]))


def test_equal_betas_reduce_to_single_factor_and_units_scale(sample):
    """Constant betas reduce to a single factor; changing return units scales covariance."""
    betas = pd.DataFrame({name: [0.4, -0.7] for name in
                          ('beta_bear', 'beta_normal', 'beta_bull')}, index=['A', 'C'])
    betas['idio_vol'] = [0.05, 0.07]
    covariance = rg.compute_regime_mixture_covar_from_sample(sample, 'BM', 12., betas)
    beta = np.array([1., 0.4, -0.7])
    expected = np.outer(beta, beta) * sample.BM.var(ddof=0) * 12.
    expected += np.diag([0., 0.05 ** 2, 0.07 ** 2])
    np.testing.assert_allclose(covariance, expected, rtol=1e-12, atol=1e-15)
    scaled = sample.copy()
    scaled[['BM', 'A', 'C']] *= 100.
    scaled_betas = betas.copy()
    scaled_betas.idio_vol *= 100.
    np.testing.assert_allclose(
        rg.compute_regime_mixture_covar_from_sample(scaled, 'BM', 12., scaled_betas),
        covariance * 10000., rtol=1e-12)


def test_custom_partition_column_and_benchmark_only(sample):
    """Both helpers follow the supplied classification, without assuming three buckets."""
    sampled = rg.create_sampled_returns_with_regime_id(
        sample.drop(columns='regime'), 'BM', q=[0., 0.3, 1.],
        regime_ids=['Low', 'High']).rename(columns={'regime': 'state'})
    moments = rg.compute_sample_regime_moments(sampled, 'BM', regime_column='state')
    assert moments.index.tolist() == ['Low', 'High']
    covariance = rg.compute_regime_mixture_covar_from_sample(
        sampled, 'BM', 12., regime_column='state')
    assert covariance.loc['BM', 'BM'] == pytest.approx(12. * sample.BM.var(ddof=0))
    solo = rg.compute_regime_mixture_covar_from_sample(sample[['BM', 'regime']], 'BM', 12.)
    assert solo.shape == (1, 1)


def test_unclassified_rows_are_excluded_without_reclassifying(sample):
    """Missing classification omits the whole observation before all moment calculations."""
    sample.loc[sample.index[0], 'regime'] = np.nan
    expected = sample.dropna(subset=['regime'])
    actual = rg.compute_regime_mixture_covar_from_sample(sample, 'BM', 12.)
    pd.testing.assert_frame_equal(
        actual, rg.compute_regime_mixture_covar_from_sample(expected, 'BM', 12.))


@pytest.mark.parametrize('af', [0., -1., np.nan, np.inf])
def test_invalid_annualisation_raises(sample, af):
    """An invalid annualisation factor cannot create a covariance."""
    with pytest.raises(ValueError, match='af'):
        rg.compute_regime_mixture_covar_from_sample(sample, 'BM', af)


@pytest.mark.parametrize('asset', ['BM', 'A'])
def test_missing_or_infinite_classified_returns_raise(sample, asset):
    """Do not silently mix asset-specific estimation windows with benchmark moments."""
    for value in (np.nan, np.inf):
        broken = sample.copy()
        broken.loc[broken.index[0], asset] = value
        with pytest.raises(ValueError, match='finite|complete'):
            rg.compute_regime_mixture_covar_from_sample(broken, 'BM', 12.)


def test_empty_and_degenerate_regimes_raise(sample):
    """An empty bucket or an unidentified within-regime slope gets an explicit error."""
    broken = sample[sample.regime != 'Bear']
    with pytest.raises(ValueError, match='empty'):
        rg.compute_sample_regime_moments(broken, 'BM')
    broken = sample.copy()
    broken.loc[broken.regime == 'Bear', 'BM'] = -0.1
    with pytest.raises(ValueError, match='distinct'):
        rg.compute_regime_mixture_covar_from_sample(broken, 'BM', 12.)


@pytest.mark.parametrize('change', ['missing_asset', 'extra_asset', 'missing_beta',
                                    'negative_vol', 'nan', 'duplicate'])
def test_frozen_beta_contract(sample, change):
    """Frozen sheets must exactly cover assets, label every regime and state finite units."""
    betas = rg.compute_regime_betas(sample, 'BM', 12.)
    if change == 'missing_asset':
        betas = betas.drop(index='A')
    elif change == 'extra_asset':
        betas.loc['unknown'] = betas.iloc[0]
    elif change == 'missing_beta':
        betas = betas.drop(columns='beta_bear')
    elif change == 'negative_vol':
        betas.loc['A', 'idio_vol'] = -0.1
    elif change == 'nan':
        betas.loc['A', 'beta_bear'] = np.nan
    else:
        betas = pd.concat([betas, betas.iloc[:1]])
    with pytest.raises(ValueError):
        rg.compute_regime_mixture_covar_from_sample(sample, 'BM', 12., betas)


def test_invalid_sample_labels_raise(sample):
    """Ambiguous columns or labels cannot silently select the wrong moments."""
    for broken, benchmark in [
            (sample, 'absent'), (sample.drop(columns='regime'), 'BM'),
            (sample.iloc[:0], 'BM'),
            (pd.concat([sample, sample[['BM']]], axis=1), 'BM')]:
        with pytest.raises(ValueError):
            rg.compute_sample_regime_moments(broken, benchmark)
    broken = sample.copy()
    broken['regime'] = broken['regime'].cat.rename_categories(['Bear', 'bear', 'Bull'])
    with pytest.raises(ValueError, match='unique'):
        rg.compute_regime_mixture_covar_from_sample(broken, 'BM', 12.)


def test_covariance_rejects_total_regime_label_collision(sample):
    """Do not mistake the estimator's overall beta_total for a regime-specific slope."""
    sample['regime'] = sample['regime'].cat.rename_categories(['Total', 'Normal', 'Bull'])
    # Moments have no beta-column naming restriction.
    assert 'Total' in rg.compute_sample_regime_moments(sample, 'BM').index
    with pytest.raises(ValueError, match='reserved.*beta_total'):
        rg.compute_regime_mixture_covar_from_sample(sample, 'BM', 12.)

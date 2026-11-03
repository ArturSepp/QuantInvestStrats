"""Fixed-row QR/Bartlett references for null-centered joint bootstrap families."""

# packages
import numpy as np
import pandas as pd
import pytest
from dataclasses import replace
from scipy.stats import binomtest

# qis
from qis.datasets.synthetic import generate_synthetic_prices
from qis.models.bootstrap import bootstrap_numba
from qis.models.bootstrap.bootstrap_numba import BootstrapType
from qis.perfstats import bootstrap_significance as bootstrap
from qis.perfstats import multiple_testing, significance
from qis.perfstats.multiple_testing import MultipleTestingMethod
from qis.perfstats.returns import to_returns
from qis.utils import regression


pytestmark = pytest.mark.filterwarnings('ignore:B=.*minimum bootstrap p-value:RuntimeWarning')
CONVENTIONS = dict(family_id='bootstrap-family', return_basis='supplied excess returns',
                   cost_basis='net of supplied costs', sampling_frequency='ME')
METHODS = [MultipleTestingMethod.HOLM, MultipleTestingMethod.BHY]


def _panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Small deterministic serial/cross-correlated panel with nonzero factor means."""
    dates = pd.date_range('2000-01-31', periods=36, freq='ME', name='date')
    t = np.arange(len(dates))
    x = pd.DataFrame({'F0': 0.01 + 0.015 * np.sin(t * 0.37),
                      'F1': -0.007 + 0.012 * np.cos(t * 0.71),
                      'F2': 0.003 + 0.009 * np.sin(t * 1.13)}, index=dates)
    noise = 0.005 * np.sin(t * 0.19) + 0.002 * np.cos(t * 1.07)
    y = pd.DataFrame({'A': 0.001 + x.to_numpy() @ [0.4, -0.3, 0.2] + noise,
                      'B': -0.002 + x.to_numpy() @ [-0.1, 0.4, 0.3]
                      + 0.7 * noise + 0.003 * np.sin(t * 0.93)}, index=dates)
    y.columns.name = 'hypothesis'
    return y, x


def _indices(nobs: int, count: int = 11) -> np.ndarray:
    """Nonuniform row multisets with repeats and deliberate chronological disorder."""
    t = np.arange(nobs)
    return np.column_stack([(t + 5 * b + np.where((t + b) % 7 < 3, 4, 0)) % nobs
                            for b in range(count)])


def _qr_reference(y: np.ndarray, x: np.ndarray | None, lags: int) -> tuple:
    """Independent coefficient map and dense Bartlett covariance for intercepts."""
    nobs = len(y)
    design = np.ones((nobs, 1)) if x is None else np.column_stack((np.ones(nobs), x))
    q, r = np.linalg.qr(design, mode='reduced')
    coefficients = np.linalg.solve(r, q.T @ y)
    coefficient_map = np.linalg.solve(r, q.T)
    residuals = y - design @ coefficients
    influence = coefficient_map[0, :, None] * residuals
    distance = np.abs(np.subtract.outer(np.arange(nobs), np.arange(nobs)))
    kernel = np.maximum(1.0 - distance / (lags + 1.0), 0.0)
    covariance = nobs / (nobs - design.shape[1]) * influence.T @ kernel @ influence
    se = np.sqrt(np.diag(covariance))
    return coefficients[0], se, coefficients[0] / se


def _call(y: pd.DataFrame, x: pd.DataFrame | None, **kwargs):
    if x is None:
        return bootstrap.compute_bootstrap_mean_significance(y, **dict(CONVENTIONS, **kwargs))
    return bootstrap.compute_bootstrap_alpha_significance(y, x, **dict(CONVENTIONS, **kwargs))


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('factor_count', [0, 1, 3])
@pytest.mark.parametrize('lags', [0, 3])
@pytest.mark.parametrize('count', [11, 36])
def test_fixed_indices_match_independent_reference(method, factor_count, lags, count) -> None:
    y, all_x = _panel()
    x = None if factor_count == 0 else all_x.iloc[:, :factor_count]
    indices = _indices(len(y), count)
    result = _call(y, x, method=method, hac_lags=lags, bootstrapped_indices=indices,
                   include_diagnostics=True)
    values = y.to_numpy()
    factor_values = None if x is None else x.to_numpy()
    intercept, se, statistic = _qr_reference(values, factor_values, lags)
    null_values = values - (values.mean(axis=0) if x is None else intercept)
    statistics = np.vstack([_qr_reference(null_values[indices[:, b]],
                                          None if x is None else factor_values[indices[:, b]],
                                          lags)[2] for b in range(count)])
    np.testing.assert_allclose(result.metadata['bootstrap_statistics'], statistics,
                               rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(result.table['estimate'], intercept, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(result.table['hac_se'], se, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(result.table['statistic'], statistic, rtol=1e-9, atol=1e-12)
    counts = (np.abs(statistics) >= np.abs(statistic)).sum(axis=0)
    pvalues = (counts + 1) / (count + 1)
    np.testing.assert_array_equal(result.table['bootstrap_exceedances'], counts)
    np.testing.assert_array_equal(result.table['pvalue_raw'], pvalues)
    order = np.argsort(pvalues)
    ordered = pvalues[order]
    harmonic = sum(1 / i for i in range(1, len(y.columns) + 1))
    expected_adjusted = np.empty(len(ordered))
    for rank in range(len(ordered)):
        if method is MultipleTestingMethod.HOLM:
            value = max((len(ordered) - j) * ordered[j] for j in range(rank + 1))
        else:
            value = min(len(ordered) * harmonic * ordered[j] / (j + 1)
                        for j in range(rank, len(ordered)))
        expected_adjusted[order[rank]] = min(1.0, value)
    np.testing.assert_allclose(result.table['pvalue_adjusted'], expected_adjusted, rtol=1e-14)
    np.testing.assert_array_equal(result.table['reject'], expected_adjusted <= 0.05)
    for name, exceedances in zip(y.columns, counts):
        interval = binomtest(int(exceedances), count).proportion_ci(method='exact')
        np.testing.assert_allclose(result.table.loc[name, ['bootstrap_tail_mc_lower',
                                                          'bootstrap_tail_mc_upper']].astype(float),
                                   [interval.low, interval.high], rtol=1e-10, atol=1e-12)
    meta = result.metadata
    pd.testing.assert_index_equal(result.table.index, y.columns)
    assert meta['num_samples'] == count and meta['bootstrap_index_shape'] == (len(y), count)
    assert meta['minimum_pvalue'] == 1 / (count + 1) and meta['pvalue_source'] == 'bootstrap'
    assert meta['family_members'] == tuple(y.columns) and meta['method'] is method
    assert meta['nobs'] == len(y)
    assert meta['factor_columns'] == (() if x is None else tuple(x.columns))
    assert meta['null_center'] == ('sample_mean' if x is None else 'fitted_alpha')
    assert meta['test'] == ('bootstrap_mean' if x is None else 'bootstrap_alpha')
    assert meta['index_source'] == 'supplied'
    assert meta['bootstrap_type'] is meta['seed'] is meta['block_size'] is None
    assert meta['reference_distribution'] == 'null-centered joint bootstrap'
    assert meta['mc_confidence_level'] == 0.95 and meta['replicate_numbering'] == 'zero-based'
    assert meta['confidence_interval'] == 'unadjusted/pointwise'
    assert meta['interval_method'] == 'normal HAC' and meta['family_id'] == 'bootstrap-family'
    assert meta['return_basis'] == CONVENTIONS['return_basis']
    assert meta['cost_basis'] == CONVENTIONS['cost_basis']
    np.testing.assert_array_equal(meta['bootstrap_indices'], indices)
    assert not np.shares_memory(meta['bootstrap_indices'], indices)


@pytest.mark.parametrize('alpha', [False, True])
def test_every_draw_refits_null_paired_rows_on_original_grid(alpha, monkeypatch) -> None:
    y, x = _panel()
    x = x if alpha else None
    indices = _indices(len(y), 3)
    calls = []
    original = regression.estimate_hac_regression

    def spy(**kwargs):
        result = original(**kwargs)
        calls.append((kwargs, result))
        return result

    monkeypatch.setattr(regression, 'estimate_hac_regression', spy)
    _call(y, x, bootstrapped_indices=indices)
    assert len(calls) == 4
    observed_inputs, observed_fit = calls[0]
    center = observed_fit.coefficients.loc['Intercept'].to_numpy()
    if not alpha:
        center = observed_inputs['y'].mean().to_numpy()
    for b, (inputs, _) in enumerate(calls[1:]):
        rows = indices[:, b]
        pd.testing.assert_index_equal(inputs['y'].index, y.index)
        assert inputs['y'].index.is_unique
        assert inputs['span'] is None and inputs['hac_lags'] == 3
        np.testing.assert_allclose(inputs['y'], (observed_inputs['y'] - center).iloc[rows],
                                   rtol=1e-12, atol=1e-14)
        if alpha:
            pd.testing.assert_index_equal(inputs['x'].index, y.index)
            np.testing.assert_array_equal(inputs['x'], observed_inputs['x'].iloc[rows])
        else:
            assert inputs['x'] is None


@pytest.mark.parametrize('alpha', [False, True])
@pytest.mark.parametrize('block_size', [4, 2**53])
def test_reproducible_stationary_sampler_and_supplied_replay(
        alpha, block_size, monkeypatch) -> None:
    y, x = _panel()
    x = x if alpha else None
    original_y, original_x = y.copy(), None if x is None else x.copy()
    calls = []
    original = bootstrap_numba.generate_bootstrapped_indices

    def sampler_spy(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(bootstrap_numba, 'generate_bootstrapped_indices', sampler_spy)
    state = np.random.get_state()
    first = _call(y, x, num_samples=13, block_size=block_size, seed=20261101,
                  include_diagnostics=True)
    second = _call(y, x, num_samples=13, block_size=block_size, seed=20261101,
                   include_diagnostics=True)
    replay = _call(y, x, bootstrapped_indices=first.metadata['bootstrap_indices'],
                   num_samples=13, include_diagnostics=False)
    assert len(calls) == 2
    assert calls[0] == calls[1] == dict(num_data_index=36, bootstrap_type=BootstrapType.STATIONARY,
                                      num_samples=13, index_length=36, block_size=block_size,
                                      min_block_size=1, seed=20261101)
    pd.testing.assert_frame_equal(first.table, second.table)
    pd.testing.assert_frame_equal(first.table, replay.table)
    pd.testing.assert_frame_equal(first.metadata['bootstrap_statistics'],
                                   second.metadata['bootstrap_statistics'])
    np.testing.assert_array_equal(first.metadata['bootstrap_indices'],
                                   second.metadata['bootstrap_indices'])
    assert first.metadata['indices_sha256'] == replay.metadata['indices_sha256']
    assert first.metadata['bootstrap_type'] is BootstrapType.STATIONARY
    assert first.metadata['seed'] == 20261101 and first.metadata['block_size'] == block_size
    assert first.metadata['min_block_size'] == 1 and first.metadata['index_source'] == 'generated'
    assert 'bootstrap_statistics' not in replay.metadata
    assert 'bootstrap_indices' not in replay.metadata
    pd.testing.assert_frame_equal(y, original_y)
    if x is not None:
        pd.testing.assert_frame_equal(x, original_x)
    after = np.random.get_state()
    assert state[0] == after[0] and state[2:] == after[2:]
    np.testing.assert_array_equal(state[1], after[1])


def test_default_draw_count_is_explicit_and_never_falls_back_to_iid(monkeypatch) -> None:
    y, _ = _panel()

    def inspect_sampler(**kwargs):
        assert kwargs['num_samples'] == 9999 and kwargs['block_size'] == 6
        assert kwargs['bootstrap_type'] is BootstrapType.STATIONARY
        assert kwargs['index_length'] == len(y) and kwargs['min_block_size'] == 1
        raise RuntimeError('inspected default request before allocation')

    monkeypatch.setattr(bootstrap_numba, 'generate_bootstrapped_indices', inspect_sampler)
    with pytest.raises(RuntimeError, match='inspected default'):
        _call(y, None)


@pytest.mark.parametrize('alpha', [False, True])
def test_observed_hac_diagnostics_annualisation_and_unit_scaling(alpha) -> None:
    y, x = _panel()
    x = x if alpha else None
    indices = _indices(len(y))
    base = _call(y, x, bootstrapped_indices=indices, include_diagnostics=True)
    annual = _call(y, x, bootstrapped_indices=indices, af=12)
    observed = (significance.compute_mean_significance(y, **CONVENTIONS) if x is None
                else significance.compute_alpha_significance(y, x, **CONVENTIONS))
    fields = ['estimate', 'hac_se', 'ci_lower_pointwise', 'ci_upper_pointwise']
    np.testing.assert_array_equal(base.table[fields], observed.table[fields])
    np.testing.assert_array_equal(base.table['pvalue_hac'], observed.table['pvalue_raw'])
    np.testing.assert_array_equal(base.table['statistic'], observed.table['statistic'])
    np.testing.assert_allclose(annual.table[fields], base.table[fields] * 12)
    for field in ['statistic', 'pvalue_raw', 'pvalue_hac', 'pvalue_adjusted', 'reject']:
        np.testing.assert_array_equal(annual.table[field], base.table[field])
    for scale in [1e-100, 1e100]:
        scaled = _call(y * scale, None if x is None else x * [1e-80, 1e80, -1e20],
                       bootstrapped_indices=indices, include_diagnostics=True)
        np.testing.assert_allclose(scaled.metadata['bootstrap_statistics'],
                                   base.metadata['bootstrap_statistics'], rtol=1e-9, atol=1e-12)
        np.testing.assert_array_equal(scaled.table['pvalue_raw'], base.table['pvalue_raw'])
        np.testing.assert_allclose(scaled.table[fields] / scale, base.table[fields],
                                   rtol=1e-9, atol=1e-12)
    assert annual.metadata['af'] == 12 and annual.metadata['annualised']


def test_joint_duplicates_permutations_nullable_values_and_custom_calendar() -> None:
    y, x = _panel()
    y['A-copy'] = y['A']
    y, x = y.drop(y.index[5]).astype('Float64'), x.drop(x.index[5]).astype('Float64')
    y.columns = pd.MultiIndex.from_tuples([('s', 'A'), ('s', 'B'), ('s', 'A-copy')])
    indices = _indices(len(y), 7).astype(np.int16)
    original_indices = indices.copy()
    settings = dict(bootstrapped_indices=indices, include_diagnostics=True,
                    sampling_frequency=None, expected_index=y.index,
                    method=MultipleTestingMethod.BHY)
    result = _call(y, x, **settings)
    reordered = _call(y.iloc[:, ::-1], x.iloc[:, ::-1], **settings)
    np.testing.assert_allclose(reordered.table.reindex(y.columns).to_numpy(dtype=float),
                               result.table.to_numpy(dtype=float), rtol=1e-9, atol=1e-12)
    pd.testing.assert_series_equal(result.table.loc[('s', 'A')], result.table.loc[('s', 'A-copy')],
                                   check_names=False)
    pd.testing.assert_series_equal(result.metadata['bootstrap_statistics'].iloc[:, 0],
                                   result.metadata['bootstrap_statistics'].iloc[:, 2],
                                   check_names=False)
    pd.testing.assert_index_equal(result.metadata['expected_index'], y.index)
    assert result.metadata['family_members'] == tuple(y.columns)
    np.testing.assert_array_equal(indices, original_indices)


@pytest.mark.parametrize('factor_count', [0, 3])
def test_invalid_draw_aborts_with_ids_without_redrawing(factor_count, monkeypatch) -> None:
    y, x = _panel()
    x = None if factor_count == 0 else x
    indices = _indices(len(y), 3)
    indices[:, 1] = 0
    calls = []
    original = bootstrap._draw_statistics

    def spy(*args):
        calls.append(args[-1])
        return original(*args)

    monkeypatch.setattr(bootstrap, '_draw_statistics', spy)
    with pytest.raises(ValueError, match="replicate 1.*hypotheses.*'A'.*'B'"):
        _call(y, x, bootstrapped_indices=indices)
    assert calls == [0, 1]


def test_invalid_single_member_and_zero_se_are_named(monkeypatch) -> None:
    y, _ = _panel()
    y.iloc[:2, 0] = 0.01
    indices = _indices(len(y), 2)
    indices[:, 1] = np.arange(len(y)) % 2
    with pytest.raises(ValueError, match=r"replicate 1.*\['A'\]"):
        _call(y, None, bootstrapped_indices=indices)
    original = regression.estimate_hac_regression
    count = 0

    def zero_se(**kwargs):
        nonlocal count
        result = original(**kwargs)
        count += 1
        if count == 2:
            result.hac_se.loc['Intercept', 'B'] = 0.0
        return result

    monkeypatch.setattr(regression, 'estimate_hac_regression', zero_se)
    with pytest.raises(ValueError, match=r"replicate 0.*\['B'\]"):
        _call(y, None, bootstrapped_indices=_indices(len(y), 2))


def test_zero_exceedances_have_positive_pvalue_and_nonzero_mc_upper_bound() -> None:
    y, _ = _panel()
    y = y[['A']] + 1.0
    indices = _indices(len(y), 9)
    result = _call(y, None, bootstrapped_indices=indices)
    assert result.table['bootstrap_exceedances'].iloc[0] == 0
    assert result.table['pvalue_raw'].iloc[0] == result.metadata['minimum_pvalue'] == 0.1
    assert result.table['bootstrap_tail_mc_lower'].iloc[0] == 0.0
    assert result.table['bootstrap_tail_mc_upper'].iloc[0] > 0.0


def test_inclusive_tails_use_bootstrap_pvalues_in_correction(monkeypatch) -> None:
    y, _ = _panel()
    observed = significance.compute_mean_significance(y, **CONVENTIONS)
    corrections = []
    original = multiple_testing.adjust_pvalues

    def tied(*args):
        return -observed.table['statistic'].to_numpy()

    def correction_spy(pvalues, **kwargs):
        corrections.append(pvalues.copy())
        result = original(pvalues, **kwargs)
        if len(corrections) == 2:
            result['pvalue_adjusted'], result['reject'] = 0.7, False
        return result

    monkeypatch.setattr(bootstrap, '_draw_statistics', tied)
    monkeypatch.setattr(multiple_testing, 'adjust_pvalues', correction_spy)
    result = _call(y, None, bootstrapped_indices=_indices(len(y), 4))
    assert len(corrections) == 2  # observed HAC diagnostics, then bootstrap correction
    np.testing.assert_array_equal(corrections[-1], np.ones(2))
    np.testing.assert_array_equal(result.table['bootstrap_exceedances'], [4, 4])
    np.testing.assert_array_equal(result.table['pvalue_raw'], [1.0, 1.0])
    np.testing.assert_array_equal(result.table['pvalue_adjusted'], [0.7, 0.7])
    assert not result.table['reject'].any()
    assert (result.table['bootstrap_tail_mc_lower'] < 1).all()
    assert (result.table['bootstrap_tail_mc_upper'] == 1).all()


@pytest.mark.parametrize('method', METHODS)
def test_resolution_warning_and_equality_threshold(method) -> None:
    y, _ = _panel()
    y = y[['A']]
    with pytest.warns(RuntimeWarning, match='first-rank threshold'):
        short = _call(y, None, method=method, bootstrapped_indices=_indices(len(y), 18))
    assert not short.metadata['first_rank_reachable']
    reached = _call(y, None, method=method, bootstrapped_indices=_indices(len(y), 19))
    assert reached.metadata['first_rank_reachable']
    assert reached.metadata['first_rank_threshold'] == reached.metadata['minimum_pvalue'] == 0.05
    y = pd.concat([y, y.rename(columns={'A': 'B'})], axis=1)
    with pytest.warns(RuntimeWarning, match='BHY can still reject at higher ranks'):
        bhy = _call(y, None, method=MultipleTestingMethod.BHY,
                    bootstrapped_indices=_indices(len(y), 19))
    assert bhy.metadata['first_rank_threshold'] == pytest.approx(0.05 / (2 * 1.5))


@pytest.mark.parametrize('bad', ['list', 'float', 'bool', 'object', 'nan', 'negative', 'bound',
                                'uint_overflow', 'one_dimensional', 'transposed', 'empty',
                                'three_d'])
def test_invalid_supplied_indices_fail_before_sampler(bad, monkeypatch) -> None:
    y, _ = _panel()
    indices = _indices(len(y), 5)
    variants = dict(list=indices.tolist(), float=indices.astype(float), bool=indices.astype(bool),
                    object=indices.astype(object), nan=np.full_like(indices, np.nan, dtype=float),
                    negative=indices - len(y), bound=indices + len(y),
                    uint_overflow=np.full(indices.shape, 2**64 - 1, dtype=np.uint64),
                    one_dimensional=indices[:, 0], transposed=indices.T, empty=indices[:, :0],
                    three_d=indices[:, :, None])

    def should_not_sample(**kwargs):
        raise AssertionError('invalid supplied indices must never reach the sampler')

    monkeypatch.setattr(bootstrap_numba, 'generate_bootstrapped_indices', should_not_sample)
    with pytest.raises(ValueError, match='bootstrapped_indices'):
        _call(y, None, bootstrapped_indices=variants[bad])


@pytest.mark.parametrize('settings', [
    {'num_samples': 0}, {'num_samples': True}, {'num_samples': 2.5}, {'num_samples': '5'},
    {'num_samples': 2**64}, {'block_size': 0}, {'block_size': -1}, {'block_size': 1.5},
    {'block_size': True}, {'block_size': 2**53 + 1}, {'block_size': 2**63 - 1},
    {'seed': -1}, {'seed': 2**32}, {'seed': True}, {'seed': 1.5},
    {'include_diagnostics': 'yes'}, {'af': 0}, {'af': np.inf},
])
def test_invalid_bootstrap_settings_fail_before_sampler(settings, monkeypatch) -> None:
    y, _ = _panel()

    def should_not_sample(**kwargs):
        raise AssertionError('invalid settings must never reach Numba')

    monkeypatch.setattr(bootstrap_numba, 'generate_bootstrapped_indices', should_not_sample)
    with pytest.raises(ValueError):
        _call(y, None, **settings)


def test_common_sample_and_num_samples_conflict_validation(monkeypatch) -> None:
    y, x = _panel()

    def should_not_sample(**kwargs):
        raise AssertionError('invalid data must never reach Numba')

    monkeypatch.setattr(bootstrap_numba, 'generate_bootstrapped_indices', should_not_sample)
    with pytest.raises(ValueError, match='conflicts'):
        _call(y, x, bootstrapped_indices=_indices(len(y), 3), num_samples=4)
    for bad_y, bad_x in [(y.assign(A=np.nan), x), (y, x.assign(F0=np.inf)),
                         (y, x.iloc[::-1]), (y.iloc[1:], x), (y.iloc[::2], x.iloc[::2]),
                         (y.iloc[:23], x.iloc[:23])]:
        with pytest.raises(ValueError):
            _call(bad_y, bad_x)
    with pytest.raises(ValueError, match='factors'):
        bootstrap.compute_bootstrap_alpha_significance(y, None, **CONVENTIONS)


def test_generated_shape_and_backend_sample_are_checked(monkeypatch) -> None:
    y, _ = _panel()
    monkeypatch.setattr(bootstrap_numba, 'generate_bootstrapped_indices',
                        lambda **kwargs: _indices(len(y), 2).T)
    with pytest.raises(ValueError, match='shape'):
        _call(y, None, num_samples=2)
    original = regression.estimate_hac_regression
    calls = 0

    def missing_row(**kwargs):
        nonlocal calls
        result = original(**kwargs)
        calls += 1
        return result if calls == 1 else replace(result, nobs=result.nobs - 1)

    monkeypatch.setattr(regression, 'estimate_hac_regression', missing_row)
    with pytest.raises(ValueError, match='replicate 0.*sample'):
        _call(y, None, bootstrapped_indices=_indices(len(y), 2))


def test_frozen_synthetic_monthly_integration_and_automatic_lags(monkeypatch) -> None:
    prices = generate_synthetic_prices(start='2010-01-01', end='2019-12-31')
    returns = to_returns(prices[['SEQ_US', 'SBD_TSY', 'SBD_IG', 'SBD_HY']],
                         freq='ME', is_log_returns=False).iloc[1:]
    y, x = returns[['SBD_IG', 'SBD_HY']], returns[['SEQ_US', 'SBD_TSY']]
    indices = _indices(len(y), 5)
    calls = []
    original = regression.newey_west_lag_rule

    def lag_rule(nobs):
        calls.append(nobs)
        return original(nobs)

    monkeypatch.setattr(regression, 'newey_west_lag_rule', lag_rule)
    for factors in [None, x]:
        result = _call(y, factors, bootstrapped_indices=indices, hac_lags=None,
                       include_diagnostics=True)
        lags = original(len(y))
        center, _, _ = _qr_reference(y.to_numpy(), None if factors is None else x.to_numpy(), lags)
        for b in range(5):
            rows = indices[:, b]
            expected = _qr_reference((y.to_numpy() - center)[rows],
                                      None if factors is None else x.to_numpy()[rows], lags)[2]
            np.testing.assert_allclose(result.metadata['bootstrap_statistics'].iloc[b], expected,
                                       rtol=1e-9, atol=1e-12)
        assert result.metadata['hac_lags_requested'] is None and result.metadata['hac_lags'] == lags
    assert calls == [len(y), len(y)]

"""Independent common-sample HAC references and explicit-family table contracts."""

# packages
import numpy as np
import pandas as pd
import pytest
from dataclasses import replace
from scipy.stats import norm

# qis
from qis.datasets.synthetic import generate_synthetic_prices
from qis.perfstats import multiple_testing, sharpe_inference
from qis.perfstats.multiple_testing import MultipleTestingMethod
from qis.perfstats.returns import to_returns
from qis.perfstats.significance import (compute_alpha_significance, compute_mean_significance,
                                       compute_sharpe_significance)
from qis.perfstats.tests.sharpe_test import _moment_reference
from qis.utils import regression


CONVENTIONS = dict(family_id='prespecified-family', return_basis='supplied excess returns',
                   cost_basis='net of supplied trading costs', sampling_frequency='ME')
METHODS = [MultipleTestingMethod.HOLM, MultipleTestingMethod.BHY]
DISPLAY_COLUMNS = ['estimate', 'hac_se', 'ci_lower_pointwise', 'ci_upper_pointwise']


def _panel(nobs: int = 96) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic mathematical panel with six factors and correlated serial residuals."""
    dates = pd.date_range('2000-01-31', periods=nobs, freq='ME', name='date')
    t = np.arange(nobs, dtype=float)
    x = pd.DataFrame({f'F{j}': 0.003 * (j + 1) + 0.02 * np.sin((j + 1) * 0.173 * t)
                      + 0.006 * np.cos((j + 1) * 0.411 * t) for j in range(6)}, index=dates)
    noise = 0.008 * np.sin(t * 0.079) + 0.004 * np.cos(t * 1.131)
    y = pd.DataFrame({
        'A': 0.002 + x.to_numpy() @ np.array([0.4, -0.2, 0.1, 0.3, -0.15, 0.2]) + noise,
        'B': -0.001 + x.to_numpy() @ np.array([-0.3, 0.5, 0.2, 0.1, 0.2, -0.1])
        + 0.7 * noise + 0.002 * np.sin(t * 0.89),
        'C': 0.0003 + x.to_numpy() @ np.array([0.2, 0.1, -0.3, 0.4, -0.2, 0.1])
        - 0.5 * noise + 0.003 * np.cos(t * 0.97),
    }, index=dates)
    y.columns.name = 'hypothesis'
    x.columns.name = 'factor'
    return y, x


def _correction_reference(pvalues: np.ndarray, method: MultipleTestingMethod) -> np.ndarray:
    """Defining prefix/suffix extrema, independent of the production correction backend."""
    order = np.argsort(pvalues)
    sorted_p = pvalues[order]
    count = len(pvalues)
    harmonic = sum(1.0 / rank for rank in range(1, count + 1))
    result = np.empty(count)
    for i in range(count):
        result[order[i]] = min(1.0, max((count - j) * sorted_p[j] for j in range(i + 1))) \
            if method is MultipleTestingMethod.HOLM else min(
                1.0, min(count * harmonic * sorted_p[j] / (j + 1) for j in range(i, count)),
            )
    return result


def _reference_table(y: pd.DataFrame, x: pd.DataFrame | None, lags: int,
                     confidence: float, method: MultipleTestingMethod) -> pd.DataFrame:
    """Direct demeaned mean or QR coefficient map with a dense Bartlett time kernel."""
    values = y.to_numpy()
    nobs = len(y)
    distances = np.abs(np.subtract.outer(np.arange(nobs), np.arange(nobs)))
    kernel = np.maximum(1.0 - distances / (lags + 1.0), 0.0)
    if x is None:
        estimates = values.mean(axis=0)
        residuals = values - estimates
        covariance = (residuals.T @ kernel @ residuals) / (nobs * (nobs - 1))
    else:
        design = np.column_stack((np.ones(nobs), x.to_numpy()))
        q, r = np.linalg.qr(design, mode='reduced')
        coefficient_map = np.linalg.solve(r, q.T)
        parameters = np.linalg.solve(r, q.T @ values)
        estimates = parameters[0]
        residuals = values - design @ parameters
        intercept_influence = coefficient_map[0, :, None] * residuals
        covariance = nobs / (nobs - design.shape[1]) * (
            intercept_influence.T @ kernel @ intercept_influence
        )
    se = np.sqrt(np.diag(covariance))
    statistics = estimates / se
    pvalues = 2.0 * norm.sf(np.abs(statistics))
    adjusted = _correction_reference(pvalues, method)
    critical = norm.ppf((1.0 + confidence) / 2.0)
    return pd.DataFrame({
        'estimate': estimates, 'hac_se': se, 'statistic': statistics, 'pvalue_raw': pvalues,
        'pvalue_adjusted': adjusted, 'reject': adjusted <= 0.05,
        'ci_lower_pointwise': estimates - critical * se,
        'ci_upper_pointwise': estimates + critical * se,
    }, index=y.columns.copy())


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('lags', [0, 3, 6])
@pytest.mark.parametrize('factor_count', [0, 1, 3, 6])
def test_mean_and_alpha_tables_match_independent_reference(
        method: MultipleTestingMethod, lags: int, factor_count: int) -> None:
    y, all_factors = _panel()
    x = None if factor_count == 0 else all_factors.iloc[:, :factor_count]
    kwargs = dict(CONVENTIONS, hac_lags=lags, confidence_level=0.9, method=method)
    result = compute_mean_significance(y, **kwargs) if x is None else (
        compute_alpha_significance(y, x, **kwargs)
    )
    expected = _reference_table(y, x, lags, 0.9, method)
    assert result.table.columns.tolist() == expected.columns.tolist()
    for name in expected:
        if name == 'reject':
            np.testing.assert_array_equal(result.table[name], expected[name])
        else:
            np.testing.assert_allclose(result.table[name], expected[name], rtol=1e-9, atol=1e-12)
    pd.testing.assert_index_equal(result.table.index, y.columns)
    meta = result.metadata
    assert meta['test'] == ('mean' if x is None else 'alpha')
    assert meta['null_value'] == 0.0 and meta['alternative'] == 'two-sided'
    assert meta['reference_distribution'] == 'normal'
    assert meta['confidence_interval'] == 'unadjusted/pointwise'
    assert meta['nobs'] == len(y) and meta['n_parameters'] == factor_count + 1
    assert meta['sample_start'] == y.index[0] and meta['sample_end'] == y.index[-1]
    assert meta['hac_lags'] == meta['hac_lags_requested'] == lags
    assert meta['span'] is None and meta['weighting'] == 'equal'
    assert meta['factor_columns'] == (() if x is None else tuple(x.columns))
    assert meta['sampling_frequency'] == 'ME' and meta['expected_index'] is None
    assert meta['return_convention'] == 'simple'
    assert meta['return_basis'] == CONVENTIONS['return_basis']
    assert meta['cost_basis'] == CONVENTIONS['cost_basis']
    assert meta['family_id'] == CONVENTIONS['family_id']
    assert meta['family_members'] == tuple(y.columns) and meta['n_hypotheses'] == 3
    assert meta['method'] is method and meta['confidence_level'] == 0.9
    assert meta['significance_level'] == 0.05 and meta['af'] is None


@pytest.mark.parametrize('factor_count', [1, 3, 6])
def test_alpha_matches_legacy_full_rank_ols_helper(factor_count: int) -> None:
    y, x = _panel()
    x = x.iloc[:, :factor_count]
    result = compute_alpha_significance(y, x, **CONVENTIONS)
    for name in y:
        old = regression.estimate_ols_alpha_beta_hac(x=x, y=y[name], hac_lags=3)
        fields = ['estimate', 'hac_se', 'pvalue_raw', 'ci_lower_pointwise', 'ci_upper_pointwise']
        np.testing.assert_allclose(
            result.table.loc[name, fields].to_numpy(dtype=float),
            [old.alpha, old.alpha_hac_se, old.alpha_pvalue, *old.alpha_confidence_interval],
            rtol=1e-9, atol=1e-12,
        )


@pytest.mark.parametrize('alpha', [False, True])
def test_annualisation_and_funding_descriptions_do_not_change_inference(alpha: bool) -> None:
    y, x = _panel()
    call = compute_alpha_significance if alpha else compute_mean_significance
    args = (y, x) if alpha else (y,)
    period = call(*args, **CONVENTIONS)
    annual = call(*args, **CONVENTIONS, af=12.0)
    np.testing.assert_allclose(annual.table[DISPLAY_COLUMNS], period.table[DISPLAY_COLUMNS] * 12.0)
    for name in ['statistic', 'pvalue_raw', 'pvalue_adjusted', 'reject']:
        np.testing.assert_array_equal(annual.table[name], period.table[name])
    assert annual.metadata['af'] == 12.0 and annual.metadata['annualised']
    assert annual.metadata['nobs'] == period.metadata['nobs']
    described = call(*args, **dict(CONVENTIONS, return_basis='supplied total returns',
                                  cost_basis='gross, no additional deduction'))
    pd.testing.assert_frame_equal(described.table, period.table)
    shifted = call(*(y - 0.001, x) if alpha else (y - 0.001,), **CONVENTIONS)
    np.testing.assert_allclose(shifted.table['estimate'], period.table['estimate'] - 0.001,
                               rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(shifted.table['hac_se'], period.table['hac_se'], rtol=1e-9)


@pytest.mark.parametrize('alpha', [False, True])
@pytest.mark.parametrize('scale', [0.01, 100.0, 1e-100, 1e100, -1e-100, -1e100])
def test_unit_and_sign_changes_preserve_validity_and_pvalues(alpha: bool, scale: float) -> None:
    y, x = _panel()
    call = compute_alpha_significance if alpha else compute_mean_significance
    base = call(*(y, x) if alpha else (y,), **CONVENTIONS)
    scaled_x = x * np.array([1e-80, -1e80, 1e-30, 1e30, -1e-50, 1e50])
    scaled = call(*(y * scale, scaled_x) if alpha else (y * scale,), **CONVENTIONS)
    np.testing.assert_allclose(scaled.table['estimate'] / scale, base.table['estimate'],
                               rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(scaled.table['hac_se'] / abs(scale), base.table['hac_se'],
                               rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(scaled.table['statistic'], np.sign(scale) * base.table['statistic'],
                               rtol=1e-9, atol=1e-12)
    for name in ['pvalue_raw', 'pvalue_adjusted']:
        np.testing.assert_allclose(scaled.table[name], base.table[name], rtol=1e-9, atol=1e-12)
    np.testing.assert_array_equal(scaled.table['reject'], base.table['reject'])
    lower = 'ci_lower_pointwise' if scale > 0 else 'ci_upper_pointwise'
    upper = 'ci_upper_pointwise' if scale > 0 else 'ci_lower_pointwise'
    np.testing.assert_allclose(scaled.table['ci_lower_pointwise'] / scale, base.table[lower],
                               rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(scaled.table['ci_upper_pointwise'] / scale, base.table[upper],
                               rtol=1e-9, atol=1e-12)


def test_family_order_duplicates_and_input_mutation_contract() -> None:
    y, x = _panel()
    y['A-copy'] = y['A']
    original_y, original_x = y.copy(deep=True), x.copy(deep=True)
    rng_state = np.random.get_state()
    result = compute_alpha_significance(y, x, **CONVENTIONS, method=MultipleTestingMethod.BHY)
    pd.testing.assert_series_equal(result.table.loc['A'], result.table.loc['A-copy'],
                                   check_names=False)
    reordered = compute_alpha_significance(y.iloc[:, ::-1], x.iloc[:, ::-1], **CONVENTIONS,
                                           method=MultipleTestingMethod.BHY)
    np.testing.assert_allclose(reordered.table.reindex(y.columns).to_numpy(dtype=float),
                               result.table.to_numpy(dtype=float), rtol=1e-9, atol=1e-12)
    assert reordered.metadata['family_members'] == tuple(y.columns[::-1])
    pd.testing.assert_frame_equal(y, original_y)
    pd.testing.assert_frame_equal(x, original_x)
    after_state = np.random.get_state()
    assert rng_state[0] == after_state[0] and rng_state[2:] == after_state[2:]
    np.testing.assert_array_equal(rng_state[1], after_state[1])


def test_shared_estimator_and_correction_outputs_are_consumed(
        monkeypatch: pytest.MonkeyPatch) -> None:
    y, x = _panel()
    fit_calls, correction_calls = [], []
    original_fit = regression.estimate_hac_regression
    original_correction = multiple_testing.adjust_pvalues

    def fit_spy(**kwargs):
        fit_calls.append(kwargs)
        fitted = original_fit(**kwargs)
        fitted.pvalues.loc['Intercept'] = [0.2, 0.4, 0.6]
        fitted.confidence_lower.loc['Intercept'] = -0.2
        fitted.confidence_upper.loc['Intercept'] = 0.3
        return fitted

    def correction_spy(pvalues, **kwargs):
        correction_calls.append((pvalues, kwargs))
        corrected = original_correction(pvalues, **kwargs)
        corrected['pvalue_adjusted'] = 0.75
        corrected['reject'] = False
        return corrected

    monkeypatch.setattr(regression, 'estimate_hac_regression', fit_spy)
    monkeypatch.setattr(multiple_testing, 'adjust_pvalues', correction_spy)
    for factors in [None, x]:
        result = compute_mean_significance(y, **CONVENTIONS) if factors is None else (
            compute_alpha_significance(y, factors, **CONVENTIONS)
        )
        assert fit_calls[-1]['span'] is None and fit_calls[-1]['hac_lags'] == 3
        pd.testing.assert_index_equal(fit_calls[-1]['y'].index, y.index)
        np.testing.assert_allclose(fit_calls[-1]['y'], y / y.abs().max())
        if factors is None:
            assert fit_calls[-1]['x'] is None
        else:
            np.testing.assert_allclose(fit_calls[-1]['x'], x / x.abs().max())
        np.testing.assert_array_equal(result.table['pvalue_raw'], [0.2, 0.4, 0.6])
        np.testing.assert_array_equal(result.table['pvalue_adjusted'], [0.75] * 3)
        np.testing.assert_allclose(result.table['ci_lower_pointwise'], -0.2 * y.abs().max())
        np.testing.assert_allclose(result.table['ci_upper_pointwise'], 0.3 * y.abs().max())
        assert not result.table['reject'].any()
        assert correction_calls[-1][1]['family_id'] == CONVENTIONS['family_id']
        assert tuple(correction_calls[-1][0].index) == tuple(y.columns)
    assert len(fit_calls) == len(correction_calls) == 2


def test_optional_lag_rule_and_no_silent_lag_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    y, _ = _panel(24)
    calls = []

    def rule(nobs):
        calls.append(nobs)
        return 5

    monkeypatch.setattr(regression, 'newey_west_lag_rule', rule)
    automatic = compute_mean_significance(y, **CONVENTIONS, hac_lags=None)
    fixed = compute_mean_significance(y, **CONVENTIONS, hac_lags=5)
    pd.testing.assert_frame_equal(automatic.table, fixed.table)
    assert calls == [24] and automatic.metadata['hac_lags'] == 5
    assert automatic.metadata['hac_lags_requested'] is None
    compute_mean_significance(y, **CONVENTIONS, hac_lags=23)
    with pytest.raises(ValueError, match='hac_lags'):
        compute_mean_significance(y, **CONVENTIONS, hac_lags=24)


def test_calendar_and_alignment_are_checked_before_fitting(monkeypatch: pytest.MonkeyPatch) -> None:
    y, x = _panel()

    def should_not_fit(**kwargs):
        raise AssertionError('invalid calendars must never reach estimation')

    monkeypatch.setattr(regression, 'estimate_hac_regression', should_not_fit)
    for bad_x in [x.iloc[::-1], x.iloc[1:], x.set_axis(x.index + pd.Timedelta(days=1))]:
        with pytest.raises(ValueError, match='identical indexes'):
            compute_alpha_significance(y, bad_x, **CONVENTIONS)
    for positions in [np.delete(np.arange(len(y)), 20), np.arange(0, len(y), 2)]:
        with pytest.raises(ValueError, match='complete'):
            compute_alpha_significance(y.iloc[positions], x.iloc[positions], **CONVENTIONS)
    for bad_y in [y.iloc[::-1], pd.concat([y, y.iloc[-1:]]), y.reset_index(drop=True)]:
        with pytest.raises(ValueError):
            compute_mean_significance(bad_y, **CONVENTIONS)
    with pytest.raises(ValueError, match='explicitly'):
        compute_mean_significance(y, **dict(CONVENTIONS, sampling_frequency=None))
    with pytest.raises(ValueError, match='only one'):
        compute_mean_significance(y, **CONVENTIONS, expected_index=y.index)
    with pytest.raises(ValueError, match='expected_index'):
        compute_mean_significance(y, **dict(CONVENTIONS, sampling_frequency=None),
                                  expected_index=y.index[:-1])


def test_custom_calendar_and_nullable_data() -> None:
    y, x = _panel()
    positions = np.delete(np.arange(len(y)), [2, 17, 45])
    y, x = y.iloc[positions], x.iloc[positions]
    expected_index = y.index.copy()
    result = compute_alpha_significance(
        y.astype('Float64'), x.astype('Float64'), **dict(CONVENTIONS, sampling_frequency=None),
        expected_index=expected_index,
    )
    expected = _reference_table(y, x, 3, 0.95, MultipleTestingMethod.HOLM)
    np.testing.assert_allclose(result.table.to_numpy(dtype=float), expected.to_numpy(dtype=float),
                               rtol=1e-9, atol=1e-12)
    pd.testing.assert_index_equal(result.metadata['expected_index'], expected_index)
    assert result.metadata['expected_index'] is not expected_index


@pytest.mark.parametrize('which', ['returns', 'factors'])
@pytest.mark.parametrize('bad_value', [np.nan, np.inf, -np.inf])
def test_nonfinite_member_rejects_whole_common_sample(which: str, bad_value: float) -> None:
    y, x = _panel()
    frame = y if which == 'returns' else x
    frame.iloc[15, 1] = bad_value
    with pytest.raises(ValueError, match=str(frame.columns[1])):
        compute_alpha_significance(y, x, **CONVENTIONS)


@pytest.mark.parametrize('settings', [
    {'hac_lags': True}, {'hac_lags': 2.5}, {'hac_lags': -1}, {'hac_lags': 96},
    {'confidence_level': 0}, {'confidence_level': 1}, {'confidence_level': np.nan},
    {'confidence_level': True}, {'confidence_level': '0.95'},
    {'af': 0}, {'af': -1}, {'af': np.inf}, {'af': np.nan}, {'af': True}, {'af': '12'},
    {'family_id': ''}, {'method': 'holm'}, {'significance_level': 0},
    {'return_basis': ''}, {'cost_basis': None}, {'sampling_frequency': 'bad-frequency'},
])
def test_invalid_settings_fail(settings: dict) -> None:
    y, _ = _panel()
    with pytest.raises(ValueError):
        compute_mean_significance(y, **dict(CONVENTIONS, **settings))


def test_short_samples_bad_containers_and_bad_columns() -> None:
    y, x = _panel()
    with pytest.raises(ValueError, match='24'):
        compute_mean_significance(y.iloc[:23], **CONVENTIONS)
    compute_mean_significance(y.iloc[:24], **CONVENTIONS)
    with pytest.raises(ValueError, match='T > k'):
        many = pd.DataFrame(np.ones((24, 23)), index=y.index[:24],
                            columns=[f'factor-{i}' for i in range(23)])
        compute_alpha_significance(y.iloc[:24], many, **CONVENTIONS)
    for bad in [None, y['A'], y.to_numpy(), y.iloc[:0], y.iloc[:, :0],
                y.astype(str), y.astype(bool), y.astype(complex),
                pd.concat([y, y.iloc[:, :1]], axis=1), y.rename(columns={'A': ''})]:
        with pytest.raises(ValueError):
            compute_mean_significance(bad, **CONVENTIONS)
    for bad in [None, x['F0'], x.iloc[:, :0], x.astype(str), x.rename(columns={'F0': 'Intercept'})]:
        with pytest.raises(ValueError):
            compute_alpha_significance(y, bad, **CONVENTIONS)


@pytest.mark.parametrize('scale', [1.0, 1e-100, 1e100])
def test_degenerate_members_and_invalid_designs_are_rejected_at_any_scale(scale: float) -> None:
    y, x = _panel()
    for value in [0.0, 0.1]:
        bad_y = y.copy()
        bad_y['B'] = value
        with pytest.raises(ValueError, match='B'):
            compute_mean_significance(bad_y * scale, **CONVENTIONS)
    fitted_y = pd.DataFrame({'self': 0.002 + 0.4 * x['F0'] + 0.2 * x['F1']
                             + 1e-16 * np.sin(np.arange(len(y)))}, index=y.index)
    with pytest.raises(ValueError, match='degenerate'):
        compute_alpha_significance(fitted_y * scale, x * scale, **CONVENTIONS)
    for bad_x in [x.assign(constant=0.1), x.assign(zero=0.0), x.assign(copy=2.0 * x['F0']),
                  x.assign(combination=x['F1'] - 0.3 * x['F2']), x.assign(A=y['A'])]:
        with pytest.raises(ValueError):
            compute_alpha_significance(y * scale, bad_x * scale, **CONVENTIONS)


def test_zero_hac_se_is_not_reported_as_significant(monkeypatch: pytest.MonkeyPatch) -> None:
    y, x = _panel()
    original = regression.estimate_hac_regression

    def zero_se(**kwargs):
        result = original(**kwargs)
        result.hac_se.loc['Intercept', 'B'] = 0.0
        return result

    monkeypatch.setattr(regression, 'estimate_hac_regression', zero_se)
    with pytest.raises(ValueError, match='B'):
        compute_alpha_significance(y, x, **CONVENTIONS)


def test_frozen_synthetic_panel_end_to_end() -> None:
    prices = generate_synthetic_prices(start='2010-01-01', end='2019-12-31')
    returns = to_returns(prices[['SEQ_US', 'SBD_TSY', 'SBD_IG', 'SBD_HY']],
                         freq='ME', is_log_returns=False).iloc[1:]
    y, x = returns[['SBD_IG', 'SBD_HY']], returns[['SEQ_US', 'SBD_TSY']]
    for factors in [None, x]:
        result = compute_mean_significance(y, **CONVENTIONS) if factors is None else (
            compute_alpha_significance(y, factors, **CONVENTIONS)
        )
        expected = _reference_table(y, factors, 3, 0.95, MultipleTestingMethod.HOLM)
        np.testing.assert_allclose(result.table.to_numpy(dtype=float),
                                   expected.to_numpy(dtype=float),
                                   rtol=1e-9, atol=1e-12)


def _sharpe_panel() -> tuple[pd.DataFrame, pd.Series]:
    """Correlated serial returns with a distinct fixed benchmark."""
    y, x = _panel()
    benchmark = (0.6 * x['F2'] - 0.2 * x['F4'] + 0.001
                 + 0.001 * np.cos(np.arange(len(y)) * 0.61)).rename('Fixed benchmark')
    return y, benchmark


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('lags', [0, 3, 12])
@pytest.mark.parametrize('af', [None, 12.0, 252.0])
def test_sharpe_table_matches_paired_and_four_moment_references(
        method: MultipleTestingMethod, lags: int, af: float | None) -> None:
    y, benchmark = _sharpe_panel()
    result = compute_sharpe_significance(y, benchmark, **CONVENTIONS, hac_lags=lags,
                                         af=af, method=method, confidence_level=0.9,
                                         significance_level=0.1)
    effective_af = 1.0 if af is None else af
    pvalues = []
    for name in y:
        direct = sharpe_inference.sharpe_difference_test(y[name], benchmark, lags, effective_af)
        fields = ['sharpe_candidate', 'sharpe_benchmark', 'estimate', 'hac_se',
                  'statistic', 'pvalue_raw']
        actual = result.table.loc[name, fields].to_numpy(dtype=float)
        expected = [direct.sr1_an, direct.sr2_an, direct.diff_an, direct.se_an,
                    direct.t_stat, direct.p_value]
        np.testing.assert_array_equal(actual, expected)
        independent = _moment_reference(np.column_stack((y[name], benchmark)), lags, effective_af)
        np.testing.assert_allclose(actual, independent, rtol=1e-9, atol=1e-12)
        critical = norm.ppf(0.95)
        np.testing.assert_allclose(
            result.table.loc[name, DISPLAY_COLUMNS[-2:]].to_numpy(dtype=float),
            direct.diff_an + np.array([-1.0, 1.0]) * critical * direct.se_an,
        )
        pvalues.append(direct.p_value)
    adjusted = _correction_reference(np.array(pvalues), method)
    np.testing.assert_allclose(result.table['pvalue_adjusted'], adjusted, rtol=1e-14)
    np.testing.assert_array_equal(result.table['reject'], adjusted <= 0.1)
    pd.testing.assert_index_equal(result.table.index, y.columns)
    meta = result.metadata
    assert meta['test'] == 'sharpe_difference' and meta['null_value'] == 0.0
    assert meta['comparison'] == 'candidate_minus_benchmark' and meta['paired']
    assert meta['benchmark_name'] == benchmark.name
    assert meta['reference_distribution'] == 'normal' and meta['alternative'] == 'two-sided'
    assert meta['hac_correction'] == 'none' and meta['hac_lags'] == lags
    assert meta['hac_lags_requested'] == lags and meta['nobs'] == len(y)
    assert meta['family_members'] == tuple(y.columns) and meta['n_hypotheses'] == len(y.columns)
    assert meta['family_id'] == CONVENTIONS['family_id'] and meta['method'] is method
    assert meta['sample_start'] == y.index[0] and meta['sample_end'] == y.index[-1]
    assert meta['sampling_frequency'] == 'ME' and meta['expected_index'] is None
    assert meta['af'] == af and meta['annualised'] == (af is not None)
    assert meta['return_basis'] == CONVENTIONS['return_basis']
    assert meta['cost_basis'] == CONVENTIONS['cost_basis'] and meta['return_convention'] == 'simple'
    assert meta['sharpe_convention'] == 'arithmetic mean/population standard deviation (ddof=0)'
    assert meta['confidence_interval'] == 'unadjusted/pointwise'
    assert meta['confidence_level'] == 0.9 and meta['significance_level'] == 0.1
    assert result.table.attrs['confidence_interval'] == 'unadjusted/pointwise'


@pytest.mark.parametrize('method', METHODS)
def test_sharpe_identical_proportional_and_reordered_family(method: MultipleTestingMethod) -> None:
    y, benchmark = _sharpe_panel()
    y[benchmark.name], y['Leveraged benchmark'], y['A-copy'] = benchmark, 3.0 * benchmark, y['A']
    original_y, original_benchmark = y.copy(deep=True), benchmark.copy(deep=True)
    rng_state = np.random.get_state()
    result = compute_sharpe_significance(y, benchmark, **CONVENTIONS, method=method)
    for name in [benchmark.name, 'Leveraged benchmark']:
        row = result.table.loc[name]
        assert all(row[column] == 0.0 for column in DISPLAY_COLUMNS + ['statistic'])
        assert row['pvalue_raw'] == row['pvalue_adjusted'] == 1.0 and not row['reject']
    pd.testing.assert_series_equal(result.table.loc['A'], result.table.loc['A-copy'],
                                   check_names=False)
    reversed_family = compute_sharpe_significance(y.iloc[:, ::-1], benchmark, **CONVENTIONS,
                                                  method=method)
    np.testing.assert_array_equal(reversed_family.table.reindex(y.columns), result.table)
    assert result.metadata['n_hypotheses'] == 6
    assert reversed_family.metadata['family_members'] == tuple(y.columns[::-1])
    pd.testing.assert_frame_equal(y, original_y)
    pd.testing.assert_series_equal(benchmark, original_benchmark)
    after_state = np.random.get_state()
    assert rng_state[0] == after_state[0] and rng_state[2:] == after_state[2:]
    np.testing.assert_array_equal(rng_state[1], after_state[1])


def test_sharpe_display_units_and_funding_descriptions() -> None:
    y, benchmark = _sharpe_panel()
    base = compute_sharpe_significance(y, benchmark, **CONVENTIONS)
    annual = compute_sharpe_significance(y, benchmark, **CONVENTIONS, af=12.0)
    fields = DISPLAY_COLUMNS + ['sharpe_candidate', 'sharpe_benchmark']
    np.testing.assert_allclose(annual.table[fields], np.sqrt(12.0) * base.table[fields])
    for column in ['statistic', 'pvalue_raw', 'pvalue_adjusted', 'reject']:
        np.testing.assert_array_equal(annual.table[column], base.table[column])
    scaled = compute_sharpe_significance(y * [1e-180, 1e180, 0.01], 1e150 * benchmark,
                                         **CONVENTIONS)
    np.testing.assert_allclose(scaled.table.to_numpy(dtype=float), base.table.to_numpy(dtype=float),
                               rtol=1e-9, atol=1e-12)
    described = compute_sharpe_significance(
        y, benchmark, **dict(CONVENTIONS, return_basis='supplied total returns',
                            cost_basis='gross'),
    )
    pd.testing.assert_frame_equal(described.table, base.table)
    assert annual.metadata['nobs'] == base.metadata['nobs']


def test_sharpe_adapter_consumes_existing_results_and_full_correction(
        monkeypatch: pytest.MonkeyPatch) -> None:
    y, benchmark = _sharpe_panel()
    calls, corrections = [], []
    original = sharpe_inference.sharpe_difference_test
    correct = multiple_testing.adjust_pvalues

    def paired_spy(returns1, returns2, n_lags, af):
        calls.append((returns1.copy(), returns2.copy(), n_lags, af))
        result = original(returns1, returns2, n_lags, af)
        return replace(result, diff_an=0.5, se_an=0.25, t_stat=7.0, p_value=0.4,
                       sr1_an=1.5, sr2_an=1.0)

    def correction_spy(pvalues, **kwargs):
        corrections.append((pvalues.copy(), kwargs))
        result = correct(pvalues, **kwargs)
        result['pvalue_adjusted'], result['reject'] = 0.8, False
        return result

    monkeypatch.setattr(sharpe_inference, 'sharpe_difference_test', paired_spy)
    monkeypatch.setattr(multiple_testing, 'adjust_pvalues', correction_spy)
    result = compute_sharpe_significance(y, benchmark, **CONVENTIONS, af=12, hac_lags=5)
    assert len(calls) == 3 and len(corrections) == 1
    for position, (candidate, fixed, lags, af) in enumerate(calls):
        pd.testing.assert_series_equal(candidate, y.iloc[:, position])
        pd.testing.assert_series_equal(fixed, benchmark)
        assert (lags, af) == (5, 12)
    assert corrections[0][1]['family_id'] == CONVENTIONS['family_id']
    pd.testing.assert_index_equal(corrections[0][0].index, y.columns)
    for column, expected in [('estimate', 0.5), ('hac_se', 0.25), ('statistic', 7.0),
                             ('pvalue_raw', 0.4), ('pvalue_adjusted', 0.8),
                             ('sharpe_candidate', 1.5), ('sharpe_benchmark', 1.0)]:
        np.testing.assert_array_equal(result.table[column], [expected] * 3)
    np.testing.assert_allclose(result.table['ci_lower_pointwise'], 0.5 - norm.ppf(0.975) * 0.25)
    np.testing.assert_allclose(result.table['ci_upper_pointwise'], 0.5 + norm.ppf(0.975) * 0.25)
    assert not result.table['reject'].any()


@pytest.mark.parametrize('field', ['n_obs', 'n_lags'])
def test_sharpe_changed_pair_sample_or_lag_is_rejected(field: str,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    y, benchmark = _sharpe_panel()
    original = sharpe_inference.sharpe_difference_test

    def changed(**kwargs):
        result = original(**kwargs)
        return replace(result, **{field: getattr(result, field) - 1})

    monkeypatch.setattr(sharpe_inference, 'sharpe_difference_test', changed)
    with pytest.raises(ValueError, match='common sample or lags'):
        compute_sharpe_significance(y, benchmark, **CONVENTIONS)


def test_sharpe_calendar_alignment_and_missing_rows_fail_before_paired_calls(
        monkeypatch: pytest.MonkeyPatch) -> None:
    y, benchmark = _sharpe_panel()

    def should_not_fit(**kwargs):
        raise AssertionError('incomplete or unpaired samples must not reach estimation')

    monkeypatch.setattr(sharpe_inference, 'sharpe_difference_test', should_not_fit)
    for bad in [benchmark.iloc[::-1], benchmark.iloc[1:],
                benchmark.set_axis(benchmark.index + pd.Timedelta(days=1))]:
        with pytest.raises(ValueError, match='identical indexes'):
            compute_sharpe_significance(y, bad, **CONVENTIONS)
    for candidate in [False, True]:
        for value in [np.nan, np.inf, -np.inf]:
            bad_y, bad_benchmark = y.copy(), benchmark.copy()
            if candidate:
                bad_y.iloc[15, 1] = value
            else:
                bad_benchmark.iloc[21] = value
            with pytest.raises(ValueError, match='B' if candidate else 'benchmark'):
                compute_sharpe_significance(bad_y, bad_benchmark, **CONVENTIONS)
    for positions in [np.delete(np.arange(len(y)), 20), np.arange(0, len(y), 2)]:
        with pytest.raises(ValueError, match='complete'):
            compute_sharpe_significance(y.iloc[positions], benchmark.iloc[positions], **CONVENTIONS)
    for bad_y in [y.iloc[::-1], pd.concat([y, y.iloc[-1:]]), y.reset_index(drop=True)]:
        with pytest.raises(ValueError):
            compute_sharpe_significance(bad_y, benchmark, **CONVENTIONS)
    with pytest.raises(ValueError, match='explicitly'):
        compute_sharpe_significance(y, benchmark, **dict(CONVENTIONS, sampling_frequency=None))
    with pytest.raises(ValueError, match='only one'):
        compute_sharpe_significance(y, benchmark, **CONVENTIONS, expected_index=y.index)


def test_sharpe_custom_calendar_labels_nullable_inputs_and_lag_rule(
        monkeypatch: pytest.MonkeyPatch) -> None:
    y, benchmark = _sharpe_panel()
    positions = np.delete(np.arange(len(y)), [2, 17, 45])
    y, benchmark = y.iloc[positions].astype('Float64'), benchmark.iloc[positions].astype('Float64')
    y.columns = pd.MultiIndex.from_tuples([('strategy', 'A'), ('strategy', 'B'), ('strategy', 'C')],
                                         names=['kind', 'name'])
    calls = []

    def rule(nobs):
        calls.append(nobs)
        return 5

    monkeypatch.setattr(regression, 'newey_west_lag_rule', rule)
    result = compute_sharpe_significance(y, benchmark.rename(None),
                                         **dict(CONVENTIONS, sampling_frequency=None),
                                         expected_index=y.index, hac_lags=None)
    fixed = compute_sharpe_significance(y, benchmark.rename(None),
                                        **dict(CONVENTIONS, sampling_frequency=None),
                                        expected_index=y.index, hac_lags=5)
    pd.testing.assert_frame_equal(result.table, fixed.table)
    pd.testing.assert_index_equal(result.table.index, y.columns)
    pd.testing.assert_index_equal(result.metadata['expected_index'], y.index)
    assert result.metadata['expected_index'] is not y.index
    assert calls == [len(y)] and result.metadata['hac_lags'] == 5
    assert result.metadata['hac_lags_requested'] is None
    assert result.metadata['benchmark_name'] is None


@pytest.mark.parametrize('settings', [
    {'hac_lags': True}, {'hac_lags': 2.5}, {'hac_lags': -1}, {'hac_lags': 96},
    {'confidence_level': 0}, {'confidence_level': 1}, {'confidence_level': np.nan},
    {'confidence_level': True}, {'confidence_level': '0.95'},
    {'af': 0}, {'af': -1}, {'af': np.inf}, {'af': np.nan}, {'af': True}, {'af': '12'},
    {'family_id': ''}, {'method': 'holm'}, {'significance_level': 0},
    {'return_basis': ''}, {'cost_basis': None}, {'sampling_frequency': 'bad-frequency'},
])
def test_sharpe_invalid_settings_fail(settings: dict) -> None:
    y, benchmark = _sharpe_panel()
    with pytest.raises(ValueError):
        compute_sharpe_significance(y, benchmark, **dict(CONVENTIONS, **settings))


def test_sharpe_bad_containers_and_small_samples_fail() -> None:
    y, benchmark = _sharpe_panel()
    for bad in [None, y['A'], y.to_numpy(), y.iloc[:0], y.iloc[:, :0],
                y.astype(str), y.astype(bool), y.astype(complex),
                pd.concat([y, y.iloc[:, :1]], axis=1), y.rename(columns={'A': ''})]:
        with pytest.raises(ValueError):
            compute_sharpe_significance(bad, benchmark, **CONVENTIONS)
    for bad in [None, benchmark.to_frame(), benchmark.to_numpy(), benchmark.astype(str),
                benchmark.astype(bool), benchmark.astype(complex)]:
        with pytest.raises(ValueError):
            compute_sharpe_significance(y, bad, **CONVENTIONS)
    with pytest.raises(ValueError, match='24'):
        compute_sharpe_significance(y.iloc[:23], benchmark.iloc[:23], **CONVENTIONS)
    result = compute_sharpe_significance(y.iloc[:24], benchmark.iloc[:24], **CONVENTIONS,
                                         hac_lags=23)
    assert result.metadata['nobs'] == 24 and result.metadata['hac_lags'] == 23
    with pytest.raises(ValueError, match='hac_lags'):
        compute_sharpe_significance(y.iloc[:24], benchmark.iloc[:24], **CONVENTIONS, hac_lags=24)


@pytest.mark.parametrize('scale', [1.0, 1e-180, 1e180])
def test_sharpe_degenerate_member_rejects_whole_family(scale: float) -> None:
    y, benchmark = _sharpe_panel()
    for constant in [0.0, 0.01]:
        with pytest.raises(ValueError, match='B.*degenerate'):
            compute_sharpe_significance(y.assign(B=constant) * scale, benchmark * scale,
                                         **CONVENTIONS)
        with pytest.raises(ValueError, match='benchmark.*degenerate'):
            compute_sharpe_significance(y * scale, benchmark * 0.0 + constant * scale,
                                         **CONVENTIONS)


def test_sharpe_frozen_synthetic_panel_end_to_end() -> None:
    prices = generate_synthetic_prices(start='2010-01-01', end='2019-12-31')
    returns = to_returns(prices[['SEQ_US', 'SBD_TSY', 'SBD_IG', 'SBD_HY']],
                         freq='ME', is_log_returns=False).iloc[1:]
    result = compute_sharpe_significance(returns.drop(columns='SEQ_US'), returns['SEQ_US'],
                                         **CONVENTIONS, af=12, method=MultipleTestingMethod.BHY)
    for name in result.table.index:
        reference = _moment_reference(returns[[name, 'SEQ_US']].to_numpy(), 3, 12)
        fields = ['sharpe_candidate', 'sharpe_benchmark', 'estimate', 'hac_se',
                  'statistic', 'pvalue_raw']
        np.testing.assert_allclose(result.table.loc[name, fields].to_numpy(dtype=float), reference,
                                   rtol=1e-9, atol=1e-12)

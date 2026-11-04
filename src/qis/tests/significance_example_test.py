"""Offline numerical and path contracts for the significance walkthrough in examples/."""

# packages
import importlib
import runpy
import socket
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
import yaml

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# qis
from qis import local_path
from qis.models.bootstrap import bootstrap_numba
from qis.perfstats.sharpe_inference import sharpe_difference_test


ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.skipif(
    not (ROOT / 'examples').is_dir(), reason='examples are not shipped in the wheel',
)


@pytest.fixture
def example(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))

    def refuse_network(*args, **kwargs):
        raise AssertionError('The significance example must be entirely offline')

    monkeypatch.setattr(socket.socket, 'connect', refuse_network)
    monkeypatch.setattr(socket, 'create_connection', refuse_network)
    yield importlib.import_module('examples.perfstats.factor_and_backtest_significance')
    plt.close('all')


def _bartlett_mean_se(series: np.ndarray, lags: int) -> float:
    """Dense time-domain kernel reference, independent of the production lag-sum code."""
    centered = series - series.mean()
    distance = np.abs(np.subtract.outer(np.arange(len(series)), np.arange(len(series))))
    kernel = np.maximum(1 - distance / (lags + 1), 0)
    return np.sqrt(centered @ kernel @ centered / (len(series) * (len(series) - 1)))


def _prewhitened_mean_se(series: np.ndarray, lags: int) -> tuple[float, float]:
    """AR(1) filter, Bartlett sum on the innovations and recolouring, as in the guide."""
    e = series - series.mean()
    rho = (e[1:] @ e[:-1]) / (e[:-1] @ e[:-1])
    v = e[1:] - rho * e[:-1]
    distance = np.abs(np.subtract.outer(np.arange(len(v)), np.arange(len(v))))
    kernel = np.maximum(1 - distance / (lags + 1), 0)
    return np.sqrt((v @ kernel @ v) / (1 - rho) ** 2 / (len(e) * (len(e) - 1))), rho


def test_analytic_example_matches_independent_references(example, capsys, monkeypatch):
    def refuse_output(*args, **kwargs):
        raise AssertionError('Default analytic mode must not resolve/write output')

    monkeypatch.setattr(example.local_path, 'get_output_path', refuse_output)
    results = example.run_local(example.Locals.ANALYTIC)
    returns, benchmark = example.load_example_returns()
    assert returns.shape == (251, 9) and not returns.isna().any().any()
    assert 'SEQ_EM' not in returns and benchmark.index.equals(returns.index)
    keys = {f'{family}_{inference}_{method}' for family in ('mean', 'alpha')
            for inference in ('bartlett_hac', 'prewhitened_ar1') for method in ('HOLM', 'BHY')}
    assert set(results) == keys | {'sharpe_HOLM', 'sharpe_BHY'}
    means = results['mean_bartlett_hac_HOLM']
    np.testing.assert_allclose(means.table.estimate, 12 * returns.mean())
    responses = returns.drop(columns=example.FACTORS)
    design = np.column_stack((np.ones(len(returns)), returns[example.FACTORS]))
    coefficients = np.linalg.lstsq(design, responses.to_numpy(), rcond=None)[0]
    np.testing.assert_allclose(results['alpha_bartlett_hac_HOLM'].table.estimate,
                               12 * coefficients[0])
    first = returns.iloc[:, 0].to_numpy()
    assert means.table.hac_se.iloc[0] == pytest.approx(12 * _bartlett_mean_se(first, 3), rel=1e-10)
    prewhitened = results['mean_prewhitened_ar1_HOLM']
    se, rho = _prewhitened_mean_se(first, 3)
    assert prewhitened.table.hac_se.iloc[0] == pytest.approx(12 * se, rel=1e-10)
    assert prewhitened.table.rho.iloc[0] == pytest.approx(rho, rel=1e-10)
    np.testing.assert_array_equal(prewhitened.table.estimate, means.table.estimate)
    assert 'rho' not in means.table
    for name in returns:
        paired = sharpe_difference_test(returns[name], benchmark, n_lags=3, af=12)
        assert results['sharpe_HOLM'].table.loc[name, 'estimate'] == pytest.approx(paired.diff_an)
    for name, result in results.items():
        assert result.metadata['nobs'] == 251 and result.metadata['hac_lags'] == 3
        assert result.metadata['sample_start'] == returns.index[0]
        assert result.metadata['sample_end'] == returns.index[-1]
        assert result.table.attrs['confidence_interval'] == 'unadjusted/pointwise'
        expected = ('prewhitened_ar1' if 'prewhitened' in name else 'bartlett_hac')
        if not name.startswith('sharpe'):
            assert result.metadata['intercept_inference'] == expected
    assert results['alpha_bartlett_hac_HOLM'].metadata['factor_columns'] == tuple(example.FACTORS)
    assert results['sharpe_HOLM'].metadata['benchmark_name'] == benchmark.name
    printed = capsys.readouterr().out
    assert (f"Common sample: {returns.index[0]:%Y-%m-%d} to {returns.index[-1]:%Y-%m-%d}; "
            '251 monthly simple returns') in printed
    assert 'Direct calls reproduce the family columns' in printed
    assert len(plt.get_fignums()) == 1


def test_inference_comparison_and_direct_calls(example):
    returns, benchmark = example.load_example_returns()
    results = example.run_families(returns, benchmark)
    for family in ('mean', 'alpha'):
        table = example.compare_inference(results, family)
        bartlett = results[f'{family}_bartlett_hac_HOLM'].table
        prewhitened = results[f'{family}_prewhitened_ar1_HOLM'].table
        np.testing.assert_allclose(table.se_ratio, prewhitened.hac_se / bartlett.hac_se)
        np.testing.assert_array_equal(table.rho, prewhitened.rho)
        np.testing.assert_array_equal(table.holm_bartlett, bartlett.pvalue_adjusted)
        np.testing.assert_array_equal(
            table.bhy_prewhitened, results[f'{family}_prewhitened_ar1_BHY'].table.pvalue_adjusted)
        assert (table.se_ratio > 0.9).all() and (table.se_ratio < 1.1).all()
    direct = example.direct_calls(returns, results)
    assert direct.index.equals(returns.columns) and (direct.innovations == 250).all()
    fig = example.plot_family_summary(example.compare_inference(results, 'mean'),
                                      example.compare_inference(results, 'alpha'))
    assert len(fig.axes) == 2


def test_example_family_corrections_are_not_raw_pvalues(example):
    returns, benchmark = example.load_example_returns()
    results = example.run_families(returns, benchmark)
    for name in ('mean_bartlett_hac', 'mean_prewhitened_ar1', 'alpha_bartlett_hac', 'sharpe'):
        raw = results[f'{name}_HOLM'].table.pvalue_raw.to_numpy()
        order = np.argsort(raw)
        sorted_p = raw[order]
        n = len(raw)
        for method in ('HOLM', 'BHY'):
            if method == 'HOLM':
                adjusted = np.maximum.accumulate((n - np.arange(n)) * sorted_p)
            else:
                harmonic = sum(1 / i for i in range(1, n + 1))
                adjusted = np.minimum.accumulate(
                    (n * harmonic * sorted_p / np.arange(1, n + 1))[::-1],
                )[::-1]
            expected = np.empty(n)
            expected[order] = np.minimum(adjusted, 1)
            table = results[f'{name}_{method}'].table
            np.testing.assert_allclose(table.pvalue_raw, raw)
            np.testing.assert_allclose(table.pvalue_adjusted, expected)
            np.testing.assert_array_equal(table.reject, expected <= .05)


def test_bootstrap_families_share_one_generated_index_matrix(example, monkeypatch):
    generator = bootstrap_numba.generate_bootstrapped_indices
    generated = []

    def spy(*args, **kwargs):
        generated.append(kwargs)
        return generator(*args, **kwargs)

    monkeypatch.setattr(bootstrap_numba, 'generate_bootstrapped_indices', spy)
    returns, _ = example.load_example_returns()
    results = example.run_bootstrap_families(returns)
    assert len(generated) == 1 and len(results) == 4
    first = results['bootstrap_mean_bartlett_hac']
    assert first.metadata['index_source'] == 'generated'
    assert first.metadata['bootstrap_index_shape'] == (251, example.BOOTSTRAP_DRAWS)
    assert first.metadata['seed'] == 20261101 and first.metadata['block_size'] == 6
    assert first.metadata['first_rank_reachable']
    for name, result in results.items():
        if name != 'bootstrap_mean_bartlett_hac':
            assert result.metadata['index_source'] == 'supplied'
            assert result.metadata['indices_sha256'] == first.metadata['indices_sha256']
        np.testing.assert_allclose(
            result.table.pvalue_raw,
            (1 + result.table.bootstrap_exceedances) / (example.BOOTSTRAP_DRAWS + 1),
        )
        assert result.metadata['intercept_inference'] == (
            'prewhitened_ar1' if 'prewhitened' in name else 'bartlett_hac')
        assert ('rho' in result.table) == ('prewhitened' in name)
        assert result.metadata['interval_method'] == (
            'normal prewhitened HAC' if 'prewhitened' in name else 'normal HAC')


@pytest.mark.parametrize('explicit', [False, True])
def test_export_uses_yaml_or_explicit_override(example, tmp_path, monkeypatch, explicit):
    yaml_output = tmp_path / 'configured'
    config = tmp_path / 'private.yaml'
    config.write_text(yaml.safe_dump({'OUTPUT_PATH': str(yaml_output)}), encoding='utf-8')
    monkeypatch.setattr(local_path, '_SETTINGS_PATH', config)
    override = tmp_path / 'explicit' if explicit else None
    try:
        results = example.run_local(example.Locals.EXPORT, output_path=override)
        output = (override if explicit else yaml_output) / 'factor_and_backtest_significance'
        names = {path.name for path in output.rglob('*') if path.is_file()}
        assert names == {*(f'{name}.csv' for name in results),
                         'mean_inference_comparison.csv', 'alpha_inference_comparison.csv',
                         'family_summary.PNG'}
        if explicit:
            assert not yaml_output.exists()
        assert len(results) == 14
        for name, result in results.items():
            saved = pd.read_csv(output / f'{name}.csv', index_col=0)
            pd.testing.assert_frame_equal(saved, result.table, check_exact=False, atol=1e-14)
    finally:
        local_path.get_paths.cache_clear()


def test_main_guard_and_invalid_branch_do_not_write(example, monkeypatch, capsys):
    def refuse_output(*args, **kwargs):
        raise AssertionError('Default main guard must not write output')

    monkeypatch.setattr(example.local_path, 'get_output_path', refuse_output)
    namespace = runpy.run_path(str(ROOT / 'examples/perfstats/factor_and_backtest_significance.py'),
                               run_name='__main__')
    assert namespace['Locals'].ANALYTIC.name == 'ANALYTIC'
    assert 'Common sample:' in capsys.readouterr().out
    with pytest.raises(ValueError, match='Unknown example branch'):
        example.run_local('CALIBRATION')

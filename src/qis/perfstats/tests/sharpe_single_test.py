"""Single-Sharpe references, contracts, unchanged paired inference and chart evidence."""
from dataclasses import replace
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import toeplitz
from scipy.stats import norm

import qis
from qis.datasets.synthetic import generate_synthetic_prices
from qis.perfstats.sharpe_inference import SharpeInference, sharpe_ratio_test
from qis.utils import regression


@pytest.fixture(scope='module')
def returns():
    prices = generate_synthetic_prices(end='2006-06-30', apply_quirks=False)
    return prices.SEQ_US.pct_change(fill_method=None).iloc[1:]


@pytest.mark.parametrize('af', [1.0, 12.0, 252.0])
def test_iid_closed_form(returns, af):
    actual = sharpe_ratio_test(returns, method=SharpeInference.IID_NORMAL, af=af)
    theta = returns.mean() / returns.std(ddof=0)
    estimate = np.sqrt(af) * theta
    se = np.sqrt(af * (1 + theta ** 2 / 2) / len(returns))
    np.testing.assert_allclose(
        [actual.sr_an, actual.se_an, actual.ci_lower, actual.ci_upper, actual.p_value],
        [estimate, se, estimate - norm.ppf(.975) * se, estimate + norm.ppf(.975) * se,
         2 * norm.sf(abs(estimate / se))], rtol=1e-12)
    assert actual.n_lags == 0 and actual.n_obs == len(returns)


@pytest.mark.parametrize('lags', [0, 1, 3, 20])
def test_hac_two_raw_moment_reference(returns, lags):
    """Dense covariance of raw moments, not the implementation's scalar score path."""
    values = returns.to_numpy()
    n = len(values)
    mu, second = values.mean(), np.mean(values ** 2)
    sigma = np.sqrt(second - mu ** 2)
    gradient = np.array([second / sigma ** 3, -mu / (2 * sigma ** 3)])
    moments = np.column_stack([values - mu, values ** 2 - second])
    kernel = toeplitz(np.maximum(1 - np.arange(n) / (lags + 1), 0))
    reference_se = np.sqrt(252 * gradient @ (moments.T @ kernel @ moments) @ gradient / n ** 2)
    actual = sharpe_ratio_test(returns, method=SharpeInference.HAC, af=252, n_lags=lags)
    np.testing.assert_allclose(actual.se_an, reference_se, rtol=2e-12)
    assert actual.n_lags == lags
    # HAC(0) equals the empirical skew/kurtosis expression, not a normal assumption.
    if lags == 0:
        z = (values - mu) / sigma
        theta = mu / sigma
        variance = (1 - np.mean(z ** 3) * theta + (np.mean(z ** 4) - 1) * theta ** 2 / 4)
        assert actual.se_an == pytest.approx(np.sqrt(252 * variance / n))


def test_hac_uses_shared_estimator_and_automatic_lags(returns, monkeypatch):
    baseline = sharpe_ratio_test(returns, method=SharpeInference.HAC, af=12)
    original = regression.estimate_hac_mean
    calls = []

    def twice(y, **kwargs):
        calls.append(kwargs['hac_lags'])
        result = original(y, **kwargs)
        return replace(result, hac_se=result.hac_se * 2)

    monkeypatch.setattr(regression, 'estimate_hac_mean', twice)
    actual = sharpe_ratio_test(returns, method=SharpeInference.HAC, af=12)
    assert calls == [regression.newey_west_lag_rule(len(returns))]
    assert actual.se_an == pytest.approx(2 * baseline.se_an)
    assert actual.sr_an == baseline.sr_an


@pytest.mark.parametrize('bad_se', [0.0, -1.0, np.nan, np.inf])
def test_reject_undefined_hac_uncertainty(returns, monkeypatch, bad_se):
    original = regression.estimate_hac_mean

    def broken(y, **kwargs):
        return replace(original(y, **kwargs), hac_se=bad_se)

    monkeypatch.setattr(regression, 'estimate_hac_mean', broken)
    with pytest.raises(ValueError, match='standard error'):
        sharpe_ratio_test(returns, method=SharpeInference.HAC, af=252)


@pytest.mark.parametrize('method', list(SharpeInference))
def test_scale_sign_annualisation_null_and_interval(method, returns):
    base = sharpe_ratio_test(returns, method=method, af=1, confidence_level=.9)
    for multiplier in (1e-180, 1e180):
        scaled = sharpe_ratio_test(multiplier * returns, method=method, af=1,
                                  confidence_level=.9)
        np.testing.assert_allclose([scaled.sr_an, scaled.se_an], [base.sr_an, base.se_an])
    negative = sharpe_ratio_test(-returns, method=method, af=1, confidence_level=.9)
    assert negative.ci_lower == pytest.approx(-base.ci_upper)
    assert negative.p_value == pytest.approx(base.p_value)
    annual = sharpe_ratio_test(returns, method=method, af=252, confidence_level=.9)
    assert annual.se_an == pytest.approx(base.se_an * np.sqrt(252))
    assert annual.p_value == pytest.approx(base.p_value)
    shifted_null = sharpe_ratio_test(returns, method=method, af=1, confidence_level=.9,
                                    null_sharpe=base.sr_an)
    assert shifted_null.p_value == 1 and shifted_null.t_stat == 0
    assert shifted_null.ci_lower == base.ci_lower
    wider = sharpe_ratio_test(returns, method=method, af=1, confidence_level=.99)
    assert wider.ci_lower < base.ci_lower < base.ci_upper < wider.ci_upper


def test_dependence_changes_uncertainty_not_sharpe(returns):
    smooth = returns.ewm(alpha=.1, adjust=False).mean()
    iid = sharpe_ratio_test(smooth, method=SharpeInference.IID_NORMAL, af=252)
    hac = sharpe_ratio_test(smooth, method=SharpeInference.HAC, af=252, n_lags=20)
    assert hac.sr_an == iid.sr_an and hac.se_an > 2 * iid.se_an


@pytest.mark.parametrize('settings', [
    dict(method='hac'), dict(af=True), dict(af=0), dict(af=np.inf),
    dict(confidence_level=0), dict(confidence_level=1), dict(confidence_level=np.nan),
    dict(confidence_level=True), dict(null_sharpe=np.inf), dict(null_sharpe=True),
    dict(n_lags=-1), dict(n_lags=2.5), dict(n_lags=True), dict(n_lags=10_000),
    dict(method=SharpeInference.IID_NORMAL, n_lags=3),
])
def test_invalid_settings(returns, settings):
    kwargs = dict(method=SharpeInference.HAC, af=252)
    kwargs.update(settings)
    with pytest.raises(ValueError):
        sharpe_ratio_test(returns, **kwargs)


@pytest.mark.parametrize('defect', ['nan', 'inf', 'duplicate', 'reversed', 'nat', 'short',
                                     'constant', 'zero', 'bool', 'complex', 'frame', 'index'])
def test_invalid_data(returns, defect):
    data = returns.copy()
    if defect in ('nan', 'inf'):
        data.iloc[10] = np.nan if defect == 'nan' else np.inf
    elif defect == 'duplicate':
        data.index = data.index.where(data.index != data.index[1], data.index[0])
    elif defect == 'reversed':
        data = data.iloc[::-1]
    elif defect == 'nat':
        data.index = data.index.where(data.index != data.index[1], pd.NaT)
    elif defect == 'short':
        data = data.iloc[:23]
    elif defect in ('constant', 'zero'):
        data[:] = 1.0 if defect == 'constant' else 0.0
    elif defect == 'bool':
        data = data > 0
    elif defect == 'complex':
        data = data.astype(complex)
    elif defect == 'frame':
        data = data.to_frame()
    elif defect == 'index':
        data.index = pd.RangeIndex(len(data))
    with pytest.raises(ValueError):
        sharpe_ratio_test(data, method=SharpeInference.HAC, af=252)


def test_minimum_sample_and_nullable_dtype(returns):
    data = returns.iloc[:24].astype('Float64')
    assert sharpe_ratio_test(data, method=SharpeInference.HAC, af=252).n_obs == 24
    data.iloc[5] = pd.NA
    with pytest.raises(ValueError, match='finite'):
        sharpe_ratio_test(data, method=SharpeInference.HAC, af=252)


def test_high_confidence_and_no_input_mutation(returns):
    before = returns.copy(deep=True)
    result = sharpe_ratio_test(returns, method=SharpeInference.IID_NORMAL, af=252,
                               confidence_level=np.nextafter(1.0, 0.0))
    assert np.isfinite([result.ci_lower, result.ci_upper]).all()
    pd.testing.assert_series_equal(returns, before)


def test_existing_defaults_and_root_exports_unchanged():
    import inspect
    signature = inspect.signature(qis.sharpe_difference_test)
    assert signature.parameters['n_lags'].default == 3
    assert signature.parameters['af'].default == 12
    assert 'sharpe_ratio_test' not in qis.__all__


def test_chart_and_export(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[4]
    if not (root / 'examples').is_dir():
        pytest.skip('Repository examples do not ship in the wheel')
    monkeypatch.syspath_prepend(str(root))
    example = importlib.import_module('examples.perfstats.sharpe_confidence_intervals')
    import socket
    monkeypatch.setattr(socket, 'create_connection',
                        lambda *a, **kw: pytest.fail('example must run offline'))
    monkeypatch.setattr(example.plt, 'show', lambda: None)
    settings = tmp_path / 'private.yaml'
    settings.write_text('OUTPUT_PATH: configured-output\n', encoding='utf-8')
    figure, intervals, comparisons = example.run_local(example.Locals.EXPORT,
                                                       settings_path=settings)
    try:
        half = intervals.loc[126]
        np.testing.assert_allclose([half.sr_an, half.se_an, half.ci_lower, half.ci_upper],
                                   [2, 1.41981443, -.78278515, 4.78278515], atol=1e-8)
        np.testing.assert_allclose(intervals.sr_an, 2, atol=1e-12)
        np.testing.assert_allclose(intervals.se_an,
                                   np.sqrt((252 + 2) / intervals.index.to_numpy()), rtol=1e-12)
        assert intervals.loc[252].ci_lower > 0 and len(comparisons) == 3
        # The filled polygon must use the saved bounds, not an unrelated illustration.
        vertices = figure.axes[0].collections[0].get_paths()[0].vertices
        assert np.max(vertices[:, 1]) == pytest.approx(intervals.ci_upper.max())
        output = tmp_path / 'configured-output' / 'sharpe_confidence_intervals'
        assert (output / 'sharpe_confidence_intervals.png').stat().st_size > 10_000
        roundtrip = pd.read_csv(output / 'track_length.csv', index_col='n_obs')
        np.testing.assert_allclose(roundtrip.ci_lower, intervals.ci_lower)
        # Explicit directories bypass YAML, but display-only execution resolves neither.
        monkeypatch.setattr(example, 'build_example', lambda: (figure, intervals, comparisons))
        example.run_local(example.Locals.EXPORT, output_path=tmp_path / 'explicit',
                          settings_path=tmp_path / 'absent.yaml')
        monkeypatch.setattr(example.local_path, 'get_output_path',
                            lambda *a, **kw: pytest.fail('display mode must not resolve paths'))
        example.run_local(example.Locals.CHART)
    finally:
        example.plt.close(figure)

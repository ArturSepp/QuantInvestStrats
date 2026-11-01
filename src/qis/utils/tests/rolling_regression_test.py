"""Independent endpoint references and timing contracts for rolling HAC estimates."""

# packages
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

# qis
from qis.datasets.synthetic import generate_synthetic_prices
from qis.perfstats.returns import to_returns
from qis.utils import regression
from qis.utils.dates import TimePeriod
from qis.utils.rolling_regression import RollingHacStatus, estimate_rolling_hac_regression


STATISTICS = ('coefficients', 'hac_se', 'pvalues', 'confidence_lower', 'confidence_upper')
MODES = [(None, None), (18, None), (None, 12.0), (18, 12.0)]


def _panel(nobs: int = 72) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Small deterministic mathematical fixture with changing slopes and serial errors."""
    t = np.arange(nobs, dtype=float)
    dates = pd.date_range('2010-01-31', periods=nobs, freq='ME', name='date')
    x = pd.DataFrame({'Market': np.sin(0.7 * t), 'Value': np.cos(0.37 * t)}, index=dates)
    y = pd.DataFrame({
        'A': 0.1 + (0.2 + t / 60.0) * x['Market'] + 0.3 * x['Value'] + np.cos(0.83 * t),
        'B': -0.2 + 0.7 * x['Market'] - 0.4 * x['Value'] + np.sin(1.13 * t),
    }, index=dates)
    return y, x


def _dense_reference(y: np.ndarray, x: np.ndarray, span: float | None,
                     lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Weighted QR coefficients and a dense time-kernel sandwich, without lag loops."""
    nobs = len(y)
    design = np.column_stack([np.ones(nobs), x])
    weights = (np.ones(nobs) if span is None
               else (1.0 - 2.0 / (span + 1.0)) ** np.arange(nobs - 1, -1, -1))
    q, r = np.linalg.qr(np.sqrt(weights[:, None]) * design, mode='reduced')
    parameters = np.linalg.solve(r, q.T @ (np.sqrt(weights[:, None]) * y))
    residuals = y - design @ parameters
    bread = np.linalg.solve(r, np.linalg.solve(r.T, np.eye(design.shape[1])))
    influence = np.concatenate([
        (weights[:, None] * design * residuals[:, j, None]) @ bread
        for j in range(y.shape[1])
    ], axis=1)
    lag = min(lags, nobs - 1)
    distances = np.abs(np.subtract.outer(np.arange(nobs), np.arange(nobs)))
    kernel = np.maximum(1.0 - distances / (lag + 1.0), 0.0)
    covariance = nobs / (nobs - design.shape[1]) * influence.T @ kernel @ influence
    return parameters, covariance


@pytest.mark.parametrize('window,span', MODES)
@pytest.mark.parametrize('factor_count', [0, 1, 2])
def test_all_modes_delegate_exact_samples_and_match_references(
        window: int | None, span: float | None, factor_count: int,
        monkeypatch: pytest.MonkeyPatch) -> None:
    y, factors = _panel()
    x = None if factor_count == 0 else factors.iloc[:, :factor_count]
    dates = y.index[[0, 10, 17, 23, 38, 71]]
    calls = []
    original = regression.estimate_hac_regression

    def tracked(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(regression, 'estimate_hac_regression', tracked)
    result = estimate_rolling_hac_regression(
        y=y, x=x, window=window, span=span, min_periods=12, fit_dates=dates,
        hac_lags=4, confidence_level=0.9, include_covariance=True,
    )
    pd.testing.assert_index_equal(result.coefficients.index, dates)
    assert len(calls) == 4
    for date in dates:
        row = y.index.get_loc(date)
        first = 0 if window is None else max(0, row - window + 1)
        metadata = result.diagnostics.loc[date]
        assert metadata['nobs'] == row + 1 - first
        assert metadata['actual_start'] == y.index[first]
        assert metadata['actual_end'] == date
        if row < 11:
            assert metadata['status'] is RollingHacStatus.WARMUP
            assert all(getattr(result, name).loc[date].isna().all() for name in STATISTICS)
            assert date not in result.parameter_covariance
            continue
        assert metadata['status'] is RollingHacStatus.FITTED
        sample_y = y.iloc[first:row + 1]
        sample_x = None if x is None else x.iloc[first:row + 1]
        call = calls.pop(0)
        pd.testing.assert_frame_equal(call['y'], sample_y)
        if x is None:
            assert call['x'] is None
        else:
            pd.testing.assert_frame_equal(call['x'], sample_x)
        expected = original(y=sample_y, x=sample_x, span=span,
                            hac_lags=4, confidence_level=0.9)
        for name in STATISTICS:
            np.testing.assert_array_equal(
                getattr(result, name).loc[date], getattr(expected, name).to_numpy().T.ravel(),
            )
        pd.testing.assert_frame_equal(
            result.parameter_covariance[date], expected.parameter_covariance,
        )
        assert metadata['hac_lags'] == expected.hac_lags
        assert metadata['effective_nobs'] == expected.effective_nobs
        x_array = np.empty((len(sample_y), 0)) if x is None else sample_x.to_numpy()
        coefficients, covariance = _dense_reference(sample_y.to_numpy(), x_array, span, 4)
        np.testing.assert_allclose(result.coefficients.loc[date], coefficients.T.ravel(),
                                   rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(result.parameter_covariance[date], covariance,
                                   rtol=1e-10, atol=1e-12)
        se = np.sqrt(np.diag(covariance))
        np.testing.assert_allclose(result.hac_se.loc[date], se, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(result.pvalues.loc[date],
                                   2.0 * norm.sf(np.abs(coefficients.T.ravel() / se)),
                                   rtol=1e-10, atol=1e-12)
        if x is None:
            weights = (np.ones(len(sample_y)) if span is None else
                       (1.0 - 2.0 / (span + 1.0)) ** np.arange(len(sample_y) - 1, -1, -1))
            np.testing.assert_allclose(result.coefficients.loc[date],
                                       np.average(sample_y, weights=weights, axis=0), atol=1e-12)


@pytest.mark.parametrize('window,span', MODES)
def test_future_changes_appends_and_output_filtering_preserve_history(
        window: int | None, span: float | None) -> None:
    y, x = _panel()
    settings = dict(window=window, span=span, min_periods=12)
    baseline = estimate_rolling_hac_regression(y=y, x=x, **settings)
    changed_y, changed_x = y.copy(), x.copy()
    changed_y.iloc[40:] += 100.0
    changed_x.iloc[40:] *= -7.0
    changed = estimate_rolling_hac_regression(y=changed_y, x=changed_x, **settings)
    prefix = estimate_rolling_hac_regression(y=y.iloc[:40], x=x.iloc[:40], **settings)
    output_dates = y.index[17::7]
    scheduled = estimate_rolling_hac_regression(y=y, x=x, fit_dates=output_dates, **settings)
    period = TimePeriod(start=y.index[30], end=y.index[59])
    limited = estimate_rolling_hac_regression(y=y, x=x, time_period=period, **settings)
    for name in STATISTICS:
        full = getattr(baseline, name)
        pd.testing.assert_frame_equal(full.iloc[:40], getattr(changed, name).iloc[:40])
        pd.testing.assert_frame_equal(full.iloc[:40], getattr(prefix, name))
        pd.testing.assert_frame_equal(full.loc[output_dates], getattr(scheduled, name))
        pd.testing.assert_frame_equal(period.locate(full), getattr(limited, name))
    pd.testing.assert_frame_equal(period.locate(baseline.diagnostics), limited.diagnostics)


@pytest.mark.parametrize('span', [None, 12.0])
def test_window_boundary_and_expanding_ewma_older_observations(span: float | None) -> None:
    y, _ = _panel()
    date = y.index[-1:]
    baseline = estimate_rolling_hac_regression(y=y, window=18, span=span, fit_dates=date)
    assert baseline.diagnostics.iloc[0]['status'] is RollingHacStatus.FITTED
    assert baseline.diagnostics.iloc[0]['nobs'] == 18
    assert np.isfinite(baseline.coefficients.to_numpy()).all()
    old_changed = y.copy()
    old_changed.iloc[:-18] += 1000.0
    outside = estimate_rolling_hac_regression(y=old_changed, window=18, span=span, fit_dates=date)
    for name in STATISTICS:
        pd.testing.assert_frame_equal(getattr(baseline, name), getattr(outside, name))
    boundary_changed = y.copy()
    boundary_changed.iloc[-18] += 10.0
    boundary = estimate_rolling_hac_regression(y=boundary_changed, window=18, span=span,
                                               fit_dates=date)
    assert not np.allclose(baseline.coefficients, boundary.coefficients)
    if span is not None:
        expanding = estimate_rolling_hac_regression(y=old_changed, span=span, fit_dates=date)
        weights = (1.0 - 2.0 / (span + 1.0)) ** np.arange(len(y) - 1, -1, -1)
        np.testing.assert_allclose(expanding.coefficients.iloc[0],
                                   np.average(old_changed, weights=weights, axis=0), atol=1e-12)
        assert not np.allclose(expanding.coefficients, baseline.coefficients)


def test_warmup_defaults_min_periods_lag_cap_labels_and_covariance_storage() -> None:
    y, x = _panel()
    expanding = estimate_rolling_hac_regression(y=y['A'], x=x, hac_lags=100)
    rolling = estimate_rolling_hac_regression(y=y['A'], x=x, window=8, hac_lags=100)
    explicit = estimate_rolling_hac_regression(y=y['A'], x=x, window=8, min_periods=4,
                                               hac_lags=100, include_covariance=True)
    assert expanding.coefficients.iloc[:23].isna().all().all()
    assert expanding.diagnostics.iloc[23]['status'] is RollingHacStatus.FITTED
    assert rolling.coefficients.iloc[:7].isna().all().all()
    assert rolling.diagnostics.iloc[7]['status'] is RollingHacStatus.FITTED
    assert explicit.diagnostics.iloc[3]['hac_lags'] == 3
    assert explicit.diagnostics.iloc[7]['hac_lags'] == 7
    assert pd.isna(explicit.diagnostics.iloc[0]['requested_start'])
    assert explicit.diagnostics.iloc[0]['actual_start'] == y.index[0]
    assert explicit.diagnostics.iloc[7]['requested_start'] == y.index[0]
    assert explicit.diagnostics.iloc[8]['requested_start'] == y.index[1]
    expected_columns = pd.MultiIndex.from_product(
        [['A'], ['Intercept', 'Market', 'Value']], names=['equation', 'parameter'],
    )
    pd.testing.assert_index_equal(explicit.coefficients.columns, expected_columns)
    no_covariance = estimate_rolling_hac_regression(y=y['A'], x=x, window=8, min_periods=4,
                                                    hac_lags=100)
    assert no_covariance.parameter_covariance is None
    assert len(explicit.parameter_covariance) == len(y) - 3
    assert explicit.confidence_level == 0.95
    for name in (*STATISTICS, 'diagnostics'):
        pd.testing.assert_frame_equal(getattr(explicit, name), getattr(no_covariance, name))


def test_eligible_errors_are_dated_and_shared_effective_size_guard_is_preserved() -> None:
    y, x = _panel()
    x.iloc[20:28] = 1.0
    with pytest.raises(ValueError, match=f'{y.index[27].date()}.*singular'):
        estimate_rolling_hac_regression(y=y, x=x, window=8, fit_dates=y.index[27:28])
    with pytest.raises(ValueError, match=f'{y.index[23].date()}.*effective sample size'):
        estimate_rolling_hac_regression(y=y, x=x, span=2.5, fit_dates=y.index[23:24])
    y.iloc[20:28, 0] = 0.0
    with pytest.raises(ValueError, match=f'{y.index[27].date()}.*constant response'):
        estimate_rolling_hac_regression(y=y, window=8, fit_dates=y.index[27:28])


@pytest.mark.parametrize('settings', [
    {'window': True}, {'window': 4.5}, {'window': 0}, {'window': -3}, {'window': 3},
    {'min_periods': False}, {'min_periods': 4.1}, {'min_periods': 3},
    {'window': 5, 'min_periods': 6}, {'span': 1.0}, {'span': np.inf}, {'span': True},
    {'hac_lags': -1}, {'hac_lags': 1.5}, {'confidence_level': 1.0},
    {'include_covariance': 'yes'}, {'time_period': '2020'},
])
def test_invalid_configuration_is_rejected_even_without_eligible_dates(settings: dict) -> None:
    y, x = _panel()
    with pytest.raises((ValueError, TypeError)):
        estimate_rolling_hac_regression(y=y, x=x, fit_dates=y.index[:0], **settings)


def test_missing_values_and_gaps_cannot_be_compressed_before_windowing() -> None:
    y, x = _panel()
    dates = y.index[-1:]
    for frame, column in [(y, 'A'), (x, 'Market')]:
        broken = frame.copy()
        broken.iloc[2, broken.columns.get_loc(column)] = np.nan
        with pytest.raises(ValueError, match='finite'):
            estimate_rolling_hac_regression(y=broken if frame is y else y,
                                             x=broken if frame is x else x,
                                             window=18, fit_dates=dates)
    positions = np.delete(np.arange(len(y)), 20)
    with pytest.raises(ValueError, match='complete'):
        estimate_rolling_hac_regression(y=y.iloc[positions], x=x.iloc[positions],
                                         sampling_frequency='ME')
    # Equally spaced omissions cannot masquerade as a new, coarser declared calendar.
    with pytest.raises(ValueError, match='complete'):
        estimate_rolling_hac_regression(y=y.iloc[::2], sampling_frequency='ME')
    no_frequency = y.copy()
    no_frequency.index = pd.DatetimeIndex(list(y.index), name=y.index.name)
    with pytest.raises(ValueError, match='declare'):
        estimate_rolling_hac_regression(y=no_frequency)
    declared = estimate_rolling_hac_regression(y=no_frequency, sampling_frequency='ME')
    attached = estimate_rolling_hac_regression(y=y)
    pd.testing.assert_frame_equal(declared.coefficients, attached.coefficients, check_freq=False)


def test_custom_calendar_alignment_and_invalid_schedules() -> None:
    y, x = _panel()
    positions = np.delete(np.arange(len(y)), [3, 12, 25])
    custom_y, custom_x = y.iloc[positions], x.iloc[positions]
    result = estimate_rolling_hac_regression(y=custom_y, x=custom_x,
                                             expected_index=custom_y.index)
    assert result.diagnostics.iloc[23]['nobs'] == 24
    with pytest.raises(ValueError, match='only one'):
        estimate_rolling_hac_regression(y=y, sampling_frequency='ME', expected_index=y.index)
    with pytest.raises(ValueError, match='expected_index'):
        estimate_rolling_hac_regression(y=custom_y, expected_index=y.index)
    with pytest.raises(ValueError, match='indexes'):
        estimate_rolling_hac_regression(y=y, x=x.iloc[::-1])
    for broken in [y.iloc[::-1], pd.concat([y, y.iloc[-1:]]), y.reset_index(drop=True), y.iloc[:0]]:
        with pytest.raises(ValueError):
            estimate_rolling_hac_regression(y=broken, sampling_frequency='ME')
    for dates in [y.index[::-1], y.index[[2, 2]], pd.date_range('2000-01-31', periods=2, freq='ME'),
                  pd.DatetimeIndex([pd.NaT]), list(y.index)]:
        with pytest.raises(ValueError, match='fit_dates'):
            estimate_rolling_hac_regression(y=y, fit_dates=dates)
    empty = estimate_rolling_hac_regression(y=y, fit_dates=y.index[:0], include_covariance=True)
    assert empty.coefficients.empty
    assert empty.diagnostics.columns.tolist()[-1] == 'status'
    assert empty.parameter_covariance == {}


def test_shared_core_outputs_are_consumed_and_failures_are_not_swallowed(
        monkeypatch: pytest.MonkeyPatch) -> None:
    y, x = _panel()
    original = regression.estimate_hac_regression

    def sentinel(**kwargs):
        result = original(**kwargs)
        result.coefficients.iloc[:, :] = 42.0
        return result

    monkeypatch.setattr(regression, 'estimate_hac_regression', sentinel)
    result = estimate_rolling_hac_regression(y=y, x=x, fit_dates=y.index[-1:])
    assert (result.coefficients == 42.0).all().all()

    def fail(**kwargs):
        raise ValueError('deliberate shared fit failure')

    monkeypatch.setattr(regression, 'estimate_hac_regression', fail)
    with pytest.raises(ValueError, match=f'{y.index[-1].date()}.*deliberate shared fit failure'):
        estimate_rolling_hac_regression(y=y, x=x, fit_dates=y.index[-1:])


def test_frozen_synthetic_panel_supports_multiresponse_multifactor_rolling() -> None:
    prices = generate_synthetic_prices(start='2015-01-01', end='2019-12-31')
    returns = to_returns(prices[['SEQ_US', 'SBD_TSY', 'SBD_IG', 'SBD_HY']],
                         freq='ME', is_log_returns=False).iloc[1:]
    y, x = returns[['SBD_IG', 'SBD_HY']], returns[['SEQ_US', 'SBD_TSY']]
    result = estimate_rolling_hac_regression(y=y, x=x, window=24, span=18,
                                             sampling_frequency='ME', fit_dates=y.index[-1:])
    expected, _ = _dense_reference(y.iloc[-24:].to_numpy(), x.iloc[-24:].to_numpy(), 18, 3)
    np.testing.assert_allclose(result.coefficients.iloc[0], expected.T.ravel(),
                               rtol=1e-9, atol=1e-12)

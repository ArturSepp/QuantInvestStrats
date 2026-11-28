"""Independent references and input contracts for summaries of dated signal ICs."""
from io import StringIO
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from qis.perfstats.signal_ic import compute_signal_ic_summary


def _dated(values, **columns):
    return pd.DataFrame({
        'horizon': 1, 'sample': 'common', 'signal': 'momentum', 'target': 'adjusted',
        'realisation_date': pd.date_range('2020-01-31', periods=len(values), freq='D'),
        'ic': values, **columns,
    })


def _dense_reference(x, weights, lags=3, level=0.95):
    """Direct mean, unbiased dispersion and dense Bartlett quadratic form."""
    x = np.asarray(x)
    w = np.asarray(weights) / np.sum(weights)
    n = len(x)
    mean = np.dot(w, x)
    sd = np.sqrt(np.average((x - mean) ** 2, weights=w) / (1 - np.sum(w ** 2)))
    lags = min(n - 1, lags)
    distances = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    kernel = np.maximum(1 - distances / (lags + 1), 0.0)
    scores = w * (x - mean)
    se = np.sqrt(n / (n - 1) * (scores @ kernel @ scores))
    z = NormalDist().inv_cdf((1 + level) / 2)
    return mean, sd, mean / sd, np.dot(w, x > 0), 1 / np.sum(w ** 2), se, mean-z*se, mean+z*se


@pytest.mark.parametrize(('settings', 'weighting', 'weights', 'mean'), [
    ({}, 'equal', [1, 1, 1], 1 / 6),
    ({'rolling_span': 2}, 'rolling', [1, 1], 1 / 8),
    ({'ewma_span': 3}, 'ewma', [1, 2, 4], 9 / 28),
])
def test_worked_example_and_dense_bartlett(settings, weighting, weights, mean):
    frame = _dated([0.25, -0.50, 0.75], n_assets=[6, 3, 9])
    report = compute_signal_ic_summary(frame, warmup_periods=0, **settings)
    row = report.summary.iloc[0]
    x = frame.ic.iloc[-len(weights):]
    expected = _dense_reference(x, weights)
    assert row.mean_ic == pytest.approx(mean)
    actual = row[['mean_ic', 'std_ic', 'ic_ir', 'hit_rate', 'effective_n', 'ic_hac_se',
                  'ic_lower', 'ic_upper']].to_numpy(dtype=float)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(report.dated.weight.dropna(), weights / np.sum(weights))
    assert (row.weighting, row.inference, row.n_windows) == (weighting, 'ok', 3)
    assert row.effective_hac_lags == len(weights) - 1
    assert row.hac_lags == 3
    assert row.n_selected == len(weights)
    assert row.mean_n_assets == frame.n_assets.tail(len(weights)).mean()
    assert report.dated.paired_weight.isna().all()
    assert row.n_paired_valid == row.n_paired_selected == row.paired_effective_n == 0
    assert pd.isna(row.mean_ic_minus_reference)
    warm = compute_signal_ic_summary(frame, **settings).summary.iloc[0]
    assert warm.warmup_periods == 12 and warm.inference == 'insufficient_dates'
    assert warm.mean_ic == row.mean_ic and pd.isna(warm.ic_hac_se)


@pytest.mark.parametrize('policy', ['rolling', 'ewma'])
def test_paired_sample_has_its_own_dates_weights_and_hac(policy):
    frame = _dated([0.2, 0.3, np.nan, -0.5, 0.7, np.inf],
                   ic_minus_reference=[-0.1, np.nan, np.nan, 0.4, -0.2, np.nan])
    if policy == 'ewma':
        settings = {'ewma_span': 3}
        ic_dates, ic_weights = [0, 1, 3, 4], [1, 2, 4, 8]
        pair_dates, pair_weights = [0, 3, 4], [1, 2, 4]
    else:
        settings = {'rolling_span': 3}
        ic_dates, ic_weights = [1, 3, 4], [1, 1, 1]
        pair_dates, pair_weights = [0, 3, 4], [1, 1, 1]
    report = compute_signal_ic_summary(frame, warmup_periods=0, confidence_level=0.9,
                                       hac_lags=1, **settings)
    row = report.summary.iloc[0]
    assert row.n_valid == 4 and row.n_paired_valid == 3
    assert row.paired_inference == 'ok' and row.paired_effective_hac_lags == 1
    assert report.dated.weight.dropna().index.tolist() == ic_dates
    assert report.dated.paired_weight.dropna().index.tolist() == pair_dates
    np.testing.assert_allclose(report.dated.weight.dropna(), ic_weights / np.sum(ic_weights))
    np.testing.assert_allclose(report.dated.paired_weight.dropna(),
                               pair_weights / np.sum(pair_weights))
    expected = _dense_reference(frame.loc[pair_dates, 'ic_minus_reference'], pair_weights,
                                lags=1, level=0.9)
    np.testing.assert_allclose(
        [row.mean_ic_minus_reference, row.paired_effective_n, row.difference_hac_se,
         row.difference_lower, row.difference_upper],
        [expected[i] for i in (0, 4, 5, 6, 7)], atol=1e-12, rtol=1e-12)


def test_cutoff_grouping_and_csv_round_trip():
    a = _dated([0.1, -0.3, 0.9], method='spearman', reference_target='raw')
    a['formation_date'] = pd.Timestamp('2019-12-31')
    a['realisation_date'] = pd.to_datetime(['2020-02-29', '2020-05-31', '2020-08-31'])
    a['horizon'] = 3
    b = a.assign(ic=-a.ic, method='pearson')
    frame = pd.concat([a.assign(fund='A'), b.assign(fund='B')]).sample(frac=1, random_state=3)
    frame['weight'], frame['paired_weight'] = 99.0, 99.0
    frame = pd.read_csv(StringIO(frame.to_csv(index=False)))
    original = frame.copy()
    result = compute_signal_ic_summary(frame, group_by=('fund',), as_of='2020-07-31', ewma_span=3)
    from_window = compute_signal_ic_summary(
        frame.loc[frame.realisation_date <= '2020-07-31'], group_by=['fund'],
        ewma_span=3, as_of='2020-07-31')
    pd.testing.assert_frame_equal(result.dated, from_window.dated)
    pd.testing.assert_frame_equal(result.summary, from_window.summary)
    pd.testing.assert_frame_equal(original, frame)
    assert result.dated.realisation_date.is_monotonic_increasing
    summaries = result.summary.set_index('fund')
    assert summaries.at['A', 'mean_ic'] == pytest.approx(-1 / 6)
    assert summaries.at['B', 'mean_ic'] == pytest.approx(1 / 6)
    assert summaries.at['B', 'method'] == 'pearson'
    assert summaries.at['A', 'as_of'] == pd.Timestamp('2020-07-31')
    assert result.dated.paired_weight.isna().all()
    inclusive = compute_signal_ic_summary(a, as_of='2020-05-31').summary.iloc[0]
    assert inclusive.n_windows == 2
    empty = compute_signal_ic_summary(a, as_of='2020-01-31')
    assert empty.dated.empty and empty.summary.empty
    assert empty.summary.columns.tolist() == [c for c in result.summary.columns if c != 'fund']
    assert compute_signal_ic_summary(empty.dated).summary.empty


def test_warmup_depends_on_effective_count_with_an_inclusive_equal_weight_boundary():
    frame = _dated([0.1, 0.2] * 6, ic_minus_reference=[np.nan] + [-0.1, 0.2] * 5 + [0.2])
    equal = compute_signal_ic_summary(frame).summary.iloc[0]
    rolling = compute_signal_ic_summary(frame, rolling_span=12).summary.iloc[0]
    ewma = compute_signal_ic_summary(frame, ewma_span=24).summary.iloc[0]
    assert equal.inference == rolling.inference == 'ok'
    assert equal.paired_inference == 'insufficient_dates'
    assert rolling.paired_inference == 'insufficient_history'
    assert pd.isna(equal.difference_lower) and np.isfinite(equal.mean_ic_minus_reference)
    assert pd.isna(rolling.mean_ic_minus_reference) and rolling.n_paired_selected == 0
    assert ewma.n_selected == 12 and ewma.effective_n < 12
    assert ewma.inference == 'insufficient_dates' and pd.isna(ewma.ic_lower)
    short_window = compute_signal_ic_summary(frame, rolling_span=6).summary.iloc[0]
    assert short_window.inference == 'insufficient_dates' and short_window.n_selected == 6

    long = _dated([0.1, 0.2] * 500)
    for span in (6, 12):
        row = compute_signal_ic_summary(long, ewma_span=span).summary.iloc[0]
        assert row.effective_n == pytest.approx(span)
        assert row.inference == 'insufficient_dates' and pd.isna(row.ic_hac_se)
    assert compute_signal_ic_summary(long, ewma_span=24).summary.iloc[0].inference == 'ok'


def test_fractional_span_and_resummarizing_dated_weights():
    frame = _dated(pd.array([0.25, None, -0.5, 0.75], dtype='Float64'),
                   n_assets=pd.array([3, 4, None, 7], dtype='Int64'))
    report = compute_signal_ic_summary(frame, ewma_span=2.5, warmup_periods=0)
    expected = _dense_reference([0.25, -0.5, 0.75], [9, 21, 49])
    row = report.summary.iloc[0]
    np.testing.assert_allclose([row.mean_ic, row.std_ic, row.ic_hac_se],
                               [expected[i] for i in (0, 1, 5)], rtol=1e-12)
    assert row.mean_n_assets == 5
    rolled = compute_signal_ic_summary(report.dated, rolling_span=2)
    direct = compute_signal_ic_summary(frame, rolling_span=2)
    pd.testing.assert_frame_equal(rolled.dated, direct.dated)
    pd.testing.assert_frame_equal(rolled.summary, direct.summary)


def test_span_just_above_one_preserves_dispersion_with_one_effective_date():
    report = compute_signal_ic_summary(_dated([-0.1, 0.3]), ewma_span=np.nextafter(1.0, 2.0),
                                       warmup_periods=0)
    row = report.summary.iloc[0]
    assert row.mean_ic == pytest.approx(0.3)
    # For two points the unbiased weighted sample variance is (x1-x2)**2 / 2.
    assert row.std_ic == pytest.approx(0.4 / np.sqrt(2))
    assert row.inference == 'insufficient_dates' and pd.isna(row.ic_hac_se)


@pytest.mark.parametrize(('values', 'settings', 'status', 'n', 'mean', 'std', 'hit'), [
    ([np.nan, np.inf], {}, 'insufficient_dates', 0, np.nan, np.nan, np.nan),
    ([np.nan, 0.0], {}, 'insufficient_dates', 1, 0.0, np.nan, 0.0),
    ([0.1, 0.2], {'rolling_span': 3}, 'insufficient_history', 0, np.nan, np.nan, np.nan),
    ([0.2, 0.2, 0.2], {}, 'zero_dispersion', 3, 0.2, 0.0, 1.0),
    ([-0.1, -0.1], {'ewma_span': 5}, 'zero_dispersion', 2, -0.1, 0.0, 0.0),
])
def test_undefined_estimates(values, settings, status, n, mean, std, hit):
    row = compute_signal_ic_summary(_dated(values), **settings).summary.iloc[0]
    assert row.inference == status and row.n_selected == n
    np.testing.assert_allclose([row.mean_ic, row.std_ic, row.hit_rate], [mean, std, hit])
    assert pd.isna(row.ic_ir) and pd.isna(row.ic_hac_se) and pd.isna(row.ic_upper)
    if n < 2:
        assert row.effective_n == n


@pytest.mark.parametrize('kwargs', [
    {'ewma_span': True}, {'ewma_span': 1.0}, {'ewma_span': np.inf},
    {'ewma_span': 3.0, 'rolling_span': 2}, {'rolling_span': 2.5}, {'rolling_span': False},
    {'rolling_span': 1}, {'warmup_periods': True}, {'warmup_periods': 1.5},
    {'warmup_periods': -1}, {'hac_lags': False}, {'hac_lags': -1},
    {'confidence_level': 1.0}, {'confidence_level': np.nan}, {'as_of': pd.NaT},
])
def test_invalid_settings(kwargs):
    with pytest.raises((TypeError, ValueError)):
        compute_signal_ic_summary(_dated([0.1]), **kwargs)


def test_ambiguous_inputs_raise():
    frame = _dated([0.1, -0.1])
    bad_dates = frame.assign(realisation_date=['2020-01-31', '2020-02-30'])
    for bad in (frame.assign(realisation_date=pd.NaT), bad_dates,
                frame.assign(realisation_date='2020-02-29'), frame.drop(columns='sample'),
                frame.assign(reference_target=['raw', 'other']), frame.assign(ic=['bad', '1'])):
        with pytest.raises(ValueError):
            compute_signal_ic_summary(bad)

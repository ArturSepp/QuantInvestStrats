"""Numerical and timing contracts for matched cross-sectional ICs on public synthetic data."""
import numpy as np
import pandas as pd
import pytest

from qis.perfstats.signal_diagnostics import compute_ic_timeseries, estimate_signal_diagnostics
from qis.perfstats.signal_ic import compute_matched_signal_ic, compute_signal_ic_summary


def _window():
    return pd.DataFrame({
        'horizon': 1, 'formation_date': '2021-01-31', 'realisation_date': '2021-02-28',
        'asset': list('ABCDEFGHIJ'),
        'eligible': pd.array([True] * 7 + [False, True, None], dtype='boolean'),
        'low_beta': [2, 2, 6, 1, 4, 5, 3, 8, 9, 10],
        'momentum': [2, 5, 4, 1, 6, 3, np.nan, 7, 8, 10],
        'raw': [3, 1, 2, 5, 4, 6, 7, 8, -np.inf, 10],
        'market': [2, 4, 4, 1, 6, np.nan, 5, 2, 1, 2],
        'factor': [8, 4, 9, 5, np.inf, 7, 6, 3, 2, 1],
    })


def _report(frame, **kwargs):
    return compute_matched_signal_ic(
        frame, signals=['low_beta', 'momentum'], targets=['market', 'raw', 'factor'],
        reference_target='raw', **kwargs)


def _correlation(values, outcomes, method):
    """Enumerate average ranks and form the centered cosine independently of scipy."""
    def centered(series):
        x = np.asarray(series, dtype=float)
        if method == 'spearman':
            x = np.array([1 + (x < v).sum() + ((x == v).sum() - 1) / 2 for v in x])
        return x - x.mean()

    x, y = centered(values), centered(outcomes)
    return np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y))


@pytest.mark.parametrize('method', ['spearman', 'pearson'])
def test_samples_ties_breadth_and_paired_differences(method):
    frame = _window().sample(frac=1, random_state=14)
    original = frame.copy(deep=True)
    result = _report(frame, min_obs_per_date=3, method=method)
    result = result.set_index(['sample', 'signal', 'target'])
    names = frame.set_index('asset')
    samples = {
        'common': ('ABCD', ['raw', 'market', 'factor']),
        'raw_vs_market': ('ABCDE', ['raw', 'market']),
        'raw_vs_factor': ('ABCDF', ['raw', 'factor']),
        'available_raw': ('ABCDEF', ['raw']),
    }
    expected_keys = set()
    for sample, (assets, targets) in samples.items():
        section = names.loc[list(assets)]
        for signal in ['low_beta', 'momentum']:
            reference = _correlation(section[signal], section.raw, method)
            for target in targets:
                expected_keys.add((sample, signal, target))
                row = result.loc[(sample, signal, target)]
                ic = _correlation(section[signal], section[target], method)
                assert row.n_assets == len(assets)
                assert (row.status, row.reference_target, row.method) == ('ok', 'raw', method)
                assert row.ic == pytest.approx(ic, abs=1e-12)
                if target == 'raw':
                    assert pd.isna(row.ic_minus_reference)
                else:
                    assert row.ic_minus_reference == pytest.approx(ic - reference, abs=1e-12)
    assert set(result.index) == expected_keys and result.index.is_unique
    pd.testing.assert_frame_equal(frame, original)


def test_one_gap_at_one_horizon_preserves_other_groups_samples_and_dates():
    base = _window().iloc[:5].copy()
    base['factor'] = [2, 5, 3, 1, 4]
    blocks = []
    for fund in ('A', 'B'):
        for horizon, end in [(1, '2021-02-28'), (3, '2021-04-30')]:
            window = base.assign(fund=fund, horizon=horizon, realisation_date=end)
            if fund == 'B':
                window['raw'] *= -1
            blocks.append(window)
    panel = pd.concat(blocks)  # repeated DataFrame index across windows
    options = dict(group_by=['fund'], min_obs_per_date=3)
    complete = _report(panel, **options)
    key = (panel.fund == 'A') & (panel.horizon == 3) & (panel.asset == 'A')
    panel.loc[key, 'market'] = np.nan
    with_gap = _report(panel, **options)

    def affected(table):
        return ((table.fund == 'A') & (table.horizon == 3)
                & table['sample'].isin(['common', 'raw_vs_market']))

    assert complete.loc[affected(complete), 'n_assets'].eq(5).all()
    assert with_gap.loc[affected(with_gap), 'n_assets'].eq(4).all()
    pd.testing.assert_frame_equal(complete.loc[~affected(complete)],
                                  with_gap.loc[~affected(with_gap)])
    for _, group in with_gap.groupby(['fund', 'horizon']):
        assert set(group['sample']) == {'common', 'raw_vs_market', 'raw_vs_factor', 'available_raw'}
    summary = compute_signal_ic_summary(with_gap, group_by=['fund']).summary
    assert len(summary) == len(with_gap)
    assert summary.n_windows.eq(1).all()
    assert summary.n_valid.eq(1).all()

    later = panel.assign(formation_date='2022-01-31', realisation_date='2022-04-30', low_beta=-1)
    combined = _report(pd.concat([panel, later]), **options)
    earlier = combined[combined.formation_date < '2022-01-01'].reset_index(drop=True)
    pd.testing.assert_frame_equal(earlier, with_gap)


def test_reasons_are_ordered_and_reference_scale_detects_factor_cancellation():
    frame = _window().iloc[:5].copy()
    frame['momentum'] = 1
    frame['raw'] = [0.25, 0.5, 0.75, 1, 0.75]
    threshold = 8 * np.finfo(float).eps
    frame['market'] = [0, threshold / 4, threshold / 2, threshold, 0]
    frame['factor'] = frame.raw - frame.raw

    def common(data, **kwargs):
        out = _report(data, **kwargs)
        return out.loc[out['sample'] == 'common'].set_index(['signal', 'target'])

    values = common(frame)
    assert values.loc[('momentum', 'market'), 'status'] == 'constant_score'
    assert values.loc[('low_beta', 'market'), 'status'] == 'constant_target'
    assert values.loc[('low_beta', 'factor'), 'status'] == 'constant_target'
    assert values.loc[('low_beta', 'raw'), 'status'] == 'ok'
    assert values.loc[('low_beta', 'market'), ['ic', 'ic_minus_reference']].isna().all()
    frame.loc[frame.index[3], 'market'] = np.nextafter(threshold, np.inf)
    assert common(frame).loc[('low_beta', 'market'), 'status'] == 'ok'
    assert common(frame, min_obs_per_date=6).status.eq('insufficient_assets').all()

    frame['raw'] = 1
    undefined_ref = common(frame)
    row = undefined_ref.loc[('low_beta', 'market')]
    assert row.status == 'ok' and np.isfinite(row.ic) and pd.isna(row.ic_minus_reference)
    none = _report(frame.assign(eligible=False))
    assert len(none) == 16 and none.n_assets.eq(0).all()
    assert none.status.eq('insufficient_assets').all() and none.ic.isna().all()


@pytest.mark.parametrize('method', ['spearman', 'pearson'])
def test_formation_scores_agree_with_legacy_one_period_ic(method):
    rng = np.random.default_rng(4329)
    dates = pd.date_range('2020-01-31', periods=12, freq='ME')
    assets = list('ABCDEFGH')
    scores = pd.DataFrame(rng.normal(size=(len(dates), len(assets))), index=dates, columns=assets)
    noise = rng.normal(size=scores.shape)
    returns = 0.025 * scores.shift(1).fillna(0.0) + 0.02 * noise
    legacy_result = estimate_signal_diagnostics({'ME': returns}, scores, horizons=(1,))
    legacy = compute_ic_timeseries(legacy_result, method=method)['1']
    blocks = [pd.DataFrame({
        'asset': assets, 'horizon': 1, 'formation_date': start, 'realisation_date': end,
        'eligible': True, 'score': scores.loc[start].to_numpy(), 'raw': returns.loc[end].to_numpy(),
    }) for start, end in zip(dates[:-1], dates[1:])]
    dated = compute_matched_signal_ic(pd.concat(blocks), signals=['score'], targets=['raw'],
                                      reference_target='raw', method=method)
    common = dated.loc[dated['sample'] == 'common'].set_index('realisation_date')
    assert common.index.tolist() == legacy.index.tolist() == list(dates[1:])
    np.testing.assert_array_equal(common.n_assets, legacy.n)
    np.testing.assert_allclose(common.ic, legacy.IC, atol=1e-12, rtol=1e-12)
    assert dated.ic_minus_reference.isna().all()


def test_scale_and_breadth_are_specific_to_each_matched_sample():
    frame = _window().iloc[:3].assign(raw=[1.0, 2.0, 1e14], market=[0.0, 0.01, np.nan])
    args = dict(signals=['momentum'], targets=['market', 'raw'], reference_target='raw',
                min_obs_per_date=2)
    rows = compute_matched_signal_ic(frame, **args).set_index(['sample', 'target'])
    assert rows.at[('raw_vs_market', 'market'), 'n_assets'] == 2
    assert rows.at[('raw_vs_market', 'market'), 'ic'] == pytest.approx(1.0)
    assert rows.at[('available_raw', 'raw'), 'n_assets'] == 3
    unavailable = compute_matched_signal_ic(frame.assign(market=np.nan), **args)
    assert unavailable.loc[unavailable['sample'] == 'available_raw', 'status'].item() == 'ok'
    assert unavailable.loc[unavailable['sample'] != 'available_raw', 'n_assets'].eq(0).all()


def test_single_arbitrary_reference_and_empty_input():
    frame = _window().rename(columns={'raw': 'total_return', 'momentum': 'signal'})
    args = dict(signals=['signal'], targets=['total_return'], reference_target='total_return')
    result = compute_matched_signal_ic(frame, **args)
    assert result['sample'].tolist() == ['common', 'available_total_return']
    empty = compute_matched_signal_ic(frame.iloc[:0], **args)
    assert empty.empty and empty.columns.tolist() == result.columns.tolist()
    assert compute_signal_ic_summary(empty).summary.empty


@pytest.mark.parametrize('settings', [
    {'signals': []}, {'signals': 'momentum'}, {'targets': []}, {'targets': ['raw', 'raw']},
    {'signals': ['raw']}, {'reference_target': 'missing'}, {'signals': ['unavailable']},
    {'group_by': ['signal']}, {'group_by': 'fund'}, {'method': 'kendall'},
    {'min_obs_per_date': True}, {'min_obs_per_date': 1}, {'min_obs_per_date': 2.5},
])
def test_invalid_parameters(settings):
    args = dict(signals=['momentum'], targets=['raw'], reference_target='raw')
    with pytest.raises(ValueError):
        compute_matched_signal_ic(_window(), **{**args, **settings})


def test_invalid_observations():
    frame = _window()
    invalid = [
        pd.concat([frame, frame.iloc[:1]]), frame.assign(realisation_date=frame.formation_date),
        frame.assign(formation_date=pd.NaT), frame.assign(asset=None),
        frame.drop(columns='eligible'), frame.assign(factor=complex(0, 1)),
        frame.assign(low_beta='not a score'),
        pd.concat([frame, frame[['raw']]], axis=1),
    ]
    for panel in invalid:
        with pytest.raises(ValueError):
            _report(panel)

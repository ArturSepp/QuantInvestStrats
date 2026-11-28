"""Decision windows, native increments and point-in-time factor attribution on synthetic panels."""
import math
import warnings

import numpy as np
import pandas as pd
import pytest

from qis.perfstats.forward_return_targets import (
    ForwardReturnAdjustment, compute_forward_return_targets,
)
from qis.perfstats.signal_ic import compute_matched_signal_ic, compute_signal_ic_summary


def _mixed_panels():
    months = pd.date_range('2020-12-31', periods=7, freq='ME')
    quarters = months[[0, 3, 6]]
    assets = {
        'ME': pd.DataFrame({'A': [np.nan, .011, -.006, .025, .020, .030, -.010],
                            'B': [99.0, -.010, .020, .010, -.010, -.020, .040]}, index=months),
        'QE': pd.DataFrame({'Q': [99.0, .040, -.010]}, index=quarters),
    }
    # Rate increments are changes in the supplied factor units. Only closing labels are needed.
    factors = {
        'ME': pd.DataFrame({'rate': [.01, .04, -.02, -.01, 0, .02],
                            'market': [.005, -.010, .025, .020, -.010, .030]}, index=months[1:]),
        'QE': pd.DataFrame({'market': [.020, .040], 'rate': [.030, .010]}, index=quarters[1:]),
    }
    opening = pd.DataFrame({'market': [.0, -.5, 1.25], 'rate': [.5, .4, -.2]},
                           index=['Q', 'B', 'A'])
    next_quarter = opening * 2
    next_day = opening + 17
    snapshots = {quarters[1]: next_quarter, months[0] + pd.Timedelta(days=1): next_day,
                 months[0]: opening, months[0] - pd.Timedelta(days=7): opening + 3}
    benchmark = ForwardReturnAdjustment(
        factor_increments_by_freq={k: v[['market']] for k, v in factors.items()},
        loadings_by_date={k: v[['market']] for k, v in snapshots.items()})
    multi = ForwardReturnAdjustment(factor_increments_by_freq=factors, loadings_by_date=snapshots)
    return months, quarters, assets, {'benchmark': benchmark, 'multi': multi}


def test_frozen_loadings_and_native_sums_against_scalar_reference():
    months, decisions, assets, models = _mixed_panels()
    targets = compute_forward_return_targets(decisions, assets, horizons=(1, 2), adjustments=models)
    index = ['horizon', 'formation_date', 'realisation_date', 'asset']
    assert len(targets) == 9 and not targets.duplicated(index).any()
    output = targets.set_index(index)
    for h, start, end, monthly_ends, quarterly_ends in [
        (1, 0, 1, [1, 2, 3], [1]), (1, 1, 2, [4, 5, 6], [2]),
        (2, 0, 2, [1, 2, 3, 4, 5, 6], [1, 2]),
    ]:
        formation, realisation = decisions[start], decisions[end]
        for asset, cadence in [('A', 'ME'), ('B', 'ME'), ('Q', 'QE')]:
            ends = months[monthly_ends] if cadence == 'ME' else decisions[quarterly_ends]
            raw = math.fsum(assets[cadence].at[day, asset] for day in ends)
            row = output.loc[(h, formation, realisation, asset)]
            assert row.raw_status == 'ok' and row.cadence == cadence
            assert row.raw == pytest.approx(raw, rel=1e-12, abs=1e-14)
            for name, model in models.items():
                # Explicitly choose the known boundary snapshot and sum scalar products by label.
                exposure = model.loadings_by_date[formation].loc[asset]
                increments = model.factor_increments_by_freq[cadence]
                explained = math.fsum(exposure[f] * increments.at[day, f]
                                      for day in ends for f in exposure.index)
                assert row[name + '_snapshot_date'] == formation and row[name + '_status'] == 'ok'
                np.testing.assert_allclose([row[name + '_explained'], row[name]],
                                           [explained, raw - explained], rtol=1e-12, atol=1e-14)
    first = output.loc[(1, decisions[0], decisions[1], 'A')]
    assert first.raw == pytest.approx(.030)
    assert first.benchmark_explained == pytest.approx(.025)
    assert first.benchmark == pytest.approx(.005)
    long = output.loc[(2, decisions[0], decisions[2], 'A')]
    next_period = output.loc[(1, decisions[1], decisions[2], 'A')]
    assert long.benchmark != pytest.approx(first.benchmark + next_period.benchmark)


def test_later_publication_cannot_change_earlier_payoff():
    months, decisions, assets, models = _mixed_panels()
    original = models['multi']
    changed = {date: matrix * 99 for date, matrix in original.loadings_by_date.items()
               if date > decisions[0]}
    altered = ForwardReturnAdjustment(
        original.factor_increments_by_freq, {**original.loadings_by_date, **changed})
    def run(model):
        frame = compute_forward_return_targets(decisions, assets, horizons=(1, 2),
                                               adjustments={'adj': model})
        return frame[frame.formation_date == months[0]]
    pd.testing.assert_frame_equal(run(original), run(altered))


def test_status_precedence_and_component_availability():
    dates = pd.date_range('2021-01-31', periods=3, freq='ME')
    assets = pd.DataFrame({'A': [1, np.nan, 2], 'B': [np.nan, .2, .3],
                           'C': [0, .1, .1], 'D': [0, .3, .4]}, index=dates)
    factors = pd.DataFrame({'x': [1, 2], 'unused': [np.nan, np.nan]}, index=dates[1:])
    old = pd.DataFrame({'x': [10., 10., 10., 10.]}, index=list('ABCD'))
    current = pd.DataFrame({'x': [2., np.inf, 0.]}, index=list('ABC'))
    snapshots = {dates[0] - pd.Timedelta(days=1): old, dates[0]: current}
    models = {
        'present': ForwardReturnAdjustment({'ME': factors}, snapshots),
        'bad_factor': ForwardReturnAdjustment({'ME': factors.assign(x=[1, -np.inf])}, snapshots),
        'no_cadence': ForwardReturnAdjustment({}, snapshots),
        'no_factor_label': ForwardReturnAdjustment({'ME': factors[['unused']]}, snapshots),
        'no_snapshot': ForwardReturnAdjustment({}, {dates[1]: current}),
    }
    rows = compute_forward_return_targets(dates, {'ME': assets}, horizons=(2,),
                                          adjustments=models).set_index('asset')
    assert rows.at['A', 'raw_status'] == 'missing_return'
    for name in models:
        assert rows.at['A', name + '_status'] == 'missing_return' and pd.isna(rows.at['A', name])
    assert rows.at['A', 'present_explained'] == 6
    assert rows.at['A', 'present_snapshot_date'] == dates[0]
    assert rows.at['B', 'present_status'] == rows.at['D', 'present_status'] == 'missing_loading'
    assert pd.isna(rows.at['B', 'present_explained']) and pd.isna(rows.at['D', 'present'])
    assert rows.at['C', 'present_status'] == 'ok' and rows.at['C', 'present_explained'] == 0
    assert rows.at['C', 'present'] == rows.at['C', 'raw']
    for name in ['bad_factor', 'no_cadence', 'no_factor_label']:
        assert rows.loc[['B', 'C', 'D'], name + '_status'].eq('missing_factor_return').all()
    assert rows.loc[['B', 'C', 'D'], 'no_snapshot_status'].eq('missing_snapshot').all()
    assert rows.no_snapshot_snapshot_date.isna().all()


def test_missing_endpoint_and_missing_interior_mark_have_different_reasons():
    months, decisions, assets, models = _mixed_panels()
    model = models['multi']
    def run(frame):
        return compute_forward_return_targets(decisions[:2], {'ME': frame}, horizons=(1,),
                                              adjustments={'adj': model}).set_index('asset')
    with pytest.warns(UserWarning, match="cadence 'ME', horizon 1"):
        off = run(assets['ME'].drop(index=decisions[0]))
    assert off.raw_status.eq('native_endpoints').all()
    assert off.adj_status.eq('native_endpoints').all()
    inside = run(assets['ME'].drop(index=months[2]))
    assert inside.raw_status.eq('missing_return').all()
    assert inside.adj_explained.notna().all() and inside.adj.isna().all()


@pytest.mark.parametrize(('start', 'expected'), [
    ('2019-12-31', {1: 0, 3: 4, 6: 2}),
    ('2020-01-31', {1: 0, 3: 0, 6: 0}),
])
def test_phase_anchor_and_one_warning_per_unaligned_pair(start, expected):
    dates = pd.date_range(start, periods=13, freq='ME')
    quarter_ends = dates[dates.is_quarter_end]
    assets = {'QE': pd.DataFrame({'q': .1}, index=quarter_ends),
              'ME': pd.DataFrame({'m': .01}, index=dates)}
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter('always')
        output = compute_forward_return_targets(dates, assets, horizons=(1, 3, 6))
    messages = [str(w.message) for w in observed if issubclass(w.category, UserWarning)]
    assert sorted(messages) == sorted(f"No aligned native window for cadence 'QE', horizon {h}"
                                       for h, n in expected.items() if not n)
    for h, n in expected.items():
        window = output[(output.asset == 'q') & (output.horizon == h)]
        assert len(window) == 12 // h and window.raw_status.eq('ok').sum() == n
        assert window.raw_status.eq('native_endpoints').sum() == len(window) - n
        assert window.loc[window.raw_status == 'native_endpoints', 'raw'].isna().all()
        np.testing.assert_allclose(window.raw.dropna(), h / 3 * .1)
    assert output[output.asset == 'm'].raw_status.eq('ok').all()


def test_irregular_second_friday_decisions_and_no_partial_final_window():
    dates = pd.DatetimeIndex(['2021-01-08', '2021-02-12', '2021-03-12', '2021-04-09',
                              '2021-05-14'])
    assets = {'WOM-2FRI': pd.DataFrame({'a': [9, 1, 2, 4, 8]}, index=dates)}
    out = compute_forward_return_targets(dates[[0, 2, 3, 4]], assets, horizons=(1, 2))
    assert out.raw.tolist() == [3., 4., 8., 7.]
    assert list(out.columns) == ['horizon', 'formation_date', 'realisation_date', 'asset',
                                 'cadence', 'raw', 'raw_status']
    assert out.iloc[-1].realisation_date == dates[3]


def test_forward_targets_feed_matched_ic_and_summary():
    _, decisions, assets, models = _mixed_panels()
    targets = compute_forward_return_targets(decisions, assets, horizons=(1,),
                                            adjustments={'hedged': models['benchmark']})
    panel = targets.assign(score=targets.asset.map({'A': .1, 'B': .2, 'Q': .3}), eligible=True)
    dated = compute_matched_signal_ic(panel, signals=['score'], targets=['raw', 'hedged'],
                                      reference_target='raw', min_obs_per_date=3)
    summary = compute_signal_ic_summary(dated).summary
    assert len(summary) == 5 and summary.n_valid.eq(2).all()
    assert summary.loc[summary.target == 'hedged', 'n_paired_valid'].eq(2).all()


@pytest.mark.parametrize('horizons', [(), (1, 1), (0,), (-1,), (True,), (1.5,)])
def test_invalid_horizons(horizons):
    _, dates, assets, _ = _mixed_panels()
    with pytest.raises(ValueError, match='horizons'):
        compute_forward_return_targets(dates, assets, horizons=horizons)


def test_invalid_names_snapshot_shape_and_ambiguous_inputs():
    _, dates, assets, models = _mixed_panels()
    m = models['multi']
    for names in [['asset'], ['raw_status'], ['a_snapshot_date', 'a'], ['a', 'a_explained']]:
        with pytest.raises(ValueError, match='collides'):
            compute_forward_return_targets(dates, assets, adjustments=dict.fromkeys(names, m))
    bad = [pd.DataFrame(index=['A']), pd.DataFrame([[1, 2]], index=['A'], columns=['x', 'x']),
           pd.DataFrame([[1], [2]], index=['A', 'A'], columns=['x'])]
    for matrix in bad:
        with pytest.raises(ValueError):
            compute_forward_return_targets(dates, assets,
                                           adjustments={'model': ForwardReturnAdjustment(
                                               {}, {dates[0]: matrix})})
    with pytest.raises(ValueError, match='one native cadence'):
        compute_forward_return_targets(dates, {**assets, 'D': assets['ME'][['A']]})
    with pytest.raises(ValueError, match='unique'):
        compute_forward_return_targets(dates, {'ME': pd.concat([assets['ME'], assets['ME']])})
    for bad_dates in (dates[::-1], dates.insert(0, dates[0]), dates.insert(0, pd.NaT)):
        with pytest.raises(ValueError, match='decision_dates'):
            compute_forward_return_targets(bad_dates, assets)


def test_short_schedule_retains_output_schema():
    _, dates, assets, models = _mixed_panels()
    with pytest.warns(UserWarning, match="cadence 'ME', horizon 2"):
        empty = compute_forward_return_targets(dates[:1], {'ME': assets['ME']}, horizons=(2,),
                                              adjustments={'hedged': models['benchmark']})
    assert empty.empty
    assert empty.columns[-4:].tolist() == ['hedged', 'hedged_explained', 'hedged_snapshot_date',
                                           'hedged_status']


def test_snapshot_availability_preserves_timezone_and_native_factor_units():
    dates = pd.date_range('2021-03-01', periods=3, freq='D', tz='Europe/Zurich')
    release = dates[0].tz_convert('UTC') + pd.Timedelta(hours=1)
    assets = {'D': pd.DataFrame({'bond': [.0, .01, .02]}, index=dates)}
    # Rate increments in basis points, signed loading in return per basis point.
    rates = pd.DataFrame({'bp': [-5, 2]}, index=dates[1:].tz_convert('UTC'))
    model = ForwardReturnAdjustment({'D': rates},
                                    {release: pd.DataFrame({'bp': [-.003]}, index=['bond'])})
    out = compute_forward_return_targets(dates, assets, horizons=(1,), adjustments={'alpha': model})
    assert out.alpha_status.tolist() == ['missing_snapshot', 'ok']
    assert pd.isna(out.alpha_snapshot_date.iloc[0])
    assert out.alpha_snapshot_date.iloc[1] == release
    assert out.alpha_snapshot_date.dt.tz == dates.tz
    assert out.alpha_explained.iloc[1] == pytest.approx(-.006)
    assert out.alpha.iloc[1] == pytest.approx(.026)


def test_unrepresentable_sum_raises_instead_of_emitting_an_infinite_target():
    dates = pd.date_range('2021-01-31', periods=3, freq='ME')
    assets = {'ME': pd.DataFrame({'a': [0., 1e308, 1e308]}, index=dates)}
    with pytest.raises(FloatingPointError):
        compute_forward_return_targets(dates, assets, horizons=(2,))

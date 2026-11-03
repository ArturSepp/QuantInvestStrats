"""Stage 5 experiment contracts, independent references and resume protections."""

import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

runner = pytest.importorskip('qis.perfstats.run_local.significance_run',
                            reason='development runner is intentionally absent from wheels')


def test_plan_has_exact_baseline_counts_grid_and_frozen_seeds():
    first = runner.frozen_plan()
    assert first == runner.frozen_plan()
    baseline = [c for c in first['cases'] if c['baseline']]
    assert len(baseline) == 4
    assert {(c['nobs'], c['phi'], c['rho'], c['lags'], c['block']) for c in baseline} == {
        (720, 0., 0., 0, 1), (720, 0., .5, 0, 1),
        (2400, .3, 0., 27, 27), (2400, .3, .5, 27, 27),
    }
    for case in baseline:
        assert case['hac_reps'] == 2000 and case['bootstrap_reps'] == 500
        assert len(case['seeds']['hac']) == 2000 and len(case['seeds']['bootstrap']) == 500
    assert first['num_samples'] == 1999 and first['families'] == [6, 12]
    assert first['factor_count'] == 3 and first['burn_in'] == 1000
    stress = [c for c in first['cases'] if not c['baseline']]
    assert {c['block'] for c in stress if c['nobs'] == 2400} >= {3, 6, 12}
    assert any(c['nobs'] < 720 for c in stress) and any(c['tail_df'] == 5 for c in stress)
    assert any(c['lags'] == 3 and c['nobs'] == 2400 for c in stress)
    # Stream/order independence is a property of the frozen integer seed mapping.
    seeds = [tuple(pair) for c in first['cases'] for pairs in c['seeds'].values() for pair in pairs]
    assert len(seeds) == len(set(seeds))
    expected = int(np.random.SeedSequence([20261102, 0, 0, 0, 0]).generate_state(1)[0])
    assert baseline[0]['seeds']['hac'][0][0] == expected


def test_offline_references_and_direct_reuse():
    result = runner.offline_checks()
    assert result['reuse_verified'] and result['max_absolute_error'] < 1e-9
    assert result['factor_counts'] == [0, 1, 3, 6]
    assert result['frozen_market_integration_nobs'] > 24


def test_checkpoint_rejects_corruption_identity_or_experiment(tmp_path):
    payload = dict(case='A', engine='hac', replicate=3, seeds=[1, 2])
    path = tmp_path / 'rep.json'
    runner._write(path, dict(payload=payload, sha256=runner._digest(payload), experiment='x'))
    assert runner._load_checkpoint(path, 'x', payload) == payload
    with pytest.raises(RuntimeError, match='checkpoint'):
        runner._load_checkpoint(path, 'different', payload)
    with pytest.raises(RuntimeError, match='checkpoint'):
        runner._load_checkpoint(path, 'x', dict(payload, seeds=[2, 1]))
    record = json.loads(path.read_text())
    record['payload']['seconds'] = 999
    runner._write(path, record)
    with pytest.raises(RuntimeError, match='checkpoint'):
        runner._load_checkpoint(path, 'x', payload)


def test_real_hac_checkpoint_resume_and_changed_plan_refusal(tmp_path, monkeypatch):
    for name in runner.THREAD_ENV:
        monkeypatch.setenv(name, '1')
    plan = runner.frozen_plan()
    case = copy.deepcopy(plan['cases'][0])
    case.update(hac_reps=1, bootstrap_reps=0, baseline=False)
    case['seeds'] = dict(hac=case['seeds']['hac'][:1], bootstrap=[])
    plan['cases'] = [case]
    monkeypatch.setattr(runner, 'frozen_plan', lambda: plan)
    monkeypatch.setattr(runner, 'offline_checks', lambda: {'checked_elsewhere': True})
    runner._calibrate(tmp_path, 1)
    checkpoint = next((tmp_path / 'checkpoints').glob('*.json'))
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    runner._calibrate(tmp_path, 1)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before
    sessions = [json.loads(p.read_text()) for p in tmp_path.glob('session_*.json')]
    assert any(s['resumed'] == 1 and s['pending'] == 0 for s in sessions)
    plan['num_samples'] = 1998
    with pytest.raises(RuntimeError, match='frozen plan/source/environment changed'):
        runner._calibrate(tmp_path, 1)


def test_gate_preserves_failures_and_uses_replication_not_pooled_members():
    table = pd.DataFrame({'estimate': np.zeros(12), 'hac_se': np.ones(12)})
    families = runner._family_rows(table, None, runner.SignificanceDgp(24))
    records = []
    for i in range(100):
        rows = copy.deepcopy(families)
        for f in rows:
            # One false null rejection in 40% of replications. Mixture FDP = 1/2,
            # not V/M and not ratio of expected V to expected R.
            null = np.array(f['effect']) == 0
            j = int(np.flatnonzero(null)[0])
            f['pvalue'][j] = .001 if i < 40 else 1.
            f['noncoverage'][j] = i < 40
            for method in runner.METHODS:
                reject = [False] * f['members']
                reject[j] = i < 40
                if f['mixture']:
                    reject[0] = i < 40
                f['corrections'][method.name]['reject'] = reject
        records.append(dict(case='base', engine='hac', models={'mean': {'families': rows}}))
    summary = runner._summarize({'cases': [dict(id='base', baseline=True)]}, records)
    assert not summary['baseline_pass']
    failures = summary['baseline_failures']
    assert any(f['metric'] == 'HOLM' for f in failures)
    assert any(f['metric'] == 'BHY' for f in failures)
    assert any(f['metric'].startswith('marginal_size') for f in failures)
    assert any(f['metric'].startswith('pointwise_noncoverage') for f in failures)
    mixture = next(g for g in summary['groups'] if g['mixture'])
    assert mixture['corrections']['BHY']['fdr']['estimate'] == pytest.approx(.2)
    expected = .05 + 3*np.sqrt(.05*.95/100)
    assert mixture['binomial_upper_gate'] == pytest.approx(expected)
    for group in summary['groups']:
        assert group['replications'] == 100


def test_outputs_refuse_onedrive(tmp_path):
    with pytest.raises(ValueError, match='C-local'):
        runner._root(tmp_path / 'OneDrive' / 'outputs')


def test_worker_source_change_is_rejected(tmp_path, monkeypatch):
    source = tmp_path / 'numeric.py'
    source.write_text('original')
    monkeypatch.setattr(runner, '_WORKER_HASHES', {
        source: hashlib.sha256(source.read_bytes()).hexdigest(),
    })
    runner._check_worker_source()
    source.write_text('changed')
    with pytest.raises(RuntimeError, match='numerical source changed'):
        runner._check_worker_source()


def test_concurrent_experiment_lock_is_not_removed(tmp_path, monkeypatch):
    monkeypatch.setenv('AGENT_LOCAL_ROOT', str(tmp_path))
    output = tmp_path / 'task'
    output.mkdir()
    lock = output / 'active.lock'
    lock.write_text('{"pid":123}')
    with pytest.raises(RuntimeError, match='experiment locked'):
        runner.run_local(runner.Locals.CALIBRATION, output_dir=output, workers=1)
    assert lock.read_text() == '{"pid":123}'

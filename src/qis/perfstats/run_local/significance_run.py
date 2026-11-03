"""Offline significance references and frozen, resumable desktop calibration.

No network or provider data. CALIBRATION uses the actual QIS HAC estimators,
Holm/BHY and stationary-bootstrap routine. M=6 is nested in M=12; constant
intercept shifts reuse null studentized draws only after direct-call checks.
Outputs require an explicit C-local path or AGENT_LOCAL_ROOT. An existing
experiment refuses a changed plan, source, Python or numerical environment.
"""

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from enum import Enum
import hashlib
from importlib.metadata import version
import json
import multiprocessing
import os
from pathlib import Path
import platform
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

import qis
from qis.datasets.significance import SignificanceDgp, generate_significance_sample
from qis.datasets.synthetic import generate_synthetic_prices
from qis.models.bootstrap.bootstrap_numba import BootstrapType, generate_bootstrapped_indices
from qis.perfstats import bootstrap_significance as bootstrap
from qis.perfstats import significance
from qis.perfstats.multiple_testing import MultipleTestingMethod, adjust_pvalues
from qis.perfstats.returns import to_returns


class Locals(Enum):
    """Offline checks or the full opt-in Monte Carlo experiment."""

    OFFLINE_CHECKS = 1
    CALIBRATION = 2


CONVENTIONS = dict(family_id='stage5', return_basis='fractional simple; zero funding',
                   cost_basis='gross; no costs', sampling_frequency='ME')
METHODS = (MultipleTestingMethod.HOLM, MultipleTestingMethod.BHY)
THREAD_ENV = ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS')
_WORKER_HASHES = {}
_WORKER_SOURCE = None


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _write(path: Path, value: object) -> None:
    """Atomically replace one generated record, never an experiment's frozen inputs."""
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False),
                         encoding='utf-8')
    temporary.replace(path)


def _root(output_dir: str | Path | None) -> Path:
    selected = output_dir or os.environ.get('QIS_SIGNIFICANCE_OUTPUT_DIR')
    if selected is None:
        local = os.environ.get('AGENT_LOCAL_ROOT')
        if local is None:
            raise ValueError('set AGENT_LOCAL_ROOT or an explicit C-local output_dir')
        selected = Path(local) / 'analyses/factor-backtest-significance/outputs/stage5'
    root = Path(selected).resolve()
    if 'onedrive' in str(root).lower() or (os.name == 'nt' and root.drive.lower() != 'c:'):
        raise ValueError('outputs must be C-local, outside OneDrive')
    allowed = os.environ.get('AGENT_LOCAL_ROOT')
    if allowed is None or not root.is_relative_to(Path(allowed).resolve()) or root == Path(allowed).resolve():
        raise ValueError('outputs must be in a task subdirectory of AGENT_LOCAL_ROOT')
    root.mkdir(parents=True, exist_ok=True)
    return root


def _source() -> dict:
    root = Path(__file__).resolve().parents[4]
    imported = Path(qis.__file__).resolve()
    if not imported.is_relative_to(root / 'src'):
        raise RuntimeError(f'wrong QIS import: {imported}')
    paths = sorted((root / 'src/qis').rglob('*.py')) + [root / 'pyproject.toml']
    hashes = {str(p.relative_to(root)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in paths}
    return dict(hashes=hashes, python=platform.python_version(),
                packages={p: version(p) for p in ('numpy', 'pandas', 'scipy', 'statsmodels',
                                                 'numba', 'llvmlite')})


def frozen_plan() -> dict:
    """Return the predeclared matrix, seeds, budgets, effects and gates; no outcomes read.

    Returns:
        JSON-safe experiment specification. Every baseline has 2000 HAC and 500
        bootstrap outer replications, B=1999, both mean/three-factor alpha, both
        M=6/12, and all-null/half-nonzero families. Stress budgets are separate.
    """
    cases = []
    for rho in (0.0, 0.5):
        for nobs, phi, lags, block in ((720, 0.0, 0, 1), (2400, 0.3, 27, 27)):
            cases.append(dict(id=f'base_T{nobs}_rho{rho}', baseline=True, nobs=nobs,
                              phi=phi, rho=rho, lags=lags, block=block, tail_df=None,
                              hac_reps=2000, bootstrap_reps=500))
    stress = [(2400, 3, 27, None, 200, 25),
              (2400, 27, 3, None, 0, 25), (2400, 27, 6, None, 0, 25),
              (2400, 27, 12, None, 0, 25), (120, 3, 6, None, 200, 25),
              (720, 0, 1, 5.0, 200, 25)]
    for i, (nobs, lags, block, tails, hac, boot) in enumerate(stress):
        cases.append(dict(id=f'stress_{i}', baseline=False, nobs=nobs,
                          phi=0.0 if tails else 0.3, rho=0.5, lags=lags,
                          block=block, tail_df=tails, hac_reps=hac, bootstrap_reps=boot))
    for ci, case in enumerate(cases):
        dgp = _dgp(case)
        case['population'] = dict(
            null_mean=[0.] * 12, null_alpha=[0.] * 12,
            innovation_covariance=dgp.innovation_covariance.tolist(),
            stationary_covariance=dgp.stationary_covariance.tolist(),
            long_run_covariance=dgp.long_run_covariance.tolist(),
            mixture_effects={str(m): _effects(dgp, m, True).tolist() for m in (6, 12)},
            factors='three zero-mean IID Gaussian, volatility .015; independent of innovations',
            burn_in=1000, tail_normalization='sqrt((df-2)/df); common and idiosyncratic',
        )
        case['seeds'] = {}
        for ei, engine in enumerate(('hac', 'bootstrap')):
            case['seeds'][engine] = [
                [int(np.random.SeedSequence([20261102, ci, ei, r, stream]).generate_state(
                    1, dtype=np.uint32)[0]) for stream in (0, 1)]
                for r in range(case[f'{engine}_reps'])
            ]
    return dict(schema=1, master_seed=20261102, cases=cases, num_samples=1999,
                families=[6, 12], factor_count=3, burn_in=1000, innovation_std=0.02,
                target=0.05, confidence_level=0.95,
                effects='first M/2 have alternating +/-3*sqrt(diag(LRV)/T); rest zero',
                reuse='nested M; constant shifts of intercepts; shared data/indices mean/alpha',
                gates='each null marginal size and pointwise noncoverage; Holm FWER; BHY FDR',
                bound='target+3*sqrt(target*(1-target)/R); mixture FDR target+3*empirical SE',
                coverage='normal HAC pointwise only; bootstrap does NOT construct bootstrap CIs',
                power='reported with uncertainty; no minimum gate',
                stress='diagnostic only, never substituted for baseline budgets',
                seed_law='SeedSequence([20261102,case,engine,replicate,stream]); uint32',
                sampling='monthly artificial grid; fractional simple; no RF/cost subtraction')


def _dgp(case: dict) -> SignificanceDgp:
    return SignificanceDgp(case['nobs'], correlation=case['rho'],
                           ar_coefficient=case['phi'], tail_df=case['tail_df'])


def _effects(dgp: SignificanceDgp, members: int, mixture: bool) -> np.ndarray:
    effects = np.zeros(members)
    if mixture:
        count = members // 2
        effects[:count] = (3 * np.sqrt(np.diag(dgp.long_run_covariance)[:count] / dgp.nobs)
                           * np.where(np.arange(count) % 2, -1, 1))
    return effects


def _family_rows(table: pd.DataFrame, statistics: np.ndarray | None,
                 dgp: SignificanceDgp) -> list[dict]:
    """Derive nested/shifted families, preserving full-family correction and actual SEs."""
    records = []
    for members in (6, 12):
        for mixture in (False, True):
            effect = _effects(dgp, members, mixture)
            estimate = table['estimate'].to_numpy()[:members] + effect
            se = table['hac_se'].to_numpy()[:members]
            statistic = estimate / se
            p_hac = 2 * norm.sf(np.abs(statistic))
            counts = None if statistics is None else (
                np.abs(statistics[:, :members]) >= np.abs(statistic)).sum(axis=0)
            pvalue = p_hac if counts is None else (1 + counts) / (len(statistics) + 1)
            corrections = {}
            labels = table.index[:members]
            for method in METHODS:
                correction = adjust_pvalues(pd.Series(pvalue, index=labels),
                                            family_id='stage5', method=method,
                                            significance_level=0.05)
                corrections[method.name] = dict(
                    adjusted=correction['pvalue_adjusted'].tolist(),
                    reject=correction['reject'].tolist(),
                )
            records.append(dict(members=members, mixture=mixture, effect=effect.tolist(),
                                estimate=estimate.tolist(), se=se.tolist(), p_hac=p_hac.tolist(),
                                pvalue=pvalue.tolist(), corrections=corrections,
                                exceedances=None if counts is None else counts.tolist(),
                                noncoverage=(np.abs((estimate-effect)/se) > norm.ppf(.975)).tolist()))
    return records


def _worker_init(source: dict) -> None:
    """Bind spawned workers to the frozen export and verify numerical source per task."""
    global _WORKER_HASHES, _WORKER_SOURCE
    if _source() != source:
        raise RuntimeError('worker imported changed source/environment')
    root = Path(__file__).resolve().parents[4]
    modules = ('datasets/significance.py', 'perfstats/significance.py',
               'perfstats/bootstrap_significance.py', 'perfstats/multiple_testing.py',
               'utils/regression.py', 'models/bootstrap/bootstrap_numba.py',
               'perfstats/run_local/significance_run.py')
    _WORKER_HASHES = {root / 'src/qis' / name: source['hashes'][f'src/qis/{name}']
                      for name in modules}
    _WORKER_SOURCE = _digest(source)


def _check_worker_source() -> None:
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != digest
           for path, digest in _WORKER_HASHES.items()):
        raise RuntimeError('numerical source changed during worker execution')


def _replicate(job: tuple) -> dict:
    _check_worker_source()
    case, engine, replicate, seeds, count = job
    started = time.perf_counter()
    dgp = _dgp(case)
    sample = generate_significance_sample(dgp, seeds[0])
    indices = None
    if engine == 'bootstrap':
        indices = generate_bootstrapped_indices(
            num_data_index=dgp.nobs, bootstrap_type=BootstrapType.STATIONARY,
            num_samples=count, index_length=dgp.nobs, block_size=case['block'],
            min_block_size=1, seed=seeds[1],
        )
    models = {}
    for model, y, x in [('mean', sample.mean_returns, None),
                         ('alpha', sample.alpha_returns, sample.factors)]:
        kwargs = dict(CONVENTIONS, hac_lags=case['lags'])
        if indices is None:
            result = (significance.compute_mean_significance(y, **kwargs) if x is None
                      else significance.compute_alpha_significance(y, x, **kwargs))
            statistics = None
        else:
            kwargs.update(bootstrapped_indices=indices, include_diagnostics=True)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='B=.*minimum bootstrap p-value')
                result = (bootstrap.compute_bootstrap_mean_significance(y, **kwargs) if x is None
                          else bootstrap.compute_bootstrap_alpha_significance(y, x, **kwargs))
            statistics = result.metadata['bootstrap_statistics'].to_numpy()
        models[model] = dict(families=_family_rows(result.table, statistics, dgp),
                             statistics_sha256=None if statistics is None else hashlib.sha256(
                                 statistics.astype('<f8').tobytes()).hexdigest())
    _check_worker_source()
    return dict(case=case['id'], engine=engine, replicate=replicate, seeds=seeds,
                worker_source=_WORKER_SOURCE,
                seconds=time.perf_counter()-started, models=models,
                indices_sha256=None if indices is None else hashlib.sha256(
                    indices.astype('<i8').tobytes()).hexdigest())


def _qr_reference(y: np.ndarray, x: np.ndarray | None, lags: int) -> tuple:
    """Independent dense Bartlett / QR intercept inference, never used by calibration."""
    design = np.ones((len(y), 1)) if x is None else np.column_stack((np.ones(len(y)), x))
    q, r = np.linalg.qr(design, mode='reduced')
    coefficients = np.linalg.solve(r, q.T @ y)
    influence = np.linalg.solve(r, q.T)[0, :, None] * (y-design @ coefficients)
    distance = np.abs(np.subtract.outer(np.arange(len(y)), np.arange(len(y))))
    kernel = np.maximum(1-distance/(lags+1), 0)
    se = np.sqrt(np.diag(influence.T @ kernel @ influence) * len(y)/(len(y)-design.shape[1]))
    return coefficients[0], se


def offline_checks() -> dict:
    """Execute independent references, sampler mapping, reuse and frozen-panel integration.

    Returns:
        Maximum discrepancies and checked dimensions; raises AssertionError on disagreement.
    """
    discrepancies = []
    population_errors = []
    for phi in (0., .3, -.3):
        for rho in (0., .5):
            dgp = SignificanceDgp(96, correlation=rho, ar_coefficient=phi)
            # Independent innovation loading matrix and convergent autocovariance sums.
            loading = .02*np.column_stack((np.full(12, np.sqrt(rho)),
                                           np.sqrt(1-rho)*np.eye(12)))
            q = loading @ loading.T
            stationary = q * sum(phi**(2*k) for k in range(1000))
            lrv = stationary * (1+2*sum(phi**k for k in range(1, 1000)))
            for actual, reference in ((dgp.innovation_covariance, q),
                                       (dgp.stationary_covariance, stationary),
                                       (dgp.long_run_covariance, lrv)):
                np.testing.assert_allclose(actual, reference, rtol=1e-13, atol=1e-16)
                population_errors.append(float(np.max(np.abs(actual-reference))))
    for factors in (0, 1, 3, 6):
        sample = generate_significance_sample(SignificanceDgp(96, factor_count=max(1, factors)),
                                              8123 + factors)
        x = None if factors == 0 else sample.factors
        y = sample.mean_returns if x is None else sample.alpha_returns
        for lags in (0, 3, 6):
            result = (significance.compute_mean_significance(y, **CONVENTIONS, hac_lags=lags)
                      if x is None else significance.compute_alpha_significance(
                          y, x, **CONVENTIONS, hac_lags=lags))
            reference = _qr_reference(y.to_numpy(), None if x is None else x.to_numpy(), lags)
            actual = result.table[['estimate', 'hac_se']].to_numpy().T
            np.testing.assert_allclose(actual, reference, rtol=1e-9, atol=1e-12)
            discrepancies.append(float(np.max(np.abs(actual-reference))))
    # Verify BOTH nested families and shifts against direct public calls, including
    # canonical SEs and every studentized draw, not only the corrected decisions.
    sample = generate_significance_sample(SignificanceDgp(60, ar_coefficient=.3), 99123)
    indices = generate_bootstrapped_indices(num_data_index=60,
        bootstrap_type=BootstrapType.STATIONARY, num_samples=19, index_length=60,
        block_size=6, min_block_size=1, seed=9182)
    for alpha in (False, True):
        x = sample.factors if alpha else None
        y = sample.alpha_returns if alpha else sample.mean_returns
        def call(panel):
            settings = dict(CONVENTIONS, hac_lags=3, bootstrapped_indices=indices,
                            include_diagnostics=True)
            return (bootstrap.compute_bootstrap_mean_significance(panel, **settings) if x is None
                    else bootstrap.compute_bootstrap_alpha_significance(panel, x, **settings))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            base = call(y)
            draws = base.metadata['bootstrap_statistics'].to_numpy()
            derived = _family_rows(base.table, draws, SignificanceDgp(60, ar_coefficient=.3))
            for row in derived:
                members = row['members']
                direct = call(y.iloc[:, :members] + np.array(row['effect']))
                for key, column in [('estimate', 'estimate'), ('se', 'hac_se'),
                                     ('pvalue', 'pvalue_raw')]:
                    np.testing.assert_allclose(row[key], direct.table[column], rtol=1e-9,
                                               atol=1e-12)
                np.testing.assert_allclose(draws[:, :members],
                    direct.metadata['bootstrap_statistics'], rtol=1e-9, atol=1e-12)
            center = (y.mean().to_numpy() if x is None else base.table['estimate'].to_numpy())
            expected = []
            for rows in indices.T:
                intercept, se = _qr_reference((y.to_numpy()-center)[rows],
                                              None if x is None else x.to_numpy()[rows], 3)
                expected.append(intercept/se)
            np.testing.assert_allclose(draws, expected, rtol=1e-9, atol=1e-12)
            discrepancies.append(float(np.max(np.abs(draws-np.array(expected)))))
    p = pd.Series([.001, .013, .07, .2, .8, .03], index=list('abcdef'))
    order = np.argsort(p.to_numpy())
    ranked = p.to_numpy()[order]
    for method in METHODS:
        penalty = len(p) * sum(1/i for i in range(1, len(p)+1))
        raw = ((len(p)-np.arange(len(p)))*ranked if method is MultipleTestingMethod.HOLM
               else penalty*ranked/np.arange(1, len(p)+1))
        cumulative = (np.maximum.accumulate(raw) if method is MultipleTestingMethod.HOLM
                      else np.minimum.accumulate(raw[::-1])[::-1])
        reference = np.empty(len(p))
        reference[order] = np.minimum(cumulative, 1)
        actual = adjust_pvalues(p, family_id='reference', method=method)
        np.testing.assert_allclose(actual['pvalue_adjusted'], reference, rtol=1e-14)
    prices = generate_synthetic_prices()
    returns = to_returns(prices[['SEQ_US', 'SBD_TSY', 'SBD_IG']], freq='ME',
                          is_log_returns=False).iloc[1:]
    integration = significance.compute_mean_significance(returns, **CONVENTIONS)
    assert integration.metadata['nobs'] == len(returns) and np.isfinite(integration.table).all().all()
    return dict(max_absolute_error=max(discrepancies), rtol=1e-9, atol=1e-12,
                population_reference='innovation loading matrix; 1000-term covariance sums',
                population_max_absolute_error=max(population_errors),
                population_targets='means and alphas zero by symmetric innovations and exogeneity',
                factor_counts=[0, 1, 3, 6], lags=[0, 3, 6], bootstrap_reference_draws=19,
                reuse_verified=True, frozen_market_integration_nobs=len(returns))


def _mean_uncertainty(values: np.ndarray) -> dict:
    estimate = float(values.mean())
    se = float(values.std(ddof=1)/np.sqrt(len(values))) if len(values) > 1 else 0.0
    return dict(estimate=estimate, se=se, lower=max(0., estimate-1.96*se),
                upper=min(1., estimate+1.96*se))


def _summarize(plan: dict, completed: list[dict]) -> dict:
    groups = {}
    for payload in completed:
        for model, values in payload['models'].items():
            for family in values['families']:
                key = (payload['case'], payload['engine'], model,
                       family['members'], family['mixture'])
                groups.setdefault(key, []).append(family)
    results, failures = [], []
    for key, families in sorted(groups.items()):
        case_id, engine, model, members, mixture = key
        baseline = next(c['baseline'] for c in plan['cases'] if c['id'] == case_id)
        n = len(families)
        null = np.array(families[0]['effect']) == 0
        raw = np.array([f['pvalue'] for f in families]) <= .05
        nc = np.array([f['noncoverage'] for f in families])
        bound = .05 + 3*np.sqrt(.05*.95/n)
        row = dict(case=case_id, engine=engine, model=model, members=members,
                   mixture=mixture, replications=n, baseline=baseline, binomial_upper_gate=bound,
                   null_mask=null.tolist(),
                   marginal=[_mean_uncertainty(raw[:, j]) for j in range(members)],
                   pointwise_noncoverage=[_mean_uncertainty(nc[:, j]) for j in range(members)],
                   corrections={})
        for j in range(members):
            if baseline and null[j] and raw[:, j].mean() > bound:
                failures.append(dict(key=key, metric=f'marginal_size_H{j}',
                                     observed=float(raw[:, j].mean()), bound=bound))
            if baseline and nc[:, j].mean() > bound:
                failures.append(dict(key=key, metric=f'pointwise_noncoverage_H{j}',
                                     observed=float(nc[:, j].mean()), bound=bound))
        for method in METHODS:
            decisions = np.array([f['corrections'][method.name]['reject'] for f in families])
            false = decisions[:, null].sum(axis=1)
            fdp = false / np.maximum(decisions.sum(axis=1), 1)
            metrics = dict(fwer=_mean_uncertainty((false > 0).astype(float)),
                           fdr=_mean_uncertainty(fdp),
                           power=None if not mixture else _mean_uncertainty(
                               decisions[:, ~null].mean(axis=1)))
            metric = metrics['fwer'] if method is MultipleTestingMethod.HOLM else metrics['fdr']
            limit = (bound if method is MultipleTestingMethod.HOLM or not mixture
                     else .05 + 3*metric['se'])
            metrics.update(gate_bound=float(limit), gate_pass=bool(metric['estimate'] <= limit))
            if baseline and not metrics['gate_pass']:
                failures.append(dict(key=key, metric=method.name, observed=metric['estimate'],
                                     bound=limit))
            row['corrections'][method.name] = metrics
        results.append(row)
    return dict(groups=results, baseline_failures=failures, baseline_pass=not failures)


def _load_checkpoint(path: Path, experiment: str, identity: dict) -> dict:
    record = json.loads(path.read_text(encoding='utf-8'))
    payload = record['payload']
    if (record['experiment'] != experiment or record['sha256'] != _digest(payload)
            or any(payload[k] != v for k, v in identity.items())):
        raise RuntimeError(f'incompatible or corrupt checkpoint: {path}')
    return payload


def _calibrate(root: Path, workers: int) -> dict:
    plan, source = frozen_plan(), _source()
    frozen = dict(plan=plan, source=source)
    experiment = _digest(frozen)
    freeze_path = root / 'frozen.json'
    if freeze_path.exists():
        if json.loads(freeze_path.read_text(encoding='utf-8')) != frozen:
            raise RuntimeError('refusing resume: frozen plan/source/environment changed')
    else:
        # Exclusive creation freezes all seeds and gates before the first outcome.
        with freeze_path.open('x', encoding='utf-8') as handle:
            json.dump(frozen, handle, indent=2, sort_keys=True, allow_nan=False)
    _write(root / 'offline_checks.json', offline_checks())
    checkpoints = root / 'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    jobs, completed = [], []
    for case in plan['cases']:
        for engine in ('hac', 'bootstrap'):
            for replicate, seeds in enumerate(case['seeds'][engine]):
                path = checkpoints / f"{case['id']}_{engine}_{replicate:04d}.json"
                identity = dict(case=case['id'], engine=engine, replicate=replicate, seeds=seeds)
                if path.exists():
                    completed.append(_load_checkpoint(path, experiment, identity))
                else:
                    jobs.append(((case, engine, replicate, seeds, plan['num_samples']), path))
    # Stable scheduling order is not a seed source. Bound queued work to 2*workers.
    # Long bootstrap jobs first; HAC jobs can finish while the last bootstrap jobs drain.
    jobs.sort(key=lambda item: (item[0][1] != 'bootstrap', item[0][0]['id'], item[0][2]))
    start = time.perf_counter()
    session = dict(started_utc=datetime.now(timezone.utc).isoformat(), host=platform.node(),
                   workers=workers, logical_cpus=os.cpu_count(), resumed=len(completed),
                   pending=len(jobs), experiment=experiment,
                   threads={name: os.environ.get(name) for name in THREAD_ENV})
    session_path = root / f"session_{time.time_ns()}.json"
    _write(session_path, session)
    print(f'Frozen {experiment}; resumed={len(completed)}; pending={len(jobs)}; '
          f'workers={workers}; bundle={root}', flush=True)
    remaining = iter(jobs)
    futures = {}
    try:
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_worker_init, initargs=(source,)) as pool:
            for _ in range(min(2*workers, len(jobs))):
                job, path = next(remaining)
                futures[pool.submit(_replicate, job)] = path
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    path = futures.pop(future)
                    payload = future.result()
                    _write(path, dict(experiment=experiment, sha256=_digest(payload), payload=payload))
                    completed.append(payload)
                    if len(completed) % 25 == 0:
                        print(f'checkpointed={len(completed)} elapsed={time.perf_counter()-start:.1f}s',
                              flush=True)
                    following = next(remaining, None)
                    if following is not None:
                        job, destination = following
                        futures[pool.submit(_replicate, job)] = destination
    except BaseException as exception:
        session.update(status='interrupted_or_failed', exception=repr(exception),
                       seconds=time.perf_counter()-start, completed=len(completed))
        _write(session_path, session)
        raise
    summary = _summarize(plan, completed)
    if _source() != source:
        _write(root / 'source_changed.json', dict(expected=source, actual=_source()))
        raise RuntimeError('source/environment changed during experiment; results retained')
    summary.update(experiment=experiment, completed=len(completed),
                   expected=sum(c['hac_reps']+c['bootstrap_reps'] for c in plan['cases']),
                   total_replication_seconds=sum(p['seconds'] for p in completed))
    _write(root / 'summary.json', summary)
    session.update(status='gates_passed' if summary['baseline_pass'] else 'gates_failed',
                   seconds=time.perf_counter()-start, completed=len(completed))
    _write(session_path, session)
    checksums = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(root.rglob('*')) if p.is_file()
                 and p.name not in ('checksums.json', 'active.lock')}
    _write(root / 'checksums.json', checksums)
    if summary['completed'] != summary['expected'] or not summary['baseline_pass']:
        raise RuntimeError(f'calibration gates FAILED; preserved outcomes at {root / "summary.json"}')
    return summary


def run_local(local: Locals = Locals.OFFLINE_CHECKS, output_dir: str | Path | None = None,
              workers: int | None = None) -> dict:
    """Run offline references or the complete frozen calibration on this machine.

    Args:
        local: OFFLINE_CHECKS (default) or explicit CALIBRATION.
        output_dir: C-local output directory, or QIS_SIGNIFICANCE_OUTPUT_DIR.
        workers: One through eight processes, at most the logical CPU count.
            Default is min(8, half the logical CPUs). Numerical threads are one.

    Returns:
        Reference evidence or completed calibration summary. Baseline gate failures
        raise after writing their complete outcomes; no seeds or gates are changed.
    """
    root = _root(output_dir)
    if local not in (Locals.OFFLINE_CHECKS, Locals.CALIBRATION):
        raise ValueError('unsupported local case')
    limit = min(8, os.cpu_count() or 1)
    workers = min(limit, max(1, (os.cpu_count() or 1)//2)) if workers is None else workers
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= limit:
        raise ValueError(f'workers must be an integer in [1,{limit}]')
    for name in THREAD_ENV:
        os.environ[name] = '1'
    lock = root / 'active.lock'
    try:
        with lock.open('x', encoding='utf-8') as handle:
            json.dump(dict(host=platform.node(), pid=os.getpid()), handle)
    except FileExistsError as exception:
        raise RuntimeError('experiment locked; verify recorded host/PID before removing '
                           'a stale active.lock; do not start concurrent writers') from exception
    try:
        if local is Locals.OFFLINE_CHECKS:
            result = offline_checks()
            _write(root / 'offline_checks.json', result)
            print(json.dumps(result), flush=True)
            return result
        return _calibrate(root, workers)
    finally:
        lock.unlink()


if __name__ == '__main__':
    run_local(local=Locals.OFFLINE_CHECKS)

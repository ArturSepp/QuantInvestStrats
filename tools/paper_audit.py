"""
generate the record of every number the JOSS paper quotes.

A count pasted into prose goes stale silently: the repository moves, the sentence does not, and
nothing fails. Five counts in ``paper.md`` had drifted within two days of being written. This
script measures each of them and writes ``docs/audit/paper_numbers.json``;
``qis/tests/test_paper_audit.py`` then fails when the record, the repository and the paper
disagree.

Metrics are of two kinds. A **live** metric is measurable from the installed package or the
working tree, so the test recomputes it and compares. A **recorded** metric needs ``git`` or a
network clone of a consumer, neither of which a test may require, so the test checks only that
the record carries a value and the commit it was measured at.

Run it from the repository root::

    python tools/paper_audit.py            # rewrite docs/audit/paper_numbers.json
    python tools/paper_audit.py --check    # exit 1 unless the record is the truth, write nothing

``--check`` writes nothing and fails closed. It exits 1 on three conditions: a recorded value
that moved, a measurement that could not be taken at all, and a record whose metric names are not
the names measured here. The second and third matter more than the first. The comparison iterates
over the metrics this run measured, so losing a whole measurement class - an absent
``consumers.json``, a collection that will not run - used to drop those metrics out of the
comparison and return 0, and an incomplete check that reports success is worse than no check.
Plain write mode is unaffected: it prints the same warnings and writes the record anyway, because
a partial record is still the best measurement available and its gaps are visible in the file.
``--check`` is not wired into CI: the git metrics move on every commit, so a CI gate on them would
fail on every push. ``qis/tests/test_paper_audit.py`` enforces the part that holds still.

``measured_at_commit`` is the commit that was checked out when the record was written, which is
the parent of the commit that carries the record. It cannot be otherwise: writing the value would
change the tree it names.
"""
# packages
import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
RECORD_PATH: Path = REPO_ROOT.joinpath('docs', 'audit', 'paper_numbers.json')
CONSUMERS_PATH: Path = REPO_ROOT.joinpath('docs', 'audit', 'consumers.json')
PAPER_PATH: Path = REPO_ROOT.joinpath('paper.md')
BACKTESTER_PATH: Path = REPO_ROOT.joinpath('qis', 'portfolio', 'backtester.py')

# a [TODO: ...] marker is a note to the author, not manuscript text, so it counts against
# neither the word guidance nor the unrecorded-number check
TODO_BLOCK = re.compile(r'\[TODO:.*?\]', flags=re.S)

AI_COAUTHOR: str = 'Claude'  # the trailer identity disclosed in the paper's AI usage section


@dataclass
class Metric:
    """
    one number the paper is allowed to quote.

    Attributes:
        value: the measured value
        how: the command or rule that produced it, short enough to read in the json
        live: True when a test can recompute it without git, a network or a subprocess
        paper_phrase: the exact substring paper.md must contain, or None when the paper states
            the quantity without a figure
    """
    value: Any
    how: str
    live: bool = False
    paper_phrase: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        """the record entry, in a stable key order."""
        return dict(value=self.value, how=self.how, live=self.live,
                    paper_phrase=self.paper_phrase)


@dataclass
class AuditResult:
    """
    Attributes:
        commit: the commit the measurement was taken at
        metrics: metric name to measurement
        warnings: measurements that could not be taken, with the reason
    """
    commit: str
    metrics: Dict[str, Metric] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def _git(*arguments: str) -> str:
    """stdout of a git command run in the repository root, stripped."""
    completed = subprocess.run(['git', *arguments],
                               cwd=str(REPO_ROOT),
                               capture_output=True,
                               text=True,
                               check=True)
    return completed.stdout.strip()


def paper_body(text: str) -> str:
    """
    the paper without its YAML front matter.

    JOSS counts the body against its 750-1750 word guidance, and the metadata block is not part
    of it. The block is the region between the first two lines that are exactly ``---``.

    Args:
        text: the full contents of paper.md

    Returns:
        the body text, or the whole input when there is no front matter
    """
    lines = text.splitlines()
    body = text
    if len(lines) > 0 and lines[0].strip() == '---':
        for index in range(1, len(lines)):
            if lines[index].strip() == '---':
                body = '\n'.join(lines[index + 1:])
                break
    return TODO_BLOCK.sub('', body)


def count_exported_symbols() -> Dict[str, int]:
    """
    the size of the public namespace, split into callables and module bindings.

    ``qis.__all__`` is the definition of public, fixed at the end of ``qis/__init__.py``. It is
    used rather than ``dir(qis)`` because importing a submodule binds its name on the package,
    so ``dir(qis)`` depends on what a process has imported and ``__all__`` does not. The module
    bindings are subpackages that the wildcard re-exports leave visible; they are counted
    separately because they are not API.

    Returns:
        a mapping with keys ``exported_symbols`` and ``exported_module_bindings``
    """
    import types
    import qis
    names = list(qis.__all__)
    modules = [name for name in names if isinstance(getattr(qis, name), types.ModuleType)]
    return dict(exported_symbols=len(names), exported_module_bindings=len(modules))


def count_source_lines(path: Path) -> Dict[str, int]:
    """
    physical and nonblank line counts of a source file.

    ``wc -l`` counts newline characters, which is one less than the number of lines when a file
    does not end in a newline. The paper quotes a line count, so the physical count is the one
    that has to be right.

    Args:
        path: the file to measure

    Returns:
        a mapping with keys ``physical`` and ``nonblank``
    """
    lines = path.read_text(encoding='utf-8').splitlines()
    return dict(physical=len(lines), nonblank=sum(1 for line in lines if len(line.strip()) > 0))


def collect_test_count() -> Optional[int]:
    """
    the number of tests pytest collects, or None when collection fails.

    Collection is a subprocess because running it in-process from a test would recurse.
    """
    completed = subprocess.run([sys.executable, '-m', 'pytest', '--collect-only', '-q'],
                               cwd=str(REPO_ROOT),
                               capture_output=True,
                               text=True)
    if completed.returncode != 0:
        return None
    total = 0
    for line in completed.stdout.splitlines():
        match = re.match(r'^\S+\.py: (\d+)$', line.strip())
        if match is not None:
            total += int(match.group(1))
    return total if total > 0 else None


def measure() -> AuditResult:
    """
    take every measurement the paper depends on.

    Returns:
        the metrics, the commit they were taken at, and the measurements that could not be made
    """
    result = AuditResult(commit=_git('rev-parse', 'HEAD'))

    exports = count_exported_symbols()
    result.metrics['exported_symbols'] = Metric(
        value=exports['exported_symbols'],
        how='len(qis.__all__)',
        live=True,
        paper_phrase=str(exports['exported_symbols']))
    result.metrics['exported_module_bindings'] = Metric(
        value=exports['exported_module_bindings'],
        how='the subpackage names the wildcard re-exports leave in the namespace',
        live=True)

    from qis.api import CORE_API, core_api_names
    result.metrics['core_symbols'] = Metric(
        value=len(core_api_names()),
        how='len(qis.api.core_api_names())',
        live=True,
        paper_phrase=f'{len(core_api_names())} symbols')
    result.metrics['capability_groups'] = Metric(
        value=len(CORE_API),
        how='len(qis.api.CORE_API)',
        live=True,
        paper_phrase=f'{len(CORE_API)} capability groups')

    backtester = count_source_lines(BACKTESTER_PATH)
    result.metrics['backtester_physical_lines'] = Metric(
        value=backtester['physical'],
        how='physical lines of qis/portfolio/backtester.py',
        live=True,
        paper_phrase=f"backtester is {backtester['physical']} lines")
    result.metrics['backtester_nonblank_lines'] = Metric(
        value=backtester['nonblank'],
        how='nonblank lines of qis/portfolio/backtester.py',
        live=True)

    words = len(paper_body(PAPER_PATH.read_text(encoding='utf-8')).split())
    result.metrics['paper_body_words'] = Metric(
        value=words,
        how='whitespace-separated tokens in paper.md after the YAML front matter',
        live=True)

    collected = collect_test_count()
    if collected is None:
        result.warnings.append('pytest collection failed; collected_tests left unmeasured')
    else:
        result.metrics['collected_tests'] = Metric(
            value=collected,
            how='pytest --collect-only -q, summed over files, core install',
            live=False)

    result.metrics['commits'] = Metric(
        value=int(_git('rev-list', '--count', 'HEAD')),
        how='git rev-list --count HEAD',
        live=False)
    dates = _git('log', '--format=%ad', '--date=format:%Y-%m').splitlines()
    result.metrics['active_months'] = Metric(
        value=len(set(dates)),
        how='distinct YYYY-MM with at least one commit',
        live=False)
    first_date = _git('log', '--reverse', '--format=%ad', '--date=format:%Y-%m').splitlines()[0]
    last_date = dates[0]
    first_year, first_month = (int(part) for part in first_date.split('-'))
    last_year, last_month = (int(part) for part in last_date.split('-'))
    result.metrics['span_months'] = Metric(
        value=(last_year - first_year) * 12 + (last_month - first_month) + 1,
        how='calendar months from the first commit to the last, inclusive',
        live=False)
    result.metrics['first_commit_month'] = Metric(
        value=first_date,
        how='git log --reverse --format=%ad --date=format:%Y-%m',
        live=False)
    result.metrics['last_commit_month'] = Metric(
        value=last_date,
        how='git log -1 --format=%ad --date=format:%Y-%m',
        live=False)

    trailers = _git('log', '--format=%H %(trailers:key=Co-authored-by,valueonly)').splitlines()
    ai_commits = [line for line in trailers if AI_COAUTHOR.lower() in line.lower()]
    result.metrics['ai_coauthored_commits'] = Metric(
        value=len(ai_commits),
        how=f'commits carrying a Co-authored-by trailer naming {AI_COAUTHOR}',
        live=False)
    if len(ai_commits) > 0:
        ai_dates = _git('log', '--format=%ad %(trailers:key=Co-authored-by,valueonly)',
                        '--date=format:%Y-%m-%d').splitlines()
        ai_dates = [line.split()[0] for line in ai_dates if AI_COAUTHOR.lower() in line.lower()]
        result.metrics['ai_coauthored_first_day'] = Metric(
            value=ai_dates[-1], how='earliest AI co-authored commit', live=False)
        result.metrics['ai_coauthored_last_day'] = Metric(
            value=ai_dates[0], how='latest AI co-authored commit', live=False)

    if CONSUMERS_PATH.is_file():
        consumers = json.loads(CONSUMERS_PATH.read_text(encoding='utf-8'))
        for name, entry in consumers.get('consumers', {}).items():
            result.metrics[f'consumer_{name}_symbols'] = Metric(
                value=entry['distinct_symbols'],
                how=f"tools/audit_consumers.py at {entry['commit'][:12]}",
                live=False,
                paper_phrase=str(entry['distinct_symbols']))
            result.metrics[f'consumer_{name}_call_sites'] = Metric(
                value=entry['call_sites'],
                how=f"tools/audit_consumers.py at {entry['commit'][:12]}",
                live=False,
                paper_phrase=str(entry['call_sites']))
    else:
        result.warnings.append(f'{CONSUMERS_PATH.name} absent; consumer counts left unmeasured')

    return result


def to_record(result: AuditResult) -> Dict[str, Any]:
    """the json document written to docs/audit/paper_numbers.json."""
    return dict(
        generated_by='tools/paper_audit.py',
        measured_at_commit=result.commit,
        note=('every number paper.md quotes. Regenerate with python tools/paper_audit.py; '
              'qis/tests/test_paper_audit.py fails when this file and the repository disagree.'),
        metrics={name: metric.to_json() for name, metric in sorted(result.metrics.items())})


def main() -> int:
    """write or check the record; returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true',
                        help='exit 1 when the record is out of date, incomplete, or carries a '
                             'different set of metrics, instead of rewriting it')
    arguments = parser.parse_args()

    result = measure()
    for warning in result.warnings:
        print(f'warning: {warning}', file=sys.stderr)
    record = to_record(result)
    serialised = json.dumps(record, indent=2) + '\n'

    if arguments.check:
        if not RECORD_PATH.is_file():
            print(f'{RECORD_PATH} does not exist; run python tools/paper_audit.py',
                  file=sys.stderr)
            return 1
        if len(result.warnings) > 0:
            print(f'{len(result.warnings)} measurement(s) above could not be taken, so this '
                  f'comparison would cover less than the record does', file=sys.stderr)
            return 1
        current = json.loads(RECORD_PATH.read_text(encoding='utf-8'))
        measured_names = set(record['metrics'])
        recorded_names = set(current.get('metrics', {}))
        if measured_names != recorded_names:
            for name in sorted(recorded_names - measured_names):
                print(f'{name}: in {RECORD_PATH.name}, not measured here', file=sys.stderr)
            for name in sorted(measured_names - recorded_names):
                print(f'{name}: measured here, not in {RECORD_PATH.name}', file=sys.stderr)
            return 1
        moved = [name for name, metric in record['metrics'].items()
                 if current['metrics'].get(name, {}).get('value') != metric['value']]
        if len(moved) > 0:
            for name in moved:
                was = current['metrics'].get(name, {}).get('value')
                print(f'{name}: recorded {was}, measured {record["metrics"][name]["value"]}',
                      file=sys.stderr)
            return 1
        print(f'{len(record["metrics"])} metrics agree with {RECORD_PATH.name}')
        return 0

    RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECORD_PATH.write_text(serialised, encoding='utf-8')
    print(f'wrote {len(record["metrics"])} metrics to {RECORD_PATH}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

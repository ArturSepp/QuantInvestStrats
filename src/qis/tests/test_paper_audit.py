"""
the numbers the paper quotes are the numbers the repository has.

``paper.md`` quoted eleven capability groups against twelve, a 273-line backtester against 274,
929 tests against 988, and 278 commits against 289 - all within two days of being written, and
none of them failing anything. A count in prose next to code that moves is the same defect as a
docstring next to a signature that moves, and it gets the same treatment: it is generated, it is
recorded, and the record is enforced.

``tools/paper_audit.py`` writes ``docs/audit/paper_numbers.json``. This file checks three things
against it:

  1. every **live** metric the manuscript actually quotes still has the recorded value;
  2. every metric carrying a ``paper_phrase`` has that phrase in ``paper.md``, so a number cannot
     be corrected in the record and left stale in the manuscript;
  3. ``paper.md`` quotes no large number that the record does not know about, which is what stops
     a new hand-measured count from being pasted in.

A **live** metric is one measurable from the installed package or the working tree; ``live`` is a
statement about how a number can be obtained, not about whether it is enforced. Check 1 enforces
only the subset carrying a ``paper_phrase``, because a metric the manuscript never quotes cannot
make the manuscript wrong - it can only turn the build red. Enforcing all seven made two builds
red in the first three days and caught nothing either time: once because two words of prose moved
``paper_body_words``, once because a module docstring moved a line count. A check that fires when
a proxy moves but the claim stays true is the kind that gets routed around, which is the same
reason the lint job in ``.github/workflows/ci.yml`` gates changed lines rather than whole files.
The unquoted metrics are still measured, still recorded and still refreshed; nothing asserts on
their values.

A **recorded** metric needs ``git`` or a network clone of a consumer repository. Neither may run
in a test, so those are checked for presence and provenance rather than recomputed. Refresh them
with ``python tools/paper_audit.py`` before cutting a release tag.
"""
# packages
import json
import re
from pathlib import Path
from typing import Any, Dict, List
import pytest
# qis / project
import qis

REPO_ROOT: Path = Path(qis.__file__).resolve().parents[2]
RECORD_PATH: Path = REPO_ROOT.joinpath('docs', 'audit', 'paper_numbers.json')
PAPER_PATH: Path = REPO_ROOT.joinpath('paper.md')
BACKTESTER_PATH: Path = REPO_ROOT.joinpath('src', 'qis', 'portfolio', 'backtester.py')
IS_REPOSITORY_CHECKOUT: bool = REPO_ROOT.joinpath('pyproject.toml').is_file()

pytestmark = pytest.mark.skipif(
    not IS_REPOSITORY_CHECKOUT,
    reason='repository audit records are absent from an installed wheel')

# integers of three digits or more that appear in paper.md and are not measurements of this
# repository. Each needs a reason: the list is what keeps check 3 from becoming a rubber stamp.
NON_METRIC_NUMBERS: Dict[str, str] = {
    '250': 'the sample length in the bootstrap example',
    '400': 'the number of draws in the bootstrap example',
    '260': 'periods per year used to annualise the bootstrap bias',
    '2022': 'the year of the first commit',
    '2023': 'a cited publication year',
    '2025': 'a cited publication year',
    '2026': 'the year of writing and of cited publications',
    '1994': 'Politis and Romano, cited for the stationary bootstrap',
    '310': 'the python floor written as 3.10 without its separator',
    '314': 'the python ceiling written as 3.14 without its separator',
}

# a [TODO: ...] marker is a note to the author, not manuscript text, so it counts against
# neither the word guidance nor the unrecorded-number check
TODO_BLOCK = re.compile(r'\[TODO:.*?\]', flags=re.S)

LARGE_NUMBER = re.compile(r'(?<![\d.])(\d{3,})(?![\d.])')


def _record() -> Dict[str, Any]:
    """the generated record, or a skip when it has not been generated in this checkout."""
    if not RECORD_PATH.is_file():
        pytest.skip(f'{RECORD_PATH} is absent; run python tools/paper_audit.py')
    return json.loads(RECORD_PATH.read_text(encoding='utf-8'))


def _live_measurements() -> Dict[str, Any]:
    """
    recompute the metrics that need neither git nor a network.

    Kept deliberately small and duplicated from ``tools/paper_audit.py`` rather than imported:
    a check that calls the code it is checking cannot fail when that code is wrong, and ``tools``
    is not an importable package.

    Returns:
        metric name to freshly measured value
    """
    import types
    from qis.api import CORE_API, core_api_names
    names = list(qis.__all__)
    modules = [name for name in names if isinstance(getattr(qis, name), types.ModuleType)]
    backtester_lines = BACKTESTER_PATH.read_text(encoding='utf-8').splitlines()
    paper_text = PAPER_PATH.read_text(encoding='utf-8')
    return {
        'exported_symbols': len(names),
        'exported_module_bindings': len(modules),
        'core_symbols': len(core_api_names()),
        'capability_groups': len(CORE_API),
        'backtester_physical_lines': len(backtester_lines),
        'backtester_nonblank_lines': sum(1 for line in backtester_lines if len(line.strip()) > 0),
        'paper_body_words': len(_paper_body(paper_text).split()),
    }


def _paper_body(text: str) -> str:
    """the paper without its YAML front matter; the same rule tools/paper_audit.py applies."""
    lines = text.splitlines()
    body = text
    if len(lines) > 0 and lines[0].strip() == '---':
        for index in range(1, len(lines)):
            if lines[index].strip() == '---':
                body = '\n'.join(lines[index + 1:])
                break
    return TODO_BLOCK.sub('', body)


def test_record_exists_and_names_its_commit() -> None:
    """the record says which commit it was measured at, or it is not evidence."""
    record = _record()
    assert len(record.get('measured_at_commit', '')) == 40, (
        'docs/audit/paper_numbers.json has no full commit sha; regenerate it with '
        'python tools/paper_audit.py')
    assert len(record.get('metrics', {})) >= 10, 'the record is nearly empty; regenerate it'


LIVE_METRICS: List[str] = sorted(_live_measurements()) if IS_REPOSITORY_CHECKOUT else []


def test_live_metrics_are_recorded() -> None:
    """every metric measured here remains present in the generated record."""
    record = _record()
    missing = sorted(set(LIVE_METRICS) - set(record['metrics']))
    assert not missing, f'{missing} are measured here but absent from the record; regenerate it'


def _quoted_live_metrics() -> List[str]:
    """live metrics whose values are quoted in the paper."""
    record = _record()
    return [metric for metric in LIVE_METRICS
            if record['metrics'].get(metric, {}).get('paper_phrase') is not None]


QUOTED_LIVE_METRICS: List[str] = _quoted_live_metrics() if IS_REPOSITORY_CHECKOUT else []


@pytest.mark.parametrize('metric', QUOTED_LIVE_METRICS)
def test_quoted_live_metric_matches_the_record(metric: str) -> None:
    """a metric the paper quotes still has the value the record and the paper carry."""
    record = _record()
    recorded = record['metrics'][metric]['value']
    measured = _live_measurements()[metric]
    assert measured == recorded, (
        f'{metric}: the repository says {measured}, docs/audit/paper_numbers.json says '
        f'{recorded}. Run python tools/paper_audit.py and correct paper.md if the number '
        f'appears there')


def _phrases() -> List[str]:
    """the exact strings paper.md is required to contain."""
    record = _record()
    return sorted({metric['paper_phrase'] for metric in record['metrics'].values()
                   if metric.get('paper_phrase')})


PAPER_PHRASES: List[str] = _phrases() if IS_REPOSITORY_CHECKOUT else []


@pytest.mark.parametrize('phrase', PAPER_PHRASES)
def test_paper_quotes_the_recorded_value(phrase: str) -> None:
    """
    a number corrected in the record is corrected in the manuscript too.

    Without this the record becomes a second place for the truth to live, and the paper keeps
    the old figure - which is exactly the failure the record exists to prevent.
    """
    paper = PAPER_PATH.read_text(encoding='utf-8')
    assert phrase in paper, (
        f'paper.md does not contain {phrase!r}, which docs/audit/paper_numbers.json records as '
        f'the current value')


def test_paper_quotes_no_unrecorded_measurement() -> None:
    """
    every large number in the paper is either a recorded measurement or an allowed constant.

    This is the check that catches the next stale count rather than the ones already found: a
    hand-measured figure pasted into the manuscript has no entry in the record, so it fails here
    on the first run.
    """
    record = _record()
    recorded_values = {str(metric['value']) for metric in record['metrics'].values()}
    body = _paper_body(PAPER_PATH.read_text(encoding='utf-8'))
    # a bibliography key such as sepp2026robust is not a number the paper quotes
    body = re.sub(r'@\w+', '', body)
    unrecorded = sorted({found for found in LARGE_NUMBER.findall(body)
                         if found not in recorded_values and found not in NON_METRIC_NUMBERS})
    assert not unrecorded, (
        f'paper.md quotes {unrecorded}, which is neither in docs/audit/paper_numbers.json nor '
        f'in NON_METRIC_NUMBERS. Measure it in tools/paper_audit.py, or add it to the allowed '
        f'list with the reason it is not a measurement')

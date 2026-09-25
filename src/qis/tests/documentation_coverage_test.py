"""Every core analytics symbol is explained by at least one methodology chapter.

``src/qis/api.py`` groups the documented core by capability. For the analytics capabilities the
promise of documentation means a chapter that names the symbol, so its formula and conventions
are one search away. Plotting, date, file and DataFrame helpers are documented by their
docstrings and the shared plotting note, and are excluded here.
"""

# packages
import re
import runpy
from pathlib import Path
from typing import List

import pytest

# qis
import qis
from qis.api import CORE_API


REPO_ROOT: Path = Path(__file__).resolve().parents[3]
CHECKER_PATH: Path = REPO_ROOT.joinpath('tools', 'check_docs.py')
if not CHECKER_PATH.is_file():
    pytest.skip('Documentation sources are not shipped in the wheel.', allow_module_level=True)
PAGES: List[str] = sorted(runpy.run_path(str(CHECKER_PATH))['METHODOLOGY_PAGES'])
PACKAGED_NOTES: List[Path] = sorted(REPO_ROOT.joinpath('src', 'qis', 'docs').glob('*.md'))
EXCLUDED_GROUPS = frozenset({
    'Plots', 'Dates, schedules and annualisation', 'DataFrame utilities', 'File and figure output',
})
ANALYTICS = [(group, name) for group, names in CORE_API.items()
             if group not in EXCLUDED_GROUPS for name in names]


def _corpus() -> str:
    """Text of every methodology chapter and packaged note."""
    texts = [REPO_ROOT.joinpath('docs', page).read_text(encoding='utf-8') for page in PAGES]
    texts += [path.read_text(encoding='utf-8') for path in PACKAGED_NOTES]
    return '\n'.join(texts)


CORPUS = _corpus()


def test_analytics_groups_are_found() -> None:
    """The excluded set names real groups, so the coverage list is not silently empty."""
    assert EXCLUDED_GROUPS <= set(CORE_API)
    assert len(ANALYTICS) >= 80


@pytest.mark.parametrize('group,name', ANALYTICS, ids=[name for _, name in ANALYTICS])
def test_core_analytics_symbol_is_documented(group: str, name: str) -> None:
    """A core analytics symbol is named in a methodology chapter or packaged note.

    Args:
        group: capability group in ``CORE_API``
        name: exported symbol
    """
    assert re.search(rf'\b{re.escape(name)}\b', CORPUS), (
        f'{name} ({group}) is not named in any methodology chapter')


CATALOGUE: Path = REPO_ROOT.joinpath('docs', 'performance_statistics.md')


@pytest.mark.parametrize('member', list(qis.PerfStat.__members__))
def test_perf_stat_member_is_in_the_catalogue(member: str) -> None:
    """Every ``PerfStat`` column has an entry in the performance-statistic catalogue.

    Args:
        member: ``PerfStat`` member name, e.g. ``SHARPE_RF0``
    """
    text = CATALOGUE.read_text(encoding='utf-8')
    assert re.search(rf'`(?:PerfStat\.)?{member}`', text), (
        f'PerfStat.{member} has no entry in performance_statistics.md')

"""Every methodology reference is a verbatim entry of the single bibliography.

``docs/bibliography.md`` holds each cited work once, in one style. A chapter's References section
is a numbered list whose items begin with a bibliography entry, word for word, and may append a
note on what the work contributes to that chapter. This keeps one spelling of every citation and
one place to correct it.
"""

# packages
import re
import runpy
from pathlib import Path
from typing import List

import pytest


REPO_ROOT: Path = Path(__file__).resolve().parents[3]
BIBLIOGRAPHY: Path = REPO_ROOT.joinpath('docs', 'bibliography.md')
CHECKER_PATH: Path = REPO_ROOT.joinpath('tools', 'check_docs.py')
if not BIBLIOGRAPHY.is_file() or not CHECKER_PATH.is_file():
    pytest.skip('Documentation sources are not shipped in the wheel.', allow_module_level=True)
PAGES: List[str] = sorted(runpy.run_path(str(CHECKER_PATH))['METHODOLOGY_PAGES'])
PENDING_MARK = ' [pending publisher check]'
SOFTWARE_CITATION = (
    'Sepp, A. qis: Performance analytics, portfolio backtesting, risk analysis, and factsheet '
    'reporting in Python. [Software citation metadata]'
    '(https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).')


def _normalise(text: str) -> str:
    """Collapse whitespace so that line wrapping does not change an entry."""
    return re.sub(r'\s+', ' ', text).strip()


def list_items(text: str, marker: str) -> List[str]:
    """Items of the Markdown lists in ``text`` whose bullets match ``marker``.

    Args:
        text: Markdown source
        marker: regular expression for the list bullet, e.g. ``-`` or ``\\d+\\.``

    Returns:
        each item with its continuation lines joined
    """
    items: List[str] = []
    current = None
    for line in text.splitlines():
        match = re.match(rf'^(?:{marker})\s+(.*)$', line)
        if match:
            if current is not None:
                items.append(_normalise(current))
            current = match[1]
        elif current is not None and line.startswith(' ') and line.strip():
            current += ' ' + line.strip()
        elif current is not None:
            items.append(_normalise(current))
            current = None
    if current is not None:
        items.append(_normalise(current))
    return items


def bibliography_entries() -> List[str]:
    """The canonical entries, without their pending-verification mark."""
    text = BIBLIOGRAPHY.read_text(encoding='utf-8')
    body = text.split('\n## ', 1)[1]
    entries = []
    for item in list_items(body, '-'):
        entries.append(item[:-len(PENDING_MARK)] if item.endswith(PENDING_MARK) else item)
    return entries


def references_section(page: str) -> str:
    """The References section of one methodology page."""
    text = REPO_ROOT.joinpath('docs', page).read_text(encoding='utf-8')
    return text.split('\n## References\n', 1)[1]


def test_bibliography_has_unique_entries() -> None:
    """Each work appears once, and the software citation is among the entries."""
    entries = bibliography_entries()
    assert len(entries) >= 20
    assert len(entries) == len(set(entries)), 'duplicate bibliography entry'
    assert SOFTWARE_CITATION in entries


@pytest.mark.parametrize('page', PAGES)
def test_references_are_numbered_bibliography_entries(page: str) -> None:
    """A chapter cites only bibliography entries, as a numbered list including the software.

    Args:
        page: methodology page file name under ``docs/``
    """
    section = references_section(page)
    bullets = list_items(section, '-')
    assert not bullets, f'{page}: use a numbered reference list, not bullets'
    items = list_items(section, r'\d+\.')
    assert items, f'{page}: no numbered references found'
    entries = bibliography_entries()
    for item in items:
        assert any(item.startswith(entry) for entry in entries), (
            f'{page}: reference is not a bibliography entry: {item[:120]}')
    assert any(item.startswith(SOFTWARE_CITATION) for item in items), (
        f'{page}: cite the software with the canonical entry')

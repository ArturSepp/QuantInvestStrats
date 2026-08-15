"""
the three release version locations agree.

A release touches ``pyproject.toml``, ``CITATION.cff`` and the ``@software`` BibTeX entry in
``README.md``, and nothing has connected them. They are the same drift shape as a stale count in
prose: three hand-maintained copies of one fact, where updating two of them leaves no trace. In
the sibling ``optimalportfolios`` repository the three read 6.3.0, 6.2.0 and versionless at the
same commit, and a reader following ``CITATION.cff`` would have cited a release two versions
behind the code.

``date-released`` is checked for shape rather than value. A bare year passes a human's glance and
sorts wrong wherever the field is read as a date, Zenodo included.

The checks are skipped when the repository root is not on disk, which is the case for an
installed wheel: these three files are packaging metadata, not package data, so there is nothing
to compare against.
"""
# packages
import re
from pathlib import Path
from typing import Optional
import pytest
import yaml
# qis / project
import qis

REPO_ROOT: Path = Path(qis.__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    not REPO_ROOT.joinpath('pyproject.toml').is_file(),
    reason='repository root not on disk (installed wheel), so there is no metadata to compare')

PROJECT_VERSION = re.compile(r'^\s*version\s*=\s*["\']([^"\']+)["\']', flags=re.M)
SOFTWARE_ENTRY = re.compile(r'@software\{.*?\n\}', flags=re.S)
BIBTEX_VERSION = re.compile(r'version\s*=\s*\{([^}]+)\}')


def _pyproject_version() -> str:
    """
    the ``[project] version`` string.

    Parsed with a regular expression rather than ``tomllib``, which arrived in 3.11 while the
    supported floor is 3.10 - the same reason ``test_documentation.py`` reads the dependency
    table that way.

    Returns:
        the version

    Raises:
        AssertionError: when the table is no longer where the parser expects it
    """
    text = REPO_ROOT.joinpath('pyproject.toml').read_text(encoding='utf-8')
    match = PROJECT_VERSION.search(text.split('[project]', 1)[-1])
    assert match is not None, "pyproject.toml no longer has a '[project] version' entry"
    return match.group(1)


def _citation_field(name: str) -> Optional[str]:
    """
    one top-level field of ``CITATION.cff``.

    Args:
        name: the key to read

    Returns:
        the value as a string, or None when the key is absent
    """
    data = yaml.safe_load(REPO_ROOT.joinpath('CITATION.cff').read_text(encoding='utf-8'))
    value = data.get(name)
    return None if value is None else str(value)


def _readme_bibtex_version() -> str:
    """
    the ``version`` field of the ``@software`` entry in the README.

    Returns:
        the version

    Raises:
        AssertionError: when the entry is missing or carries no version field
    """
    text = REPO_ROOT.joinpath('README.md').read_text(encoding='utf-8')
    entry = SOFTWARE_ENTRY.search(text)
    assert entry is not None, "README.md no longer has an '@software' BibTeX entry"
    match = BIBTEX_VERSION.search(entry.group(0))
    assert match is not None, (
        "the '@software' entry in README.md carries no version field, so a reader citing it "
        "cannot say which release they used")
    return match.group(1).strip()


def test_citation_cff_matches_pyproject() -> None:
    """the version a citing reader copies is the version that was released."""
    citation = _citation_field('version')
    pyproject = _pyproject_version()
    assert citation == pyproject, (
        f"CITATION.cff says {citation}, pyproject.toml says {pyproject}")


def test_readme_bibtex_matches_pyproject() -> None:
    """the README's software entry names the same release as the package metadata."""
    readme = _readme_bibtex_version()
    pyproject = _pyproject_version()
    assert readme == pyproject, (
        f"the README @software entry says {readme}, pyproject.toml says {pyproject}")


def test_citation_cff_date_released_is_a_date() -> None:
    """``date-released`` is an ISO date; a bare year reads as one and sorts as none."""
    date_released = _citation_field('date-released')
    assert date_released is not None, "CITATION.cff has no date-released field"
    assert re.fullmatch(r'\d{4}-\d{2}-\d{2}', date_released), (
        f"date-released must be YYYY-MM-DD, got {date_released!r}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

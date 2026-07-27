"""
the documentation agrees with the repository it documents.

Two checks of the same shape, each asserting something a reader is told against the thing it
describes: a link points at a file that exists, and the README's core dependency list is the
list ``pip install qis`` actually resolves.

**Links.** The README told a reader to run ``qis/examples/performances.py`` and to look in
``qis/examples/notebooks``. Neither path had existed since the examples were reorganised into
subdirectories, and nothing failed, because a dead link in a markdown file is invisible until
somebody clicks it. The same class covers image references, which are the most numerous links in
the README and the easiest to strand when a figure is renamed.

Two kinds of link are checked, both resolvable without a network:

  * a relative path, as in ``![image info](qis/examples/figures/perf1.PNG)``;
  * a ``github.com/<owner>/<repo>/blob/<ref>/<path>`` url pointing back into this repository,
    which is how the README refers to its own example scripts.

Everything else - an external url, an anchor, a mailto - is skipped. A test may not reach the
network, so an external link is not this file's business. ``docs/_included/`` and ``docs/api/``
are skipped too: ``docs/conf.py`` writes both at build time, so they are absent from a checkout
by design rather than by accident.

**Dependencies.** The README listed ``yfinance`` and ``pandas-datareader`` under "Core
dependencies" while ``pyproject.toml`` had both in the ``[data]`` extra. A reader planning an
install therefore saw two packages that a core install does not pull, and the project's own rule
is that library code never imports either. The list is hand-written prose next to a machine-read
table, which is the same drift shape as a stale count, so it is checked rather than trusted.
"""
# packages
import re
from pathlib import Path
from typing import List, NamedTuple
import pytest
# qis / project
import qis

REPO_ROOT: Path = Path(qis.__file__).resolve().parent.parent

# the markdown files a reader actually reads; generated output and vendored notebooks are not
DOCUMENT_GLOBS: List[str] = ['README.md', 'CONTRIBUTING.md', 'AGENTS.md', 'CHANGELOG.md',
                             'paper.md', 'docs/*.md', 'qis/docs/*.md']

# written by docs/conf.py during the sphinx build, so absent from a checkout by design
GENERATED_PREFIXES: List[str] = ['docs/_included/', 'docs/api/']

SELF_URL_PATTERN = re.compile(
    r'https://github\.com/ArturSepp/QuantInvestStrats/(?:blob|tree)/[^/\s)]+/([^\s)#]+)')
MARKDOWN_LINK_PATTERN = re.compile(r'!?\[[^\]]*\]\(([^)\s]+)')


class Link(NamedTuple):
    """
    Attributes:
        document: the markdown file the link was found in, relative to the repository root
        target: the path the link resolves to, relative to the repository root
        raw: the link as written, for the failure message
    """
    document: str
    target: str
    raw: str


def _documents() -> List[Path]:
    """every markdown document in scope, sorted."""
    paths: List[Path] = []
    for pattern in DOCUMENT_GLOBS:
        paths.extend(sorted(REPO_ROOT.glob(pattern)))
    return [path for path in paths if path.is_file()]


def _links_in(path: Path) -> List[Link]:
    """
    the repository-internal links one document contains.

    A github blob url names a path from the repository root. A relative markdown link names a
    path from the directory of the document it is written in, which is not the same thing:
    ``qis/docs/gallery.md`` linking to ``images/multi_asset.png`` means
    ``qis/docs/images/multi_asset.png``.

    Targets are normalised to forward slashes with ``as_posix``. ``str()`` on a ``Path`` gives
    backslashes on Windows, and every prefix compared against a target here is written with
    forward slashes, so the skip list silently matched nothing on Windows and passed five
    generated paths through to the existence check.

    Args:
        path: the markdown file

    Returns:
        one entry per link that should resolve to a file in this repository
    """
    text = path.read_text(encoding='utf-8', errors='replace')
    document = path.relative_to(REPO_ROOT).as_posix()
    directory = path.parent
    links: List[Link] = []
    for match in SELF_URL_PATTERN.finditer(text):
        links.append(Link(document=document, target=match.group(1), raw=match.group(0)))
    for match in MARKDOWN_LINK_PATTERN.finditer(text):
        target = match.group(1)
        if target.startswith(('http://', 'https://', 'mailto:', '#')):
            continue
        target = target.split('#')[0]
        if len(target) == 0:
            continue
        try:
            from_root = directory.joinpath(target).resolve().relative_to(REPO_ROOT)
        except ValueError:
            continue  # a link that escapes the repository is not this file's business
        links.append(Link(document=document, target=from_root.as_posix(), raw=match.group(0)))
    return links


def _all_links() -> List[Link]:
    """every checkable link across the documents in scope."""
    return [link for path in _documents() for link in _links_in(path)]


ALL_LINKS: List[Link] = _all_links()
LINK_IDS: List[str] = [f'{link.document}:{link.target}' for link in ALL_LINKS]


def test_documents_are_found() -> None:
    """the globs still match something, so a green run is not an empty run."""
    documents = _documents()
    assert len(documents) >= 5, f"only {len(documents)} documents matched {DOCUMENT_GLOBS}"
    assert len(ALL_LINKS) >= 20, f"only {len(ALL_LINKS)} internal links found; the patterns broke"


@pytest.mark.parametrize('prefix', GENERATED_PREFIXES)
def test_generated_prefix_is_exercised(prefix: str) -> None:
    """
    the skip list still matches something.

    A skip that matches nothing is invisible: it passes on every platform and silently stops
    excluding what it was written to exclude. That is exactly how the Windows path separator got
    through - targets were built with backslashes, these prefixes are written with forward
    slashes, and five generated paths went to the existence check instead of being skipped.
    """
    assert any(link.target.startswith(prefix) for link in ALL_LINKS), (
        f"no link starts with {prefix!r}, so this entry in GENERATED_PREFIXES excludes nothing. "
        f"Either the documentation stopped linking to generated output, or link targets are no "
        f"longer normalised to forward slashes")


@pytest.mark.parametrize('link', ALL_LINKS, ids=LINK_IDS)
def test_internal_link_resolves(link: Link) -> None:
    """a link into this repository points at a file or directory that is here."""
    if any(link.target.startswith(prefix) for prefix in GENERATED_PREFIXES):
        pytest.skip(f'{link.target} is written by docs/conf.py at build time')
    resolved = REPO_ROOT.joinpath(link.target)
    assert resolved.exists(), (
        f"{link.document} links to {link.target}, which does not exist. "
        f"Written as {link.raw!r}")


README_PATH: Path = REPO_ROOT.joinpath('README.md')
PYPROJECT_PATH: Path = REPO_ROOT.joinpath('pyproject.toml')

README_DEPENDENCY_BLOCK = re.compile(r'^Core dependencies:\n((?:    .+\n)+)', flags=re.M)
PYPROJECT_DEPENDENCY_BLOCK = re.compile(r'^dependencies = \[\n(.*?)^\]', flags=re.M | re.S)
REQUIREMENT_NAME = re.compile(r'^\s*"?([A-Za-z][A-Za-z0-9_.-]*)')


def _readme_core_dependencies() -> List[str]:
    """
    the distribution names the README lists as core, lower-cased.

    ``python`` is dropped: it is a interpreter floor rather than a dependency, and
    ``pyproject.toml`` carries it as ``requires-python``.

    Returns:
        the names, sorted

    Raises:
        AssertionError: when the block is no longer where the parser expects it
    """
    match = README_DEPENDENCY_BLOCK.search(README_PATH.read_text(encoding='utf-8'))
    assert match is not None, "README.md no longer has a 'Core dependencies:' block"
    names = []
    for line in match.group(1).splitlines():
        found = REQUIREMENT_NAME.match(line)
        if found is not None and found.group(1).lower() != 'python':
            names.append(found.group(1).lower())
    return sorted(names)


def _pyproject_dependencies() -> List[str]:
    """
    the distribution names in the project's ``dependencies`` table, lower-cased.

    Parsed with a regular expression rather than ``tomllib``, which arrived in 3.11 while the
    supported floor is 3.10.

    Returns:
        the names, sorted
    """
    match = PYPROJECT_DEPENDENCY_BLOCK.search(PYPROJECT_PATH.read_text(encoding='utf-8'))
    assert match is not None, "pyproject.toml no longer has a 'dependencies = [' table"
    names = []
    for line in match.group(1).splitlines():
        found = REQUIREMENT_NAME.match(line)
        if found is not None:
            names.append(found.group(1).lower())
    return sorted(names)


def test_readme_core_dependencies_match_pyproject() -> None:
    """
    the README's core list is the list a core install resolves.

    An extra in the README is the defect that prompted this: a reader is told to expect a package
    that ``pip install qis`` does not pull, and a contributor is told that library code may import
    it. A missing one is the mirror, and equally wrong.
    """
    readme = _readme_core_dependencies()
    pyproject = _pyproject_dependencies()
    only_in_readme = sorted(set(readme) - set(pyproject))
    only_in_pyproject = sorted(set(pyproject) - set(readme))
    assert not only_in_readme and not only_in_pyproject, (
        f"README.md lists {only_in_readme} as core and pyproject.toml does not; "
        f"pyproject.toml requires {only_in_pyproject} and the README does not list them")

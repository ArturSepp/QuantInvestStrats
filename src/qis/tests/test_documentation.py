"""
the documentation agrees with the repository it documents.

Four checks of the same shape, each asserting something a reader is told against the thing it
describes: a link points at a file that exists, an in-page anchor points at a section that
exists, the README's core dependency list is the list ``pip install qis`` actually resolves, and
every name a README code block uses is a name the blocks above it bound.

**Links.** The README told a reader to run ``examples/performances.py`` and to look in
``examples/notebooks``. Neither path had existed since the examples were reorganised into
subdirectories, and nothing failed, because a dead link in a markdown file is invisible until
somebody clicks it. The same class covers image references, which are the most numerous links in
the README and the easiest to strand when a figure is renamed.

Two kinds of link are checked, both resolvable without a network:

  * a relative path, as in ``![image info](examples/figures/perf1.PNG)``;
  * a ``github.com/<owner>/<repo>/blob/<ref>/<path>`` url pointing back into this repository,
    which is how the README refers to its own example scripts.

Everything else - an external url, a mailto - is skipped. A test may not reach the network, so an
external link is not this file's business. ``docs/_included/`` and ``docs/api/`` are skipped too:
``docs/conf.py`` writes both at build time, so they are absent from a checkout by design rather
than by accident.

**Anchors.** The README's table of contents is thirteen ``#anchor`` links into its own body. A
section can be renamed, merged or deleted without the entry above it moving, and the result is a
contents line that scrolls nowhere - the same invisibility as a dead path link, on the part of the
document a reader meets first. An anchor resolves when the document defines ``<a name="...">`` or
``id="..."`` with that value, or when a heading slugifies to it the way GitHub renders one.

**Dependencies.** The README listed ``yfinance`` and ``pandas-datareader`` under "Core
dependencies" while ``pyproject.toml`` had both in the ``[data]`` extra. A reader planning an
install therefore saw two packages that a core install does not pull, and the project's own rule
is that library code never imports either. The list is hand-written prose next to a machine-read
table, which is the same drift shape as a stale count, so it is checked rather than trusted.

**Code blocks.** The README's first example imported ``matplotlib``, ``seaborn``, ``yfinance`` and
``qis``, and then used ``PerfStat`` twice without importing it. A reader pasting the blocks in the
order they are written got ``NameError`` at the performance table, and the example script the
block was copied from carries the import the README lacked. The blocks are therefore read as one
script: every bare name loaded in a block must be a builtin, a name bound in a block above it, or
a name bound somewhere in the same block.

The resolution is static, and that is a deliberate departure from executing the blocks. They call
``yfinance``, so running them needs the network, which no test in this repository may reach.
Resolving names against an accumulated namespace catches the whole undefined-name class - the
defect above included - without leaving the process. It catches nothing past the name layer: a
keyword argument that does not exist or a shape mismatch is invisible to it, exactly as it is to
the static checks in ``test_examples.py``.
"""
# packages
import ast
import builtins
import re
from pathlib import Path
from typing import List, NamedTuple, Set
import pytest
# qis / project
import qis

REPO_ROOT: Path = Path(qis.__file__).resolve().parents[2]
IS_REPOSITORY_CHECKOUT: bool = REPO_ROOT.joinpath('pyproject.toml').is_file()

pytestmark = pytest.mark.skipif(
    not IS_REPOSITORY_CHECKOUT,
    reason='repository documentation is absent from an installed wheel')

# the markdown files a reader actually reads; generated output and vendored notebooks are not
DOCUMENT_GLOBS: List[str] = ['README.md', 'CONTRIBUTING.md', 'AGENTS.md', 'CHANGELOG.md',
                             'paper.md', 'docs/*.md', 'src/qis/docs/*.md']

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
    ``src/qis/docs/gallery.md`` linking to ``images/multi_asset.png`` means
    ``src/qis/docs/images/multi_asset.png``.

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


EXPLICIT_ANCHOR_PATTERN = re.compile(r'(?:name|id)\s*=\s*"([^"]+)"')
HEADING_PATTERN = re.compile(r'^#{1,6}\s+(.+?)\s*$', flags=re.M)
# GitHub's heading slug: drop the markup and everything that is not a word character, a space or a
# hyphen, lower-case the rest, then join on hyphens
SLUG_STRIP = re.compile(r'[^\w\s-]')


class Anchor(NamedTuple):
    """
    Attributes:
        document: the markdown file the link was found in, relative to the repository root
        target: the anchor name, without the leading '#'
        raw: the link as written, for the failure message
    """
    document: str
    target: str
    raw: str


def _slugify(heading: str) -> str:
    """the anchor GitHub generates for a heading."""
    text = re.sub(r'<[^>]+>', '', heading)          # an inline <a name="..."></a> is not title text
    text = SLUG_STRIP.sub('', text).strip().lower()
    return re.sub(r'[\s_]+', '-', text)


def _anchor_targets_in(text: str) -> set:
    """every anchor a document defines: explicit name/id attributes, and heading slugs."""
    targets = set(EXPLICIT_ANCHOR_PATTERN.findall(text))
    targets.update(_slugify(heading) for heading in HEADING_PATTERN.findall(text))
    return targets


def _anchors_in(path: Path) -> List[Anchor]:
    """
    the in-page anchor links one document contains.

    Args:
        path: the markdown file

    Returns:
        one entry per '#...' link, which should name a section of the same document
    """
    text = path.read_text(encoding='utf-8', errors='replace')
    document = path.relative_to(REPO_ROOT).as_posix()
    anchors: List[Anchor] = []
    for match in MARKDOWN_LINK_PATTERN.finditer(text):
        target = match.group(1)
        if not target.startswith('#') or len(target) == 1:
            continue
        anchors.append(Anchor(document=document, target=target[1:], raw=match.group(0)))
    return anchors


ALL_ANCHORS: List[Anchor] = [anchor for path in _documents() for anchor in _anchors_in(path)]
ANCHOR_IDS: List[str] = [f'{anchor.document}:#{anchor.target}' for anchor in ALL_ANCHORS]


def test_anchors_are_found() -> None:
    """the anchor pattern still matches something, so a green run is not an empty run."""
    assert len(ALL_ANCHORS) >= 10, (
        f"only {len(ALL_ANCHORS)} in-page anchor links found; the README table of contents alone "
        f"carries thirteen, so the pattern broke")


@pytest.mark.parametrize('anchor', ALL_ANCHORS, ids=ANCHOR_IDS)
def test_in_page_anchor_resolves(anchor: Anchor) -> None:
    """an anchor link names a section of the document it is written in."""
    text = REPO_ROOT.joinpath(anchor.document).read_text(encoding='utf-8', errors='replace')
    targets = _anchor_targets_in(text)
    assert anchor.target in targets, (
        f"{anchor.document} links to #{anchor.target}, which no heading or anchor in that "
        f"document defines. Written as {anchor.raw!r}")


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


README_PYTHON_BLOCK = re.compile(r'^```[ \t]*python[ \t]*\n(.*?)^```', flags=re.M | re.S)


class CodeBlock(NamedTuple):
    """
    Attributes:
        index: the block's position in the document, counting from zero
        line: the line the opening fence is written on, for the failure message
        source: the block's contents, without the fences
    """
    index: int
    line: int
    source: str


def _readme_python_blocks() -> List[CodeBlock]:
    """every ```python block in the README, in document order."""
    text = README_PATH.read_text(encoding='utf-8')
    blocks: List[CodeBlock] = []
    for index, match in enumerate(README_PYTHON_BLOCK.finditer(text)):
        blocks.append(CodeBlock(index=index,
                                line=text.count('\n', 0, match.start()) + 1,
                                source=match.group(1)))
    return blocks


def _bound_names(tree: ast.AST) -> Set[str]:
    """
    every name a block binds, wherever in the block it binds it.

    Scope is deliberately flattened: a name bound inside a function body counts as bound for the
    whole block. A README block is read rather than executed, and the alternative is a scope
    analyser catching a class of defect that has never appeared in this file.

    Args:
        tree: the parsed block

    Returns:
        assignment, loop, with, comprehension, except, import, def, class and argument names
    """
    names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname if alias.asname is not None
                          else alias.name.split('.')[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name is not None:
            names.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
    return names


def _loaded_names(tree: ast.AST) -> Set[str]:
    """
    every bare name a block reads.

    An attribute is not one: ``qis.plot_prices`` loads ``qis`` and nothing else, which is what
    keeps this from asserting that the whole public surface exists - ``test_examples.py`` does
    that, against the installed package rather than against a namespace built from prose.

    Args:
        tree: the parsed block

    Returns:
        the names in load context
    """
    return {node.id for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}


README_BLOCKS: List[CodeBlock] = _readme_python_blocks() if IS_REPOSITORY_CHECKOUT else []
BLOCK_IDS: List[str] = [f'README.md:{block.line}' for block in README_BLOCKS]


def test_readme_python_blocks_are_found() -> None:
    """the fence pattern still matches something, so a green run is not an empty run."""
    assert len(README_BLOCKS) >= 4, (
        f"only {len(README_BLOCKS)} ```python blocks found in README.md; the first example "
        f"section alone carries four, so the fence pattern broke")


@pytest.mark.parametrize('block', README_BLOCKS, ids=BLOCK_IDS)
def test_readme_block_resolves_its_names(block: CodeBlock) -> None:
    """
    a reader pasting the README's blocks in order never meets a name that is not there.

    The namespace a block is checked against is the builtins plus everything the blocks above it
    bind, which is the namespace that reader has. Names bound later in the same block count, so
    the check is about the document's order rather than the statement order inside one block.
    """
    available = set(dir(builtins))
    for earlier in README_BLOCKS[:block.index]:
        try:
            available |= _bound_names(ast.parse(earlier.source))
        except SyntaxError:
            continue  # that block fails its own case; it is not reported twice
    try:
        tree = ast.parse(block.source)
    except SyntaxError as error:
        raise AssertionError(
            f"README.md line {block.line} is fenced as python and does not parse: {error}. "
            f"Fence it as the language it is written in, or correct it") from error
    unresolved = sorted(_loaded_names(tree) - _bound_names(tree) - available)
    assert not unresolved, (
        f"README.md line {block.line} uses {unresolved}, which nothing above it binds and which "
        f"is not a builtin. A reader pasting the blocks in order gets NameError here")

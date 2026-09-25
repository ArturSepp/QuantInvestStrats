"""Execute the Python worked examples in the methodology articles.

Every ```` ```python ```` or ``~~~python`` block in a methodology page runs in one namespace per
page, in source order, so a later block may continue an earlier one. The assertions inside the
blocks are the numerical contract of each article: a calculation change that invalidates a
number quoted in the prose fails here rather than silently leaving the text wrong.

A block preceded by the line ``<!-- docs-test: skip -->`` is schematic and is not executed; the
comment is invisible in every rendered viewer. Network access is blocked while blocks run, so an
example cannot quietly depend on a data vendor. The repository root is placed on ``sys.path``
because some articles import their canonical example from ``examples/``.
"""

# packages
import re
import runpy
import socket
from pathlib import Path
from typing import List, NamedTuple

import matplotlib
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT: Path = Path(__file__).resolve().parents[3]
CHECKER_PATH: Path = REPO_ROOT.joinpath('tools', 'check_docs.py')
if not CHECKER_PATH.is_file():
    pytest.skip('Documentation sources are not shipped in the wheel.', allow_module_level=True)
CHECKER = runpy.run_path(str(CHECKER_PATH))
SKIP_MARKER = '<!-- docs-test: skip -->'
OPENING_FENCE = re.compile(r'^ {0,3}(`{3,}|~{3,})[ \t]*python[ \t]*$')


class CodeBlock(NamedTuple):
    """One executable example.

    Attributes:
        line: one-based source line of the opening fence
        code: the block's source, without its fences
        skipped: whether the block is marked schematic
    """
    line: int
    code: str
    skipped: bool


def python_blocks(text: str) -> List[CodeBlock]:
    """Extract top-level Python fences from a Markdown page.

    Args:
        text: complete Markdown source

    Returns:
        the Python blocks in source order; fences of other languages are ignored
    """
    lines = text.splitlines()
    blocks: List[CodeBlock] = []
    index = 0
    while index < len(lines):
        match = OPENING_FENCE.match(lines[index])
        if match is None:
            index += 1
            continue
        fence = match[1]
        previous = next((lines[i].strip() for i in range(index - 1, -1, -1)
                         if lines[i].strip()), '')
        closing = re.compile(rf'^ {{0,3}}{re.escape(fence[0])}{{{len(fence)},}}[ \t]*$')
        end = index + 1
        while end < len(lines) and closing.match(lines[end]) is None:
            end += 1
        blocks.append(CodeBlock(line=index + 1, code='\n'.join(lines[index + 1:end]),
                                skipped=previous == SKIP_MARKER))
        index = end + 1
    return blocks


PAGES: List[str] = sorted(CHECKER['METHODOLOGY_PAGES'])


def test_blocks_are_found() -> None:
    """The fence pattern still matches, so a green run is not an empty run."""
    count = sum(len(python_blocks(REPO_ROOT.joinpath('docs', page).read_text(encoding='utf-8')))
                for page in PAGES)
    assert count >= 15, f'only {count} Python blocks found across methodology pages'


def test_skip_marker_is_recognised() -> None:
    """A marked block is reported as skipped and an unmarked one is not."""
    source = f'{SKIP_MARKER}\n\n```python\nundefined_name\n```\n\n~~~python\nx = 1\n~~~\n'
    assert [block.skipped for block in python_blocks(source)] == [True, False]


@pytest.mark.parametrize('page', PAGES)
def test_methodology_examples_execute(page: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every unmarked block of one page runs, cumulatively and offline.

    Args:
        page: methodology page file name under ``docs/``
        monkeypatch: pytest fixture used for the path and the network guard
    """
    blocks = [block for block in python_blocks(
        REPO_ROOT.joinpath('docs', page).read_text(encoding='utf-8')) if not block.skipped]
    if len(blocks) == 0:
        pytest.skip(f'{page} has no executable Python block')

    def refuse(*args, **kwargs):
        raise OSError('documentation examples must run offline')

    monkeypatch.syspath_prepend(str(REPO_ROOT))
    monkeypatch.setattr(socket.socket, 'connect', refuse)
    monkeypatch.setattr(socket, 'create_connection', refuse)
    namespace = {'__name__': f'docs_{Path(page).stem}'}
    try:
        for block in blocks:
            try:
                exec(compile(block.code, f'docs/{page}:{block.line}', 'exec'), namespace)
            except Exception as error:
                raise AssertionError(
                    f'docs/{page} block at line {block.line} failed: {error!r}') from error
    finally:
        plt.close('all')

"""
qis docstrings are Google-style, and this is what keeps them that way.

The stack carries one deliberate exception to the house convention: ``factorlasso`` uses
numpydoc, because it is sklearn-compatible and its readers arrive from a numpydoc ecosystem.
That exception is per-package, and it works precisely because it is per-package - a reader is
inside one package at a time. Mixing the two *within* qis is the version that confuses, and it
had happened seven times before this check existed.

Both styles render, since ``napoleon`` reads either, so nothing fails loudly when a numpydoc
block appears. The section headings are the whole difference:

    Google                          numpydoc
    Args:                           Parameters
        name: description           ----------
                                    name : type
                                        description

This test looks for the underlined-heading form anywhere in a qis docstring and fails on it.
Scope is the whole package plus root-level ``examples/`` in a repository checkout, since the
examples are documentation too.
"""
# packages
import ast
import re
from pathlib import Path
from typing import List, Tuple
import pytest
# qis / project
import qis

PACKAGE_ROOT = Path(qis.__file__).parent
REPO_ROOT = Path(qis.__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT.joinpath('examples')
IS_REPOSITORY_CHECKOUT = REPO_ROOT.joinpath('pyproject.toml').is_file()

# a numpydoc section is a heading on its own line underlined by dashes. Matching the underline
# is what makes this specific: the bare word "Returns" is ordinary prose in a docstring.
NUMPYDOC_SECTION = re.compile(
    r'^[ \t]*(Parameters|Returns|Yields|Raises|Warns|Attributes|Other Parameters|See Also|'
    r'Notes|References|Examples)[ \t]*\n[ \t]*-{3,}[ \t]*$',
    flags=re.MULTILINE)


def _python_files() -> List[Path]:
    """every package module, plus repository examples when they are available."""
    files = list(PACKAGE_ROOT.rglob('*.py'))
    if IS_REPOSITORY_CHECKOUT:
        files.extend(EXAMPLES_ROOT.rglob('*.py'))
    return sorted(files)


FILES = _python_files()
FILE_IDS = [str(path.relative_to(REPO_ROOT)) for path in FILES]


def _numpydoc_docstrings(path: Path) -> List[Tuple[str, int, str]]:
    """
    find every docstring in a module that uses a numpydoc section heading.

    Args:
        path: module to scan

    Returns:
        the owner name, its line number and the section heading found, one entry per offender
    """
    try:
        tree = ast.parse(path.read_text(encoding='utf-8', errors='ignore'), filename=str(path))
    except SyntaxError:  # a module that does not parse is test_examples.py's problem, not this one
        return []
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        docstring = ast.get_docstring(node) or ''
        match = NUMPYDOC_SECTION.search(docstring)
        if match is not None:
            offenders.append((getattr(node, 'name', '<module>'),
                              getattr(node, 'lineno', 1),
                              match.group(1)))
    return offenders


@pytest.mark.parametrize('path', FILES, ids=FILE_IDS)
def test_module_has_no_numpydoc_sections(path: Path) -> None:
    """a qis docstring uses ``Args:`` and ``Returns:``, not underlined numpydoc headings."""
    offenders = _numpydoc_docstrings(path)
    assert not offenders, (
        f"{path.name} has numpydoc section headings: "
        + ', '.join(f"{name} (line {lineno}): '{section}'" for name, lineno, section in offenders)
        + ". qis is Google-style; see the docstring convention in AGENTS.md. factorlasso is the "
          "one package that keeps numpydoc")


def test_the_check_recognises_a_numpydoc_section() -> None:
    """
    the pattern matches the form it is meant to catch, and not ordinary prose.

    Without this the test above passes trivially if the regex is ever broken, and a check that
    cannot fail is worse than no check because it reads as a guarantee.
    """
    numpydoc = 'summary.\n\nParameters\n----------\nx : int\n    the argument\n'
    google = 'summary.\n\nArgs:\n    x: the argument\n\nReturns:\n    the result\n'
    prose = 'summary.\n\nReturns the result, which the Parameters section would describe.\n'
    assert NUMPYDOC_SECTION.search(numpydoc) is not None
    assert NUMPYDOC_SECTION.search(google) is None
    assert NUMPYDOC_SECTION.search(prose) is None

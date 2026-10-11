"""Execute prompt-style docstrings as part of the normal, offline pytest suite.

Discover source docstrings before importing modules so optional backends without examples
are not imported. Each example gets only its module globals, as in standard doctest; missing
input data must be supplied in the documentation itself. Installed wheels cover their shipped
modules, while checkouts also cover development runners and repository examples.
"""
import ast
import doctest
import importlib
import io
from pathlib import Path
import socket

import matplotlib.pyplot as plt
import pytest

import qis


PACKAGE_ROOT = Path(qis.__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent


def _example_cases():
    """Collect executable docstrings, including any added in future modules."""
    roots = [(PACKAGE_ROOT, 'qis')]
    if (REPO_ROOT / 'pyproject.toml').is_file():
        roots.append((REPO_ROOT / 'examples', 'examples'))
    parser = doctest.DocTestParser()
    for root, prefix in roots:
        for path in sorted(root.rglob('*.py')):
            relative = path.relative_to(root)
            if 'tests' in relative.parts:
                continue
            parts = list(relative.with_suffix('').parts)
            if parts[-1] == '__init__':
                parts.pop()
            module_name = '.'.join([prefix, *parts])
            tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                         ast.AsyncFunctionDef)):
                    continue
                doc = ast.get_docstring(node, clean=False) or ''
                if not parser.get_examples(doc):
                    continue
                name = getattr(node, 'name', '<module>')
                line = node.body[0].lineno - 1
                yield pytest.param(module_name, str(path), name, line, doc,
                                   id=f'{module_name}.{name}:{line + 1}')


@pytest.mark.parametrize('module_name,filename,name,line,doc', list(_example_cases()))
def test_docstring_examples_execute(module_name, filename, name, line, doc, monkeypatch):
    """Run every documented command with exact output checks and no network access."""
    def no_network(*args, **kwargs):
        raise AssertionError('Docstring examples must run without network access')

    monkeypatch.setattr(socket.socket, 'connect', no_network)
    monkeypatch.setattr(socket, 'create_connection', no_network)
    monkeypatch.setattr(plt, 'show', lambda: None)
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    existing_figures = set(plt.get_fignums())
    try:
        module = importlib.import_module(module_name)
        example = doctest.DocTestParser().get_doctest(
            doc, vars(module).copy(), f'{module_name}.{name}', filename, line)
        assert not any(command.options.get(doctest.SKIP) for command in example.examples), (
            'Provide runnable docstring examples instead of skipping them')
        output = io.StringIO()
        result = doctest.DocTestRunner().run(example, out=output.write)
        assert result.attempted == len(example.examples)
        assert result.failed == 0, output.getvalue()
    finally:
        for figure in set(plt.get_fignums()) - existing_figures:
            plt.close(figure)

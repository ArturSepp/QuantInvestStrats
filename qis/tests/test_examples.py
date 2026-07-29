"""
The examples are documentation, so they are tested like documentation.

``qis/examples/`` had three calls to symbols that were implemented, documented and not exported,
and every one of them raised ``AttributeError`` the moment anybody ran the file. Nothing caught
it, because nothing executes the examples: most of them pull prices from ``yfinance``, so they
cannot run on a core install or in CI.

This file closes that gap without needing the network. Four checks are static and run against
every example; the fifth executes the examples that need no data.

  1. the file parses,
  2. every ``qis.<name>`` reference resolves against the installed package,
  3. every ``from qis... import X`` names a module that exists on disk, and a symbol that
     exists in it when the module can be imported,
  4. every keyword argument passed to a ``qis`` callable exists in its signature,
  5. an example that needs no data actually runs, in a temporary directory.

Checks 2 and 4 are the two defect classes the plot smoke test found in exported API: a symbol
that is not there, and a keyword that is not there. Check 3 resolves module paths on the
filesystem rather than by importing, so an example whose module imports an optional extra is
still checked for existence.

An optional dependency is a skip, never a failure - the suite must pass on a core install.

What this does not cover, stated so the green tick is not read as more than it is:

  * most examples are never executed, only read. A runtime error past the import and
    signature layer - a shape mismatch, a bad column name - is not caught.
  * 88 of the 368 exported callables take ``**kwargs`` and accept any keyword, so check 4 skips
    them. ``plot_time_series_2ax`` is one of them, which is why the ``trend_line`` typo was
    silent; only the plot smoke test finds that class.
"""
# packages
import ast
import importlib
import inspect
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import pytest
# qis / project
import qis

EXAMPLES_DIR = Path(qis.__file__).parent.joinpath('examples')

# an example that mentions any of these reaches a data vendor and cannot run unattended
NETWORK_MARKERS = ('yfinance', 'yf.', 'bbg_fetch', 'pandas_datareader', 'blpapi')

# aliases an example may bind the top-level package to; only the plain one is checked, since
# the others alias submodules whose contents are internal and free to change
QIS_ALIAS = 'qis'

RUN_TIMEOUT_SECONDS = 300


def _example_files() -> List[Path]:
    """every example module, sorted, excluding __init__ and the shared helpers package."""
    return sorted(path for path in EXAMPLES_DIR.rglob('*.py') if path.name != '__init__.py')


def _is_offline(path: Path) -> bool:
    """True when the example reaches no data vendor and can be executed here."""
    source = path.read_text(encoding='utf-8', errors='ignore')
    return not any(marker in source for marker in NETWORK_MARKERS)


EXAMPLE_FILES = _example_files()
OFFLINE_EXAMPLE_FILES = [path for path in EXAMPLE_FILES if _is_offline(path)]
EXAMPLE_IDS = [str(path.relative_to(EXAMPLES_DIR)) for path in EXAMPLE_FILES]
OFFLINE_IDS = [str(path.relative_to(EXAMPLES_DIR)) for path in OFFLINE_EXAMPLE_FILES]


def _parse(path: Path) -> ast.Module:
    """parse an example, reporting the file rather than a bare SyntaxError."""
    return ast.parse(path.read_text(encoding='utf-8'), filename=str(path))


def _qis_attribute_references(tree: ast.Module) -> Set[str]:
    """collect the names in every ``qis.<name>`` expression."""
    names = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == QIS_ALIAS):
            names.add(node.attr)
    return names


def _qis_import_targets(tree: ast.Module) -> Set[Tuple[str, Optional[str]]]:
    """collect ``(module, symbol)`` for every import that reaches into qis."""
    targets = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module is not None and node.module.split('.')[0] == QIS_ALIAS:
                for alias in node.names:
                    targets.add((node.module, None if alias.name == '*' else alias.name))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split('.')[0] == QIS_ALIAS:
                    targets.add((alias.name, None))
    return targets


def _module_exists_on_disk(module: str) -> bool:
    """True when a dotted qis module path has a file or package directory behind it."""
    relative = Path(*module.split('.')[1:])  # drop the leading 'qis'
    package_root = Path(qis.__file__).parent
    return (package_root.joinpath(relative).with_suffix('.py').is_file()
            or package_root.joinpath(relative, '__init__.py').is_file())


def _import_or_none(module: str) -> Optional[object]:
    """import a module, returning None when an optional third-party dependency is missing."""
    try:
        return importlib.import_module(module)
    except ImportError:
        return None


@pytest.mark.parametrize('path', EXAMPLE_FILES, ids=EXAMPLE_IDS)
def test_example_parses(path: Path) -> None:
    """an example that does not parse is not documentation."""
    _parse(path)


@pytest.mark.parametrize('path', EXAMPLE_FILES, ids=EXAMPLE_IDS)
def test_example_qis_attributes_resolve(path: Path) -> None:
    """
    every ``qis.<name>`` in an example is exported or is a submodule.

    This is the check that would have caught ``qis.unsmooth_returns_glm``,
    ``qis.unsmooth_returns_ar1_ewma`` and ``qis.compute_ar1_unsmoothed_prices`` - all three
    implemented and documented, none exported, all three an ``AttributeError`` on the first line
    of the example that used them.
    """
    unresolved = []
    for name in sorted(_qis_attribute_references(_parse(path))):
        if hasattr(qis, name):
            continue
        if _module_exists_on_disk(f'qis.{name}'):
            continue
        unresolved.append(name)
    assert not unresolved, (f"{path.name} references qis.<name> that does not exist: "
                            f"{unresolved}. export it, or fix the example")


@pytest.mark.parametrize('path', EXAMPLE_FILES, ids=EXAMPLE_IDS)
def test_example_qis_imports_resolve(path: Path) -> None:
    """
    every qis module an example imports from exists, and the symbol exists in it.

    Module paths are resolved on the filesystem, so an example importing a module that needs an
    optional extra is still checked. The symbol check is skipped for those, since the module
    cannot be imported to look inside it.
    """
    missing_modules, missing_symbols = [], []
    targets = sorted(_qis_import_targets(_parse(path)),
                     key=lambda target: (target[0], target[1] or ''))
    for module, symbol in targets:
        if not _module_exists_on_disk(module):
            missing_modules.append(module)
            continue
        if symbol is None:
            continue
        imported = _import_or_none(module)
        if imported is None:
            continue  # an optional dependency, not a defect
        if not hasattr(imported, symbol):
            missing_symbols.append(f"{module}.{symbol}")
    assert not missing_modules, (f"{path.name} imports qis modules that do not exist: "
                                 f"{missing_modules}")
    assert not missing_symbols, f"{path.name} imports symbols that do not exist: {missing_symbols}"


@pytest.mark.parametrize('path', EXAMPLE_FILES, ids=EXAMPLE_IDS)
def test_example_keyword_arguments_exist(path: Path) -> None:
    """
    every keyword an example passes to a qis callable is in that callable's signature.

    ``plot_prices_2ax`` passed ``trend_line`` to a function taking ``trend_line1`` /
    ``trend_line2``; the argument was silently swallowed. Callables taking ``**kwargs`` accept
    anything and are skipped, so this checks the signatures where it can be checked.
    """
    unknown: List[str] = []
    for node in ast.walk(_parse(path)):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not (isinstance(function, ast.Attribute)
                and isinstance(function.value, ast.Name)
                and function.value.id == QIS_ALIAS):
            continue
        obj = getattr(qis, function.attr, None)
        if obj is None or not callable(obj):
            continue
        try:
            signature = inspect.signature(obj)
        except (ValueError, TypeError):
            continue  # a builtin or C-implemented callable with no introspectable signature
        if any(parameter.kind == parameter.VAR_KEYWORD
               for parameter in signature.parameters.values()):
            continue
        for keyword in node.keywords:
            if keyword.arg is not None and keyword.arg not in signature.parameters:
                unknown.append(f"line {node.lineno}: {function.attr}({keyword.arg}=...)")
    assert not unknown, f"{path.name} passes keywords that do not exist: {unknown}"


def _missing_optional_dependency(stderr: str) -> Optional[str]:
    """
    return the optional dependency an example could not import, or None.

    Two shapes appear: a bare ``ModuleNotFoundError`` from an ``import``, and the ``ImportError``
    a library raises when it needs an engine it cannot find - pandas says
    ``Unable to find a usable engine`` for parquet rather than naming a module. A missing qis
    module is a defect, not an extra, and is not matched here.
    """
    for name in re.findall(r"ModuleNotFoundError: No module named '([^']+)'", stderr):
        if name.split('.')[0] != QIS_ALIAS:
            return name
    import_errors = re.findall(r"^ImportError: (.+)$", stderr, flags=re.MULTILINE)
    if len(import_errors) > 0:
        return import_errors[-1]
    return None


@pytest.mark.parametrize('path', OFFLINE_EXAMPLE_FILES, ids=OFFLINE_IDS)
def test_offline_example_runs(path: Path,
                              tmp_path: Path,
                              ) -> None:
    """
    an example that needs no data vendor runs top to bottom.

    It runs in a temporary working directory, so an example that writes a figure or a PDF writes
    it there rather than into the repository. Rendering is headless.
    """
    environment: Dict[str, str] = dict(os.environ)
    environment['MPLBACKEND'] = 'Agg'
    completed = subprocess.run([sys.executable, str(path)],
                               cwd=str(tmp_path),
                               env=environment,
                               capture_output=True,
                               text=True,
                               timeout=RUN_TIMEOUT_SECONDS)
    if completed.returncode != 0:
        missing = _missing_optional_dependency(completed.stderr)
        if missing is not None:
            pytest.skip(f"{path.name} needs an optional dependency: {missing}")
        tail = '\n'.join(completed.stderr.strip().splitlines()[-15:])
        pytest.fail(f"{path.name} failed with exit code {completed.returncode}:\n{tail}")

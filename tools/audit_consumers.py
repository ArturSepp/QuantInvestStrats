"""
count how deeply the public consumers of qis use it, reproducibly.

The paper states that ``optimalportfolios`` and ``trendfollowing`` call a given number of qis
symbols at a given number of sites. A count with no script and no pinned revision is not
evidence: a reader cloning "current" consumer code a month later measures something else and
cannot tell whether the paper was wrong or the repository moved. This script is the counting
rule, and ``docs/audit/consumers.json`` is its output with the revision each count was taken at.

Counting rule, stated so that a different implementation can agree with it:

* Every ``.py`` file tracked by git in the consumer repository is parsed with ``ast``. Nothing is
  excluded - tests and examples are usage too, and any exclusion is a judgement a reader would
  have to reconstruct.
* A **call site** is one of two syntactic events: an attribute access ``<alias>.<name>`` where
  ``<alias>`` is a module alias bound by ``import qis`` or ``import qis as <alias>`` in that
  file, or one imported name in a ``from qis... import ...`` statement. A line calling two qis
  symbols is two sites; the same symbol on two lines is two sites.
* A **distinct symbol** is one ``<name>`` from either event, counted once per repository. A
  submodule path such as ``from qis.perfstats.perf import x`` contributes ``x``.
* Dynamic lookups (``getattr(qis, name)``) are not counted. There are none in these consumers,
  and a rule that cannot be checked syntactically cannot be reproduced.

One asymmetry follows from that rule and is stated rather than corrected: a file writing
``qis.TimePeriod`` ten times records ten sites, while a file importing ``TimePeriod`` once and
using it ten times records one. Counting the second form would mean tracking bound names through
scopes and would count a local variable that shadows an imported name, which is a worse error
than the asymmetry. The counts therefore measure references to the qis namespace, not uses of qis
symbols, and two consumers with different import styles are not comparable to each other.

Usage from the repository root::

    python tools/audit_consumers.py             # clone each consumer at its default branch tip
    python tools/audit_consumers.py --pinned    # clone at the commit already in the json

``--pinned`` is how a reader reproduces the published numbers; the unpinned form is how they are
refreshed before a release. Both need network access, so neither runs in the test suite: the
test checks the record's shape and that ``paper.md`` quotes it, not the counts themselves.
"""
# packages
import argparse
import ast
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
RECORD_PATH: Path = REPO_ROOT.joinpath('docs', 'audit', 'consumers.json')

# public consumers only. A private repository cannot be part of a reproducible count, however
# carefully it is measured, so private usage is described in the paper without a figure.
CONSUMER_URLS: Dict[str, str] = {
    'optimalportfolios': 'https://github.com/ArturSepp/OptimalPortfolios.git',
    'trendfollowing': 'https://github.com/ArturSepp/TrendFollowingSystems.git',
    'privateassets': 'https://github.com/ArturSepp/privateassets.git',
}

QIS_PACKAGE: str = 'qis'


@dataclass
class ConsumerUsage:
    """
    Attributes:
        commit: the revision the count was taken at
        distinct_symbols: qis names referenced at least once anywhere in the repository
        call_sites: attribute accesses plus imported names, summed over files
        files_scanned: python files parsed
        files_using_qis: python files with at least one call site
        symbols: the names, sorted, so that a reader can diff two runs
    """
    commit: str
    distinct_symbols: int
    call_sites: int
    files_scanned: int
    files_using_qis: int
    symbols: List[str]


def _run(arguments: List[str],
         cwd: Optional[Path] = None,
         ) -> str:
    """stdout of a subprocess, stripped, raising on a nonzero exit."""
    completed = subprocess.run(arguments,
                               cwd=None if cwd is None else str(cwd),
                               capture_output=True,
                               text=True,
                               check=True)
    return completed.stdout.strip()


def _qis_aliases(tree: ast.AST) -> Set[str]:
    """
    the module aliases bound to qis in one file.

    ``import qis as qis`` is the convention in this stack, but ``import qis`` and
    ``import qis as q`` both occur, and an alias that is not resolved is a call site missed.
    """
    aliases: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                if name.name == QIS_PACKAGE or name.name.startswith(f'{QIS_PACKAGE}.'):
                    aliases.add(name.asname if name.asname is not None else name.name.split('.')[0])
    return aliases


def count_file(source: str) -> Tuple[List[str], int]:
    """
    the qis names one file references, and how many times.

    Args:
        source: the file's text

    Returns:
        the referenced names with repetition removed, and the number of call sites

    Raises:
        SyntaxError: when the file does not parse under the running interpreter
    """
    tree = ast.parse(source)
    aliases = _qis_aliases(tree)
    names: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in aliases:
                names.append(node.attr)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            if module == QIS_PACKAGE or module.startswith(f'{QIS_PACKAGE}.'):
                names.extend(alias.name for alias in node.names)
    return sorted(set(names)), len(names)


def audit_checkout(checkout: Path) -> Tuple[Set[str], int, int, int]:
    """
    count qis usage across every tracked python file of a checkout.

    Args:
        checkout: the root of a git working tree

    Returns:
        the distinct symbols, the call sites, the files parsed, and the files using qis
    """
    tracked = _run(['git', 'ls-files', '*.py'], cwd=checkout).splitlines()
    symbols: Set[str] = set()
    call_sites = 0
    files_using = 0
    for relative in tracked:
        path = checkout.joinpath(relative)
        try:
            file_names, file_sites = count_file(path.read_text(encoding='utf-8', errors='replace'))
        except SyntaxError:
            print(f'warning: {relative} does not parse; skipped', file=sys.stderr)
            continue
        if file_sites > 0:
            files_using += 1
        symbols.update(file_names)
        call_sites += file_sites
    return symbols, call_sites, len(tracked), files_using


def audit_consumer(name: str,
                   url: str,
                   commit: Optional[str],
                   workspace: Path,
                   ) -> ConsumerUsage:
    """
    clone one consumer and count its qis usage.

    Args:
        name: the key used in the record
        url: the clone url
        commit: the revision to check out, or None for the default branch tip
        workspace: a directory the clone is made under

    Returns:
        the measurement for this consumer
    """
    checkout = workspace.joinpath(name)
    _run(['git', 'clone', '--quiet', url, str(checkout)])
    if commit is not None:
        _run(['git', 'checkout', '--quiet', commit], cwd=checkout)
    resolved = _run(['git', 'rev-parse', 'HEAD'], cwd=checkout)
    symbols, call_sites, files_scanned, files_using = audit_checkout(checkout)
    return ConsumerUsage(commit=resolved,
                         distinct_symbols=len(symbols),
                         call_sites=call_sites,
                         files_scanned=files_scanned,
                         files_using_qis=files_using,
                         symbols=sorted(symbols))


def main() -> int:
    """clone, count and write the record; returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pinned', action='store_true',
                        help='check out the commit already recorded rather than the branch tip')
    arguments = parser.parse_args()

    pinned: Dict[str, str] = {}
    if arguments.pinned:
        if not RECORD_PATH.is_file():
            print(f'{RECORD_PATH} does not exist, so there is nothing to pin to', file=sys.stderr)
            return 1
        existing = json.loads(RECORD_PATH.read_text(encoding='utf-8'))
        pinned = {name: entry['commit'] for name, entry in existing['consumers'].items()}

    workspace = Path(tempfile.mkdtemp(prefix='qis-consumer-audit-'))
    try:
        consumers: Dict[str, Dict[str, object]] = {}
        for name, url in CONSUMER_URLS.items():
            usage = audit_consumer(name=name,
                                   url=url,
                                   commit=pinned.get(name),
                                   workspace=workspace)
            consumers[name] = dict(url=url,
                                   commit=usage.commit,
                                   distinct_symbols=usage.distinct_symbols,
                                   call_sites=usage.call_sites,
                                   files_scanned=usage.files_scanned,
                                   files_using_qis=usage.files_using_qis,
                                   symbols=usage.symbols)
            print(f'{name}: {usage.distinct_symbols} symbols at {usage.call_sites} sites '
                  f'in {usage.files_using_qis} of {usage.files_scanned} files '
                  f'({usage.commit[:12]})')
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    record = dict(
        generated_by='tools/audit_consumers.py',
        note=('qis usage in its public consumers, at the pinned commits. Reproduce with '
              'python tools/audit_consumers.py --pinned. The counting rule is the module '
              'docstring of that script.'),
        consumers=consumers)
    RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECORD_PATH.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(f'wrote {RECORD_PATH}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

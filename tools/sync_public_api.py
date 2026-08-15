"""
regenerate ``PUBLIC_API`` in ``src/qis/api.py`` from the package namespace.

``src/qis/__init__.py`` assembles the namespace with wildcard re-exports from six subpackage
initialisers, so the public surface is a consequence of six other files rather than something
written down anywhere. ``PUBLIC_API`` writes it down. It does not decide what is public - the
namespace does - but it puts the surface in a diff, so adding or losing an export is a line a
reviewer can see rather than a number that quietly moves.

``src/qis/tests/test_core_api.py`` fails when the tuple and the namespace disagree, and names this
script in the failure message.

Run it from the repository root::

    python tools/sync_public_api.py            # rewrite the tuple in place
    python tools/sync_public_api.py --check    # exit 1 if it is out of date, write nothing
"""
# packages
import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
API_PATH: Path = REPO_ROOT.joinpath('src', 'qis', 'api.py')

TUPLE_PATTERN = re.compile(r'^PUBLIC_API: Tuple\[str, \.\.\.\] = \(\n.*?^\)\n',
                           flags=re.M | re.S)
LINE_WIDTH: int = 99


def exported_names() -> Tuple[str, ...]:
    """
    every public name in the qis namespace, sorted.

    ``qis.__all__`` rather than ``dir(qis)``: importing a submodule binds its name on the
    package, so ``dir(qis)`` grows as a session imports more of it, while ``__all__`` is fixed
    at the end of ``src/qis/__init__.py`` and is the same set every time.

    Returns:
        the names in ``qis.__all__``, sorted
    """
    import qis
    return tuple(sorted(qis.__all__))


def render(names: Tuple[str, ...]) -> str:
    """
    the ``PUBLIC_API`` assignment as source text.

    Args:
        names: the exported names, already sorted

    Returns:
        the assignment, ending in a newline, wrapped to the project's line length
    """
    lines: List[str] = []
    current = '   '
    for name in names:
        token = f" '{name}',"
        if len(current) + len(token) > LINE_WIDTH:
            lines.append(current)
            current = '   ' + token
        else:
            current += token
    lines.append(current)
    return 'PUBLIC_API: Tuple[str, ...] = (\n' + '\n'.join(lines) + '\n)\n'


def main() -> int:
    """rewrite or check the tuple; returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true',
                        help='exit 1 when the tuple is out of date instead of rewriting it')
    arguments = parser.parse_args()

    names = exported_names()
    rendered = render(names)
    source = API_PATH.read_text(encoding='utf-8')
    match = TUPLE_PATTERN.search(source)
    if match is None:
        print('PUBLIC_API assignment not found in src/qis/api.py', file=sys.stderr)
        return 1

    if match.group(0) == rendered:
        print(f'PUBLIC_API is current: {len(names)} names')
        return 0
    if arguments.check:
        print('PUBLIC_API is out of date; run python tools/sync_public_api.py', file=sys.stderr)
        return 1
    API_PATH.write_text(source.replace(match.group(0), rendered), encoding='utf-8')
    print(f'rewrote PUBLIC_API with {len(names)} names')
    return 0


if __name__ == '__main__':
    sys.exit(main())

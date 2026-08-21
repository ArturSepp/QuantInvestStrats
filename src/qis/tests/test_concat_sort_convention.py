"""
every axis=1 ``pd.concat`` in library code states ``sort=`` explicitly.

``pd.concat(objs, axis=1)`` joins the frames on their index, and whether the resulting union is
sorted has been changing under us. pandas 2.2 sorted the union of DatetimeIndexes whatever
``sort=`` said; pandas 3.0 honours an explicit ``sort=False`` and leaves the union in appearance
order, which is how a panel with an unsorted DatetimeIndex reached ``df_asfreq`` and raised
``ValueError: index must be monotonic increasing or decreasing`` from inside ``reindex``; pandas
3.0 still sorts when no ``sort=`` is passed, under a ``Pandas4Warning`` announcing that pandas 4
will not. A call that says nothing therefore means one thing today and another after the next
major release, in code where the difference is a scrambled time axis rather than an error.

So every such call states what it wants:

- ``sort=True`` where the joined index is a DatetimeIndex - prices, navs, weights, turnover,
  attribution. Chronological order is the meaning of the axis, and this is what pandas 2.2 did.
- ``sort=False`` where the joined index is a label index - tickers, groups, a melted RangeIndex.
  Row order there is the caller's, pandas has never sorted it, and sorting it alphabetically
  would reorder a table.

Only ``axis=1`` is covered. An ``axis=0`` concat joins on the columns, which in this package are
instrument or statistic labels rather than dates, and pandas 4 does not change their handling.

To confirm this check can fail, drop ``sort=True`` from any concat in ``portfolio_data.py``: the
call site is reported below by file, line and the object being concatenated. That was run before
this file was committed.
"""
# packages
import ast
from pathlib import Path
from typing import List, Tuple
# qis / project
import qis

PACKAGE_ROOT: Path = Path(qis.__file__).parent

# directories whose contents are scripts rather than library code: examples and development
# runners are read or invoked manually, while tests state their own frames
EXCLUDED_PARTS: Tuple[str, ...] = ('examples', 'tests', 'notebooks', 'run_local')


def _is_pd_concat(node: ast.Call) -> bool:
    """True for a ``pd.concat(...)`` call node"""
    func = node.func
    return (isinstance(func, ast.Attribute) and func.attr == 'concat'
            and isinstance(func.value, ast.Name) and func.value.id == 'pd')


def find_implicit_sort_sites() -> List[str]:
    """Return one line per axis=1 pd.concat call in library code that omits sort=."""
    offenders = []
    for path in sorted(PACKAGE_ROOT.rglob('*.py')):
        if any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        if path.name.endswith(('_test.py', '_tests.py')):
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_pd_concat(node):
                continue
            keywords = {kw.arg: kw for kw in node.keywords if kw.arg is not None}
            axis = keywords.get('axis')
            if axis is None or not isinstance(axis.value, ast.Constant):
                continue
            if axis.value.value not in (1, 'columns') or 'sort' in keywords:
                continue
            objs = ast.unparse(node.args[0]) if node.args else '<no positional objs>'
            rel = path.relative_to(PACKAGE_ROOT.parent).as_posix()
            offenders.append(f"{rel}:{node.lineno}: pd.concat({objs[:60]}, axis=1) omits sort=")
    return offenders


def test_axis1_concat_states_sort() -> None:
    """a concat that does not say whether it sorts means different things in pandas 3 and 4"""
    offenders = find_implicit_sort_sites()
    assert not offenders, (
            "axis=1 pd.concat without an explicit sort=; pass sort=True when the index is dates, "
            "sort=False when it is labels:\n" + '\n'.join(offenders))

"""
The documented core of the public API must stay documented.

``qis/api.py`` records which exported symbols are core: the ones a published package or qis's
own documentation calls. This file turns that record into a promise. A core symbol must resolve
from the top-level namespace and carry an ``Args`` or ``Attributes`` block, so that "the core
API is documented" is a claim a reader can check rather than an intention.

``PENDING_DOCSTRINGS`` is the backlog, and it is a ratchet in both directions: a core symbol
outside it must be documented, and a symbol inside it must still be undocumented. Writing a
docstring therefore fails the suite until the name is removed from the list, which is what keeps
the list from going stale and quietly hiding finished work.

Same shape as ``qis/plots/tests/plot_smoke_test.py`` and ``qis/tests/test_examples.py``: an
enforced invariant rather than a convention nobody runs.
"""
# packages
import inspect
import re
from typing import Any, Dict, FrozenSet, List
import pytest
# qis / project
import qis
from qis.api import CORE_API, PUBLIC_API, core_api_names

# core symbols still awaiting an Args/Attributes block. Empty: the documented core is complete,
# and it stays complete because a new core export without a docstring fails the suite below.
PENDING_DOCSTRINGS: FrozenSet[str] = frozenset()


# core symbols where an Args/Attributes block is the wrong form, with the reason. These are
# large compositional enums: the members are generated from a scheme, and a block restating
# every member would be longer than the enum and would tell a reader nothing the name does not.
PROSE_DOCUMENTED: Dict[str, str] = {
    'PerfStat': 'about 60 ColVar members; the docstring documents the ColVar scheme instead',
    'LegendStats': '29 compositional members; the docstring documents the naming vocabulary',
}


def _is_documented(obj: Any) -> bool:
    """
    True when the documentation carries an argument or attribute block.

    Google style puts constructor arguments on ``__init__`` rather than on the class, and
    ``inspect.getdoc`` on a class does not reach them, so both are checked. A numba ``@njit``
    function carries the docstring of its wrapped ``py_func``, which ``getdoc`` does find.
    """
    docs = [inspect.getdoc(obj) or '']
    if inspect.isclass(obj):
        docs.append(inspect.getdoc(getattr(obj, '__init__', None)) or '')
    if any(('Args:' in doc) or ('Attributes:' in doc) for doc in docs):
        return True
    # a callable taking no arguments has nothing to put in an Args block; what it returns is
    # the whole of its contract, so a Returns block is the documentation
    try:
        takes_arguments = len(inspect.signature(getattr(obj, 'py_func', obj)).parameters) > 0
    except (ValueError, TypeError):
        takes_arguments = True
    return not takes_arguments and any('Returns:' in doc for doc in docs)


def _documentable(name: str) -> bool:
    """
    True for a core symbol that can carry an Args/Attributes block at all.

    A numba ``@njit`` function is a ``CPUDispatcher`` rather than a function, and it carries the
    docstring of the wrapped ``py_func``; excluding it would silently drop the EWM kernels from
    the check. A module-level constant that happens to be a dataclass instance is documented on
    its class, not on the name, so it is excluded.
    """
    obj = getattr(qis, name)
    if inspect.isclass(obj) or inspect.isfunction(obj):
        return True
    return hasattr(obj, 'py_func')  # numba dispatcher


def test_core_api_is_exported() -> None:
    """every core name resolves from the top-level namespace."""
    missing = [name for name in core_api_names() if not hasattr(qis, name)]
    assert not missing, f"qis/api.py lists names that qis does not export: {missing}"


def test_core_api_has_no_duplicates() -> None:
    """a symbol belongs to one capability, so the reference cannot list it twice."""
    names = core_api_names()
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert not duplicates, f"listed under more than one capability: {duplicates}"


def test_pending_list_is_a_subset_of_the_core() -> None:
    """the backlog cannot name something that is not core."""
    stray = sorted(PENDING_DOCSTRINGS - set(core_api_names()))
    assert not stray, f"PENDING_DOCSTRINGS names non-core symbols: {stray}"


@pytest.mark.parametrize('name', sorted(set(core_api_names()) - PENDING_DOCSTRINGS
                                        - set(PROSE_DOCUMENTED)))
def test_core_symbol_is_documented(name: str) -> None:
    """a core symbol carries an Args or Attributes block."""
    if not _documentable(name):
        pytest.skip(f"{name} is a constant, not a callable")
    assert _is_documented(getattr(qis, name)), (
        f"qis.{name} is core but has no Args:/Attributes: block. Write it, or move the name "
        f"out of CORE_API if it is not core after all")


@pytest.mark.parametrize('name', sorted(PENDING_DOCSTRINGS))
def test_pending_symbol_is_still_undocumented(name: str) -> None:
    """
    the backlog must shrink when work lands.

    A name here that is now documented means the docstring was written and the list not updated.
    Remove it from PENDING_DOCSTRINGS; the check above then holds it documented forever.
    """
    if not _documentable(name):
        pytest.skip(f"{name} is a constant, not a callable")
    assert not _is_documented(getattr(qis, name)), (
        f"qis.{name} is documented now - remove it from PENDING_DOCSTRINGS")


def test_capability_groups_are_named() -> None:
    """every capability group has a title and at least one symbol."""
    empty = [title for title, names in CORE_API.items() if len(names) == 0]
    assert not empty, f"empty capability groups: {empty}"


@pytest.mark.parametrize('name', sorted(PROSE_DOCUMENTED))
def test_prose_documented_symbol_has_a_substantive_docstring(name: str) -> None:
    """
    the Args/Attributes exemption is not an exemption from documentation.

    These symbols are excused the block form, not the obligation. A one-line summary is not
    enough to describe an enum with dozens of members, so the docstring must actually explain
    the scheme.
    """
    assert name in set(core_api_names()), f"{name} is exempted but is not core"
    doc = inspect.getdoc(getattr(qis, name)) or ''
    assert len(doc.splitlines()) >= 6, (
        f"qis.{name} is exempted from the Args/Attributes rule because "
        f"{PROSE_DOCUMENTED[name]}, so its prose has to carry the description")


def _documented_argument_names(obj: Any) -> List[str]:
    """the argument names an Args: block declares, in order."""
    doc = inspect.getdoc(getattr(obj, 'py_func', obj)) or ''
    block = re.search(r'^Args:\n(.*?)(?=\n[A-Z][a-z]+:\n|\Z)', doc, flags=re.S | re.M)
    if block is None:
        return []
    return re.findall(r'^    (\w+):', block.group(1), flags=re.M)


@pytest.mark.parametrize('name', sorted(set(core_api_names()) - set(PROSE_DOCUMENTED)))
def test_documented_arguments_exist(name: str) -> None:
    """
    an Args: block cannot name an argument the function does not take.

    The mirror of the keyword check in test_examples.py. There it is a call site naming an
    argument that does not exist; here it is a docstring doing the same, which is worse,
    because a reader has no way to find out except by trying it.
    """
    if not _documentable(name):
        pytest.skip(f"{name} is a constant, not a callable")
    obj = getattr(qis, name)
    documented = _documented_argument_names(obj)
    if len(documented) == 0:
        pytest.skip(f"{name} has no Args: block")
    try:
        parameters = set(inspect.signature(getattr(obj, 'py_func', obj)).parameters)
    except (ValueError, TypeError):
        pytest.skip(f"{name} has no introspectable signature")
    stray = [argument for argument in documented if argument not in parameters]
    assert not stray, f"qis.{name} documents arguments it does not take: {stray}"


def test_public_api_matches_the_namespace() -> None:
    """
    the recorded public surface is the surface the package actually exports.

    ``PUBLIC_API`` is a literal so that adding or losing an export is a visible line in a diff
    rather than a count that moves. It is checked against ``qis.__all__`` and not against
    ``dir(qis)``: importing a submodule binds its name on the package, so ``dir(qis)`` is one
    name longer inside this file than it is in a fresh process, and a check against it would
    depend on which tests ran first.
    """
    recorded = set(PUBLIC_API)
    exported = set(qis.__all__)
    missing = sorted(exported - recorded)
    stale = sorted(recorded - exported)
    assert not missing and not stale, (
        f"qis exports {missing} which PUBLIC_API does not record, and PUBLIC_API records "
        f"{stale} which qis does not export. Run python tools/sync_public_api.py")


def test_public_api_has_no_duplicates() -> None:
    """the record is a set written as a tuple, so a name cannot appear twice."""
    duplicates = sorted({name for name in PUBLIC_API if PUBLIC_API.count(name) > 1})
    assert not duplicates, f"PUBLIC_API lists twice: {duplicates}"


def test_core_is_a_subset_of_public() -> None:
    """
    the documented core is part of the public surface, not beside it.

    A core symbol that is not exported would be documented and unreachable, which is the failure
    ``test_core_api_is_exported`` catches from the other direction; this states the containment
    between the two records rather than between a record and the namespace.
    """
    outside = sorted(set(core_api_names()) - set(PUBLIC_API))
    assert not outside, f"CORE_API names symbols outside PUBLIC_API: {outside}"

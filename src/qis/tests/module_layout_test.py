"""Enforce the boundary between automated tests and development runners."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

import qis


PACKAGE_ROOT = Path(qis.__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
IS_REPOSITORY_CHECKOUT = REPO_ROOT.joinpath("pyproject.toml").is_file()

requires_development_runners = pytest.mark.skipif(
    not IS_REPOSITORY_CHECKOUT,
    reason="development runners are intentionally absent from an installed wheel",
)


def _tree(path: Path) -> ast.Module:
    """Parse one Python module."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _is_test_shaped(path: Path) -> bool:
    """Return whether pytest imports the module by its filename."""
    return path.name.startswith("test_") or path.name.endswith("_test.py")


def _is_main_guard(node: ast.AST) -> bool:
    """Return whether a node is an ``if __name__ == '__main__'`` guard."""
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    comparison = node.test
    return (
        isinstance(comparison.left, ast.Name)
        and comparison.left.id == "__name__"
        and len(comparison.ops) == 1
        and isinstance(comparison.ops[0], ast.Eq)
        and len(comparison.comparators) == 1
        and isinstance(comparison.comparators[0], ast.Constant)
        and comparison.comparators[0].value == "__main__"
    )


def _has_pytest_candidate(tree: ast.Module) -> bool:
    """Return whether a module declares at least one pytest test candidate."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_"):
                return True
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            if any(
                isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and child.name.startswith("test_")
                for child in node.body
            ):
                return True
    return False


def _is_direct_run_local_call(node: ast.AST, optional_keywords: set[str] | None = None) -> bool:
    """Return whether a main-guard statement selects one ``Locals`` member."""
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    call = node.value
    if not isinstance(call.func, ast.Name) or call.func.id != "run_local":
        return False
    keywords = [keyword.arg for keyword in call.keywords]
    allowed = {"local"} | (optional_keywords or set())
    if call.args or keywords.count("local") != 1 or any(name not in allowed for name in keywords):
        return False
    value = next(keyword.value for keyword in call.keywords if keyword.arg == "local")
    return (
        isinstance(value, ast.Attribute)
        and isinstance(value.value, ast.Name)
        and value.value.id == "Locals"
    )


def test_pytest_modules_are_pure_automated_tests() -> None:
    """Every test-shaped module collects tests and contains no manual entry point."""
    failures: list[str] = []
    candidates = sorted(path for path in PACKAGE_ROOT.rglob("*.py") if _is_test_shaped(path))
    assert len(candidates) >= 48, "the automated suite unexpectedly disappeared"

    for path in candidates:
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        tree = _tree(path)
        definitions = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if not _has_pytest_candidate(tree):
            failures.append(f"{relative}: no pytest test candidate")
        if any(_is_main_guard(node) for node in tree.body):
            failures.append(f"{relative}: has a __main__ launcher")
        dispatcher_names = {"LocalTest", "LocalTests", "Locals", "run_local_test", "run_local"}
        if definitions.intersection(dispatcher_names):
            failures.append(f"{relative}: contains a development dispatcher")

    local_in_tests = sorted(
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path in PACKAGE_ROOT.rglob("*_local.py")
        if "tests" in path.parts
    )
    failures.extend(f"{path}: local diagnostic under tests" for path in local_in_tests)
    assert failures == [], "pytest/development boundary violations:\n" + "\n".join(failures)


@requires_development_runners
def test_source_adjacent_development_runner_layout() -> None:
    """Development runners use ``run_local/<subject>_run.py`` and the new API."""
    failures: list[str] = []
    runners = sorted(
        path
        for path in PACKAGE_ROOT.rglob("*.py")
        if "run_local" in path.parts and path.name != "__init__.py"
    )
    assert runners, "no source-adjacent development runners found"

    for path in runners:
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        tree = _tree(path)
        definitions = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if not path.name.endswith("_run.py"):
            failures.append(f"{relative}: runner name must end in _run.py")
        if "Locals" not in definitions or "run_local" not in definitions:
            failures.append(f"{relative}: expected Locals plus run_local")
        if definitions.intersection({"LocalTest", "LocalTests", "run_local_test"}):
            failures.append(f"{relative}: retains the old dispatcher API")
        if any(name.startswith("test_") for name in definitions):
            failures.append(f"{relative}: defines a pytest-shaped function")
        guards = [node for node in tree.body if _is_main_guard(node)]
        if len(guards) != 1 or not any(
            _is_direct_run_local_call(statement) for statement in guards[0].body
        ):
            failures.append(f"{relative}: main guard must select one Locals member")

    misplaced = sorted(
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path in PACKAGE_ROOT.rglob("*_run.py")
        if "run_local" not in path.parts
    )
    failures.extend(f"{path}: _run.py outside run_local" for path in misplaced)
    assert failures == [], "development-runner layout violations:\n" + "\n".join(failures)


@requires_development_runners
def test_repository_examples_use_current_plural_dispatcher_api() -> None:
    """Example dispatchers select ``Locals`` and may accept optional named inputs/outputs."""
    failures: list[str] = []
    examples_root = REPO_ROOT.joinpath("examples")

    for path in sorted(examples_root.rglob("*.py")):
        relative = path.relative_to(REPO_ROOT).as_posix()
        tree = _tree(path)
        classes = {
            node.name
            for node in tree.body
            if isinstance(node, ast.ClassDef)
        }
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if "LocalTests" in classes:
            failures.append(f"{relative}: retains the old LocalTests enum")
        if "Locals" not in classes:
            continue

        dispatcher = functions.get("run_local")
        optional_keywords: set[str] = set()
        if dispatcher is None:
            failures.append(f"{relative}: Locals enum requires run_local")
        else:
            args = dispatcher.args.args
            annotation = args[0].annotation if args else None
            if (
                not args
                or args[0].arg != "local"
                or not isinstance(annotation, ast.Name)
                or annotation.id != "Locals"
                or dispatcher.args.posonlyargs
                or len(args) - len(dispatcher.args.defaults) != 1
                or any(default is None for default in dispatcher.args.kw_defaults)
                or dispatcher.args.vararg is not None
                or dispatcher.args.kwarg is not None
            ):
                failures.append(f"{relative}: expected local: Locals plus optional named arguments")
            else:
                optional_keywords = {arg.arg for arg in args[1:] + dispatcher.args.kwonlyargs}

        guards = [node for node in tree.body if _is_main_guard(node)]
        if len(guards) != 1 or not any(
            _is_direct_run_local_call(statement, optional_keywords) for statement in guards[0].body
        ):
            failures.append(f"{relative}: main guard must select one Locals member")

    assert failures == [], "example dispatcher violations:\n" + "\n".join(failures)


@pytest.mark.parametrize("signature, call, valid", [
    ("local: Locals", "local=Locals.REPORT", True),
    ("local: Locals, output_path=None", "local=Locals.REPORT, output_path='report'", True),
    ("local: Locals, *, input_dir=None", "input_dir='data', local=Locals.REPORT", True),
    ("local: Locals, input_dir=None, output_path=None", "local=Locals.REPORT", True),
    ("local: Locals, output_path", "local=Locals.REPORT, output_path='report'", False),
    ("local: Locals, *, input_dir", "local=Locals.REPORT, input_dir='data'", False),
    ("local: Locals = None", "local=Locals.REPORT", False),
    ("local: str", "local=Locals.REPORT", False),
    ("local: Locals, *args", "local=Locals.REPORT", False),
    ("local: Locals, **kwargs", "local=Locals.REPORT", False),
    ("local: Locals", "local=Locals.REPORT, output_path='report'", False),
    ("local: Locals", "local='REPORT'", False),
    ("local: Locals", "Locals.REPORT", False),
    ("local: Locals", "**options", False),
])
def test_example_dispatcher_contract(tmp_path, monkeypatch, signature, call, valid) -> None:
    """Accept optional I/O while rejecting required extras and untyped case selection."""
    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "fixture.py").write_text(
        f"class Locals:\n    REPORT = 1\ndef run_local({signature}):\n    pass\n"
        f"if __name__ == '__main__':\n    run_local({call})\n", encoding="utf-8")
    monkeypatch.setitem(test_repository_examples_use_current_plural_dispatcher_api.__globals__,
                        "REPO_ROOT", tmp_path)
    if valid:
        test_repository_examples_use_current_plural_dispatcher_api()
    else:
        with pytest.raises(AssertionError, match="example dispatcher violations"):
            test_repository_examples_use_current_plural_dispatcher_api()


def test_component_dispatcher_rejects_example_options() -> None:
    """Optional I/O belongs to examples; component main guards retain their strict contract."""
    node = ast.parse("run_local(local=Locals.REPORT, output_path='report')").body[0]
    assert not _is_direct_run_local_call(node)


@requires_development_runners
def test_development_runners_import_on_a_core_install() -> None:
    """Every development runner imports without executing its selected case."""
    runners = sorted(
        path
        for path in PACKAGE_ROOT.rglob("*_run.py")
        if "run_local" in path.parts
    )
    for path in runners:
        module = ".".join(path.relative_to(PACKAGE_ROOT.parent).with_suffix("").parts)
        importlib.import_module(module)


def test_production_modules_do_not_retain_or_import_development_runners() -> None:
    """Production source has no dispatcher implementation or run-local dependency."""
    failures: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "tests" in path.parts or "run_local" in path.parts:
            continue
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        tree = _tree(path)
        definitions = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if definitions.intersection({"LocalTest", "LocalTests", "run_local_test"}):
            failures.append(f"{relative}: retains the old dispatcher API")
        for node in ast.walk(tree):
            module = node.module if isinstance(node, ast.ImportFrom) else None
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
            if (module and ".run_local" in module) or any(".run_local" in name for name in names):
                failures.append(f"{relative}: imports development-only run_local code")
                break
    assert failures == [], "production/development boundary violations:\n" + "\n".join(failures)

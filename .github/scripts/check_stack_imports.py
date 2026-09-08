"""Check stack dependencies and optional import boundaries without importing source."""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


def module_imports(tree):
    """Yield imports executed by a module body, including class/try/if bodies."""

    def walk(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return
        if isinstance(node, ast.If) and (
            isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
            or isinstance(node.test, ast.Attribute)
            and node.test.attr == "TYPE_CHECKING"
        ):
            for child in node.orelse:
                yield from walk(child)
            return
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        for child in ast.iter_child_nodes(node):
            yield from walk(child)

    yield from walk(tree)


def names(node):
    """Return absolute import names; relative imports are internal graph edges."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    return [node.module or ""] if not node.level else []


def permitted(path, module, rules, function=None):
    """Exceptions name both a source path and the optional module it may load."""
    return any(
        fnmatch.fnmatchcase(path, rule["path"])
        and module in rule["modules"]
        and ("function" not in rule or rule["function"] == function)
        for rule in rules
    )


def audit(root, policy):
    """Return concrete violations and the number of parsed package modules."""
    package_root = root / "src" / policy["import"]
    modules = {}
    for path in package_root.rglob("*.py"):
        relative = path.relative_to(root).as_posix()
        name = ".".join(path.relative_to(root / "src").with_suffix("").parts)
        if name.endswith(".__init__"):
            name = name[:-9]
        modules[name] = (
            relative,
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=relative),
        )
    errors, graph, isolated = [], {}, set()
    forbidden = set(policy["forbidden"])
    optional = set(policy["optional"])
    for module, (path, tree) in modules.items():
        parents = {
            child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                parent, function = node, None
                while parent in parents:
                    parent = parents[parent]
                    if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        function = parent.name
                        break
                for imported in names(node):
                    top = imported.split(".")[0]
                    if top in forbidden and not permitted(
                        path, top, policy["development_imports"], function
                    ):
                        errors.append(f"{path}:{node.lineno}: forbidden stack import {top}")
        top_level = list(module_imports(tree))
        edges = set()
        for node in top_level:
            for imported in names(node):
                top = imported.split(".")[0]
                if top in optional:
                    if permitted(path, top, policy["optional_adapters"]):
                        isolated.add(module)
                    else:
                        errors.append(
                            f"{path}:{node.lineno}: optional stack import {top} at module level"
                        )
            targets = []
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif not node.level:
                targets = [node.module or ""] + [
                    f"{node.module}.{alias.name}" for alias in node.names
                ]
            else:
                package = module if path.endswith("/__init__.py") else module.rpartition(".")[0]
                parts = package.split(".")
                base = ".".join(parts[: len(parts) - node.level + 1])
                target = ".".join(part for part in (base, node.module) if part)
                targets = [target] + [f"{target}.{alias.name}" for alias in node.names]
            for target in targets:
                pieces = target.split(".")
                edges.update(
                    ".".join(pieces[:i])
                    for i in range(1, len(pieces) + 1)
                    if ".".join(pieces[:i]) in modules
                )
        graph[module] = edges
    pending, visited = [policy["import"]], set()
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        pending.extend(graph.get(module, set()) - visited)
    for module in sorted(isolated & visited):
        errors.append(
            f"{modules[module][0]}: optional adapter is reachable during package-root import"
        )
    return sorted(set(errors)), len(modules)


def check_root(root, policy):
    """Test root import in a fresh process while recording blocked optional imports."""
    if not policy["optional"]:
        return
    probe = """import importlib.abc, json, sys
optional, source, package = json.loads(sys.argv[1])
attempts = []
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in optional:
            attempts.append(fullname)
            raise ModuleNotFoundError('optional stack import blocked: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
sys.path.insert(0, source)
__import__(package)
if attempts:
    raise SystemExit('package-root import attempted optional modules: ' + ', '.join(attempts))
"""
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            probe,
            json.dumps([policy["optional"], str(root / "src"), policy["import"]]),
        ],
        cwd=root,
        check=True,
    )


class BoundaryTests(unittest.TestCase):
    """Known-bad imports, allowed local loads, and adapter leakage regressions."""

    def check_source(self, source, adapter=None, exception=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "src/pkg"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text(source, encoding="utf-8")
            if adapter is not None:
                (package / "adapter.py").write_text(adapter, encoding="utf-8")
            policy = {
                "import": "pkg",
                "optional": ["optional_stack"],
                "forbidden": ["forbidden_stack"],
                "development_imports": [],
                "optional_adapters": [{"path": "src/pkg/adapter.py", "modules": ["optional_stack"]}]
                if exception
                else [],
            }
            return audit(root, policy)[0]

    def test_module_scope_is_rejected(self):
        self.assertTrue(self.check_source("import optional_stack\n"))

    def test_function_local_import_is_allowed(self):
        self.assertFalse(self.check_source("def load():\n    import optional_stack\n"))

    def test_guarded_module_import_is_still_rejected(self):
        self.assertTrue(
            self.check_source("try:\n    import optional_stack\nexcept ImportError:\n    pass\n")
        )

    def test_type_checking_only_import_is_allowed(self):
        self.assertFalse(
            self.check_source(
                "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import optional_stack\n"
            )
        )

    def test_isolated_named_adapter_is_allowed(self):
        self.assertFalse(self.check_source("", "import optional_stack\n", True))

    def test_root_reachable_adapter_is_rejected(self):
        self.assertTrue(
            self.check_source("from . import adapter\n", "import optional_stack\n", True)
        )

    def test_forbidden_import_is_not_hidden_by_local_scope(self):
        self.assertTrue(self.check_source("def load():\n    import forbidden_stack\n"))

    def test_maintainer_exception_requires_named_function_and_module(self):
        rules = [
            {"path": "src/pkg/universe.py", "function": "generate_data", "modules": ["bbg_fetch"]}
        ]
        self.assertTrue(permitted("src/pkg/universe.py", "bbg_fetch", rules, "generate_data"))
        self.assertFalse(permitted("src/pkg/universe.py", "bbg_fetch", rules, "other"))
        self.assertFalse(permitted("src/pkg/universe.py", "qis", rules, "generate_data"))


def main():
    """Run static policy, optional root-import probe, or the self-contained tests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--check-root", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(BoundaryTests)
        )
        return 0 if result.wasSuccessful() else 1
    policy = json.loads((args.root / ".github/stack-policy.json").read_text(encoding="utf-8"))
    errors, count = audit(args.root, policy)
    if errors:
        print("\n".join(errors))
        return 1
    if args.check_root:
        check_root(args.root, policy)
    print(f"Stack import policy passed ({count} modules)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

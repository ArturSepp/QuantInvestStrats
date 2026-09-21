"""Portable OSS checks, shared by GitHub Actions and the Desktop commit hook.

Canonical source: ArturSepp/scripts/repo_governance/oss_checks.py.
Copies are versioned in each package; the adoption audit detects drift.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath

import tomllib

VERSION = "1.0.0"
PROFILE = ".github/oss-checks.json"
GUIDE = "https://github.com/ArturSepp/ArturSepp/blob/main/docs/github_desktop.md"


class CheckFailure(Exception):
    """An actionable verification failure, not a Python traceback."""


def run(args, cwd, *, capture=False, env=None):
    """Run an argument vector without a shell."""
    result = subprocess.run(
        [str(a) for a in args], cwd=cwd, env=env, capture_output=capture, check=False
    )
    if result.returncode:
        details = result.stderr.decode("utf-8", "replace") if capture else ""
        raise CheckFailure(
            f"Command failed ({result.returncode}): {' '.join(map(str, args))}\n{details}"
        )
    return result.stdout if capture else b""


def git(root, *args):
    """Read Git state from the owning checkout."""
    return run(["git", "-c", "core.quotepath=false", *args], root, capture=True)


def entries(root, revision=None):
    """Return exact blob identities from the index or a committed tree."""
    if revision:
        raw = git(root, "ls-tree", "-rz", "--full-tree", revision)
    else:
        raw = git(root, "ls-files", "--stage", "-z")
    result = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, name = record.split(b"\t", 1)
        fields = metadata.decode("ascii").split()
        mode, oid = (fields[0], fields[2]) if revision else fields[:2]
        path = name.decode("utf-8")
        if not revision and fields[2] != "0":
            raise CheckFailure(f"Resolve the staged merge conflict first: {path}")
        safe_path(path)
        if mode not in {"100644", "100755"}:
            raise CheckFailure(f"Snapshot does not support mode {mode}: {path}")
        result.append((mode, oid, path))
    return result


def safe_path(name):
    """Reject paths that could write outside a source export."""
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise CheckFailure(f"Unsafe repository path: {name}")
    return path


def fingerprint(items):
    """Identify the selected files including modes and deletions."""
    return hashlib.sha256(json.dumps(items, ensure_ascii=False).encode()).hexdigest()


def export(root, items, destination):
    """Export all tracked blobs, including export-ignore verification inputs.

    This reads existing objects only. No temporary Git index, clone, or object store
    is created outside the original repository.
    """
    payload = "".join(f"{oid}\n" for _, oid, _ in items).encode("ascii")
    result = subprocess.run(
        ["git", "cat-file", "--batch"], input=payload, cwd=root, capture_output=True, check=False
    )
    if result.returncode:
        raise CheckFailure(result.stderr.decode("utf-8", "replace"))
    offset = 0
    for mode, oid, name in items:
        end = result.stdout.index(b"\n", offset)
        header = result.stdout[offset:end].decode().split()
        if len(header) != 3 or header[0] != oid or header[1] != "blob":
            raise CheckFailure(f"Expected a Git blob for {name}: {header}")
        length = int(header[2])
        start = end + 1
        target = destination.joinpath(*safe_path(name).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(result.stdout[start : start + length])
        if os.name != "nt" and mode == "100755":
            target.chmod(0o755)
        offset = start + length + 1


def diff(root, revision, base):
    """Return changed paths and added line ranges for the actual selected tree."""
    spec = [base, revision] if revision else ["--cached", base]
    names = git(root, "diff", "--name-only", "-z", *spec).decode().split("\0")
    patch = git(root, "diff", "--no-ext-diff", "--unified=0", *spec).decode("utf-8", "replace")
    return [name for name in names if name], added_lines(patch)


def added_lines(patch):
    """Parse zero-context hunks without treating unchanged legacy lines as new."""
    result = {}
    current = None
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            current = line[6:]
        match = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
        if match and current:
            start, count = int(match[1]), int(match[2] or "1")
            result.setdefault(current, set()).update(range(start, start + count))
    return result


def load_yaml(text):
    """Use YAML's BaseLoader so GitHub's 'on' key is not converted to True."""
    import yaml

    try:
        return yaml.load(text, Loader=yaml.BaseLoader)
    except yaml.YAMLError as error:
        raise ValueError(str(error)) from error


def source_checks(root, changed):
    """Check affected syntax and known documentation contracts without imports."""
    errors = []
    for name in changed:
        path = root / name
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in {".py", ".json", ".toml", ".yml", ".yaml", ".md", ".rst", ".cff"}:
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
            if re.search(r"^(?:<{7} |={7}$|>{7} )", text, re.MULTILINE):
                raise ValueError("unresolved merge-conflict markers")
            if suffix == ".py":
                ast.parse(text, filename=name)
            elif suffix == ".json":
                json.loads(text)
            elif suffix == ".toml":
                tomllib.loads(text)
            elif suffix in {".yml", ".yaml", ".cff"}:
                document = load_yaml(text)
                if name.startswith(".github/workflows/") and (
                    not isinstance(document, dict)
                    or not document.get("on")
                    or not document.get("jobs")
                ):
                    raise ValueError("workflow needs nonempty 'on' and 'jobs' mappings")
            elif suffix in {".md", ".rst"}:
                bad = re.findall(
                    r"https://github\.com/ArturSepp/ArturSepp/blob/main/docs/"
                    r"documentation_standard\.md#(?!user-content-)([\w-]+)",
                    text,
                )
                if bad:
                    raise ValueError(
                        f"shared-guide anchors need '#user-content-': {', '.join(bad)}"
                    )
        except (ValueError, SyntaxError, UnicodeError) as exc:
            errors.append(f"{name}: {exc}")
    if errors:
        raise CheckFailure("\n".join(errors))


def metadata_checks(root, changed):
    """Enforce the version contracts that must be changed together."""
    if not set(changed).intersection({"pyproject.toml", "CITATION.cff", "README.md"}):
        return
    if not (root / "pyproject.toml").exists():
        return
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    version = str(project["version"])
    if (root / "CITATION.cff").exists():
        citation = load_yaml((root / "CITATION.cff").read_text(encoding="utf-8"))
        if str(citation.get("version")) != version:
            raise CheckFailure(f"CITATION.cff version must match pyproject.toml ({version}).")
    if (root / "README.md").exists():
        readme = (root / "README.md").read_text(encoding="utf-8")
        for block in re.findall(r"@software\{.*?(?=\n\})", readme, re.DOTALL | re.IGNORECASE):
            match = re.search(r"\bversion\s*=\s*[\{\"]([^}\"]+)", block, re.IGNORECASE)
            if match and match[1] != version:
                raise CheckFailure(
                    f"README.md software citation version {match[1]} must be {version}."
                )


def lint(root, changed, lines, config):
    """Run the pinned Ruff, retaining each package's existing lint scope."""
    paths = [
        p
        for p in changed
        if p.endswith(".py")
        and (root / p).is_file()
        and any(fnmatch.fnmatch(p, pattern) for pattern in config["lint_paths"])
    ]
    if not paths:
        return
    command = [sys.executable, "-m", "ruff", "check", "--output-format", "json"]
    if config.get("lint_select"):
        command += ["--select", config["lint_select"]]
    result = subprocess.run(command + paths, cwd=root, capture_output=True, check=False)
    if result.returncode not in {0, 1}:
        raise CheckFailure(result.stderr.decode("utf-8", "replace"))
    findings = json.loads(result.stdout or b"[]")
    errors = []
    for finding in findings:
        relative = Path(finding["filename"]).relative_to(root).as_posix()
        row = finding["location"]["row"]
        if not config.get("lint_changed_lines") or row in lines.get(relative, set()):
            errors.append(f"{relative}:{row}: {finding['code']} {finding['message']}")
    if errors:
        raise CheckFailure("\n".join(errors))


def command_profile(root, config, phase, python, output):
    """Execute the same explicit documentation/test command vectors as CI."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src") + os.pathsep + str(root)
    environment["MPLBACKEND"] = "Agg"
    for step in config.get(phase, []):
        args = [arg.replace("{output}", str(output)) for arg in step]
        print(f"[{phase}] {' '.join(args)}", flush=True)
        run([python, *args], root, env=environment)


def validate_gate(needs, required, optional=()):
    """Fail closed if an expected job is absent, cancelled, failed or skipped."""
    errors = []
    for name in required:
        result = needs.get(name, {}).get("result", "missing")
        if result != "success":
            errors.append(f"{name}: {result}")
    for name in optional:
        result = needs.get(name, {}).get("result", "missing")
        if result not in {"success", "skipped"}:
            errors.append(f"{name}: {result}")
    if errors:
        raise CheckFailure("Required checks did not pass: " + "; ".join(errors))


def main():
    """Run a profile; hooks default to validating the exact staged index."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=["preflight", "docs", "ci", "doctor", "gate"])
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--revision")
    parser.add_argument("--base", default="HEAD")
    parser.add_argument("--working-tree", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    root = args.repo.resolve()
    if args.profile == "gate":
        config = json.loads((root / PROFILE).read_text())
        needs = json.loads(os.environ["OSS_NEEDS"])
        required = list(config["required_jobs"])
        optional = []
        if "audit" in needs:
            if needs.get("preflight", {}).get("outputs", {}).get("dependencies") == "true":
                required.append("audit")
            else:
                optional.append("audit")
        validate_gate(needs, required, optional)
        print("All required checks passed.")
        return
    if args.profile == "doctor":
        config = json.loads((root / PROFILE).read_text())
        print(f"OSS checks {VERSION}; package={config['repository']}; Python={args.python}")
        print(f"Hook path: {git(root, 'config', '--get', 'core.hooksPath').decode().strip()}")
        run([sys.executable, "-m", "ruff", "--version"], root)
        run([sys.executable, "-m", "uv", "--version"], root)
        print("Remote-only coverage: " + ", ".join(config["remote_only"]))
        print(f"Guide: {GUIDE}")
        return
    output = (
        args.output_dir
        or Path(os.environ.get("AGENT_LOCAL_ROOT", tempfile.gettempdir())) / "checks"
    )
    output = output.resolve()
    if output == root or root in output.parents:
        raise CheckFailure("Check output must be outside the source checkout.")
    output.mkdir(parents=True, exist_ok=True)
    revision = args.revision
    changed, lines = diff(root, revision, args.base)
    identities = entries(root, revision)
    digest = fingerprint(identities)
    with tempfile.TemporaryDirectory(prefix="source-", dir=output) as temporary:
        source = root if args.working_tree else Path(temporary)
        if not args.working_tree:
            export(root, identities, source)
        config = json.loads((source / PROFILE).read_text(encoding="utf-8"))
        if config["version"] != VERSION:
            raise CheckFailure(
                "Checker/profile versions differ; rerun the reviewed tooling update."
            )
        if args.profile in {"preflight", "ci"}:
            source_checks(source, changed)
            metadata_checks(source, changed)
            lint(source, changed, lines, config)
            if any(p.startswith(".github/workflows/") for p in changed):
                validator = os.environ.get("OSS_ACTIONLINT")
                if not validator:
                    raise CheckFailure(
                        "Actionlint is missing. Run Install-CommitHooks.ps1 -All -SetupTools."
                    )
                # Explicit files work on source exports without a .git directory.
                workflows = sorted((source / ".github/workflows").glob("*.y*ml"))
                run([validator, "-shellcheck=", "-pyflakes=", *workflows], source)
            if (
                set(changed).intersection({"pyproject.toml", "uv.lock"})
                and (source / "uv.lock").exists()
            ):
                run([sys.executable, "-m", "uv", "lock", "--check", "--offline"], source)
            for check in config.get("preflight", []):
                if any(fnmatch.fnmatch(p, pattern) for p in changed for pattern in check["paths"]):
                    command_profile(
                        source, {"check": [check["command"]]}, "check", args.python, output
                    )
        if args.profile in {"docs", "ci"}:
            command_profile(source, config, "docs", args.python, output)
        if args.profile == "ci":
            command_profile(source, config, "tests", args.python, output)
        if not args.working_tree and fingerprint(entries(root, revision)) != digest:
            raise CheckFailure(
                "Git selection changed during verification; commit again to check it."
            )
    if os.environ.get("GITHUB_OUTPUT"):
        dependency_change = bool(
            set(changed).intersection({"pyproject.toml", "uv.lock", ".github/workflows/audit.yml"})
        )
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as handle:
            handle.write(f"dependencies={str(dependency_change).lower()}\n")
    elapsed = time.monotonic() - started
    print(
        f"PASS {args.profile}: {len(changed)} changed paths, tree {digest[:12]}, {elapsed:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except (CheckFailure, OSError, KeyError, ValueError) as error:
        print(
            f"\nOSS check failed: {error}\nRepair the selected files and retry. Help: {GUIDE}",
            file=sys.stderr,
        )
        sys.exit(1)

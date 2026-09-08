"""Release-v1: validate one tag, inspect built artifacts and plan digest-safe uploads.

This helper never publishes packages or creates GitHub Release pages. GitHub builds
are isolated from the OIDC publishing job. All subprocess arguments are structured.
"""
from __future__ import annotations

import argparse
import datetime as dt
import email.parser
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tomllib
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

TAG_RE = re.compile(r"v(\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?)")


def version_from_tag(tag: str) -> str:
    """Reject refs, shell fragments, wildcard tags and ambiguous version syntax."""
    match = TAG_RE.fullmatch(tag)
    if not match:
        raise ValueError("Expected a named tag such as v1.2.3, v1.2.3rc1 or v1.2.3.dev1")
    return match[1]


def run(*args: str, cwd: Path | None = None) -> str:
    """Run an explicit command and return stdout with a bounded runtime."""
    return subprocess.run(list(args), cwd=cwd, check=True, text=True,
                          encoding="utf-8", capture_output=True, timeout=1800).stdout.strip()


def cff_scalar(source: str, key: str) -> str:
    """Read a simple scalar and reject duplicate/multiline ambiguous release values."""
    values = re.findall(rf"^{re.escape(key)}:\s*([^\n]+)$", source, flags=re.MULTILINE)
    if len(values) != 1:
        raise ValueError(f"CITATION.cff requires one {key}")
    value = values[0].strip().split(" #", 1)[0].strip()
    if value.startswith('"'):
        return json.loads(value)
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1].replace("''", "'")
    if not re.fullmatch(r"[0-9A-Za-z.+-]+", value):
        raise ValueError(f"Unsupported CFF {key} scalar")
    return value


def validate_metadata(root: Path, tag: str) -> dict:
    """Validate source release identity and the recorded intended release date."""
    version = version_from_tag(tag)
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    if project["version"] != version:
        raise ValueError(f"Tag {tag} differs from project version {project['version']}")
    cff = (root / "CITATION.cff").read_text(encoding="utf-8")
    if cff_scalar(cff, "version") != version:
        raise ValueError("CITATION.cff version differs from tag")
    intended = dt.date.fromisoformat(cff_scalar(cff, "date-released"))
    if intended > dt.datetime.now(dt.timezone.utc).date():
        raise ValueError("CITATION.cff intended release date is in the future")
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    pattern = rf"^##\s+(?:\[{re.escape(version)}\]|{re.escape(version)})(?:\s+.*)?$"
    if not re.search(pattern, changelog, flags=re.MULTILINE):
        raise ValueError("CHANGELOG.md has no heading for the tagged version")
    return project


def checkout_tag(root: Path, tag: str, expected_sha: str | None = None) -> dict:
    """Resolve and check ancestry before checking out exactly one release commit."""
    version_from_tag(tag)
    sha = run("git", "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}", cwd=root)
    if expected_sha is not None:
        if not re.fullmatch(r"[0-9a-f]{40}", expected_sha):
            raise ValueError("Invalid triggering object SHA")
        triggered = run("git", "rev-parse", "--verify", f"{expected_sha}^{{commit}}", cwd=root)
        if triggered != sha:
            raise ValueError("Tag moved after the triggering push; refusing a different commit")
    run("git", "merge-base", "--is-ancestor", sha, "refs/remotes/origin/main", cwd=root)
    run("git", "checkout", "--detach", sha, cwd=root)
    if run("git", "rev-parse", "HEAD", cwd=root) != sha:
        raise ValueError("Checkout does not match the resolved tag")
    project = validate_metadata(root, tag)
    epoch = run("git", "show", "-s", "--format=%ct", sha, cwd=root)
    return {"sha": sha, "version": project["version"], "epoch": epoch}


def normalized(name: str) -> str:
    """Normalize a PyPI distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_key(requirement: str) -> str:
    """Canonicalize names and specifier order for the stack's core requirements."""
    match = re.fullmatch(r"([A-Za-z0-9_.-]+)(\[[^]]+\])?([^;]*)(?:;(.*))?", requirement.strip())
    if not match:
        raise ValueError(f"Unsupported requirement metadata: {requirement}")
    specifiers = ",".join(sorted(x.strip() for x in match[3].split(",") if x.strip()))
    marker = re.sub(r"\s+", "", match[4] or "").replace("'", '"')
    return normalized(match[1]) + (match[2] or "") + specifiers + (";" + marker if marker else "")


def inspect_artifacts(dist: Path, project: dict, import_name: str) -> dict[str, str]:
    """Check one wheel plus one sdist and compute immutable filename/hash identity."""
    wheels, sdists = list(dist.glob("*.whl")), list(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("Expected exactly one wheel and one source distribution")
    with zipfile.ZipFile(wheels[0]) as wheel:
        metadata_names = [n for n in wheel.namelist() if n.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1 or f"{import_name}/__init__.py" not in wheel.namelist():
            raise ValueError("Wheel lacks the expected package or unique METADATA")
        wheel_metadata = wheel.read(metadata_names[0]).decode("utf-8")
        wheel_paths = set(wheel.namelist())
        wheel_metadata_root = metadata_names[0].rsplit("/", 1)[0]
    with tarfile.open(sdists[0], "r:gz") as archive:
        candidates = [m for m in archive.getmembers() if m.name.count("/") == 1 and m.name.endswith("/PKG-INFO")]
        if len(candidates) != 1:
            raise ValueError("Sdist lacks unique top-level PKG-INFO")
        sdist_metadata = archive.extractfile(candidates[0]).read().decode("utf-8")
        sdist_paths = set(archive.getnames())
        sdist_root = candidates[0].name.split("/", 1)[0]
    source_paths = {name.removeprefix(sdist_root + "/") for name in sdist_paths}
    if f"src/{import_name}/__init__.py" not in source_paths:
        raise ValueError("Sdist lacks the expected source package")
    prohibited = {".idea", ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
                  ".ruff_cache", ".mypy_cache", "run_local"}
    for archive_name, paths in (("Wheel", wheel_paths), ("Sdist", source_paths)):
        for name in paths:
            parts = PurePosixPath(name).parts
            runner = import_name in {"privateassets", "goal_based_allocation"} and import_name in parts and "run" in parts
            if prohibited.intersection(parts) or name.endswith((".pyc", ".pyo", ".nbc", ".nbi")) or runner:
                raise ValueError(f"{archive_name} contains a development runner, environment or cache: {name}")
    for metadata in (wheel_metadata, sdist_metadata):
        parsed = email.parser.Parser().parsestr(metadata)
        if normalized(parsed["Name"]) != normalized(project["name"]) or parsed["Version"] != project["version"]:
            raise ValueError("Built artifact identity differs from source/tag")
        if parsed["Summary"] != project["description"]:
            raise ValueError("Built artifact summary differs from source")
        urls = dict(value.split(", ", 1) for value in parsed.get_all("Project-URL", []))
        if urls != project.get("urls", {}):
            raise ValueError("Built artifact project URLs differ from source")
        if parsed["Requires-Python"] != project.get("requires-python"):
            raise ValueError("Built artifact Requires-Python differs from source")
        core = {requirement_key(value) for value in parsed.get_all("Requires-Dist", [])
                if not re.search(r"\bextra\s*==", value)}
        if core != {requirement_key(value) for value in project.get("dependencies", [])}:
            raise ValueError("Built artifact core dependency metadata differs from source")
        if set(parsed.get_all("Provides-Extra", [])) != set(project.get("optional-dependencies", {})):
            raise ValueError("Built artifact extra names differ from source")
        if isinstance(project.get("license"), str):
            if parsed["License-Expression"] != project["license"]:
                raise ValueError("Built artifact License-Expression differs from source")
            licenses = parsed.get_all("License-File", [])
            if not licenses:
                raise ValueError("Built artifact declares no shipped license file")
            for license_file in licenses:
                if f"{wheel_metadata_root}/licenses/{license_file}" not in wheel_paths or f"{sdist_root}/{license_file}" not in sdist_paths:
                    raise ValueError(f"Declared license file is missing from wheel or sdist: {license_file}")
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in wheels + sdists}


def pypi_files(name: str, version: str) -> list[dict] | None:
    """A confirmed 404 means new version; every other API error blocks publication."""
    request = urllib.request.Request(f"https://pypi.org/pypi/{name}/{version}/json",
                                     headers={"User-Agent": "ArturSepp-release-v1"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise
    if not isinstance(data, dict) or not isinstance(data.get("urls"), list) or not data["urls"]:
        raise ValueError("PyPI returned an unexpected or empty artifact list")
    return data["urls"]


def pending_uploads(hashes: dict[str, str], existing: list[dict] | None,
                    retry_existing: bool = False) -> list[str]:
    """Require matching digests; backfill tag pushes never append historical files."""
    if existing is None:
        return sorted(hashes)
    remote = {f["filename"]: f["digests"]["sha256"] for f in existing}
    if len(remote) != len(existing):
        raise ValueError("Duplicate PyPI artifact filenames")
    if not remote.keys() <= hashes.keys():
        raise ValueError("Existing release has unexpected artifact filenames")
    for name, digest in remote.items():
        if hashes[name] != digest:
            raise ValueError(f"Immutable PyPI artifact differs: {name}; never skip this mismatch")
    pending = sorted(hashes.keys() - remote.keys())
    if pending and not retry_existing:
        raise ValueError("Existing version is incomplete: use explicit retry_existing dispatch after digest review")
    return pending


def output(values: dict) -> None:
    """Emit simple validated GitHub outputs and a readable local result."""
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
            for key, value in values.items():
                if "\n" in str(value):
                    raise ValueError("Multiline output is not allowed")
                stream.write(f"{key}={value}\n")
    print(json.dumps(values, indent=2))


def main() -> None:
    """Run only the explicit stage selected by the workflow or local maintainer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["checkout", "check", "artifacts"])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--dist", type=Path)
    parser.add_argument("--import-name")
    parser.add_argument("--pending", type=Path)
    parser.add_argument("--retry-existing", action="store_true")
    args = parser.parse_args()
    if args.command == "checkout":
        if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch" and os.environ.get("GITHUB_REF") != "refs/heads/main":
            raise ValueError("Dispatch publishing from the main workflow only")
        expected = os.environ.get("GITHUB_SHA") if os.environ.get("GITHUB_EVENT_NAME") == "push" else None
        result = checkout_tag(args.root, args.tag, expected)
        if os.environ.get("GITHUB_ENV"):
            with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as stream:
                stream.write(f"SOURCE_DATE_EPOCH={result['epoch']}\n")
        output(result)
    elif args.command == "check":
        project = validate_metadata(args.root, args.tag)
        output({"name": project["name"], "version": project["version"]})
    else:
        if not args.dist or not args.pending or not args.import_name:
            parser.error("artifacts requires --dist, --pending, --import-name")
        project = validate_metadata(args.root, args.tag)
        hashes = inspect_artifacts(args.dist, project, args.import_name)
        pending = pending_uploads(hashes, pypi_files(project["name"], project["version"]), args.retry_existing)
        args.pending.mkdir(parents=True, exist_ok=False)
        for filename in pending:
            shutil.copy2(args.dist / filename, args.pending / filename)
        (args.dist / "release-manifest.json").write_text(json.dumps({"tag": args.tag, "sha": run("git", "rev-parse", "HEAD", cwd=args.root), "sha256": hashes, "pending": pending}, indent=2) + "\n", encoding="utf-8")
        output({"publish": "true" if pending else "false", "pending_count": len(pending)})


if __name__ == "__main__":
    main()

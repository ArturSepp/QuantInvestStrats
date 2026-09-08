"""Validate a prepared release; --push creates/pushes only its named tag.

Run after the release metadata commit is on origin/main. This does not build locally,
create an environment, upload a package, or create a GitHub Release page. The pushed
tag triggers release.yml. Omitting --push is a read-only dry run.
"""
import argparse
import subprocess
from pathlib import Path

from release_guard import run, validate_metadata


def main() -> None:
    """Require a clean main commit and matching remote identity before a named tag push."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    root = Path(run("git", "rev-parse", "--show-toplevel"))
    project = validate_metadata(root, args.tag)
    if run("git", "status", "--porcelain", cwd=root):
        raise SystemExit("Commit the intended changes first; the release checkout must be clean")
    if run("git", "branch", "--show-current", cwd=root) != "main":
        raise SystemExit("Run this helper from main after merging the release metadata")
    sha = run("git", "rev-parse", "HEAD", cwd=root)
    print(f"{project['name']} {project['version']}: {args.tag} -> {sha}")
    if not args.push:
        print("Dry run. --push verifies remote main, creates the named tag if absent, and pushes it.")
        return
    remote = run("git", "ls-remote", "origin", "refs/heads/main", cwd=root).split()
    if not remote or remote[0] != sha:
        raise SystemExit("Push the verified main commit before requesting its release tag")
    existing = subprocess.run(["git", "rev-parse", "--verify", f"refs/tags/{args.tag}^{{commit}}"],
                              cwd=root, capture_output=True, text=True)
    if existing.returncode == 0:
        if existing.stdout.strip() != sha:
            raise SystemExit("Existing tag points elsewhere; refusing to move it")
    else:
        run("git", "tag", "-a", args.tag, "-m", f"Release {project['version']}", cwd=root)
    run("git", "push", "origin", f"refs/tags/{args.tag}:refs/tags/{args.tag}", cwd=root)
    print("Named tag pushed. Follow the Publish package workflow; GitHub Release page is optional.")


if __name__ == "__main__":
    main()

"""Reject private package configuration in source and wheel builds.

Project metadata remains in pyproject.toml. The same stdlib-only content check
also runs on final archives in the publishing workflow, independently of Git's
index flags. No configuration values are logged or silently replaced.
"""

import runpy
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist


ROOT = Path(__file__).resolve().parent
validate_configuration = runpy.run_path(str(ROOT / ".github/scripts/release_guard.py"))[
    "validate_package_configuration"
]


def validate_tree(package: Path) -> None:
    """Check source and staged YAML, including files ignored by Git or left by a build."""
    validate_configuration([
        (path.relative_to(package).as_posix(), path.read_bytes())
        for path in sorted(package.rglob("*"))
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
    ])


class CheckedBuildPy(build_py):
    """Validate before copying source and again before the wheel is assembled."""

    def run(self):
        validate_tree(ROOT / "src/qis")
        super().run()
        if not self.editable_mode:
            validate_tree(Path(self.build_lib) / "qis")


class CheckedSdist(sdist):
    """Validate both the source and the release tree that becomes the sdist."""

    def run(self):
        validate_tree(ROOT / "src/qis")
        super().run()

    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        validate_tree(Path(base_dir) / "src/qis")


if __name__ == "__main__":
    setup(cmdclass={"build_py": CheckedBuildPy, "sdist": CheckedSdist})

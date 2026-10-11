"""Distribution checks must reject private settings without echoing their contents."""

import io
import runpy
import tarfile
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
GUARD = ROOT / ".github/scripts/release_guard.py"
if not GUARD.is_file():
    pytest.skip("Release tooling is only available in a source checkout", allow_module_level=True)

inspect_artifacts = runpy.run_path(str(GUARD))["inspect_artifacts"]
PROJECT = {
    "name": "qis",
    "version": "0.0.0",
    "description": "Release configuration test",
    "requires-python": ">=3.10",
}
METADATA = (
    "Metadata-Version: 2.4\nName: qis\nVersion: 0.0.0\n"
    "Summary: Release configuration test\nRequires-Python: >=3.10\n"
).encode()


def write_artifacts(dist, template, target=None, change=None):
    """Write small inert archives; all credential examples use the .invalid domain."""
    for kind in ("wheel", "sdist"):
        prefix = "qis/" if kind == "wheel" else "qis-0.0.0/src/qis/"
        files = [(prefix + "__init__.py", b""), (prefix + "settings.yaml", template)]
        if kind == target:
            if change == "database":
                files[1] = (
                    files[1][0],
                    template.replace(
                        b"postgresql://user:password@database:port",
                        b"postgresql://test:test-secret@db.example.invalid:5432/test",
                    ),
                )
            elif change == "path":
                files[1] = (files[1][0], template + b"\nPRIVATE_PATH: C:/Users/private/data\n")
            elif change == "missing":
                files.pop()
            elif change == "duplicate":
                files.append(files[1])
            else:
                files.append((prefix + change, b"password: test-secret\n"))
        if kind == "wheel":
            files.append(("qis-0.0.0.dist-info/METADATA", METADATA))
            with zipfile.ZipFile(dist / "qis-0.0.0-py3-none-any.whl", "w") as archive:
                for name, data in files:
                    archive.writestr(name, data)
        else:
            files.append(("qis-0.0.0/PKG-INFO", METADATA))
            with tarfile.open(dist / "qis-0.0.0.tar.gz", "w:gz") as archive:
                for name, data in files:
                    member = tarfile.TarInfo(name)
                    member.size = len(data)
                    archive.addfile(member, io.BytesIO(data))


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_placeholder_settings_pass(tmp_path, newline):
    """The reviewed template is portable between Linux and Windows builds."""
    template = (ROOT / "src/qis/settings.yaml").read_bytes().replace(b"\r\n", b"\n")
    write_artifacts(tmp_path, template.replace(b"\n", newline))
    assert len(inspect_artifacts(tmp_path, PROJECT, "qis")) == 2


@pytest.mark.parametrize("target", ["wheel", "sdist"])
@pytest.mark.parametrize(
    "change", ["database", "path", "missing", "duplicate", "private.yaml", "nested/private.yml"]
)
@pytest.mark.filterwarnings("ignore:Duplicate name:UserWarning")
def test_private_or_ambiguous_configuration_is_rejected(tmp_path, target, change):
    """Check each artifact independently, including ignored files and duplicate members."""
    template = (ROOT / "src/qis/settings.yaml").read_bytes()
    write_artifacts(tmp_path, template, target, change)
    with pytest.raises(ValueError, match="configuration") as error:
        inspect_artifacts(tmp_path, PROJECT, "qis")
    assert "test-secret" not in str(error.value)
    assert "Users/private" not in str(error.value)

"""Repository test collection must preserve documentation tooling imports."""

from pathlib import Path


def test_documentation_tooling_imports_after_collection():
    """The minimum pytest version must retain the real tooling namespace package."""
    from tools.docs_analytics import handbook

    root = Path(__file__).resolve().parents[2]
    assert Path(handbook.__file__).resolve() == root / "tools/docs_analytics/handbook.py"

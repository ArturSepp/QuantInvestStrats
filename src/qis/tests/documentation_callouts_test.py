"""Insight and Pitfall blockquotes render as admonitions; other quotes are untouched."""

from pathlib import Path
import re
import subprocess
import sys

import pytest


EXTENSION = Path(__file__).resolve().parents[3] / 'docs' / '_ext' / 'qis_callouts.py'
if not EXTENSION.is_file():
    pytest.skip('Documentation sources are not shipped in the wheel.', allow_module_level=True)


def test_labelled_blockquotes_become_admonitions(tmp_path: Path) -> None:
    """Build a small MyST site and inspect the rendered callouts."""
    pytest.importorskip('sphinx')
    pytest.importorskip('myst_parser')
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'conf.py').write_text(
        'import sys\n'
        f'sys.path.insert(0, {str(EXTENSION.parent)!r})\n'
        "extensions = ['myst_parser', 'qis_callouts']\n"
        "project = 'Callout test'\n", encoding='utf-8')
    (source / 'index.md').write_text(
        '# Callouts\n\n'
        '> **Insight.** Span sets the mean lag.\n\n'
        '> **Pitfall.** A span is not a window.\n\n'
        '> An ordinary quotation stays a quotation.\n', encoding='utf-8')
    output = tmp_path / 'html'
    result = subprocess.run(
        [sys.executable, '-m', 'sphinx', '-W', '-b', 'html', str(source), str(output)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    html = (output / 'index.html').read_text(encoding='utf-8')
    classes = set(re.findall(r'<div class="([^"]*admonition[^"]*)"', html))
    assert {frozenset(c.split()) for c in classes} == {
        frozenset({'tip', 'qis-insight', 'admonition'}),
        frozenset({'warning', 'qis-pitfall', 'admonition'})}
    assert 'admonition-title">Insight</p>' in html and 'admonition-title">Pitfall</p>' in html
    assert 'Span sets the mean lag.' in html and '<strong>Insight.</strong>' not in html
    assert html.count('<blockquote>') == 1

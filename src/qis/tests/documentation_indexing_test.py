"""Canonical URL and sitemap contracts, without importing optional documentation tools."""

from html.parser import HTMLParser
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest


EXTENSION = Path(__file__).resolve().parents[3] / 'docs' / '_ext' / 'qis_indexing.py'
if not EXTENSION.is_file():
    pytest.skip('Documentation sources are not shipped in the wheel.', allow_module_level=True)
SEO = runpy.run_path(str(EXTENSION))
BASE = 'https://quantinveststrats.readthedocs.io/en/latest/'


def make_app(tmp_path, *, base=BASE, builder='html'):
    """Make a small HTML-builder contract for dependency-free source tests."""
    return SimpleNamespace(
        config=SimpleNamespace(html_baseurl=base),
        builder=SimpleNamespace(name=builder, get_target_uri=lambda name: name + '.html'),
        env=SimpleNamespace(found_docs={
            'index', 'portfolio_breadth', 'api/generated/qis.RiskModel',
            'search', 'genindex', 'py-modindex', '_modules/qis/example',
        }),
        outdir=str(tmp_path),
    )


@pytest.mark.parametrize(('base', 'expected_base'), [
    (BASE, BASE.replace('/latest/', '/stable/')),
    (BASE.rstrip('/'), BASE.replace('/latest/', '/stable/')),
    (BASE.replace('/latest/', '/stable/'), BASE.replace('/latest/', '/stable/')),
    (BASE.replace('/latest/', '/5.30.0/'), BASE.replace('/latest/', '/5.30.0/')),
])
def test_homepage_canonical_and_sitemap_consolidate_moving_aliases(
        tmp_path, base, expected_base):
    """Use stable for moving aliases while preserving numbered release URLs."""
    app = make_app(tmp_path, base=base)
    context = {'pageurl': base.rstrip('/') + '/index.html'}
    SEO['set_canonical_url'](app, 'index', 'page.html', context, None)
    SEO['write_sitemap'](app, None)
    locations = ElementTree.parse(tmp_path / 'sitemap.xml').findall(
        './/{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
    assert context['pageurl'] == expected_base
    assert {node.text for node in locations} == {
        expected_base, expected_base + 'portfolio_breadth.html',
        expected_base + 'api/generated/qis.RiskModel.html',
    }


@pytest.mark.parametrize('builder,base,exception', [
    ('latex', BASE, None), ('linkcheck', BASE, None), ('html', '', None),
    ('html', BASE, RuntimeError('failed build')),
])
def test_no_sitemap_for_failed_or_non_html_builds(tmp_path, builder, base, exception):
    """Never create crawler metadata for failed, relative-only or non-HTML output."""
    SEO['write_sitemap'](make_app(tmp_path, builder=builder, base=base), exception)
    assert not (tmp_path / 'sitemap.xml').exists()


def test_sitemap_includes_unchanged_pages_on_incremental_builds(tmp_path):
    """Use all found documents, not only pages rendered in the current incremental build."""
    app = make_app(tmp_path)
    app.env.found_docs.add('unchanged')
    SEO['write_sitemap'](app, None)
    initial = (tmp_path / 'sitemap.xml').read_bytes()
    assert b'unchanged.html' in initial
    SEO['write_sitemap'](app, None)
    assert (tmp_path / 'sitemap.xml').read_bytes() == initial


def test_real_sphinx_theme_uses_the_same_urls_as_the_sitemap(tmp_path):
    """Exercise Sphinx events and rendered canonical tags when the docs extra is installed."""
    pytest.importorskip('sphinx')
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'conf.py').write_text(
        'import sys\n'
        f'sys.path.insert(0, {str(EXTENSION.parent)!r})\n'
        "extensions = ['qis_indexing']\n"
        f'html_baseurl = {BASE!r}\n'
        "project = 'Indexing test'\n", encoding='utf-8')
    (source / 'index.rst').write_text(
        'Home\n====\n\n.. toctree::\n\n   method\n', encoding='utf-8')
    (source / 'method.rst').write_text('Method\n======\n\nA distinct method.\n', encoding='utf-8')
    output = tmp_path / 'html'
    result = subprocess.run(
        [sys.executable, '-m', 'sphinx', '-W', '-b', 'html', str(source), str(output)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    class Canonicals(HTMLParser):
        """Collect the browser-visible canonical link declarations."""

        def __init__(self):
            super().__init__()
            self.urls = []

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == 'link' and attrs.get('rel') == 'canonical':
                self.urls.append(attrs['href'])

    stable_base = BASE.replace('/latest/', '/stable/')
    expected = {'index': stable_base, 'method': stable_base + 'method.html'}
    for name, url in expected.items():
        parser = Canonicals()
        parser.feed((output / f'{name}.html').read_text(encoding='utf-8'))
        assert parser.urls == [url]
    tree = ElementTree.parse(output / 'sitemap.xml')
    assert {node.text for node in tree.findall(
        './/{http://www.sitemaps.org/schemas/sitemap/0.9}loc')} == set(expected.values())

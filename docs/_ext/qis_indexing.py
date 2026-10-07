"""Keep canonical HTML URLs and the public page sitemap consistent.

Read the Docs serves the default version's sitemap at the domain root. Listing actual
pages gives crawlers one preferred discovery path instead of just latest/stable roots.
The moving ``latest`` and ``stable`` aliases describe the same default documentation,
so both use ``stable`` as their canonical URL, matching the site's default version.
Numbered releases keep their own
canonical base URL because their API documentation can differ.

The ``_included/`` pages mirror the notes shipped in ``src/qis/docs/``. Each repeats or points to
a handbook chapter, so they stay readable on the site but carry ``noindex`` and are left out of
the sitemap, leaving one indexed page per topic.
"""

from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin, urlsplit, urlunsplit
from xml.etree import ElementTree


SITEMAP_NAMESPACE = 'http://www.sitemaps.org/schemas/sitemap/0.9'
READTHEDOCS_ALIAS_PATHS = {'/en/latest', '/en/stable'}
NOINDEX_PREFIXES = ('_included/',)
ROBOTS_NOINDEX = '<meta name="robots" content="noindex, follow">\n'


def is_indexable(pagename: str) -> bool:
    """Return whether a page is offered to search engines.

    Args:
        pagename: Sphinx document name, without the output suffix.
    """
    return not pagename.startswith(NOINDEX_PREFIXES)


def canonical_baseurl(baseurl: str) -> str:
    """Consolidate Read the Docs' moving aliases onto the stable URL.

    Numbered release paths and non-Read-the-Docs deployments remain unchanged.
    """
    parts = urlsplit(baseurl)
    path = parts.path.rstrip('/')
    if parts.netloc.endswith('.readthedocs.io') and path in READTHEDOCS_ALIAS_PATHS:
        path = '/en/stable'
    return urlunsplit((parts.scheme, parts.netloc, path + '/', parts.query, parts.fragment))


def canonical_url(app: Any, pagename: str) -> str:
    """Return the same preferred URL for HTML metadata and sitemap entries.

    Args:
        app: Sphinx application with an HTML builder and configured html_baseurl.
        pagename: Sphinx document name, without the output suffix.

    Returns:
        Absolute URL, using the directory URL for the site's index.html alias.
    """
    uri = app.builder.get_target_uri(pagename)
    if uri == 'index.html':
        uri = ''
    return urljoin(canonical_baseurl(app.config.html_baseurl), quote(uri, safe='/%'))


def set_canonical_url(app: Any, pagename: str, templatename: str,
                      context: dict, doctree: Any) -> None:
    """Set the canonical consumed by the HTML theme, including the homepage.

    Args:
        app: Sphinx application.
        pagename: Current document name.
        templatename: HTML template selected by Sphinx; unchanged.
        context: Template variables; pageurl holds Sphinx's canonical URL.
        doctree: Parsed document, or None for generated helper pages.
    """
    if app.builder.name == 'html' and app.config.html_baseurl:
        context['pageurl'] = canonical_url(app, pagename)


def set_robots_meta(app: Any, pagename: str, templatename: str,
                    context: dict, doctree: Any) -> None:
    """Append a ``noindex, follow`` robots tag to pages kept out of the search index.

    Args:
        app: Sphinx application; unused.
        pagename: Current document name.
        templatename: HTML template selected by Sphinx; unchanged.
        context: Template variables; metatags already holds the page's MyST description.
        doctree: Parsed document, or None for generated helper pages.
    """
    if not is_indexable(pagename):
        context['metatags'] = (context.get('metatags') or '') + ROBOTS_NOINDEX


def write_sitemap(app: Any, exception: Exception | None) -> None:
    """Write a deterministic sitemap only after a successful HTML build.

    Args:
        app: Sphinx application with the complete discovered document inventory.
        exception: Build failure, if any; a failed build must not publish a sitemap.
    """
    if exception is not None or app.builder.name != 'html' or not app.config.html_baseurl:
        return
    namespace = SITEMAP_NAMESPACE
    root = ElementTree.Element(f'{{{namespace}}}urlset')
    pages = (name for name in app.env.found_docs
             if name not in {'search', 'genindex', 'py-modindex'}
             and not name.startswith('_modules/') and is_indexable(name))
    for url in sorted({canonical_url(app, name) for name in pages}):
        entry = ElementTree.SubElement(root, f'{{{namespace}}}url')
        ElementTree.SubElement(entry, f'{{{namespace}}}loc').text = url
    ElementTree.indent(root)
    ElementTree.ElementTree(root).write(
        Path(app.outdir) / 'sitemap.xml', encoding='utf-8', xml_declaration=True,
        default_namespace=namespace,
    )


def setup(app: Any) -> dict:
    """Register independent page metadata and end-of-build sitemap callbacks.

    Args:
        app: Sphinx application.

    Returns:
        Extension metadata; sitemap generation uses the complete merged inventory.
    """
    app.connect('html-page-context', set_canonical_url)
    app.connect('html-page-context', set_robots_meta)
    app.connect('build-finished', write_sitemap)
    return {'version': '1', 'parallel_read_safe': True, 'parallel_write_safe': True}

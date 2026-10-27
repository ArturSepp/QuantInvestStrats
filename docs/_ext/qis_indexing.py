"""Keep canonical HTML URLs and the public page sitemap consistent.

Read the Docs serves the default version's sitemap at the domain root. Listing actual
pages gives crawlers one preferred discovery path instead of just latest/stable roots.
``stable``, the site's default version, is the one offered to search engines. Every version
keeps its own canonical URL, because a page can exist in one version and not another.

``latest`` builds main ahead of the last release, so a chapter or exported symbol added since
the tag exists there and not yet on ``stable``. Pointing its canonical at ``stable`` sent
crawlers to a URL that answered 404. ``latest`` therefore carries ``noindex`` on every page and
publishes no sitemap. Its pages reach the index through ``stable`` once they are released.

The ``_included/`` pages mirror the notes shipped in ``src/qis/docs/``. Each repeats or points to
a handbook chapter, so they stay readable on the site but carry ``noindex`` and are left out of
the sitemap, leaving one indexed page per topic.
"""

from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote, urljoin, urlsplit, urlunsplit
from xml.etree import ElementTree


SITEMAP_NAMESPACE = 'http://www.sitemaps.org/schemas/sitemap/0.9'
READTHEDOCS_UNRELEASED_PATHS = {'/en/latest'}
NOINDEX_PREFIXES = ('_included/',)
ROBOTS_NOINDEX = '<meta name="robots" content="noindex, follow">\n'


def is_unreleased(baseurl: str) -> bool:
    """Return whether a build documents main ahead of the last release.

    Args:
        baseurl: The build's ``html_baseurl``.
    """
    parts = urlsplit(baseurl)
    return (parts.netloc.endswith('.readthedocs.io')
            and parts.path.rstrip('/') in READTHEDOCS_UNRELEASED_PATHS)


def is_indexable(baseurl: str, pagename: str) -> bool:
    """Return whether a page is offered to search engines.

    Args:
        baseurl: The build's ``html_baseurl``.
        pagename: Sphinx document name, without the output suffix.
    """
    return not is_unreleased(baseurl) and not pagename.startswith(NOINDEX_PREFIXES)


def canonical_baseurl(baseurl: str) -> str:
    """Return the build's own base URL with the trailing slash that ``urljoin`` needs.

    Args:
        baseurl: The build's ``html_baseurl``.
    """
    parts = urlsplit(baseurl)
    path = parts.path.rstrip('/')
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
        app: Sphinx application with a configured html_baseurl.
        pagename: Current document name.
        templatename: HTML template selected by Sphinx; unchanged.
        context: Template variables; metatags already holds the page's MyST description.
        doctree: Parsed document, or None for generated helper pages.
    """
    if not is_indexable(app.config.html_baseurl, pagename):
        context['metatags'] = (context.get('metatags') or '') + ROBOTS_NOINDEX


def write_sitemap(app: Any, exception: Exception | None) -> None:
    """Write a deterministic sitemap only after a successful HTML build of a released version.

    Args:
        app: Sphinx application with the complete discovered document inventory.
        exception: Build failure, if any; a failed build must not publish a sitemap.
    """
    baseurl = app.config.html_baseurl
    if (exception is not None or app.builder.name != 'html' or not baseurl
            or is_unreleased(baseurl)):
        return
    namespace = SITEMAP_NAMESPACE
    root = ElementTree.Element(f'{{{namespace}}}urlset')
    pages = (name for name in app.env.found_docs
             if name not in {'search', 'genindex', 'py-modindex'}
             and not name.startswith('_modules/') and is_indexable(baseurl, name))
    for url in sorted({canonical_url(app, name) for name in pages}):
        entry = ElementTree.SubElement(root, f'{{{namespace}}}url')
        ElementTree.SubElement(entry, f'{{{namespace}}}loc').text = url
    ElementTree.indent(root)
    ElementTree.ElementTree(root).write(
        Path(app.outdir) / 'sitemap.xml', encoding='utf-8', xml_declaration=True,
        default_namespace=namespace,
    )


PACKAGE = 'qis'


def public_api_names() -> list[str]:
    """Return verified exports from the same contract used by the API producer."""
    from importlib import import_module
    import json

    package = import_module(PACKAGE)
    if PACKAGE == 'optimalportfolios':
        inventory = json.loads(
            (Path(__file__).resolve().parents[2] / 'tools' / 'docs_inventory.json')
            .read_text(encoding='utf-8')
        )
        names = {name for group in inventory['symbols'].values() for name in group}
        names.update(inventory['external_symbols'])
    else:
        names = set(package.__all__)
    for name in names:
        if name.startswith('_') or not hasattr(package, name):
            raise ValueError(f'Invalid public API inventory entry: {PACKAGE}.{name}')
    return sorted(names)


def release_channel(baseurl: str) -> str:
    """Describe the configured build channel without inferring a PyPI version.

    Args:
        baseurl: The build's own configured URL.
    """
    channel = urlsplit(baseurl).path.rstrip('/').rsplit('/', 1)[-1]
    if channel == 'latest':
        return 'Development documentation; main may be ahead of PyPI'
    if channel == 'stable':
        return 'Released documentation; compare its build version with your installation'
    return 'Version-specific documentation; compare its build version with your installation'


def set_release_navigation(app, pagename, templatename, context, doctree) -> None:
    """Show the build channel and link to channel roots and generated navigation.

    Args:
        app: Running Sphinx application.
        pagename: Current document name.
        templatename: Current template; unchanged.
        context: HTML template variables, including the rendered body.
        doctree: Parsed document; unused.
    """
    from html import escape

    base = app.config.html_baseurl
    if app.builder.name != 'html' or not base or 'body' not in context:
        return
    root = urljoin(base.rstrip('/') + '/', '../')
    links = ''
    if urlsplit(base).netloc.endswith('.readthedocs.io'):
        links = (f'<a href="{escape(urljoin(root, "stable/"), quote=True)}">Stable</a> Ã‚Â· '
                 f'<a href="{escape(urljoin(root, "latest/"), quote=True)}">Latest</a> Ã‚Â· ')
    index = escape(urljoin(base.rstrip('/') + '/', 'llms.txt'), quote=True)
    version = escape(str(getattr(app.config, 'release', 'unspecified')))
    notice = ('<aside class="admonition note oss-release-navigation" '
              'aria-label="Documentation version">'
              f'<p>{escape(release_channel(base))}. Build version: {version}.</p>'
              f'<p>{links}<a href="{index}">Markdown / LLM index</a></p></aside>')
    context['body'] = notice + context['body']


def write_llms(app, exception) -> None:
    """Generate deterministic Markdown from discovered pages and verified API exports.

    Args:
        app: Successful HTML application with the merged Sphinx environment.
        exception: Build failure, if any; failed builds publish no index.
    """
    base = app.config.html_baseurl
    if (exception is not None or getattr(app, 'statuscode', 0)
            or app.builder.name != 'html' or not base):
        return
    base = base.rstrip('/') + '/'
    def page_url(name):
        uri = app.builder.get_target_uri(name)
        return urljoin(base, quote('' if uri == 'index.html' else uri, safe='/%'))

    pages = sorted(name for name in app.env.found_docs
                   if name not in {'search', 'genindex', 'py-modindex'}
                   and not name.startswith(('_', 'generated/', 'api/generated/')))
    titles = getattr(app.env, 'titles', {})
    lines = [f'# {PACKAGE} documentation', '', '> ' + release_channel(base) + '.', '',
             'Build version: ' + str(getattr(app.config, 'release', 'unspecified')) + '.', '',
             '## Canonical documentation', '']
    for name in pages:
        title = titles[name].astext() if name in titles else name
        title = title.replace('[', '').replace(']', '').replace('\n', ' ')
        lines.append(f'- [{title}]({page_url(name)})')
    lines += ['', f'- [Public API exports]({urljoin(base, "llms-api.txt")})', '']
    names = public_api_names()
    objects = {entry[0]: entry for entry in app.env.get_domain('py').get_objects()}
    api = [f'# {PACKAGE} public API exports', '', '> ' + release_channel(base) + '.', '',
           'Generated from the API producer contract; links use actual Sphinx targets.', '']
    for name in names:
        full = f'{PACKAGE}.{name}'
        entry = objects.get(full)
        if entry is not None:
            _, _, _, docname, anchor, _ = entry
            url = page_url(docname) + ('#' + quote(anchor, safe='.-_') if anchor else '')
            api.append(f'- [{full}]({url})')
        else:
            # Some exported constants have no object directive. Do not invent an anchor.
            api.append(f'- `{full}` (exported; no individual Sphinx target)')
    out = Path(app.outdir)
    (out / 'llms.txt').write_text('\n'.join(lines), encoding='utf-8')
    (out / 'llms-api.txt').write_text('\n'.join(api) + '\n', encoding='utf-8')


def read_redirect_manifest(path: Optional[Path] = None) -> dict:
    """Read the reviewed Read the Docs rules without changing hosting settings.

    Args:
        path: Optional manifest path; defaults to this repository's redirect inventory.

    Returns:
        Manifest containing unique page redirects and their replacement URLs.
    """
    import json

    path = path or Path(__file__).resolve().parents[2] / "tools" / "docs_redirects.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not manifest.get("redirects"):
        raise ValueError("Expected a nonempty version-1 redirect manifest")
    site = manifest.get("site", "")
    if site != "https://quantinveststrats.readthedocs.io":
        raise ValueError("Redirect inventory must belong to QuantInvestStrats")
    sources = set()
    for rule in manifest["redirects"]:
        source, target = rule["from_url"], urlsplit(rule["to_url"])
        if (not source.startswith("/") or source.startswith(("//", "/en/"))
                or not source.endswith(".html") or ".." in source.split("/")
                or any(char in source for char in "*?#") or source in sources):
            raise ValueError(f"Invalid or duplicate legacy page URL: {source}")
        if (target.scheme != "https" or not target.hostname or target.username
                or target.password or target.query or ".." in target.path.split("/")):
            raise ValueError(f"Invalid replacement URL: {rule['to_url']}")
        if (rule.get("redirect_type") != "page" or rule.get("http_status") != 301
                or rule.get("force") is not False or rule.get("enabled") is not True):
            raise ValueError("Legacy rules must be enabled, non-forced permanent page redirects")
        if rule["to_url"].split("#")[0] == site + "/en/stable" + source:
            raise ValueError(f"Self-redirect: {source}")
        sources.add(source)
    return manifest


def validate_redirect_targets(app: Any, exception: Optional[Exception]) -> None:
    """Fail a successful HTML build if a local redirect destination has disappeared.

    Args:
        app: Sphinx application with the finished HTML output.
        exception: Earlier build failure; failed and non-HTML builds are ignored.
    """
    from html.parser import HTMLParser

    if exception is not None or app.builder.name != "html":
        return
    path = getattr(getattr(app, "config", None), "qis_redirect_manifest", None)
    manifest = read_redirect_manifest(Path(path) if path else None)

    class Anchors(HTMLParser):
        def __init__(self):
            super().__init__()
            self.ids = set()

        def handle_starttag(self, tag, attrs):
            value = dict(attrs).get("id")
            if value:
                self.ids.add(value)

    for rule in manifest["redirects"]:
        url = urlsplit(rule["to_url"])
        if url.netloc != urlsplit(manifest["site"]).netloc:
            continue  # External destinations are checked by the shared live HTTP verifier.
        parts = url.path.strip("/").split("/", 2)
        if len(parts) != 3 or parts[0] != "en":
            raise ValueError(f"Unexpected local documentation path: {rule['to_url']}")
        target = Path(app.outdir) / parts[2]
        if not target.is_file():
            raise ValueError(f"Missing redirect destination: {rule['to_url']}")
        if url.fragment:
            anchors = Anchors()
            anchors.feed(target.read_text(encoding="utf-8"))
            if url.fragment not in anchors.ids:
                raise ValueError(f"Missing redirect destination anchor: {rule['to_url']}")


def setup(app: Any) -> dict:
    """Register independent page metadata and end-of-build sitemap callbacks.

    Args:
        app: Sphinx application.

    Returns:
        Extension metadata; sitemap generation uses the complete merged inventory.
    """
    app.add_config_value("qis_redirect_manifest", None, "env")
    app.connect('html-page-context', set_canonical_url)
    app.connect('html-page-context', set_robots_meta)
    app.connect('build-finished', write_sitemap)
    app.connect('html-page-context', set_release_navigation)
    app.connect('build-finished', write_llms)
    app.connect('build-finished', validate_redirect_targets)
    return {'version': '1', 'parallel_read_safe': True, 'parallel_write_safe': True}

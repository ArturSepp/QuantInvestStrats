"""
sphinx configuration for the qis documentation

Two things are generated at build time rather than checked in:

1. ``api/index.rst`` is written from the symbols exported by ``src/qis/__init__.py``. The public API
   is defined as the export list, so the reference follows the exports rather than the module
   tree, and moving a module between subpackages costs no documentation change.
2. ``_included/`` mirrors ``src/qis/docs/``. Those notes ship inside the package and are referenced
   from docstrings by their package path, so they stay where they are and are copied in here
   with their images, keeping one source of truth.
"""

# packages
import hashlib
import json
import tomllib
import inspect
import os
import shutil
import stat
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use('Agg')  # the build host has no display; must precede any qis import

DOCS_DIR = Path(__file__).parent
REPO_ROOT = DOCS_DIR.parent
sys.path.insert(0, str(REPO_ROOT.joinpath('src')))
sys.path.insert(0, str(DOCS_DIR.joinpath('_ext')))

project = 'qis'
author = 'Artur Sepp'
copyright = '2026, Artur Sepp'
release = tomllib.loads(
    (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
)["project"]["version"]
version = '.'.join(release.split('.')[:2])

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'myst_parser',
    'qis_indexing',
    'qis_callouts',
]

# the house convention is Google-style; factorlasso uses numpydoc and is documented separately
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_admonition_for_notes = True
# render an Attributes: block as :ivar: fields on the class rather than as separate attribute
# directives, which would otherwise duplicate the members autodoc already emits for a dataclass
napoleon_use_ivar = True

autosummary_generate = True
autodoc_typehints = 'description'  # signatures stay readable; types render in the body
autodoc_member_order = 'bysource'
autodoc_default_options = {'members': True, 'undoc-members': True, 'show-inheritance': True}

# ``linkify`` is excluded because it needs an extra dependency.
myst_enable_extensions = ['colon_fence', 'deflist', 'dollarmath']
myst_heading_anchors = 3

# Reviewed reference inventories keep strict HTML builds independent of external availability.
_INVENTORY_DIR = REPO_ROOT / '.github' / 'intersphinx'
_inventory_manifest = json.loads((_INVENTORY_DIR / 'manifest.json').read_text(encoding='utf-8'))
for _name, _record in _inventory_manifest['inventories'].items():
    _inventory = _INVENTORY_DIR / f'{_name}.inv'
    if hashlib.sha256(_inventory.read_bytes()).hexdigest() != _record['sha256']:
        raise RuntimeError(f'Reference inventory hash differs from its manifest: {_name}')

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', str(_INVENTORY_DIR / 'python.inv')),
    'numpy': ('https://numpy.org/doc/stable/', str(_INVENTORY_DIR / 'numpy.inv')),
    'pandas': ('https://pandas.pydata.org/docs/', str(_INVENTORY_DIR / 'pandas.inv')),
    'matplotlib': ('https://matplotlib.org/stable/', str(_INVENTORY_DIR / 'matplotlib.inv')),
}
# SciPy is a runtime dependency, but no public annotation links to its documentation. Omitting its
# unreliable inventory avoids network-only documentation failures without removing rendered links.

# DOI redirects commonly terminate at publisher sites that reject automated CI requests with
# HTTP 403 even though the canonical DOI remains valid. The Frongello paper is likewise retained
# as a reader-facing historical reference while its host has an expired TLS certificate. Source
# links back into this repository are validated locally and excluded from remote checks because
# GitHub throttles the many concurrent unauthenticated requests. Keep all of these links rendered,
# but do not make releases depend on those external server policies.
linkcheck_ignore = [
    r"https://doi\.org/.*",
    r"https://frongello\.com/support/Works/JPMSpring2002\.pdf",
    r"https://github\.com/ArturSepp/QuantInvestStrats/(?:blob|tree)/main/.*",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = {'.rst': 'restructuredtext', '.md': 'markdown'}

html_theme = 'furo'
html_title = 'qis - performance analytics, backtesting, and factsheet reporting'
html_baseurl = (os.environ.get("READTHEDOCS_CANONICAL_URL")
                or "https://quantinveststrats.readthedocs.io/en/latest/")
html_static_path = []

# The PDF prints the methodology chapters as one book. xelatex reads the Unicode minus signs,
# arrows and Greek letters of the prose directly, where pdflatex needs a mapping for each.
latex_engine = 'xelatex'
latex_documents = [('index', 'qis.tex', 'The qis analytics handbook', author, 'manual')]
latex_elements = {'papersize': 'a4paper'}

# suppress the warning autosummary emits for symbols that are re-exported under a short name
suppress_warnings = ['autosummary']

API_HEADER = """..
   Generated by docs/conf.py from the qis export list and src/qis/api.py. Do not edit by hand.

API reference
=============

Public means exported from ``src/qis/__init__.py``. This page is regenerated on every build, so it
cannot drift from the exports.

The reference is in two parts. The **core API** is the documented surface: every symbol there
carries an ``Args`` or ``Attributes`` block, and the boundary is drawn by measured usage - a
symbol is core when a package that depends on qis, or qis's own examples and documentation, call
it. ``src/qis/tests/test_core_api.py`` enforces that. **Also exported** lists everything else that
can be imported: usable, but with no prose promise and no stability guarantee beyond the
CHANGELOG.

Grouping in the core section is by capability rather than by defining module, so moving a symbol
between subpackages costs no documentation change.

"""

CORE_HEADING = """Core API
--------

"""

REST_HEADING = """Also exported
-------------

Exported and importable, not part of the documented core. Signatures may change without a
deprecation path.

"""

SECTION_ORDER = ['Enums', 'Dataclasses', 'Classes', 'Functions']

# the handbook chapters that derive the formulas behind each core capability; the API page links
# to them so that a reader of a signature is one click from its methodology
CAPABILITY_CHAPTERS: Dict[str, List[str]] = {
    'Instrument portfolio stress': ['portfolio_stress', 'stress_testing_with_options'],
    'Performance statistics': ['performance_statistics', 'performance_analytics_and_sharpe',
                               'drawdowns', 'benchmark_relative_performance', 'returns_and_navs',
                               'signal_diagnostics', 'turnover_conventions'],
    'Portfolio and backtesting': ['portfolio_backtesting', 'risk_contributions',
                                  'factor_risk_models', 'tracking_error_and_risk',
                                  'portfolio_breadth'],
    'Factor stress testing': ['stress_testing'],
    'Factsheets and reporting': ['factsheets_and_reporting', 'frequency_convention_note'],
    'EWM estimation': ['ewm_estimators', 'covariance_correlation_pca', 'risk_adjusted_returns'],
    'Market data and FX': ['fx_hedging_and_market_data'],
    'Regime reporting': ['regime_conditional_performance'],
    'Bootstrap': ['reproducibility'],
    'Unsmoothing': ['private_asset_unsmoothing', 'serial_dependence'],
    'Dates, schedules and annualisation': ['frequency_convention_note',
                                           'notation_and_conventions'],
}


def _classify(name: str,
              obj: Any,
              ) -> str:
    """
    Bucket an exported symbol for the reference page.

    Args:
        name: exported name
        obj: the exported object

    Returns:
        one of ``SECTION_ORDER``
    """
    import dataclasses
    from enum import Enum
    if inspect.isclass(obj):
        if issubclass(obj, Enum):
            return 'Enums'
        if dataclasses.is_dataclass(obj):
            return 'Dataclasses'
        return 'Classes'
    return 'Functions'


def _autosummary_block(names: List[str],
                       underline: str,
                       title: str,
                       count_label: str,
                       chapters: List[str] = (),
                       ) -> List[str]:
    """
    Render one titled autosummary block.

    Args:
        names: symbol names, without the ``qis.`` prefix
        underline: the character to underline the title with, which sets the heading level
        title: section heading
        count_label: the noun after the count, e.g. ``'documented'``
        chapters: handbook pages, without suffix, that explain the methodology of the block

    Returns:
        rst lines
    """
    # rst is whitespace-significant: the blank line after the option block is what makes the
    # entries content rather than options, and without it the toctree stays empty
    lines = [f"{title}\n{underline * len(title)}\n\n",
             f"{len(names)} {count_label}.\n\n"]
    if chapters:
        links = ', '.join(f':doc:`/{page}`' for page in chapters)
        lines.append(f"Methodology: {links}.\n\n")
    lines.append(".. autosummary::\n   :toctree: generated\n   :nosignatures:\n\n")
    lines.extend(f"   qis.{name}\n" for name in names)
    lines.append("\n")
    return lines


def _write_api_index() -> None:
    """Write ``api/index.rst`` from the current qis export list, split core against the rest."""
    import qis
    from qis.api import CORE_API, core_api_names

    documentable = []
    for name in sorted(dir(qis)):
        if name.startswith('_'):
            continue
        obj = getattr(qis, name)
        if inspect.ismodule(obj):
            continue
        if not (inspect.isclass(obj) or inspect.isfunction(obj) or inspect.isbuiltin(obj)):
            continue
        documentable.append((name, obj))

    core = set(core_api_names())
    lines = [API_HEADER, CORE_HEADING]
    for capability, names in CORE_API.items():
        listed = [name for name in names if any(name == n for n, _ in documentable)]
        if len(listed) == 0:
            continue
        lines.extend(_autosummary_block(names=listed, underline='~', title=capability,
                                        count_label='symbols',
                                        chapters=CAPABILITY_CHAPTERS.get(capability, [])))

    lines.append(REST_HEADING)
    sections: Dict[str, List[str]] = {key: [] for key in SECTION_ORDER}
    for name, obj in documentable:
        if name in core:
            continue
        sections[_classify(name=name, obj=obj)].append(name)
    for section in SECTION_ORDER:
        names = sections[section]
        if len(names) == 0:
            continue
        lines.extend(_autosummary_block(names=names, underline='~', title=section,
                                        count_label='exported'))

    api_dir = DOCS_DIR.joinpath('api')
    api_dir.mkdir(exist_ok=True)
    api_dir.joinpath('index.rst').write_text(''.join(lines), encoding='utf-8')


def _remove_readonly_tree_entry(
        function: Any,
        path: str,
        _exc_info: Any,
) -> None:
    """Retry removal of a generated documentation entry after making it writable.

    Args:
        function: removal function that failed inside ``shutil.rmtree``
        path: generated file or directory carrying a Windows read-only attribute
        _exc_info: exception tuple supplied by ``shutil.rmtree``; unused after the retry
    """
    os.chmod(path, os.stat(path).st_mode | stat.S_IWRITE)
    function(path)


def _mirror_package_notes() -> None:
    """Copy ``src/qis/docs/`` into ``docs/_included/`` for the Sphinx toctree."""
    source = REPO_ROOT.joinpath('src', 'qis', 'docs')
    target = DOCS_DIR.joinpath('_included')
    if target.exists():
        shutil.rmtree(target, onerror=_remove_readonly_tree_entry)
    if source.exists():
        shutil.copytree(source, target)


def setup(app: Any) -> None:
    """Generate the API index and mirror the package notes before the read phase."""
    _mirror_package_notes()
    _write_api_index()

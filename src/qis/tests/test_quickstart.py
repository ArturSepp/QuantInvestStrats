"""The documentation quickstart executes its authoritative offline example.

The executable script is the source of truth. ``docs/quickstart.md`` includes it with MyST
``literalinclude`` instead of maintaining a second code copy, while the README and docs landing
page point to the same path. The test resolves that inclusion, runs it, and checks its compact
evidence of success.
"""

# packages
import json
import re
import runpy
from pathlib import Path

import pytest

# qis / project
import qis


REPO_ROOT: Path = Path(qis.__file__).resolve().parents[2]
QUICKSTART_PATH: Path = REPO_ROOT.joinpath('docs', 'quickstart.md')
DOCS_INDEX_PATH: Path = REPO_ROOT.joinpath('docs', 'index.md')
README_PATH: Path = REPO_ROOT.joinpath('README.md')
EXAMPLE_PATH: Path = REPO_ROOT.joinpath(
    'examples', 'getting_started', 'offline_quickstart.py'
)
EXAMPLE_REPOSITORY_PATH = 'examples/getting_started/offline_quickstart.py'
NOTEBOOK_PATH: Path = REPO_ROOT.joinpath('notebooks', 'offline_quickstart_colab.ipynb')
COLAB_URL = (
    'https://colab.research.google.com/github/ArturSepp/QuantInvestStrats/'
    'blob/main/notebooks/offline_quickstart_colab.ipynb'
)
STABLE_QUICKSTART_URL = (
    'https://quantinveststrats.readthedocs.io/en/stable/quickstart.html'
)
CANONICAL_SOURCE_URL = (
    'https://github.com/ArturSepp/QuantInvestStrats/blob/main/'
    'examples/getting_started/offline_quickstart.py'
)

LITERALINCLUDE = re.compile(r'^```\{literalinclude\}\s+([^\n]+)$', flags=re.MULTILINE)


def _included_example() -> Path:
    """Resolve the quickstart's one literalinclude, or skip outside a checkout."""
    if not QUICKSTART_PATH.is_file():
        pytest.skip(f'{QUICKSTART_PATH} is absent; this test runs from a repository checkout')
    matches = LITERALINCLUDE.findall(QUICKSTART_PATH.read_text(encoding='utf-8'))
    assert len(matches) == 1, f'expected one authoritative literalinclude, got {matches}'
    return QUICKSTART_PATH.parent.joinpath(matches[0]).resolve()


def test_quickstart_references_one_authoritative_example() -> None:
    """The quickstart, landing page, and README all resolve to the same script."""
    included = _included_example()
    assert included == EXAMPLE_PATH.resolve()
    assert EXAMPLE_REPOSITORY_PATH in DOCS_INDEX_PATH.read_text(encoding='utf-8')
    assert EXAMPLE_REPOSITORY_PATH in README_PATH.read_text(encoding='utf-8')


def test_colab_notebook_is_a_clean_mirror_of_authoritative_example() -> None:
    """The hosted notebook adds setup only; its workflow cannot drift from D6."""
    if not README_PATH.is_file():
        pytest.skip('the repository-only Colab notebook is absent from installed wheels')
    if not NOTEBOOK_PATH.is_file():
        pytest.fail(f'approved Colab notebook is missing: {NOTEBOOK_PATH}')

    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding='utf-8'))
    assert notebook['nbformat'] == 4
    cells = notebook['cells']
    code_cells = [cell for cell in cells if cell['cell_type'] == 'code']
    workflow_cells = [
        cell for cell in code_cells if 'd6-source' in cell.get('metadata', {}).get('tags', [])
    ]
    assert len(workflow_cells) == 1
    assert ''.join(workflow_cells[0]['source']) == EXAMPLE_PATH.read_text(encoding='utf-8')

    setup_source = '\n'.join(''.join(cell['source']) for cell in code_cells[:-1])
    assert 'https://pypi.org/simple' in setup_source
    assert 'qis==' not in setup_source
    assert "version('qis')" in setup_source
    assert 'qis.__file__' in setup_source
    assert all(cell['execution_count'] is None and cell['outputs'] == [] for cell in code_cells)
    assert all(not cell.get('attachments') for cell in cells)

    markdown = '\n'.join(
        ''.join(cell['source']) for cell in cells if cell['cell_type'] == 'markdown'
    )
    assert STABLE_QUICKSTART_URL in markdown
    assert CANONICAL_SOURCE_URL in markdown
    assert COLAB_URL in QUICKSTART_PATH.read_text(encoding='utf-8')
    assert COLAB_URL in README_PATH.read_text(encoding='utf-8')


def test_quickstart_executes_and_reports_sane_results(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
) -> None:
    """Execute the included script and validate its deterministic evidence.

    Args:
        tmp_path: isolated working directory, proving that no repository output is required.
        monkeypatch: changes the working directory for the script.
        capsys: captures the compact output contract.
    """
    monkeypatch.chdir(tmp_path)
    namespace = runpy.run_path(str(_included_example()), run_name='__main__')

    portfolio_data = namespace['portfolio_data']
    nav = portfolio_data.get_portfolio_nav()
    assert bool(nav.notna().all()), 'quickstart nav contains nans'
    assert float(nav.iloc[-1]) == pytest.approx(120.1104, abs=0.00005)

    schedule = namespace['weight_schedule']
    prices = namespace['prices']
    assert schedule.shape == (33, 3)
    assert bool((schedule.where(prices.loc[schedule.index].notna(), 0.0) == schedule).all().all())
    assert bool(schedule.sum(axis=1).eq(1.0).all())

    output = capsys.readouterr().out
    assert 'Prices: business-day frequency, shape=(2087, 3)' in output
    assert 'Final NAV: 120.1104' in output
    assert 'TE=0.0299, IR=-0.2898' in output
    assert 'no file is written here' in output

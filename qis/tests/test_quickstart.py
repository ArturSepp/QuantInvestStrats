"""
the quickstart page runs, and its backtest produces a sane nav.

``docs/quickstart.md`` shipped two defects that every existing test missed: it passed
``rebalancing_costs=10`` where the argument is fractional — 10 runs clean and produces a nav
near -3e83 — and it called ``get_navs``, which does not exist. Doc snippets are documentation
of record on the hosted site, so this module executes every fenced python block exactly as a
reader would, top to bottom in one namespace.

The nav-bound assertion is the half that catches the units defect, which raises nothing.
"""
# packages
import re
from pathlib import Path
from typing import Dict, List
import matplotlib
import pytest
# qis / project
import qis

REPO_ROOT: Path = Path(qis.__file__).resolve().parent.parent
QUICKSTART_PATH: Path = REPO_ROOT.joinpath('docs', 'quickstart.md')

FENCED_PYTHON = re.compile(r'```python\n(.*?)```', flags=re.S)


def _python_blocks() -> List[str]:
    """the fenced python blocks of the quickstart page, or a skip outside a repository checkout."""
    if not QUICKSTART_PATH.is_file():
        pytest.skip(f'{QUICKSTART_PATH} is absent; this test runs from a repository checkout')
    return FENCED_PYTHON.findall(QUICKSTART_PATH.read_text(encoding='utf-8'))


def test_quickstart_executes_and_the_nav_is_sane(tmp_path, monkeypatch) -> None:
    """
    every fenced python block on the quickstart page executes in order in one namespace.

    Args:
        tmp_path: pytest working directory; the factsheet block writes a pdf into the cwd
        monkeypatch: used to chdir into tmp_path for the duration of the test
    """
    matplotlib.use('Agg', force=True)
    monkeypatch.chdir(tmp_path)
    blocks = _python_blocks()
    assert len(blocks) >= 3, f'expected at least 3 quickstart python blocks, got {len(blocks)}'
    namespace: Dict = {}
    for index, block in enumerate(blocks):
        try:
            exec(compile(block, filename=f'quickstart.md block {index}', mode='exec'), namespace)
        except Exception as exc:
            raise AssertionError(f'quickstart block {index} raised {exc!r}:\n{block}') from exc

    assert 'portfolio_data' in namespace, 'the backtest block should define portfolio_data'
    nav = namespace['portfolio_data'].get_portfolio_nav()
    assert bool(nav.notna().all()), 'quickstart nav contains nans'
    final_nav = float(nav.iloc[-1])
    assert 0.0 < final_nav < 1.0e6, (
        f'quickstart nav ends at {final_nav!r}; an absurd value here is the symptom of a wrong '
        f'rebalancing_costs unit — the argument is fractional, 0.0010 is 10 bp')

    import matplotlib.pyplot as plt
    plt.close('all')

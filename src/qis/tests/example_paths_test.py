"""Example entry points share the YAML resolver without hidden writes or CLI parsing."""

import ast
import importlib
from pathlib import Path

import matplotlib
import pytest
import yaml

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from qis import local_path

ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.skipif(
    not (ROOT / 'examples').is_dir(), reason='examples are not shipped in the wheel')


@pytest.fixture
def paths(tmp_path, monkeypatch):
    """Configure synthetic directories without touching the developer's settings."""
    output = tmp_path / 'reports'
    resources = tmp_path / 'resources'
    resources.mkdir()
    config = tmp_path / 'private.yaml'
    config.write_text(yaml.safe_dump({
        'OUTPUT_PATH': str(output), 'RESOURCE_PATH': str(resources)}), encoding='utf-8')
    monkeypatch.setenv('QIS_SETTINGS_PATH', str(config))
    monkeypatch.syspath_prepend(str(ROOT))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, 'show', lambda: None)
    yield output, resources
    local_path.get_paths.cache_clear()
    plt.close('all')


@pytest.mark.parametrize('name,branch,subdir', [
    ('brinson_attribution', 'REPORT', 'brinson_attribution'),
    ('account_equity_stress', 'REPORT', 'account_equity_stress'),
    ('composite_payoff_stress', 'REPORT', 'composite_payoff_stress'),
    ('factor_stress_testing', 'REPORT', 'factor_stress_testing'),
    ('instrument_portfolio_stress', 'REPORT', 'instrument_portfolio_stress'),
    ('model_layer_attribution_simulated', 'EXPORT', None),
])
@pytest.mark.parametrize('explicit', [False, True])
def test_export_dispatcher_paths(name, branch, subdir, explicit, paths, monkeypatch, tmp_path):
    """Check the real dispatcher with a stubbed expensive report producer."""
    module = importlib.import_module(f'examples.portfolios.{name}')
    monkeypatch.setattr(module, 'run_example', lambda **kwargs: kwargs)
    override = tmp_path / 'override' if explicit else None
    result = module.run_local(getattr(module.Locals, branch), output_path=override)
    expected = override if explicit else paths[0] / subdir if subdir else paths[0]
    assert result['output_dir'] == expected
    if explicit:
        assert not paths[0].exists()
    if name in ('account_equity_stress', 'composite_payoff_stress',
                'instrument_portfolio_stress'):
        assert not expected.exists(), 'fresh-path report writers must create their destination'


@pytest.mark.parametrize('name', [
    'brinson_attribution', 'account_equity_stress', 'composite_payoff_stress',
    'factor_stress_testing', 'instrument_portfolio_stress',
])
def test_check_branch_has_no_output(name, paths, monkeypatch):
    module = importlib.import_module(f'examples.portfolios.{name}')
    monkeypatch.setattr(module, 'run_example', lambda **kwargs: kwargs)
    assert module.run_local(module.Locals.CHECK)['output_dir'] is None
    assert not paths[0].exists()


def test_dendrogram_exports_to_yaml(paths):
    from examples.plots.cluster_dendrograms import Locals, run_local

    run_local(Locals.EXPORT)
    assert {file.name for file in paths[0].iterdir()} == {
        'standalone_dendrogram.png', 'cluster_dendrograms.png', 'cluster_membership.csv'}
    assert all(file.stat().st_size > 100 for file in paths[0].iterdir())


def test_dendrogram_exports_without_configuration(tmp_path, monkeypatch):
    """Exercise the real run_local workflow with a new user's portable defaults."""
    user_home = tmp_path / 'home'
    work = tmp_path / 'work'
    work.mkdir()
    monkeypatch.setenv('HOME', str(user_home))
    monkeypatch.setenv('USERPROFILE', str(user_home))
    monkeypatch.delenv('QIS_SETTINGS_PATH', raising=False)
    monkeypatch.syspath_prepend(str(ROOT))
    monkeypatch.chdir(work)
    local_path.get_paths.cache_clear()
    try:
        from examples.plots.cluster_dendrograms import Locals, run_local

        run_local(Locals.EXPORT)
        output = user_home / '.qis/outputs'
        assert {file.name for file in output.iterdir()} == {
            'standalone_dendrogram.png', 'cluster_dendrograms.png', 'cluster_membership.csv'}
        assert all(file.stat().st_size > 100 for file in output.iterdir())
        assert not (user_home / '.qis/resources').exists()
        assert list(work.iterdir()) == []
    finally:
        local_path.get_paths.cache_clear()
        plt.close('all')


@pytest.mark.parametrize('explicit', [False, True])
def test_fx_local_inputs_use_resource_path(explicit, paths, monkeypatch, tmp_path):
    from examples.market_data import fx_hedging_example as example

    source = tmp_path / 'override' if explicit else paths[1]
    source.mkdir(exist_ok=True)
    for key in ('fx_spots', 'domestic_rates', 'usd_assets'):
        (source / f'fx_hedging_data_{key}.csv').write_text('stub\n', encoding='utf-8')
    used = []

    def load_fx(local_path):
        used.append(local_path)
        return 'spots', 'rates'

    def load_assets(local_path):
        used.append(local_path)
        return 'assets'

    monkeypatch.setattr(example, 'load_fx_rates_data', load_fx)
    monkeypatch.setattr(example, 'load_usd_assets', load_assets)
    actual = example.load_example_inputs(source if explicit else None, use_local_data=True)
    assert actual == ('spots', 'rates', 'assets')
    assert used == [str(source), str(source)]


def test_fresh_report_protection(paths):
    from examples.portfolios.composite_payoff_stress import run_example

    paths[0].mkdir()
    marker = paths[0] / 'existing.txt'
    marker.write_text('keep', encoding='utf-8')
    with pytest.raises(FileExistsError, match='fresh'):
        run_example(paths[0])
    assert marker.read_text(encoding='utf-8') == 'keep'


def test_no_duplicate_directory_helpers_or_cwd_exports():
    assert not (ROOT / 'examples/_helpers/output.py').exists()
    failures = []
    for source in (ROOT / 'examples').rglob('*.py'):
        tree = ast.parse(source.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(alias.name in ('argparse', 'tempfile') for alias in node.names):
                    failures.append(f'{source.name}: duplicate directory/CLI infrastructure')
            if isinstance(node, ast.ImportFrom) and node.module == 'examples._helpers.output':
                failures.append(f'{source.name}: removed helper import')
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ('save_fig', 'save_figs_to_pdf', 'save_df_to_excel'):
                    keywords = {kw.arg: kw.value for kw in node.keywords}
                    value = keywords.get('local_path')
                    if value is None or isinstance(value, ast.Constant) and value.value is None:
                        failures.append(f'{source.name}:{node.lineno}: implicit output path')
    assert not failures, '\n'.join(failures)

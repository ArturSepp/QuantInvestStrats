"""CI partition contracts keep numerical tests in mixed reporting modules on every platform."""

import ast
import importlib.util
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


PLUGIN = Path(__file__).resolve().parents[3] / '.github' / 'ci_test_suites.py'
if not PLUGIN.exists():
    pytest.skip('CI routing is repository tooling, not part of the wheel', allow_module_level=True)
spec = importlib.util.spec_from_file_location('ci_test_suites_contract', PLUGIN)
routing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(routing)


@pytest.mark.parametrize('nodeid', [
    'portfolio/reports/tests/factsheet_facade_returns_test.py::'
    'test_daily_returns_compound_from_a_base_observation',
    'portfolio/reports/tests/asset_display_test.py::'
    'test_additive_series_conserve_each_date_and_keep_true_total',
    'portfolio/reports/tests/strategy_factsheet_test.py::'
    'test_brinson_assigns_interaction_to_selection_and_preserves_active_return',
    'portfolio/reports/tests/strategy_factsheet_test.py::'
    'test_rolling_benchmark_alpha_uses_lagged_displayed_betas_and_matching_span',
    'portfolio/reports/tests/strategy_factsheet_test.py::'
    'test_portfolio_betas_use_point_in_time_ewma_mean_adjustment',
    'portfolio/reports/tests/large_universe_factsheet_test.py::'
    'test_large_universe_top_bottom_contributors',
    'tests/test_reporting_conventions.py::test_total_return_is_frequency_invariant',
    'plots/tests/utils_test.py::test_group_colors_include_unobserved_categories_without_future_warning',
    'plots/tests/legend_missing_ratios_test.py::'
    'test_ratio_helper_uses_post_inception_windows_and_defines_all_missing',
    'portfolio/reports/tests/strategy_factsheet_test.py::test_future_numerical_regression',
])
def test_mixed_module_numerical_contracts_keep_full_matrix(nodeid):
    assert routing.suite_for('src/qis/' + nodeid) == 'numerical'


def test_explicit_rendering_selectors_name_existing_tests():
    source = PLUGIN.parents[1] / 'src' / 'qis'
    for filename in routing.PLOT_FILES:
        assert (source / 'plots' / 'tests' / filename).is_file()
    for filename, selected in routing.FACTSHEET_TESTS.items():
        tree = ast.parse((source / filename).read_text(encoding='utf-8'))
        functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        assert selected <= functions, (
            f'{filename}: stale rendering selectors {selected - functions}')


@pytest.mark.parametrize(('nodeid', 'suite'), [
    ('perfstats/tests/returns_numpy_test.py::test_returns[float64]', 'numerical'),
    ('plots/tests/plot_smoke_test.py::test_exported_plot_function_draws[plot_df_table]', 'plots'),
    ('plots/tests/plot_smoke_test.py::'
     'test_exported_plot_function_draws[qis/docs/table.md]', 'plots'),
    ('portfolio/reports/tests/large_universe_factsheet_test.py::'
     'test_large_universe_appendices[no-groups-risk]', 'factsheets'),
    ('tests/test_reporting_conventions.py::test_panel_frequency_labels[DAILY]', 'factsheets'),
    ('tests/test_reporting_goldens.py::test_multi_asset_report_layout[MONTHLY]', 'factsheets'),
])
def test_partition_is_stable_across_checkout_and_wheel_paths(nodeid, suite):
    for prefix in ('src/qis/', 'C:\\site-packages\\qis\\', '/tmp/checkout/src/qis/'):
        assert routing.suite_for(prefix + nodeid) == suite
        assert routing.shard_for(prefix + nodeid, 2) == routing.shard_for(nodeid, 2)


def _select(nodeids, suite='all', shard=0, count=1):
    options = {'--ci-suite': suite, '--ci-shard': shard, '--ci-shards': count}
    config = SimpleNamespace(
        getoption=options.__getitem__,
        hook=SimpleNamespace(pytest_deselected=lambda **kwargs: None),
    )
    items = [SimpleNamespace(nodeid=nodeid) for nodeid in nodeids]
    routing.pytest_collection_modifyitems(config, items)
    return [item.nodeid for item in items]


def test_partitions_cover_every_item_once_and_default_pytest_stays_whole():
    nodeids = [
        'src/qis/perfstats/tests/perf_stats_test.py::test_total_return',
        'src/qis/plots/tests/plot_smoke_test.py::test_exported_plot_function_draws[plot_df_table]',
    ] + [
        f'src/qis/tests/test_reporting_goldens.py::test_multi_asset_report_layout[{freq}]'
        for freq in ('DAILY', 'WEEKLY', 'MONTHLY', 'QUARTERLY')
    ]
    parts = [_select(nodeids, 'numerical'), _select(nodeids, 'plots')]
    parts += [_select(nodeids, 'factsheets', shard, 2) for shard in range(2)]
    flattened = [nodeid for part in parts for nodeid in part]
    assert sorted(flattened) == sorted(nodeids)
    assert len(flattened) == len(set(flattened))
    assert _select(nodeids) == nodeids


def test_numerical_shards_cover_the_unsplit_suite_once():
    nodeids = [f'src/qis/perfstats/tests/returns_numpy_test.py::test_return[{n}]'
               for n in range(200)]
    nodeids.append('src/qis/plots/tests/plot_smoke_test.py::test_exported_plot_function_draws')
    whole = _select(nodeids, 'numerical')
    parts = [_select(nodeids, 'numerical', shard, 2) for shard in range(2)]
    assert all(parts)
    assert sorted(parts[0] + parts[1]) == sorted(whole)
    assert not set(parts[0]) & set(parts[1])
    assert _select(nodeids) == nodeids


def _ci_jobs():
    path = PLUGIN.with_name('workflows').joinpath('ci.yml')
    return yaml.safe_load(path.read_text(encoding='utf-8'))['jobs']


def test_every_supported_matrix_cell_runs_both_numerical_shards():
    job = _ci_jobs()['test']
    cells = job['strategy']['matrix']['include']
    expected = {('ubuntu-latest', version) for version in ('3.11', '3.12', '3.13', '3.14')}
    expected |= {('windows-latest', '3.12'), ('macos-latest', '3.12')}
    assert Counter((cell['os'], cell['python-version'], cell['shard']) for cell in cells) == {
        (os, version, shard): 1 for os, version in expected for shard in range(2)
    }
    command = next(step['run'] for step in job['steps'] if step.get('name') ==
                   'Run numerical and mixed-contract tests')
    assert '--ci-shard ${{ matrix.shard }} --ci-shards 2' in command


def test_extras_and_installed_wheel_keep_complete_numerical_coverage():
    jobs = _ci_jobs()
    extras = jobs['test-extras']
    assert extras['strategy']['matrix']['shard'] == [0, 1]
    execute = next(step for step in extras['steps'] if step.get('name') ==
                   'Run numerical tests with coverage')
    assert '--ci-shard ${{ matrix.shard }} --ci-shards 2' in execute['run']
    assert execute['env']['COVERAGE_FILE'].endswith('.coverage.numerical-${{ matrix.shard }}')
    upload = next(step['with'] for step in extras['steps'] if step.get('name') ==
                  'Retain numerical coverage')
    assert upload['name'] == 'ci-coverage-numerical-${{ matrix.shard }}'
    cells = jobs['wheel-tests']['strategy']['matrix']['include']
    assert Counter((cell['shard'], cell['shards']) for cell in cells
                   if cell['suite'] == 'numerical') == {
        (0, 2): 1, (1, 2): 1,
    }
    assert set(jobs['coverage']['needs']) == {'test-extras', 'rendering'}


@pytest.mark.parametrize(('suite', 'shard', 'count'), [
    ('plots', -1, 2), ('factsheets', 2, 2), ('plots', 0, 0),
    ('numerical', 2, 2), ('all', 1, 2),
])
def test_invalid_shards_fail_instead_of_silently_skipping_tests(suite, shard, count):
    with pytest.raises(pytest.UsageError):
        _select([], suite, shard, count)

"""CI partition contracts keep numerical tests in mixed reporting modules on every platform."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


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
    ('plots/tests/plot_smoke_test.py::test_exported_plot_function_draws[plot_df_table]', 'plots'),
    ('plots/tests/plot_smoke_test.py::'
     'test_exported_plot_function_draws[qis/docs/table.md]', 'plots'),
    ('portfolio/reports/tests/large_universe_factsheet_test.py::'
     'test_large_universe_appendices[no-groups-risk]', 'factsheets'),
    ('tests/test_reporting_conventions.py::test_panel_frequency_labels[DAILY]', 'factsheets'),
    ('tests/test_reporting_goldens.py::test_multi_asset_report_layout[MONTHLY]', 'factsheets'),
])
def test_rendering_partition_is_stable_across_checkout_and_wheel_paths(nodeid, suite):
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


@pytest.mark.parametrize(('suite', 'shard', 'count'), [
    ('plots', -1, 2), ('factsheets', 2, 2), ('plots', 0, 0),
    ('numerical', 0, 2), ('all', 1, 2),
])
def test_invalid_shards_fail_instead_of_silently_skipping_tests(suite, shard, count):
    with pytest.raises(pytest.UsageError):
        _select([], suite, shard, count)

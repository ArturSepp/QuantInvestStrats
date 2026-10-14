"""Opt-in CI partitioning; ordinary pytest and installed-wheel runs still run everything.

Only named rendering workloads leave the full Python/OS matrix. Mixed modules default to
the numerical suite so new accounting, estimation and reporting-contract tests keep matrix
coverage. Each collected item belongs to exactly one suite and one stable shard.
"""

import hashlib

import pytest


PLOT_FILES = {
    'boxplot_test.py', 'errorbar_label_alignment_test.py', 'gantt_data_history_test.py',
    'lineplot_xy_legend_test.py', 'plot_smoke_test.py', 'scatter_annotations_test.py',
    'scatter_overlays_test.py', 'stackplot_annotation_options_test.py',
    'stackplot_missing_total_test.py', 'stackplot_nullable_input_test.py',
    'table_data_colors_test.py',
}
FACTSHEET_TESTS = {
    'portfolio/reports/tests/large_universe_factsheet_test.py': {
        'test_large_universe_summary', 'test_large_universe_appendices',
        'test_large_universe_facade', 'test_large_universe_benchmark_without_brinson',
        'test_large_universe_twenty_per_tail',
    },
    'portfolio/reports/tests/strategy_factsheet_test.py': {
        'test_generate_strategy_signal_factsheet_by_instrument_preserves_figures',
        'test_signal_instrument_batch_bounds_registered_pages',
        'test_generate_strategy_benchmark_factsheet_plt_preserves_figures',
        'test_factsheet_generators_preserve_caller_figures_after_later_plot_error',
        'test_strategy_benchmark_factsheet_renders_history_before_recent_default',
        'test_strategy_and_multi_asset_factsheets_use_recent_ra_start_date',
        'test_long_history_warns_limits_summary_and_appends_full_heatmap',
        'test_full_history_heatmap_layout_adapts_to_year_rows',
        'test_more_than_ten_groups_use_strategy_only_summary_tables',
        'test_strategy_benchmark_factsheet_limits_grouped_ra_tables',
        'test_multi_strategy_factsheet_limits_grouped_sharpe',
        'test_wide_ra_table_uses_multiline_short_names',
        'test_brinson_page_uses_requested_layout_titles_and_regime_backgrounds',
        'test_generate_strategy_benchmark_factsheet_labels_brinson_trading_cost_basis',
    },
    'tests/test_reporting_conventions.py': {
        'test_panel_frequency_labels', 'test_portfolio_report_labels',
        'test_facade_returns_figures', 'test_facade_writes_pdf',
    },
}


def canonical_nodeid(nodeid):
    """Remove checkout/site-packages prefixes, including Windows path separators."""
    return nodeid.replace('\\', '/').rsplit('qis/', 1)[-1]


def suite_for(nodeid):
    """Assign explicit rendering cases; keep every other test in the full matrix."""
    path, _, test = canonical_nodeid(nodeid).partition('::')
    name = test.split('[', 1)[0]
    if path.startswith('plots/tests/') and path.rsplit('/', 1)[-1] in PLOT_FILES:
        return 'plots'
    if path in {
        'tests/test_reporting_goldens.py',
        'portfolio/reports/tests/brinson_var_format_test.py',
    } or name in FACTSHEET_TESTS.get(path, ()):
        return 'factsheets'
    return 'numerical'


def shard_for(nodeid, count):
    """Use a stable digest rather than Python's process-randomised hash."""
    digest = hashlib.sha256(canonical_nodeid(nodeid).encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big') % count


def pytest_addoption(parser):
    group = parser.getgroup('qis CI suites')
    group.addoption('--ci-suite', choices=('all', 'numerical', 'plots', 'factsheets'),
                    default='all', help='Opt-in CI suite; normal pytest remains unpartitioned.')
    group.addoption('--ci-shard', type=int, default=0, help='Zero-based rendering shard.')
    group.addoption('--ci-shards', type=int, default=1, help='Number of rendering shards.')


def pytest_collection_modifyitems(config, items):
    suite = config.getoption('--ci-suite')
    shard = config.getoption('--ci-shard')
    count = config.getoption('--ci-shards')
    if count < 1 or not 0 <= shard < count:
        raise pytest.UsageError('CI shard must satisfy 0 <= --ci-shard < --ci-shards')
    if suite in ('all', 'numerical') and (shard != 0 or count != 1):
        raise pytest.UsageError('Only rendering suites may be sharded')
    if suite == 'all':
        return
    selected, deselected = [], []
    for item in items:
        target = selected if (suite_for(item.nodeid) == suite
                              and shard_for(item.nodeid, count) == shard) else deselected
        target.append(item)
    config.hook.pytest_deselected(items=deselected)
    items[:] = selected

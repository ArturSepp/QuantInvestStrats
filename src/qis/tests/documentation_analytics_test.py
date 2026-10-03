"""Repository-only checks for analytics coverage, output safety and bundle integrity."""

import copy
import json
import runpy
import socket
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RUNNER = ROOT / 'tools/docs_analytics/run.py'
if not RUNNER.is_file():
    pytest.skip('Documentation analytics tooling is not shipped in wheels.',
                allow_module_level=True)
RUN = runpy.run_path(str(RUNNER))
MANIFEST = json.loads((RUNNER.parent / 'manifest.json').read_text(encoding='utf-8'))


def test_registered_images_cover_documentation():
    assert len(RUN['load_manifest']()['assets']) == 73


def test_fx_hedging_examples_have_independent_numerical_checks():
    """The chapter's hedge decisions, lagged payoffs and performance agree with references."""
    from tools.docs_analytics import handbook

    result = handbook.fx_hedging_results(MANIFEST['producers']['handbook']['parameters'])
    assert all(result['checks'].values())
    assert result['navs'].index[0].strftime('%Y-%m-%d') == '2010-12-31'
    assert len(result['performance']) == 24  # four assets, six reference-currency strategies
    assert result['hedges'].dropna().ge(0.0).all().all()
    assert result['hedges'].dropna().le(1.0).all().all()


def test_fx_hedging_examples_reject_same_period_hedge_execution(monkeypatch):
    """An independently reconstructed forward payoff detects a deliberately unlagged hedge."""
    import pandas as pd
    import qis
    from tools.docs_analytics import handbook

    original = qis.compute_performance_of_local_ccy_asset_in_reference_ccy

    def remove_decision_lag(**kwargs):
        """Inject the next hedge decision so the canonical shift applies the current one."""
        if isinstance(kwargs['hedge_ratio'], pd.Series):
            kwargs['hedge_ratio'] = kwargs['hedge_ratio'].shift(-1)
        return original(**kwargs)

    monkeypatch.setattr(qis, 'compute_performance_of_local_ccy_asset_in_reference_ccy',
                        remove_decision_lag)
    result = handbook.fx_hedging_results(MANIFEST['producers']['handbook']['parameters'])
    assert not result['checks']['lagged_forward_payoff']


@pytest.fixture
def frozen_empirical(tmp_path, monkeypatch):
    """Prepare public aggregate statistics and eight synthetic stand-ins for preview bytes."""
    import hashlib
    from tools.docs_analytics import cash_rate_case_study

    monkeypatch.setattr(cash_rate_case_study, 'ROOT', tmp_path)
    spec = copy.deepcopy(MANIFEST['producers']['cash_rate_case_study'])
    for item in spec['parameters']['frozen_images']:
        path = tmp_path / 'docs/images' / item['filename']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'preview stand-in for producer hash validation')
        item['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return cash_rate_case_study, spec


def test_frozen_empirical_hash_contract(frozen_empirical):
    """Reject any alteration to an already reviewed historical preview."""
    module, spec = frozen_empirical
    first = module.ROOT / 'docs/images' / spec['parameters']['frozen_images'][0]['filename']
    first.write_bytes(b'changed preview')
    with pytest.raises(ValueError, match='hash mismatch'):
        module.produce(spec)


@pytest.mark.parametrize('defect', ['duplicate', 'r_squared', 'rmse', 'nonfinite'])
def test_frozen_empirical_summary_contract(frozen_empirical, defect):
    """Reject an incomplete or mathematically inconsistent public aggregate record."""
    module, spec = frozen_empirical
    records = spec['parameters']['statistics']
    if defect == 'duplicate':
        records.append(records[0])
    elif defect == 'r_squared':
        records[0]['R_squared'] = 1.1
    elif defect == 'rmse':
        records[0]['RMSE_bp'] = 0.0
    else:
        records[0]['Bias_bp'] = float('nan')
    with pytest.raises(ValueError):
        module.produce(spec)


def test_frozen_empirical_produces_public_inputs_only(frozen_empirical):
    """Complete offline generation requires no private observation panel."""
    module, spec = frozen_empirical
    result = module.produce(spec)
    assert len(result['figures']) == 8
    assert len(result['tables']['statistics']) == 24
    assert all(result['checks'].values())
    assert result['summary']['raw_vendor_data_distributed'] is False


@pytest.fixture
def frozen_hedged_indices(tmp_path, monkeypatch):
    """Prepare aggregate records and stand-in bytes without reading private inputs."""
    import hashlib
    from tools.docs_analytics import hedged_index_case_study

    monkeypatch.setattr(hedged_index_case_study, 'ROOT', tmp_path)
    spec = copy.deepcopy(MANIFEST['producers']['hedged_index_case_study'])
    for item in spec['parameters']['frozen_images']:
        path = tmp_path / 'docs/images' / item['filename']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'hedged-index preview stand-in for hash checks')
        item['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hedged_index_case_study, spec


def test_hedged_empirical_produces_public_inputs_only(frozen_hedged_indices):
    """Reproduce aggregate records offline without claiming to refit vendor history."""
    module, spec = frozen_hedged_indices
    result = module.produce(spec)
    assert len(result['figures']) == 21
    assert len(result['tables']['statistics']) == 42
    assert all(result['checks'].values())
    assert result['summary']['raw_vendor_data_distributed'] is False


@pytest.mark.parametrize('defect', [
    'duplicate', 'r_squared', 'rmse', 'nonfinite', 'moment_identity', 'cagr_identity',
    'sample_support', 'image_hash', 'missing_image',
])
def test_hedged_empirical_rejects_invalid_records(frozen_hedged_indices, defect):
    """Reject corrupt figures, sample support and inconsistent numerical aggregates."""
    module, spec = frozen_hedged_indices
    records = spec['parameters']['statistics']
    recent = next(row for row in records if row['Window'] == module.RECENT)
    if defect == 'duplicate':
        records.append(records[0])
    elif defect == 'r_squared':
        records[0]['R_squared'] = 1.1
    elif defect == 'rmse':
        records[0]['RMSE_monthly_bp'] = 0.0
    elif defect == 'nonfinite':
        records[0]['Beta'] = float('nan')
    elif defect == 'moment_identity':
        records[0]['Tracking_error_pa_bp'] += 1.0
    elif defect == 'cagr_identity':
        records[0]['CAGR_difference_bp'] += 1.0
    elif defect == 'sample_support':
        recent['Months'] = 68
    elif defect == 'image_hash':
        first = spec['parameters']['frozen_images'][0]['filename']
        (module.ROOT / 'docs/images' / first).write_bytes(b'changed figure')
    else:
        spec['parameters']['frozen_images'].pop()
    with pytest.raises((ValueError, AssertionError)):
        module.produce(spec)


@pytest.fixture
def frozen_unhedged_indices(tmp_path, monkeypatch):
    """Prepare both-FX aggregate records and synthetic stand-ins for preview bytes."""
    import hashlib
    from tools.docs_analytics import unhedged_index_case_study

    monkeypatch.setattr(unhedged_index_case_study, 'ROOT', tmp_path)
    spec = copy.deepcopy(MANIFEST['producers']['unhedged_index_case_study'])
    for item in spec['parameters']['frozen_images']:
        path = tmp_path / 'docs/images' / item['filename']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'unhedged-index preview stand-in for hash validation')
        item['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return unhedged_index_case_study, spec


def test_unhedged_empirical_produces_public_inputs_only(frozen_unhedged_indices):
    """Validate both FX inputs offline without reading private vendor histories."""
    module, spec = frozen_unhedged_indices
    result = module.produce(spec)
    assert len(result['figures']) == 22
    assert len(result['tables']['statistics']) == 44
    assert len(result['tables']['pairs']) == 22
    assert len(result['tables']['currency_diagnostics']) == 3
    assert all(result['checks'].values())
    assert result['summary']['independent_WMR_replication'] is False
    assert result['summary']['raw_vendor_data_distributed'] is False


@pytest.mark.parametrize('defect', [
    'duplicate', 'r_squared', 'holdout_r_squared', 'holdout_te', 'nonfinite',
    'moment_identity', 'cagr_identity', 'sample_support', 'circular_anchor',
    'image_hash', 'missing_image', 'currency_support', 'pair_identity',
])
def test_unhedged_empirical_rejects_invalid_records(frozen_unhedged_indices, defect):
    """Reject corrupt evidence, both-FX summaries and circular diagnostic anchors."""
    module, spec = frozen_unhedged_indices
    parameters = spec['parameters']
    records = parameters['statistics']
    recent = next(row for row in records if row['Window'] == module.RECENT)
    if defect == 'duplicate':
        records.append(records[0])
    elif defect == 'r_squared':
        records[0]['R_squared'] = 1.1
    elif defect == 'holdout_r_squared':
        records[0]['Index_FX_holdout_R_squared'] = 1.1
    elif defect == 'holdout_te':
        records[0]['Index_FX_holdout_TE_pa_bp'] = -1.0
    elif defect == 'nonfinite':
        records[0]['Beta'] = float('nan')
    elif defect == 'moment_identity':
        records[0]['Tracking_error_pa_bp'] += 1.0
    elif defect == 'cagr_identity':
        records[0]['CAGR_difference_bp'] += 1.0
    elif defect == 'sample_support':
        recent['Months'] = 68
    elif defect == 'circular_anchor':
        pair = parameters['pairs'][0]
        pair['Index_FX_anchor_base'] = pair['Base_ticker']
        pair['Index_FX_anchor_target'] = pair['Observed_ticker']
    elif defect == 'image_hash':
        first = parameters['frozen_images'][0]['filename']
        (module.ROOT / 'docs/images' / first).write_bytes(b'changed figure')
    elif defect == 'missing_image':
        parameters['frozen_images'].pop()
    elif defect == 'currency_support':
        parameters['currency_diagnostics'][0]['Months'] = 68
    else:
        records[0]['Base_name'] = 'Different index'
    with pytest.raises((ValueError, AssertionError)):
        module.produce(spec)


@pytest.mark.parametrize('source', [
    '![caption](images/new.png)',
    '<img alt="caption" src="images/new.png">',
    '<img alt=caption src=images/new.png>',
    '\x60\x60\x60{image} images/new.png\n:alt: caption\n\x60\x60\x60',
    ':::{figure} images/new.png\ncaption\n:::',
    '![caption][new]\n\n[new]: images/new.png',
    '![new][]\n\n[new]: images/new.png',
    '![new]\n\n[new]: images/new.png',
])
def test_unregistered_image_syntax_fails_coverage(tmp_path, source):
    (tmp_path / 'docs').mkdir()
    (tmp_path / 'docs/page.md').write_text(source, encoding='utf-8')
    with pytest.raises(ValueError, match='unregistered'):
        RUN['check_coverage']({'assets': [], 'non_analytics': []}, tmp_path)


def test_code_examples_and_comments_are_not_images():
    source = ('\x60\x60\x60\x60markdown\n\x60\x60\x60{image} ignored.png\n'
              '\x60\x60\x60\n![example](ignored.png)\n\x60\x60\x60\x60\n'
              '<!-- ![hidden](ignored.png) -->')
    assert RUN['image_references'](source) == []


@pytest.mark.parametrize('field,value,match', [
    ('producer', 'missing', 'Unknown producer'),
    ('path', '../escape.png', 'Unsafe relative'),
    ('path', 'other/image.png', 'allowlist'),
])
def test_invalid_asset_is_rejected(tmp_path, field, value, match):
    manifest = copy.deepcopy(MANIFEST)
    manifest['assets'][0][field] = value
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match=match):
        RUN['load_manifest'](path)


@pytest.mark.parametrize('relative', ['../escape', 'C:/escape', r'C:\escape', '/escape', 'a/../b'])
def test_nonportable_output_names_are_rejected(relative):
    with pytest.raises(ValueError):
        RUN['relative_path'](relative)


def test_output_boundary_rejects_overwrite_repo_and_onedrive(tmp_path, monkeypatch):
    local = tmp_path / 'AgentWork'
    local.mkdir()
    monkeypatch.setenv('AGENT_LOCAL_ROOT', str(local))
    root = local / 'source'
    root.mkdir()
    valid = local / 'task' / 'bundle'
    assert RUN['output_boundary'](valid, root) == valid.resolve()
    for rejected in (local, tmp_path / 'outside', root / 'bundle', local / 'OneDrive/task'):
        with pytest.raises(ValueError):
            RUN['output_boundary'](rejected, root)
    valid.mkdir(parents=True)
    with pytest.raises(ValueError, match='overwrite'):
        RUN['output_boundary'](valid, root)


def test_missing_output_root_is_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv('AGENT_LOCAL_ROOT', raising=False)
    with pytest.raises(ValueError, match='AGENT_LOCAL_ROOT'):
        RUN['output_boundary'](tmp_path / 'bundle')


def test_output_link_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv('AGENT_LOCAL_ROOT', str(tmp_path))
    link = tmp_path / 'linked'
    try:
        link.symlink_to(tmp_path / 'elsewhere', target_is_directory=True)
    except OSError:
        pytest.skip('This host does not permit creating symlinks')
    with pytest.raises(ValueError, match='Linked'):
        RUN['output_boundary'](link / 'bundle')


def test_offline_guard_restores_network_functions():
    original = socket.create_connection
    with RUN['offline'](), pytest.raises(RuntimeError, match='offline'):
        socket.create_connection(('example.invalid', 443))
    assert socket.create_connection is original


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    from PIL import Image
    monkeypatch.setenv('AGENT_LOCAL_ROOT', str(tmp_path))
    root, output = tmp_path / 'source', tmp_path / 'bundle'
    root.mkdir()
    output.mkdir()
    manifest = copy.deepcopy(MANIFEST)
    for rel in RUN['expected_files'](manifest):
        path = output / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == '.png':
            img = Image.new('RGB', (120, 120), 'white')
            img.putpixel((0, 0), (0, 0, 0))
            img.save(path)
        else:
            path.write_text('name,value\nexample,1\n', encoding='utf-8')
    record = {
        'schema_version': 1, 'status': 'complete', 'manifest': manifest,
        'source': RUN['source_fingerprint'](root),
        'outputs': {rel: RUN['file_record'](output / rel)
                    for rel in RUN['expected_files'](manifest)},
        'producers': {name: {'checks': dict.fromkeys(spec['checks'], True)}
                      for name, spec in manifest['producers'].items()},
    }
    RUN['write_json'](output / 'analytics_manifest.json', record)
    return output, manifest, root, record


def test_complete_bundle_validates(bundle):
    output, manifest, root, record = bundle
    assert RUN['validate_bundle'](output, manifest, root) == record


@pytest.mark.parametrize('defect', ['missing_image', 'edited_table', 'extra_file',
                                    'failed_check', 'missing_producer', 'source_drift'])
def test_bundle_defects_are_rejected(bundle, defect):
    output, manifest, root, record = bundle
    if defect == 'missing_image':
        (output / 'images/multi_asset.png').unlink()
    elif defect == 'edited_table':
        (output / 'tables/gallery/performance.csv').write_text('x,y\n2,3\n', encoding='utf-8')
    elif defect == 'extra_file':
        (output / 'unexpected.txt').write_text('unexpected', encoding='utf-8')
    elif defect == 'failed_check':
        record['producers']['gallery']['checks']['finite_positive_navs'] = False
    elif defect == 'missing_producer':
        del record['producers']['gallery']
    else:
        (root / 'pyproject.toml').write_text('changed source', encoding='utf-8')
    RUN['write_json'](output / 'analytics_manifest.json', record)
    with pytest.raises(ValueError):
        RUN['validate_bundle'](output, manifest, root)



def test_units_reference_detects_a_wrong_nav():
    from types import SimpleNamespace
    import pandas as pd
    check = runpy.run_path(str(RUNNER.parent / 'gallery.py'))['check_units_identity']
    prices = pd.DataFrame({'first': [50.0, 52.0, 48.0], 'second': [50.0, 49.0, 51.0]})
    portfolio = SimpleNamespace(
        prices=prices, units=pd.DataFrame(1.0, index=prices.index, columns=prices.columns),
        nav=pd.Series([100.0, 101.0, 99.0]),
    )
    assert check(portfolio) == 0.0
    portfolio.nav.iloc[1] = 102.0  # deliberate defect in the test fixture, not production code
    with pytest.raises(AssertionError):
        check(portfolio)


def test_model_displayed_values_have_independent_checks(monkeypatch):
    import matplotlib.pyplot as plt
    adapter = runpy.run_path(str(RUNNER.parent / 'model_layer.py'))
    example = adapter['example']
    original = example.plot_return_bridge

    def altered_bridge(attribution):
        figure = original(attribution)
        figure.axes[0].patches[2].set_height(0.25)  # corrupt the displayed risk-layer alpha
        return figure

    monkeypatch.setattr(example, 'plot_return_bridge', altered_bridge)
    try:
        with RUN['offline'](), pytest.raises(AssertionError):
            adapter['produce'](MANIFEST['producers']['model_layer'])
    finally:
        plt.close('all')

def test_focused_preview_preserves_values_and_positional_dates():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from tools.docs_analytics.style import gallery_preview

    fig, axes = plt.subplots(2, 2)
    nav, table_ax, drawdown, exposure = axes.flat
    dates = pd.date_range('2018-12-31', periods=3, freq='YE')
    nav.plot(dates, [1.0, 1.1, 1.2], label='Example: statistics')
    nav.set_title('Cumulative performance')
    drawdown.plot(dates, [0.0, -0.1, -0.05], label='Example, max dd=-10%')
    drawdown.set_title('Running Drawdowns')
    exposure.stackplot(range(6), [0.5, 0.6, 0.5, 0.5, 0.6, 0.5], [0.5, 0.4, 0.5, 0.5, 0.4, 0.5],
                       labels=['First: mean', 'Second: mean'])
    exposure.set_xticks(range(6), ['', 'Dec-18', '', 'Dec-19', '', 'Dec-20'])
    exposure.set_title('Exposures (ME-freq)')
    values = [['Example', '20%', '9.54%', '12.00%', '0.80', '-10%']]
    table_ax.table(
        cellText=values,
        colLabels=['Series', 'Total return', 'P.a. return', 'An. vol', 'Sharpe (rf=0)', 'Max DD'],
    )
    table_ax.set_title('RA performance table')
    try:
        gallery_preview(fig, title='Fixture', detail='Exposures (')
        np.testing.assert_array_equal(nav.lines[0].get_ydata(), [1.0, 1.1, 1.2])
        np.testing.assert_array_equal(drawdown.lines[0].get_ydata(), [0.0, -0.1, -0.05])
        assert [label.get_text() for label in exposure.get_xticklabels()] == ['Dec-18', 'Dec-20']
        cells = table_ax.tables[0].get_celld()
        assert [cells[1, column].get_text().get_text() for column in range(6)] == values[0]
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for cell in cells.values():
            text_box = cell.get_text().get_window_extent(renderer)
            cell_box = cell.get_window_extent(renderer)
            assert text_box.width <= cell_box.width
            assert text_box.height <= cell_box.height
    finally:
        plt.close(fig)


def test_focused_table_rejects_changed_column_meanings():
    import matplotlib.pyplot as plt
    from tools.docs_analytics.style import _compact_table

    fig, ax = plt.subplots()
    ax.table(cellText=[['Example', '1', '2', '3', '4', '5']],
             colLabels=['Series', 'Turnover', 'Return', 'Risk', 'Sharpe', 'Max DD'])
    try:
        with pytest.raises(ValueError, match='meanings'):
            _compact_table(ax)
    finally:
        plt.close(fig)


def test_publication_copies_only_previews_and_detects_edits(bundle):
    output, manifest, root, _ = bundle
    publisher = runpy.run_path(str(RUNNER.parent / 'publish.py'))
    published = publisher['publish_bundle'](output, root, manifest)
    assert publisher['verify_published'](root, manifest) == published
    assert not (root / 'tables').exists()
    assert {path.relative_to(root).as_posix() for path in root.rglob('*') if path.is_file()} == {
        *(asset['path'] for asset in manifest['assets']), 'docs/images/analytics_manifest.json',
    }
    image = root / manifest['assets'][0]['path']
    image.write_bytes(b'altered preview')
    with pytest.raises((ValueError, OSError)):
        publisher['verify_published'](root, manifest)


def test_publication_rolls_back_after_a_write_failure(bundle, monkeypatch):
    output, manifest, root, _ = bundle
    publisher = runpy.run_path(str(RUNNER.parent / 'publish.py'))
    originals = {}
    for asset in manifest['assets']:
        path = root / asset['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        originals[path] = ('previous ' + asset['id']).encode()
        path.write_bytes(originals[path])
    replace = publisher['os'].replace
    calls = []

    def fail_second(source, target):
        calls.append(target)
        if len(calls) == 2:
            raise OSError('deliberate publication interruption')
        replace(source, target)

    monkeypatch.setattr(publisher['os'], 'replace', fail_second)
    with pytest.raises(OSError, match='deliberate'):
        publisher['publish_bundle'](output, root, manifest)
    assert all(path.read_bytes() == content for path, content in originals.items())
    assert not (root / 'docs/images/analytics_manifest.json').exists()


def test_publication_validates_before_touching_the_destination(bundle):
    output, manifest, root, _ = bundle
    publisher = runpy.run_path(str(RUNNER.parent / 'publish.py'))
    (output / 'tables/gallery/performance.csv').write_text('edited', encoding='utf-8')
    with pytest.raises(ValueError):
        publisher['publish_bundle'](output, root, manifest)
    assert not (root / 'docs').exists()
